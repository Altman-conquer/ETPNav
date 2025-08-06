import traceback
import gc
import os
import sys
import random
import uuid
import warnings
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List
import jsonlines
from loguru import logger as mylogger

import cv2
import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, poll_checkpoint_folder

from habitat_extensions.utils import navigator_video_frame
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

from .waypoint_pred.utils import split_instruction, query_qwen, BALL_COLORS, instruction_to_token, add_text_to_frame

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

mylogger.add("./data/logs/info.log", level="INFO", backtrace=True, diagnose=True)
query_qwen_pool = ThreadPoolExecutor(max_workers=8)

torch.autograd.set_detect_anomaly(True)

@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len)  # * 0.97 transfered gt path got 0.96 spl

        self.graph_map_memory = {} # {'scene_name': GraphMap}

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],  # Back
                'Down': [-math.pi / 2, 0 + shift, 0],  # Down
                'Front': [0, 0 + shift, 0],  # Front
                'Right': [0, math.pi / 2 + shift, 0],  # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],  # Left
                'Up': [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
            self,
            config: Config,
            load_from_ckpt: bool,
            observation_space: Space,
            action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        # cwp_fn = 'data/wp_pred/semantic_check_val_best_avg_wayscore'
        # cwp_fn = 'data/wp_pred/rgb_semantic_check_val_best_avg_wayscore'
        assert self.config.MODEL.task_type == 'r2r'
        self.waypoint_predictor.load_state_dict(
            torch.load(cwp_fn, map_location=torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS, 'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                                  output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")))
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                                                        device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                                                                device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params / 1e6:.2f} MB. Trainable: {params_t / 1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path': self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles': batch_angles[i],
                    'distances': batch_distances[i],
                    'candidate_length': candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"] * self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError

        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts, nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12 - np.sum(cand_idxes))

            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }

    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0] * len(ghost_vp_ids)

            if gmap.merge_map:
                node_vp_ids_int = [int(vp) for vp in node_vp_ids]
                node_vp_ids_int.sort()

                gmap_visited_masks = [0] + [0] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

                if len(node_vp_ids) > 0:
                    for index, vp in enumerate(node_vp_ids_int):
                        if index in gmap.visited_node:
                            gmap_visited_masks[1 + index] = 1

                    # for visited_node_index in gmap.visited_node:
                    #     try:
                    #         gmap_visited_masks[1 + visited_node_index] = 1
                    #     except Exception as e:
                    #         logger.error(f"Error in gmap_visited_masks: {e}, visited_node: {gmap.visited_node}, node_vp_ids: {node_vp_ids}")
                    #         logger.error(traceback.format_exc())
            else:
                gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j + 1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST

            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))

        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                        self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                            split=self.split, role=role
                        ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every = self.config.IL.log_every
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter - idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            # for param in self.policy.parameters():
            #     if param.grad is not None:
            #         param.grad.zero_()

            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                # self.rollout('train', ml_weight, sample_ratio)
                self.customize_rollout_memory_between_episodes('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward()  # self.loss.backward()
            self.scaler.step(self.optimizer)  # self.optimizer.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx + 1}/{interval}'})

        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
            self,
            checkpoint_path: str,
            writer: TensorboardWriter,
            checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path

        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],  # Back
                'Down': [-math.pi / 2, 0 + shift, 0],  # Down
                'Front': [0, 0 + shift, 0],  # Front
                'Right': [0, math.pi / 2 + shift, 0],  # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],  # Left
                'Up': [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False,  # unseen: 11006
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                    sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total, dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k, v in aggregated_states.items():
                v = torch.tensor(v * num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v) / total).item()
                aggregated_states[k] = v

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    def customize_eval_checkpoint(
            self,
            checkpoint_path: str,
            writer: TensorboardWriter,
            checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path

        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],  # Back
                'Down': [-math.pi / 2, 0 + shift, 0],  # Down
                'Front': [0, 0 + shift, 0],  # Front
                'Right': [0, math.pi / 2 + shift, 0],  # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],  # Left
                'Up': [math.pi / 2, 0 + shift, 0],  # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            # episodes_allowed=['417'],
            auto_reset_done=False,  # unseen: 11006
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
        # while len(self.stat_eps) < 200:
        #     self.customize_rollout('eval')
            self.customize_rollout_tod_down_map('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                    sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total, dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k, v in aggregated_states.items():
                v = torch.tensor(v * num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v) / total).item()
                aggregated_states[k] = v

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    def customize_eval(self):
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                    len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                    len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION', 'STEPS_TAKEN', 'COLLISIONS']
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]

        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()

        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")

            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))

        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS

        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_val_traj()

        with TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                # evaluate singe checkpoint
                # proposed_index = get_checkpoint_id(
                #     self.config.EVAL.CKPT_PATH_DIR
                # )
                # if proposed_index is not None:
                #     ckpt_idx = proposed_index
                # else:
                #     ckpt_idx = 0
                self.customize_eval_checkpoint(
                    self.config.EVAL.CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=self.get_ckpt_id(self.config.EVAL.CKPT_PATH_DIR),
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1  # eval start index
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.EVAL.CKPT_PATH_DIR, prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    if self.local_rank < 1:
                        logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self.customize_eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=self.get_ckpt_id(current_ckpt),
                    )

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}  # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict

        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k, v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori'] * self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos,
                               self.config.IL.loc_noise,
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                        'nodes': list(gmap.node_pos.values()),
                        'ghosts': list(gmap.ghost_aug_pos.values()),
                        'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,
                                                      self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())

    def customize_rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()

        # 指定episode进行测试
        # assert self.envs.num_envs <= 1
        current_episode_name = self.envs.call_at(0, "get_episode")['name']
        # if current_episode_name != '2azQ1b91cZZ-1213':
        #     mylogger.debug(f"skip episode: {current_episode_name}")
        #     return #

        # tmp_rgb = self.envs.call_at(0, "get_rgb_frame")
        # tmp_depth = self.envs.call_at(0, "get_depth_frame")
        # cv2.imwrite("tmp/depth.png",
        #             cv2.normalize(tmp_depth[:, :, 0], None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16))


        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        use_llm = True
        rgb_frame_history = [{} for _ in range(self.envs.num_envs)]
        current_instruction_id = [1 for _ in range(self.envs.num_envs)]
        vp_history = [{} for _ in range(self.envs.num_envs)]
        llm_result = [{} for _ in range(self.envs.num_envs)]
        different_vp = [[] for _ in range(self.envs.num_envs)]
        insist_vp = [[] for _ in range(self.envs.num_envs)]
        dir_name = ['' for _ in range(self.envs.num_envs)]

        if not os.path.exists(f'tmp'):
            os.makedirs(f'tmp')
        for i in range(self.envs.num_envs):
            dir_name[i] = self.envs.call_at(i, "get_episode")['name']

            if not os.path.exists(f'tmp/{dir_name[i]}'):
                os.makedirs(f'tmp/{dir_name[i]}')

        if use_llm:
            instructions = [instruction.instruction_text for instruction in self.envs.call(['get_instruction'] * self.envs.num_envs)]
            split_instructions: list[str] = [split_instruction(instruction) for instruction in instructions]
            split_atomic_instructions = [instructions.strip().split('\n') for instructions in split_instructions]
            mylogger.info(f"Instruction: {instructions}")
            mylogger.info(f"Split Instructions: {split_instructions}")

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos,
                               self.config.IL.loc_noise,
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            try:
                query_qwen_futures = []
                for i, gmap in enumerate(self.gmaps):
                    waypoint_color_mapping = self.envs.call_at(i, "add_object", {
                        "ghost_positions": gmap.new_ghost_pos})

                    new_ghost_nodes = list(gmap.new_ghost_pos.values())

                    llm_result[i]['waypoint_color_mapping'] = waypoint_color_mapping
                    llm_result[i]['new_ghost_nodes'] = new_ghost_nodes

                    rgb_frame = self.envs.call_at(i, "get_rgb_frame", {"ghost_positions": new_ghost_nodes, 'flat': True})

                    if stepk > 0: # mask image in the back of the agent
                        # width = rgb_frame.shape[1]
                        # black_width = int(width * 0.15)
                        #
                        # # 将左右两边各15%的区域设置为全黑色
                        # rgb_frame[:, :black_width, :] = 0
                        # rgb_frame[:, -black_width:, :] = 0

                        width = rgb_frame.shape[1]
                        black_width = int(width * 0.15)

                        # Remove the left and right 15% of the image
                        rgb_frame = rgb_frame[:, black_width:-black_width, :]

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    unique_id = uuid.uuid4()
                    output_file_name = f'tmp/{dir_name[i]}/{timestamp}.png'
                    cv2.imwrite(output_file_name, rgb_frame)
                    rgb_frame_history[i][stepk] = output_file_name
                    mylogger.info(f"save image to {output_file_name}")
                    # print(f"save image to {output_file_name}")

                    if False:
                        pass
                    # if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    #     pass
                    else:

                        ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        selected_waypoint_color = None
                        if ghost_vp in gmap.new_ghost_pos:
                            for color in waypoint_color_mapping.keys():
                                if ghost_vp == waypoint_color_mapping[color]['vp_id']:
                                    selected_waypoint_color = color
                                    break
                        llm_result[i]['selected_waypoint_color'] = selected_waypoint_color

                        if use_llm:
                            prompt = ''
                            if stepk > 0:
                                prompt += f"The first image is the previous step, and the second image is the current step. "
                                prompt += f"In the previous step, the instruction is instruction {current_instruction_id[i] if llm_result[i]['current_goal_finished'] == 'yes' else current_instruction_id[i]}, and you choose the waypoint {llm_result[i]['waypoint_chosen'] if llm_result[i]['waypoint_chosen'] != 'unknown' else llm_result[i]['etp_selected_waypoint_color']}. "
                            prompt += f"There are {len(gmap.new_ghost_pos)} waypoint in the image, their color are {BALL_COLORS[:len(new_ghost_nodes)]}, respectively. Be aware that some waypoint might not be visible in the current image. "
                            prompt += (f"Your instructions are {split_instructions[i]}\n"
                                       f"You are currently at instruction {current_instruction_id[i]}, which means your current goal is {split_atomic_instructions[i][current_instruction_id[i] - 1]}")
                            prompt += f"You need to tell whether the current goal is finished, "

                            if current_instruction_id[i] == len(split_atomic_instructions[i]):
                                prompt += (
                                    f"if it is, you don't need to choose any waypoint since all instructions are finished and output the Waypoint chosen in the answer to be 'unneeded', "
                                    f"if not, you need to tell the color of the waypoint you should go to that can finish instruction {current_instruction_id[i]}. "
                                )
                            else:
                                prompt += (
                                    f"if it is, you need to switch to the instruction {current_instruction_id[i] + 1} ({split_atomic_instructions[i][current_instruction_id[i]]}) and choose waypoint that can finish instruction {current_instruction_id[i] + 1}, "
                                    f"if not, you need to tell the color of the waypoint you should go to that can finish instruction {current_instruction_id[i]}. "
                                    f"you also need to check if the place described in the next instruction appears in the image, if so, you can also switch to the next instruction with current goal finished. "
                                )

                            # prompt += '\nIf the instruction requires you to stop after reaching something, such as "Stop once you reach the bed," you should keep moving until close enough to the target object. \n'
                            prompt += 'If the instruction requires you to stop after reaching something, such as "Stop once you reach the bed," you should keep moving until there are no closer waypoints to choose. '
                            prompt += "You need to output all waypoint color that you can see in current image, if you can't see any waypoint in the image, please output \"unknown\" when choosing waypoint. \n"

                            if selected_waypoint_color is None:
                                prompt += (
                                    f"In addition, if you cannot decide which waypoint to choose, output the Waypoint chosen in the answer to be 'unknown'."
                                )

                                if current_instruction_id[i] == len(split_atomic_instructions[i]):
                                    prompt += (
                                        f"Answer in format:"
                                        f"Analysis: <analysis>, "
                                        f"Current goal finished: <yes/no>, "
                                        f"Waypoint color visible: [red, green, blue, yellow, purple], " # 'red', 'green', 'blue', 'yellow', 'purple'
                                        f"Waypoint chosen: <waypoint color/unknown/unneeded>, "
                                    )
                                else:
                                    prompt += (
                                        f"Answer in format:"
                                        f"Analysis: <analysis>, "
                                        f"Current goal finished: <yes/no>, "
                                        f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                        f"Waypoint chosen: <waypoint color>, "
                                    )
                            else:
                                prompt += (
                                    f"In addition, the waypoint selected by an expert is in color {selected_waypoint_color}. "
                                    f"If the waypoint you choose is the same as the expert, please analyze why it is correct. "
                                    f"If it is not the same, then please analyze why it is wrong and give the waypoint you choose. "
                                    f"If you cannot decide which waypoint to choose, please choose the waypoint selected by the expert. "
                                )

                                if current_instruction_id[i] == len(split_atomic_instructions[i]):
                                    prompt += (
                                        f"Answer in format:"
                                        f"Analysis: <analysis>, "
                                        f"Current goal finished: <yes/no>, "
                                        f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                        f"Waypoint chosen: <waypoint color/unneeded>, "
                                    )
                                else:
                                    prompt += (
                                        f"Answer in format:"
                                        f"Analysis: <analysis>, "
                                        f"Current goal finished: <yes/no>, "
                                        f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                        f"Waypoint chosen: <waypoint color>, "
                                    )


                            if stepk > 0:
                                qwen_image_paths = [
                                    os.path.join(os.getcwd(), rgb_frame_history[i][stepk - 1]),
                                    os.path.join(os.getcwd(), output_file_name)]
                            else:
                                qwen_image_paths = [os.path.join(os.getcwd(), output_file_name)]
                            mylogger.info(f"Query Qwen, image_paths: {qwen_image_paths}, Prompt: \n {prompt}")

                            # result = query_qwen(qwen_image_paths, prompt)
                            query_qwen_futures.append(query_qwen_pool.submit(query_qwen, qwen_image_paths, prompt))


                        # self.envs.call_at(i, "remove_object")

                for i, gmap in enumerate(self.gmaps):
                    new_ghost_nodes = llm_result[i]['new_ghost_nodes']
                    selected_waypoint_color = llm_result[i]['selected_waypoint_color']

                    result = query_qwen_futures[i].result()

                    # parse the result
                    try:
                        analysis = result.split("Analysis: ")[1].split("Current goal finished:")[0]
                        current_goal_finished = result.split("Current goal finished: ")[1].split(",")[0].split('\n')[
                            0].lower().strip()
                    except Exception as e:
                        mylogger.error(f"error when split result: {result}")
                        mylogger.error(traceback.format_exc())
                        analysis = ''
                        current_goal_finished = 'no'

                    try:
                        waypoint_visible = \
                        result.split("Waypoint color visible: ")[1].split('\n')[0].strip().split("Waypoint chosen: ")[
                            0].strip(', [].').split(',')
                        waypoint_visible = [waypoint.strip() for waypoint in waypoint_visible]
                    except:
                        mylogger.error(
                            f"error when split waypoint_visible: {result.split('Waypoint color visible: ')[1]}")
                        waypoint_visible = []
                        mylogger.error(traceback.format_exc())

                    try:
                        waypoint_chosen = \
                        result.split("Waypoint chosen: ")[1].split(",")[0].split('\n')[0].lower().strip().split(' ')[0]
                    except Exception as e:
                        mylogger.error(f"error when split waypoint_chosen: {result.split('Waypoint chosen: ')[1]}")
                        mylogger.error(traceback.format_exc())
                        waypoint_chosen = 'unknown'

                    mylogger.info(
                        f"Analysis: {analysis}\n"
                        f"Current goal finished: {current_goal_finished}\n"
                        f"Waypoint put in scene: {BALL_COLORS[:len(new_ghost_nodes)]}\n"
                        f"Waypoint color visible: {waypoint_visible}\n"
                        f"Waypoint chosen: {waypoint_chosen}\n"
                        f"Waypoint chosen valid: {waypoint_chosen in waypoint_visible}\n"
                        f"Waypoint chosen same as expert: {waypoint_chosen == selected_waypoint_color}\n"
                    )
                    # print(f"Analysis: {analysis}")
                    # print(f"Current goal finished: {current_goal_finished}")
                    # print(f"Waypoint chosen: {waypoint_chosen}")
                    # print(f"Waypoint chosen same as expert: {waypoint_chosen == selected_waypoint_color}")

                    if current_goal_finished not in ['yes', 'no']:
                        mylogger.error(f"current_goal_finished should be yes or no, but got {current_goal_finished}")
                        current_goal_finished = 'no'

                    if waypoint_chosen not in BALL_COLORS[:len(new_ghost_nodes)] and waypoint_chosen not in ['unknown',
                                                                                                             'unneeded']:
                        mylogger.error(
                            f"waypoint_chosen should be in {BALL_COLORS[:len(new_ghost_nodes)]}, but got {waypoint_chosen}")
                        waypoint_chosen = 'unknown'
                    elif waypoint_chosen not in waypoint_visible and waypoint_chosen not in ['unknown', 'unneeded']:
                        mylogger.error(
                            f"waypoint_chosen should be in waypoint_visible {waypoint_visible}, but got {waypoint_chosen}")
                        waypoint_chosen = 'unknown'

                    # assert current_goal_finished in ['yes', 'no'], f"current_goal_finished should be yes or no, but got {current_goal_finished}"
                    # assert waypoint_chosen in BALL_COLORS[:len(new_ghost_nodes)] or waypoint_chosen in ['unknown', 'unneeded'], f"waypoint_chosen should be in {BALL_COLORS[:len(new_ghost_nodes)]}, but got {waypoint_chosen}"
                    if current_goal_finished == 'yes':
                        current_instruction_id[i] = current_instruction_id[i] + 1

                    llm_result[i]['analysis'] = analysis
                    llm_result[i]['current_goal_finished'] = current_goal_finished
                    llm_result[i]['waypoint_chosen'] = waypoint_chosen
                    llm_result[i][
                        'etp_selected_waypoint_color'] = selected_waypoint_color if selected_waypoint_color in waypoint_visible else 'unknown'
                    llm_result[i]['waypoint_chosen_same_as_expert'] = waypoint_chosen == selected_waypoint_color

                # make equiv action
                env_actions = []
                use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
                for i, gmap in enumerate(self.gmaps):
                    # if llm_result[i]['waypoint_chosen'] == 'unneeded' or llm_result[i]['waypoint_chosen'] == 'unknown':
                    #     print('123')

                    if (llm_result[i]['waypoint_chosen'] == 'unneeded' and current_instruction_id[i] >= len(
                            split_atomic_instructions[i])) \
                            or (current_instruction_id[i] > len(split_atomic_instructions[i])) \
                            or (llm_result[i]['waypoint_chosen'] == 'unknown' and (
                            cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i])):
                        stop_vp = vp_history[i][stepk - 1]
                        stop_pos = gmap.ghost_aug_pos[stop_vp]
                        _, front_vp = gmap.front_to_ghost_dist(stop_vp)
                        front_pos = gmap.node_pos[front_vp]

                        # if self.config.IL.back_algo == 'control':
                        #     back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        #     back_path = back_path[1:]
                        # else:
                        #     back_path = None
                        back_path = None

                        # vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                        # stop_scores = [s[1] for s in vp_stop_scores]
                        # stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                        # stop_pos = gmap.node_pos[stop_vp]
                        # if self.config.IL.back_algo == 'control':
                        #     back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        #     back_path = back_path[1:]
                        # else:
                        #     back_path = None

                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                            'stop_by': 'llm',
                            'different_vp': different_vp[i],
                            'insist_vp': insist_vp[i],
                        }
                        # print('action 0')
                        env_actions.append(
                            {
                                'action': {
                                    'act': 0,
                                    'cur_vp': cur_vp[i],
                                    'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                    'back_path': back_path,
                                    'tryout': use_tryout,
                                },
                                'vis_info': vis_info,
                            }
                        )

                        mylogger.info(f"Stop by llm")
                    elif (cpu_a_t[i] == 0 and llm_result[i]['waypoint_chosen'] == 'unknown') or stepk == self.max_len - 1 or no_vp_left[i]:
                        # stop at node with max stop_prob
                        vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                        stop_scores = [s[1] for s in vp_stop_scores]
                        stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                        stop_pos = gmap.node_pos[stop_vp]
                        if self.config.IL.back_algo == 'control':
                            back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                            back_path = back_path[1:]
                        else:
                            back_path = None
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                            'stop_by': 'etpnav',
                            'different_vp': different_vp[i],
                            'insist_vp': insist_vp[i],
                        }
                        # print('action 0')

                        env_actions.append(
                            {
                                'action': {
                                    'act': 0,
                                    'cur_vp': cur_vp[i],
                                    'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                    'back_path': back_path,
                                    'tryout': use_tryout,
                                },
                                'vis_info': vis_info,
                            }
                        )

                        mylogger.info(f"Stop by etpnav")
                    else:
                        # ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        # ghost_vp = llm_result[i]['waypoint_color_mapping'][llm_result[i]['waypoint_chosen']]['vp_id']

                        if llm_result[i]['waypoint_chosen_same_as_expert'] or llm_result[i]['waypoint_chosen'] == 'unknown':
                            ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        # elif nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]] == llm_result[i].get('prev_etp_vp') and llm_result[i].get('prev_etp_vp') is not None:
                        #     ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        #     if llm_result[i]['waypoint_chosen_same_as_expert'] is False and llm_result[i]['waypoint_chosen'] != 'unknown':
                        #         try:
                        #             insist_vp[i].append(gmap.ghost_aug_pos[llm_result[i]['waypoint_color_mapping'][
                        #                 llm_result[i]['waypoint_chosen']]['vp_id']])
                        #         except Exception as e:
                        #             mylogger.error(traceback.format_exc())
                        else:
                            ghost_vp = llm_result[i]['waypoint_color_mapping'][llm_result[i]['waypoint_chosen']]['vp_id']

                            etp_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                            if etp_vp is not None and etp_vp in gmap.ghost_aug_pos:
                                different_vp[i].append(gmap.ghost_aug_pos[etp_vp])

                        llm_result[i]['prev_etp_vp'] = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]

                        if ghost_vp is None:
                            mylogger.error(f"ghost_vp is None, cpu_a_t: {cpu_a_t[i]}, ghost_vp: {ghost_vp}, waypoint_color_mapping: {llm_result[i]['waypoint_color_mapping']}")

                        vp_history[i][stepk] = ghost_vp
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                        _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                        front_pos = gmap.node_pos[front_vp]

                        if self.config.VIDEO_OPTION:
                            teacher_action_cpu = teacher_actions[i].cpu().item()
                            if teacher_action_cpu in [0, -100]:
                                teacher_ghost = None
                            else:
                                teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                            vis_info = {
                                'nodes': list(gmap.node_pos.values()),
                                'ghosts': list(gmap.ghost_aug_pos.values()),
                                'predict_ghost': ghost_pos,
                                'teacher_ghost': teacher_ghost,
                                'different_vp': different_vp[i],
                                'insist_vp': insist_vp[i],
                            }
                        else:
                            vis_info = None
                        # teleport to front, then forward to ghost
                        if self.config.IL.back_algo == 'control':
                            back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                            back_path = back_path[1:]
                        else:
                            back_path = None
                        # print('action 4')
                        env_actions.append(
                            {
                                'action': {
                                    'act': 4,
                                    'cur_vp': cur_vp[i],
                                    'front_vp': front_vp, 'front_pos': front_pos,
                                    'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                    'back_path': back_path,
                                    'tryout': use_tryout,
                                },
                                'vis_info': vis_info,
                            }
                        )

                        prev_vp[i] = front_vp
                        if self.config.MODEL.consume_ghost:
                            gmap.delete_ghost(ghost_vp)


                    # assert self.envs.num_envs <= 1:
                        # frame = self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info})
                        # rgb_frame = self.envs.call_at(0, "get_rgb_frame", {"ghost_positions": list(gmap.new_ghost_pos.values())})
                        # waypoint_nms = wp_outputs['waypoint_nms'][0].detach().cpu().numpy()
                        #
                        # np.save('tmp/waypoint_nms.npy', waypoint_nms)
                        # cv2.imwrite(f'tmp/{current_episode_name}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}-waypoint.png', self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info, "append_frame": False}))
                        # cv2.imwrite('tmp/rgb.png', rgb_frame)

                outputs = self.envs.step(env_actions)

                for i in range(len(self.gmaps)):
                    self.envs.call_at(i, "remove_object")

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            except Exception as e:
                mylogger.error(traceback.format_exc())


            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        # from etpnav
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

                        # from customize
                        rgb_frame_history.pop(i)
                        current_instruction_id.pop(i)
                        vp_history.pop(i)
                        llm_result.pop(i)
                        different_vp.pop(i)
                        insist_vp.pop(i)
                        dir_name.pop(i)
                        instructions.pop(i)
                        split_instructions.pop(i)
                        split_atomic_instructions.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,
                                                      self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())

    # 将分割后的指令发给etpnav
    def customize_rollout_etp_atomic_instruction(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()


        use_llm = False
        rgb_frame_history = [{} for _ in range(self.envs.num_envs)]
        current_instruction_id = [1 for _ in range(self.envs.num_envs)]
        vp_history = [{} for _ in range(self.envs.num_envs)]
        llm_result = [{} for _ in range(self.envs.num_envs)]
        different_vp = [[] for _ in range(self.envs.num_envs)]
        insist_vp = [[] for _ in range(self.envs.num_envs)]
        dir_name = ['' for _ in range(self.envs.num_envs)]

        if not os.path.exists(f'tmp'):
            os.makedirs(f'tmp')
        for i in range(self.envs.num_envs):
            dir_name[i] = self.envs.call_at(i, "get_episode")['name']

            if not os.path.exists(f'tmp/{dir_name[i]}'):
                os.makedirs(f'tmp/{dir_name[i]}')

        instructions = [instruction.instruction_text for instruction in self.envs.call(['get_instruction'] * self.envs.num_envs)]
        split_instructions: list[str] = [split_instruction(instruction) for instruction in instructions]
        split_atomic_instructions = [instructions.strip().split('\n') for instructions in split_instructions]
        incremental_instructions = [
            ['\n'.join(split_atomic_instruction[:i + 1]) for i in range(len(split_atomic_instruction))]
            for split_atomic_instruction in split_atomic_instructions
        ]
        incremental_instructions_token = [
            [instruction_to_token(instruction) for instruction in incremental_instruction]
            for incremental_instruction in incremental_instructions
        ]

        mylogger.info(f"Instruction: {instructions}")
        mylogger.info(f"Split Instructions: {split_instructions}")

        # 指定episode进行测试
        # assert self.envs.num_envs <= 1
        current_episode_name = self.envs.call_at(0, "get_episode")['name']
        # if current_episode_name != '2azQ1b91cZZ-1213':
        #     mylogger.debug(f"skip episode: {current_episode_name}")
        #     return #


        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        for i in range(len(observations)):
            observations[i]['instruction'] = incremental_instructions_token[i][current_instruction_id[i] - 1]


        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos,
                               self.config.IL.loc_noise,
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs

            while True:
                # encode instructions
                all_txt_ids = batch['instruction']
                all_txt_masks = (all_txt_ids != instr_pad_id)
                all_txt_embeds = self.policy.net(
                    mode='language',
                    txt_ids=all_txt_ids,
                    txt_masks=all_txt_masks,
                )

                txt_masks = all_txt_masks[not_done_index]
                txt_embeds = all_txt_embeds[not_done_index]

                # cand waypoint prediction
                wp_outputs = self.policy.net(
                    mode="waypoint",
                    waypoint_predictor=self.waypoint_predictor,
                    observations=batch,
                    in_train=(mode == 'train' and self.config.IL.waypoint_aug),
                )

                # pano encoder
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({
                    'mode': 'panorama',
                })
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                  torch.sum(pano_masks, 1, keepdim=True)

                # get vp_id, vp_pos of cur_node and cand_ndoe
                cur_pos, cur_ori = self.get_pos_ori()
                cur_vp, cand_vp, cand_pos = [], [], []
                for i in range(self.envs.num_envs):
                    cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                        cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                    )
                    cur_vp.append(cur_vp_i)
                    cand_vp.append(cand_vp_i)
                    cand_pos.append(cand_pos_i)

                if mode == 'train' or self.config.VIDEO_OPTION:
                    cand_real_pos = []
                    for i in range(self.envs.num_envs):
                        cand_real_pos_i = [
                            self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                            for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                        ]
                        cand_real_pos.append(cand_real_pos_i)
                else:
                    cand_real_pos = [None] * self.envs.num_envs

                for i in range(self.envs.num_envs):
                    cur_embeds = avg_pano_embeds[i]
                    cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                    self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                               cur_vp[i], cur_pos[i], cur_embeds,
                                               cand_vp[i], cand_pos[i], cand_embeds,
                                               cand_real_pos[i])

                nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
                nav_inputs.update({
                    'mode': 'navigation',
                    'txt_embeds': txt_embeds,
                    'txt_masks': txt_masks,
                })
                no_vp_left = nav_inputs.pop('no_vp_left')
                nav_outs = self.policy.net(**nav_inputs)
                nav_logits = nav_outs['global_logits']
                nav_probs = F.softmax(nav_logits, 1)
                for i, gmap in enumerate(self.gmaps):
                    gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

                # random sample demo
                # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
                # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
                # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

                # determine action
                if feedback == 'sample':
                    c = torch.distributions.Categorical(nav_probs)
                    a_t = c.sample().detach()
                    a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
                elif feedback == 'argmax':
                    a_t = nav_logits.argmax(dim=-1)
                else:
                    raise NotImplementedError
                cpu_a_t = a_t.cpu().numpy()

                tag = False
                for i, gmap in enumerate(self.gmaps):
                    if cpu_a_t[i] == 0 and current_instruction_id[i] < len(split_atomic_instructions[i]) - 1:
                        observations[i]['instruction'] = incremental_instructions_token[i][current_instruction_id[i]]
                        current_instruction_id[i] += 1
                        tag = True

                if tag:
                    batch = batch_obs(observations, self.device)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)
                else:
                    break

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            try:
                if use_llm:
                    query_qwen_futures = []
                    for i, gmap in enumerate(self.gmaps):
                        waypoint_color_mapping = self.envs.call_at(i, "add_object", {
                            "ghost_positions": gmap.new_ghost_pos})

                        new_ghost_nodes = list(gmap.new_ghost_pos.values())

                        llm_result[i]['waypoint_color_mapping'] = waypoint_color_mapping
                        llm_result[i]['new_ghost_nodes'] = new_ghost_nodes

                        rgb_frame = self.envs.call_at(i, "get_rgb_frame", {"ghost_positions": new_ghost_nodes, 'flat': True})

                        if stepk > 0: # mask image in the back of the agent
                            # width = rgb_frame.shape[1]
                            # black_width = int(width * 0.15)
                            #
                            # # 将左右两边各15%的区域设置为全黑色
                            # rgb_frame[:, :black_width, :] = 0
                            # rgb_frame[:, -black_width:, :] = 0

                            width = rgb_frame.shape[1]
                            black_width = int(width * 0.15)

                            # Remove the left and right 15% of the image
                            rgb_frame = rgb_frame[:, black_width:-black_width, :]

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        unique_id = uuid.uuid4()
                        output_file_name = f'tmp/{dir_name[i]}/{timestamp}.png'
                        cv2.imwrite(output_file_name, rgb_frame)
                        rgb_frame_history[i][stepk] = output_file_name
                        mylogger.info(f"save image to {output_file_name}")
                        # print(f"save image to {output_file_name}")

                        ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        selected_waypoint_color = None
                        if ghost_vp in gmap.new_ghost_pos:
                            for color in waypoint_color_mapping.keys():
                                if ghost_vp == waypoint_color_mapping[color]['vp_id']:
                                    selected_waypoint_color = color
                                    break
                        llm_result[i]['selected_waypoint_color'] = selected_waypoint_color

                        prompt = ''
                        if stepk > 0:
                            prompt += f"The first image is the previous step, and the second image is the current step. "
                            prompt += f"In the previous step, the instruction is instruction {current_instruction_id[i] if llm_result[i]['current_goal_finished'] == 'yes' else current_instruction_id[i]}, and you choose the waypoint {llm_result[i]['waypoint_chosen'] if llm_result[i]['waypoint_chosen'] != 'unknown' else llm_result[i]['etp_selected_waypoint_color']}. "
                        prompt += f"There are {len(gmap.new_ghost_pos)} waypoint in the image, their color are {BALL_COLORS[:len(new_ghost_nodes)]}, respectively. Be aware that some waypoint might not be visible in the current image. "
                        prompt += (f"Your instructions are {split_instructions[i]}\n"
                                   f"You are currently at instruction {current_instruction_id[i]}, which means your current goal is {split_atomic_instructions[i][current_instruction_id[i] - 1]}")
                        prompt += f"You need to tell whether the current goal is finished, "

                        if current_instruction_id[i] == len(split_atomic_instructions[i]):
                            prompt += (
                                f"if it is, you don't need to choose any waypoint since all instructions are finished and output the Waypoint chosen in the answer to be 'unneeded', "
                                f"if not, you need to tell the color of the waypoint you should go to that can finish instruction {current_instruction_id[i]}. "
                            )
                        else:
                            prompt += (
                                f"if it is, you need to switch to the instruction {current_instruction_id[i] + 1} ({split_atomic_instructions[i][current_instruction_id[i]]}) and choose waypoint that can finish instruction {current_instruction_id[i] + 1}, "
                                f"if not, you need to tell the color of the waypoint you should go to that can finish instruction 1. "
                                f"you also need to check if the place described in the next instruction appears in the image, if so, you can also switch to the next instruction with current goal finished. "
                            )

                        # prompt += '\nIf the instruction requires you to stop after reaching something, such as "Stop once you reach the bed," you should keep moving until close enough to the target object. \n'
                        prompt += 'If the instruction requires you to stop after reaching something, such as "Stop once you reach the bed," you should keep moving until there are no closer waypoints to choose. '
                        prompt += "You need to output all waypoint color that you can see in current image, if you can't see any waypoint in the image, please output \"unknown\" when choosing waypoint. \n"

                        if selected_waypoint_color is None:
                            prompt += (
                                f"In addition, if you cannot decide which waypoint to choose, output the Waypoint chosen in the answer to be 'unknown'."
                            )

                            if current_instruction_id[i] == len(split_atomic_instructions[i]):
                                prompt += (
                                    f"Answer in format:"
                                    f"Analysis: <analysis>, "
                                    f"Current goal finished: <yes/no>, "
                                    f"Waypoint color visible: [red, green, blue, yellow, purple], " # 'red', 'green', 'blue', 'yellow', 'purple'
                                    f"Waypoint chosen: <waypoint color/unknown/unneeded>, "
                                )
                            else:
                                prompt += (
                                    f"Answer in format:"
                                    f"Analysis: <analysis>, "
                                    f"Current goal finished: <yes/no>, "
                                    f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                    f"Waypoint chosen: <waypoint color>, "
                                )
                        else:
                            prompt += (
                                f"In addition, the waypoint selected by an expert is in color {selected_waypoint_color}. "
                                f"If the waypoint you choose is the same as the expert, please analyze why it is correct. "
                                f"If it is not the same, then please analyze why it is wrong and give the waypoint you choose. "
                                f"If you cannot decide which waypoint to choose, please choose the waypoint selected by the expert. "
                            )

                            if current_instruction_id[i] == len(split_atomic_instructions[i]):
                                prompt += (
                                    f"Answer in format:"
                                    f"Analysis: <analysis>, "
                                    f"Current goal finished: <yes/no>, "
                                    f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                    f"Waypoint chosen: <waypoint color/unneeded>, "
                                )
                            else:
                                prompt += (
                                    f"Answer in format:"
                                    f"Analysis: <analysis>, "
                                    f"Current goal finished: <yes/no>, "
                                    f"Waypoint color visible: [red, green, blue, yellow, purple], "
                                    f"Waypoint chosen: <waypoint color>, "
                                )


                        if stepk > 0:
                            qwen_image_paths = [
                                os.path.join(os.getcwd(), rgb_frame_history[i][stepk - 1]),
                                os.path.join(os.getcwd(), output_file_name)]
                        else:
                            qwen_image_paths = [os.path.join(os.getcwd(), output_file_name)]
                        mylogger.info(f"Query Qwen, image_paths: {qwen_image_paths}, Prompt: \n {prompt}")

                        # result = query_qwen(qwen_image_paths, prompt)
                        query_qwen_futures.append(query_qwen_pool.submit(query_qwen, qwen_image_paths, prompt))


                        # self.envs.call_at(i, "remove_object")
                    for i, gmap in enumerate(self.gmaps):
                        new_ghost_nodes = llm_result[i]['new_ghost_nodes']
                        selected_waypoint_color = llm_result[i]['selected_waypoint_color']

                        result = query_qwen_futures[i].result()

                        # parse the result
                        try:
                            analysis = result.split("Analysis: ")[1].split("Current goal finished:")[0]
                            current_goal_finished = result.split("Current goal finished: ")[1].split(",")[0].split('\n')[
                                0].lower().strip()
                        except Exception as e:
                            mylogger.error(f"error when split result: {result}")
                            mylogger.error(traceback.format_exc())
                            analysis = ''
                            current_goal_finished = 'no'

                        try:
                            waypoint_visible = \
                            result.split("Waypoint color visible: ")[1].split('\n')[0].strip().split("Waypoint chosen: ")[
                                0].strip(', [].').split(',')
                            waypoint_visible = [waypoint.strip() for waypoint in waypoint_visible]
                        except:
                            mylogger.error(
                                f"error when split waypoint_visible: {result.split('Waypoint color visible: ')[1]}")
                            waypoint_visible = []
                            mylogger.error(traceback.format_exc())

                        try:
                            waypoint_chosen = \
                            result.split("Waypoint chosen: ")[1].split(",")[0].split('\n')[0].lower().strip().split(' ')[0]
                        except Exception as e:
                            mylogger.error(f"error when split waypoint_chosen: {result.split('Waypoint chosen: ')[1]}")
                            mylogger.error(traceback.format_exc())
                            waypoint_chosen = 'unknown'

                        mylogger.info(
                            f"Analysis: {analysis}\n"
                            f"Current goal finished: {current_goal_finished}\n"
                            f"Waypoint put in scene: {BALL_COLORS[:len(new_ghost_nodes)]}\n"
                            f"Waypoint color visible: {waypoint_visible}\n"
                            f"Waypoint chosen: {waypoint_chosen}\n"
                            f"Waypoint chosen valid: {waypoint_chosen in waypoint_visible}\n"
                            f"Waypoint chosen same as expert: {waypoint_chosen == selected_waypoint_color}\n"
                        )
                        # print(f"Analysis: {analysis}")
                        # print(f"Current goal finished: {current_goal_finished}")
                        # print(f"Waypoint chosen: {waypoint_chosen}")
                        # print(f"Waypoint chosen same as expert: {waypoint_chosen == selected_waypoint_color}")

                        if current_goal_finished not in ['yes', 'no']:
                            mylogger.error(f"current_goal_finished should be yes or no, but got {current_goal_finished}")
                            current_goal_finished = 'no'

                        if waypoint_chosen not in BALL_COLORS[:len(new_ghost_nodes)] and waypoint_chosen not in ['unknown',
                                                                                                                 'unneeded']:
                            mylogger.error(
                                f"waypoint_chosen should be in {BALL_COLORS[:len(new_ghost_nodes)]}, but got {waypoint_chosen}")
                            waypoint_chosen = 'unknown'
                        elif waypoint_chosen not in waypoint_visible and waypoint_chosen not in ['unknown', 'unneeded']:
                            mylogger.error(
                                f"waypoint_chosen should be in waypoint_visible {waypoint_visible}, but got {waypoint_chosen}")
                            waypoint_chosen = 'unknown'

                        # assert current_goal_finished in ['yes', 'no'], f"current_goal_finished should be yes or no, but got {current_goal_finished}"
                        # assert waypoint_chosen in BALL_COLORS[:len(new_ghost_nodes)] or waypoint_chosen in ['unknown', 'unneeded'], f"waypoint_chosen should be in {BALL_COLORS[:len(new_ghost_nodes)]}, but got {waypoint_chosen}"
                        if current_goal_finished == 'yes':
                            current_instruction_id[i] = current_instruction_id[i] + 1

                        llm_result[i]['analysis'] = analysis
                        llm_result[i]['current_goal_finished'] = current_goal_finished
                        llm_result[i]['waypoint_chosen'] = waypoint_chosen
                        llm_result[i][
                            'etp_selected_waypoint_color'] = selected_waypoint_color if selected_waypoint_color in waypoint_visible else 'unknown'
                        llm_result[i]['waypoint_chosen_same_as_expert'] = waypoint_chosen == selected_waypoint_color

                # make equiv action
                env_actions = []
                use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
                for i, gmap in enumerate(self.gmaps):
                    if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                        # stop at node with max stop_prob
                        vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                        stop_scores = [s[1] for s in vp_stop_scores]
                        stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                        stop_pos = gmap.node_pos[stop_vp]
                        if self.config.IL.back_algo == 'control':
                            back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                            back_path = back_path[1:]
                        else:
                            back_path = None
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                            'stop_by': 'etpnav',
                            'different_vp': different_vp[i],
                            'insist_vp': insist_vp[i],
                        }
                        # print('action 0')

                        if cpu_a_t[i] == 0 and current_instruction_id[i] < len(split_atomic_instructions[i]) - 1:
                            pass
                        else:
                            env_actions.append(
                                {
                                    'action': {
                                        'act': 0,
                                        'cur_vp': cur_vp[i],
                                        'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                        'back_path': back_path,
                                        'tryout': use_tryout,
                                    },
                                    'vis_info': vis_info,
                                }
                            )

                        mylogger.info(f"Stop by etpnav")
                    else:
                        ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                        vp_history[i][stepk] = ghost_vp
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                        _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                        front_pos = gmap.node_pos[front_vp]

                        if self.config.VIDEO_OPTION:
                            teacher_action_cpu = teacher_actions[i].cpu().item()
                            if teacher_action_cpu in [0, -100]:
                                teacher_ghost = None
                            else:
                                teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                            vis_info = {
                                'nodes': list(gmap.node_pos.values()),
                                'ghosts': list(gmap.ghost_aug_pos.values()),
                                'predict_ghost': ghost_pos,
                                'teacher_ghost': teacher_ghost,
                                'different_vp': different_vp[i],
                                'insist_vp': insist_vp[i],
                            }
                        else:
                            vis_info = None
                        # teleport to front, then forward to ghost
                        if self.config.IL.back_algo == 'control':
                            back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                            back_path = back_path[1:]
                        else:
                            back_path = None
                        # print('action 4')
                        env_actions.append(
                            {
                                'action': {
                                    'act': 4,
                                    'cur_vp': cur_vp[i],
                                    'front_vp': front_vp, 'front_pos': front_pos,
                                    'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                    'back_path': back_path,
                                    'tryout': use_tryout,
                                },
                                'vis_info': vis_info,
                            }
                        )

                        prev_vp[i] = front_vp
                        if self.config.MODEL.consume_ghost:
                            gmap.delete_ghost(ghost_vp)


                    # assert self.envs.num_envs <= 1:
                        # frame = self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info})
                        # rgb_frame = self.envs.call_at(0, "get_rgb_frame", {"ghost_positions": list(gmap.new_ghost_pos.values())})
                        # waypoint_nms = wp_outputs['waypoint_nms'][0].detach().cpu().numpy()
                        #
                        # np.save('tmp/waypoint_nms.npy', waypoint_nms)
                        frame = self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info, "append_frame": False})
                        frame = add_text_to_frame(frame, incremental_instructions[i][current_instruction_id[i] - 1])
                        cv2.imwrite(f'tmp/{current_episode_name}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}-waypoint.png', frame)
                        # cv2.imwrite('tmp/rgb.png', rgb_frame) # 'zsNo4HB9uLZ-417'

                outputs = self.envs.step(env_actions)

                for i in range(len(self.gmaps)):
                    self.envs.call_at(i, "remove_object")

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            except Exception as e:
                mylogger.error(traceback.format_exc())


            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        # from etpnav
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

                        # from customize
                        rgb_frame_history.pop(i)
                        current_instruction_id.pop(i)
                        vp_history.pop(i)
                        llm_result.pop(i)
                        different_vp.pop(i)
                        insist_vp.pop(i)
                        dir_name.pop(i)
                        instructions.pop(i)
                        split_instructions.pop(i)
                        split_atomic_instructions.pop(i)
                        incremental_instructions.pop(i)
                        incremental_instructions_token.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,
                                                      self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            for i in range(len(observations)):
                observations[i]['instruction'] = incremental_instructions_token[i][current_instruction_id[i] - 1]

            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())



    def customize_rollout_memory_between_episodes(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        current_episode_name = self.envs.call_at(0, "get_episode")['name']

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        assert self.envs.num_envs == 1, "Customize rollout only supports single env for now."


        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        if os.path.basename(self.envs.current_episodes()[0].scene_id) not in self.graph_map_memory:
            self.graph_map_memory[os.path.basename(self.envs.current_episodes()[0].scene_id)] = \
                GraphMap(have_real_pos,
                         self.config.IL.loc_noise,
                         self.config.MODEL.merge_ghost,
                         ghost_aug, merge_map=True)
        self.gmaps = [self.graph_map_memory[os.path.basename(self.envs.current_episodes()[0].scene_id)]]

        # self.gmaps = [GraphMap(have_real_pos,
        #                        self.config.IL.loc_noise,
        #                        self.config.MODEL.merge_ghost,
        #                        ghost_aug, merge_map=False) for _ in range(self.envs.num_envs)]

        for i, gmap in enumerate(self.gmaps):
            gmap.reset_but_keep_node()
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp] if vp in gmap.node_pos else gmap.old_node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                        'nodes': list(gmap.node_pos.values()),
                        'ghosts': list(gmap.ghost_aug_pos.values()),
                        'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp] if front_vp in gmap.node_pos else gmap.old_node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp] if vp in gmap.node_pos else gmap.old_node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # cv2.imwrite(
            #     f'tmp/{current_episode_name}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}-waypoint.png',
            #     self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info, "append_frame": False}))

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,
                                                      self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())


    def customize_rollout_tod_down_map(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len  # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes())
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        current_episode_name = self.envs.call_at(0, "get_episode")['name']

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        assert self.envs.num_envs == 1, "Customize rollout only supports single env for now."

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        if os.path.basename(self.envs.current_episodes()[0].scene_id) not in self.graph_map_memory:
            self.graph_map_memory[os.path.basename(self.envs.current_episodes()[0].scene_id)] = \
                GraphMap(have_real_pos,
                         self.config.IL.loc_noise,
                         self.config.MODEL.merge_ghost,
                         ghost_aug, merge_map=True)
        self.gmaps = [self.graph_map_memory[os.path.basename(self.envs.current_episodes()[0].scene_id)]]

        # self.gmaps = [GraphMap(have_real_pos,
        #                        self.config.IL.loc_noise,
        #                        self.config.MODEL.merge_ghost,
        #                        ghost_aug, merge_map=False) for _ in range(self.envs.num_envs)]

        for i, gmap in enumerate(self.gmaps):
            gmap.reset_but_keep_node()
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint_predictor,
                observations=batch,
                in_train=(mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)

            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                self.gmaps[i].update_graph(prev_vp[i], stepk + 1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float) <= sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp] if vp in gmap.node_pos else gmap.old_node_pos[vp]) for vp in
                                     gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                        'nodes': list(gmap.node_pos.values()),
                        'ghosts': list(gmap.ghost_aug_pos.values()),
                        'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp] if front_vp in gmap.node_pos else gmap.old_node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp] if vp in gmap.node_pos else gmap.old_node_pos[vp]) for vp in
                                     gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            self.envs.call_at(0, "get_top_down_map")
            # cv2.imwrite(
            #     f'tmp/{current_episode_name}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}-waypoint.png',
            #     self.envs.call_at(0, "get_plan_frame", {"vis_info": vis_info, "append_frame": False}))

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,
                                                      self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
