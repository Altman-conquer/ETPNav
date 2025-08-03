from typing import Any, Dict, Optional, Tuple, List, Union

import habitat_sim
import math
import random
import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from habitat_extensions.maps import drawpoint
from habitat_extensions.utils import generate_video, heading_from_quaternion, navigator_video_frame, \
    planner_video_frame, get_frame, UUIDS_EQ
from scipy.spatial.transform import Rotation as R
import cv2
import os
import magnum as mn

from vlnce_baselines.waypoint_pred.utils import BALL_COLORS


def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat


def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx ** 2 + dz ** 2), 1e-8)
    # xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2 * np.pi
    heading = heading % (2 * np.pi)

    return heading, xz_dist


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.prev_episode_id = "something different"

        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []
        self.plan_frames = []

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self,
                          node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)

    def get_observation_at(self,
                           source_position: List[float],
                           source_rotation: List[Union[int, np.float64]],
                           keep_agent_at_new_pose: bool = False):

        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs

    def current_dist_to_goal(self):
        init_state = self._env.sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance

    def point_dist_to_goal(self, pos):
        dist = self._env.sim.geodesic_distance(
            pos, self._env.current_episode.goals[0].position,
        )
        return dist

    def get_cand_real_pos(self, forward, angle):
        '''get cand real_pos by executing action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward // init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_pose = post_state.position

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)

        return post_pose

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        # circle_dists = np.linalg.norm(np.array(path)-current_pos, axis=1).tolist()
        return circle_dists

    def ghost_dist_to_ref(self, ghost_vp_pos, ref_path):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        if circle_bool.sum() == 0:  # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0, len(ref_path)) >= progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress]  # prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0], cand_idxes[0] + len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare == False)[0][0] - 1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx

            self.prev_sub_goal_pos = sub_goal_pos

        # ghost dis to subgoal
        ghost_dists_to_subgoal = []
        for ghost_vp, ghost_pos in ghost_vp_pos:
            dist = self._env.sim.geodesic_distance(ghost_pos, sub_goal_pos)
            ghost_dists_to_subgoal.append(dist)

        oracle_ghost_vp = ghost_vp_pos[np.argmin(ghost_dists_to_subgoal)][0]
        self.prev_episode_id = episode_id

        return oracle_ghost_vp

    def get_cand_idx(self, ref_path, angles, distances, candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0:  # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0, len(ref_path)) >= progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress]  # prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0], cand_idxes[0] + len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare == False)[0][0] - 1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx

            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
        # if curr_dist_to_goal == np.inf:

        return oracle_cand_idx  # , sub_goal_pos

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1],
                           init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward // init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)

        return post_distance

    def cand_dist_to_subgoal(self,
                             angle: float, forward: float,
                             sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1],
                           init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward // init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)

        return post_distance

    def reset(self):
        observations = self._env.reset()
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations,
                    info,
                )
            ]
        return observations

    # def wrap_act(self, act, ang, dis, cand_wp, action_wp, oracle_wp, start_p, start_h):
    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            # print('wrap_act')
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task
            )
        return observations

    def turn(self, ang, vis_info):
        ''' angle: 0 ~ 360 degree '''
        act_l = HabitatSimActions.TURN_LEFT
        act_r = HabitatSimActions.TURN_RIGHT
        uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_l].actuation.amount
        ang_degree = math.degrees(ang)
        ang_degree = round(ang_degree / uni_l) * uni_l
        observations = None

        if 180 < ang_degree <= 360:
            ang_degree -= 360
        if ang_degree >= 0:
            turns = [act_l] * (ang_degree // uni_l)
        else:
            turns = [act_r] * (-ang_degree // uni_l)

        for turn in turns:
            observations = self.wrap_act(turn, vis_info)
        return observations

    def teleport(self, pos):
        self._env.sim.set_agent_state(pos, quat_from_heading(0))

    def single_step_control(self, pos, tryout, vis_info):
        act_f = HabitatSimActions.MOVE_FORWARD
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        ang, dis = calculate_vp_rel_pos(agent_state.position, pos, heading_from_quaternion(agent_state.rotation))
        self.turn(ang, vis_info)

        ksteps = int(dis // uni_f)
        if not tryout:
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
        else:
            cnt = 0
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
                if self._env.sim.previous_step_collided:
                    break
                else:
                    cnt += 1
            # left forward step
            ksteps = ksteps - cnt
            if ksteps > 0:
                try_ang = random.choice([math.radians(90), math.radians(270)])  # left or right randomly
                self.turn(try_ang, vis_info)
                if try_ang == math.radians(90):  # from left to right
                    turn_seqs = [
                        (0, 270),  # 90, turn_left=30, turn_right=330
                        (330, 300),  # 60
                        (330, 330),  # 30
                        (300, 30),  # -30
                        (330, 60),  # -60
                        (330, 90),  # -90
                    ]
                elif try_ang == math.radians(270):  # from right to left
                    turn_seqs = [
                        (0, 90),  # -90
                        (30, 60),  # -60
                        (30, 30),  # -30
                        (60, 330),  # 30
                        (30, 300),  # 60
                        (30, 270),  # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    # do head_turns
                    self.turn(math.radians(turn_seq[0]), vis_info)
                    prev_position = self._env.sim.get_agent_state().position
                    self.wrap_act(act_f, vis_info)
                    post_posiiton = self._env.sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        # do tail_turns
                        self.turn(math.radians(turn_seq[1]), vis_info)
                        # do left forward actions
                        for _ in range(ksteps):
                            self.wrap_act(act_f, vis_info)
                            if self._env.sim.previous_step_collided:
                                break
                        break

    def multi_step_control(self, path, tryout, vis_info):
        for vp, vp_pos in path:  # path[::-1]:
            self.single_step_control(vp_pos, tryout, vis_info)

    def get_plan_frame(self, vis_info, append_frame=True):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6, 6, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        if append_frame:
            self.plan_frames.append(frame)
        return frame

    def get_rgb_frame(self, ghost_positions: list = None, flat = False):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        rgb = get_frame(observations, flat=flat)
        return rgb


    def get_depth_frame(self, ghost_positions: list = None):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        depth = get_frame(observations, frame_type='depth')
        return depth

    def get_2d_point(self, sensor_name, point_3d):
        sim = self._env.sim
        # get the scene render camera and sensor object
        visual_sensor = sim._sensors[sensor_name]
        scene_graph = sim.get_active_scene_graph()
        scene_graph.set_default_render_camera_parameters(visual_sensor._sensor_object)
        render_camera = scene_graph.get_default_render_camera()

        # use the camera and projection matrices to transform the point onto the near plane
        projected_point_3d = render_camera.projection_matrix.transform_point(
            render_camera.camera_matrix.transform_point(point_3d)
        )
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / render_camera.projection_size()[0]
        point_2d += mn.Vector2(0.5)
        point_2d *= render_camera.viewport
        return mn.Vector2i(point_2d)

    def step(self, action, vis_info, *args, **kwargs):
        act = action['act']

        if act == 4:  # high to low
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to front node
            if action['back_path'] is None:
                self.teleport(action['front_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)

            # 2. forward to ghost node
            self.single_step_control(action['ghost_pos'], action['tryout'], vis_info)
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)

        elif act == 0:  # stop
            if self.video_option:
                self.get_plan_frame(vis_info)

            if vis_info['stop_by'] == 'llm':
                # 1. back to stop node
                if action['back_path'] is None:
                    self.teleport(action['stop_pos'])
                else:
                    self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            else:
                # 1. back to stop node
                if action['back_path'] is None:
                    self.teleport(action['stop_pos'])
                else:
                    self.multi_step_control(action['back_path'], action['tryout'], vis_info)

            # 2. stop
            observations = self._env.step(act)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)

        else:
            raise NotImplementedError

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        if self.video_option and done:
            # for pano visualization
            metrics = {
                "sr": round(info["success"], 3),
                # "spl": round(info["spl"], 3),
                # "ndtw": round(info["ndtw"], 3),
                # "sdtw": round(info["sdtw"], 3),
            }

            # if 0 < info["spl"] <= 0.6:  #TODO backtrack
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics=metrics,
                tb_writer=None,
                fps=8,
            )

            metric_strs = []
            for k, v in metrics.items():
                metric_strs.append(f"{k}{v:.2f}")
            episode_id = self._env.current_episode.episode_id
            scene_id = self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            tmp_fn = os.path.join(self.video_dir, tmp_name)
            tmp = np.concatenate(self.plan_frames, axis=0)
            cv2.imwrite(tmp_fn, tmp)
            self.plan_frames = []

        return observations, reward, done, info

    def set_object_location_and_scale(self, sim, obj_id, location, scale):
        r"""
        Adds an object in front of the agent at some distance.
        """
        # agent_transform = sim.agents[0].scene_node.transformation_matrix()
        # obj_translation = agent_transform.transform_point(
        #     # np.array([0.25, 0.25, z_offset])
        #     np.array([location[0], location[2], location[1]])
        # )
        # sim.set_translation(obj_translation, obj_id)
        sim.set_translation(mn.Vector3(location), obj_id)

        obj_node = sim.get_object_scene_node(obj_id)
        xform_bb = habitat_sim.geo.get_transformed_bb(
            obj_node.cumulative_bb, obj_node.transformation
        )

        # also account for collision margin of the scene
        # scene_collision_margin = 0.04
        # y_translation = mn.Vector3(
        #     0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
        # )
        # sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)

        # scale the object
        scale_matrix = mn.Matrix4.scaling(mn.Vector3(scale))
        obj_node.transformation = obj_node.transformation @ scale_matrix

    def add_object(self, ghost_positions: dict):
        sim = self._env.sim

        # Manager of Object Attributes Templates
        obj_attr_mgr = sim.get_object_template_manager()
        obj_attr_mgr.load_configs(
            str(os.path.join('data', "test_assets/objects"))
        )

        color_mapping = {}

        for i, position_key in enumerate(ghost_positions.keys()):
            position = ghost_positions[position_key]
            color_mapping[BALL_COLORS[i]] = {
                "vp_id": position_key,
                "position": position,
            }

            obj_path = f"test_assets/objects/{BALL_COLORS[i]}_sphere"
            chair_template_id = obj_attr_mgr.load_object_configs(
                str(os.path.join('data', obj_path))
            )[0]
            chair_attr = obj_attr_mgr.get_template_by_ID(chair_template_id)
            obj_attr_mgr.register_template(chair_attr)

            current_position = sim.get_agent_state().position

            # Object's initial position 3m away from the agent.
            # object_id = sim.add_object_by_handle(chair_attr.handle)
            # self.set_object_location_and_scale(sim, object_id, -1.0)
            # sim.set_object_motion_type(
            #     habitat_sim.physics.MotionType.STATIC, object_id
            # )
            relative_position = np.array(position) - current_position
            position = np.array(position)
            scale = float(np.linalg.norm(relative_position)) * 1.25

            object_id = sim.add_object_by_handle(chair_attr.handle)
            self.set_object_location_and_scale(sim, object_id, position, scale)
            sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, object_id)

        return color_mapping

    def remove_object(self):
        sim = self._env.sim
        for obj_id in sim.get_existing_object_ids():
            sim.remove_object(obj_id)

    def get_instruction(self):
        # Get the instruction from the current episode
        return self._env.current_episode.instruction

    def get_episode(self):
        episode_id = self._env.current_episode.episode_id
        scene_id = self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
        # tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
        # tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
        return {
            'episode_id': episode_id,
            'scene_id': scene_id,
            'name': f"{scene_id}-{episode_id}",
        }


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
