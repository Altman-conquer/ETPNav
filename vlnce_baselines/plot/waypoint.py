import torch

def main():
    from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
    device = list(range(torch.cuda.device_count()))[0]
    waypoint_predictor = BinaryDistPredictor_TRM(device=device)
    cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if 'config.MODEL.task_type' == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
    waypoint_predictor.load_state_dict(
        torch.load(cwp_fn, map_location=torch.device('cpu'))['predictor']['state_dict'])
    for param in waypoint_predictor.parameters():
        param.requires_grad_(False)

    waypoint_predictor.to(device)

if __name__ == '__main__':
    main()