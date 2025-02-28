import os
import argparse
import json
import torch
import numpy as np
from eikonal_solver import EikonalSolverPointCloud
# from optimizer import optimize_velocity_field
# from save_results import save_as_vtk, save_point_cloud_as_vtk
from save_results import write_vtk_data_pointcloud
from utils import load_geo, load_data, TaskDataset
from torch.utils.data import Dataset
from optimizer import Optimizer

def run_experiment(config):
    # Extract configuration parameters
    data_config = config["data"]
    sigma_h, sigma_s = data_config["sigma_h"], data_config["sigma_s"]
    optimization_config = config["optimization"]
    output_config = config["output"]
    res_folder = "opt_baseline/" + data_config["geo"] + '_' + str(data_config["segment"]) + '_' + str(data_config["initial_id"]) + "/"
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
        print(f"Created folder: {res_folder}")
    bounds = (optimization_config["lower_bound"], optimization_config["upper_bound"])

    # Load and prepare the dataset
    geo_path = os.path.join(data_config["dir"], "geometry", f'{ data_config["geo"]}.mat')
    seg_path = os.path.join(data_config["dir"], "geometry", f'{ data_config["geo"]}.seg')
    seg_dict = data_config["segments"]["EC_AW_DC"]
    nodes, faces, segs = load_geo(geo_path, seg_path)
    at_org, velocity_org, mask, act_noise, idx = load_data(data_config["dir"], data_config["geo"], data_config["segment"], data_config["initial_id"], len(nodes), data_config["num_data"], data_config["noise"])
    if at_org is None:
        return
    
    source_point = np.where(at_org < 0.001)
    source_point = source_point[0]
    
    # Create an Eikonal solver instance
    solver = EikonalSolverPointCloud(nodes, faces, data_config["velocity_field"])
    
    
    # Perform optimization
    optimizer = Optimizer()

    slow_conduction_value = data_config["velocity_field"][1]
    normal_conduction_value = data_config["velocity_field"][0]
    
    if optimization_config["type"] == "segment":
        initial_velocity_field = np.full(len(seg_dict.keys()), (sigma_h + sigma_s)/2)
        initial_velocity_field_visual = np.full(nodes.shape[0], (sigma_h + sigma_s)/2)
        # initial_velocity_field = np.full(len(seg_dict.keys()), sigma_s)
        # initial_velocity_field_visual = np.full(nodes.shape[0], sigma_s)
        for k in seg_dict.keys():
            mask = np.where(np.isin(segs, seg_dict[k]))[0]
            initial_velocity_field_visual[mask] = initial_velocity_field[int(k)]
    else:
        initial_velocity_field = np.random.uniform(slow_conduction_value, normal_conduction_value, nodes.shape[0])
        initial_velocity_field_visual = initial_velocity_field
    # Optimization process
    optimized_velocity_field = optimizer.optimize_velocity_field(optimization_config["type"], initial_velocity_field, bounds, source_point, nodes, faces, act_noise, seg_dict, segs, (sigma_h, sigma_s))

    # Save results
    if output_config["save_activation_times"]:
        write_vtk_data_pointcloud(initial_velocity_field_visual, nodes, res_folder+f'initial_velocity_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        write_vtk_data_pointcloud(optimizer.pred_velocity_min[0], nodes, res_folder+f'pred_velocity_mid_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        write_vtk_data_pointcloud(velocity_org, nodes, res_folder+f'velocity_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        write_vtk_data_pointcloud(optimizer.pred_time_min[0], nodes, res_folder+f'pred_at_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        write_vtk_data_pointcloud(at_org, nodes, res_folder+f'at_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        write_vtk_data_pointcloud(optimized_velocity_field, nodes, res_folder+f'eval_pred_velocity_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
        # write_vtk_data_pointcloud(optimizer.pred_time_now[0], nodes, res_folder+f'eval_at_pred_geo_{data_config["geo"]}_id_{data_config["segment"]}_initial_{data_config["initial_id"]}.vtk')
    
    # Compute L2 loss for predicted activation time and velocity
    l2_loss_time = np.mean((at_org - optimizer.pred_time_min[0]) ** 2)
    l2_loss_velocity = np.mean(velocity_org - optimizer.pred_velocity_min[0]) ** 2

    # Save results to a log file
    log_file = os.path.join("opt_baseline" , "experiment_log.txt")
    with open(log_file, "a") as log:
        log.write(f"Geo: {data_config['geo']}, Segment: {data_config['segment']}, Initial ID: {data_config['initial_id']}, "
                  f"L2 Loss Time: {l2_loss_time}, L2 Loss Velocity: {l2_loss_velocity}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimization baseline experiment")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    parser.add_argument("--geo", type=str, help="Geometry (e.g., EC, DC, AW, pstov1, pstov2, pstov3)")
    parser.add_argument("--segment", type=int, help="Segment (0-4)")
    parser.add_argument("--initial_id", type=int, help="Initial ID (1-Nodes in mesh)")

    # # update this part if debugging through vscode 
    # args = parser.parse_args()

    args = argparse.Namespace()
    args.config = 'Baseline_Opt/config/opt.json'
    args.geo = 'EC'
    args.segment = 1
    args.initial_id = 0
    # Load the config file
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Update config with command-line arguments
    config["data"]["geo"] = args.geo
    config["data"]["segment"] = args.segment
    config["data"]["initial_id"] = args.initial_id

    pred_time_now = [0]
    pred_time_min = [0]
    pred_seg_min = [0]
    pred_velocity_min = [0]
    min_error = [10000]
    
    run_experiment(config)
