import pybobyqa
import numpy as np
from eikonal_solver import EikonalSolverPointCloud

class Optimizer:
    def __init__(self):
        self.pred_time_now = [0]
        self.pred_time_min = [0]
        self.pred_velocity_min = [0]
        self.min_error = [10000]
        self.pred_seg_min = [0]

    def objective_function(self, velocity_field_trial, source_points, points, faces, observed_activation_times, arg1=None, arg2=None, arg3=None):
        # solver.update_velocity_field(velocity_field_trial)
        solver = EikonalSolverPointCloud(points, faces, velocity_field_trial)
        predicted_activation_times = solver.solve(source_points)
        predicted_activation_times = predicted_activation_times/1000
        # pred_time_now[0] = predicted_activation_times
        self.pred_time_now[0] = predicted_activation_times
        error_val = np.mean((observed_activation_times - predicted_activation_times) ** 2)
        print(f"Error: {error_val}", flush=True)
        if error_val < self.min_error[0]:
            self.min_error[0] = error_val
            self.pred_time_min[0] = predicted_activation_times
            # self.pred_seg_min[0] = velocity_field_trial.copy()
            self.pred_velocity_min[0] = velocity_field_trial.copy()
        return error_val

    # TO DO double check if everything is gettnig values
    def segment_objective_function(self, z_initial, source_points, points, triangs, observed_activation_times, segs_dict, segs, sigma_values):
        sigma_h, sigma_s = sigma_values
        velocity_field = np.full(points.shape[0], sigma_s)
        for k in segs_dict.keys():
            mask = np.where(np.isin(segs, segs_dict[k]))[0]
            velocity_field[mask] = z_initial[int(k)]
        
        solver = EikonalSolverPointCloud(points, triangs, velocity_field)
        predicted_activation_times = solver.solve([source_points,])

        self.pred_time_now[0] = predicted_activation_times

        # print("Observed Activation Times:", observed_activation_times)
        # print("Predicted Activation Times:", predicted_activation_times)

        # error_val = np.sqrt(np.mean((observed_activation_times - predicted_activation_times) ** 2))
        error_val = np.mean(np.abs(observed_activation_times - predicted_activation_times))
        
        # print(f"error: {error_val}", flush=True)

        if error_val < self.min_error[0]:
            self.min_error[0] = error_val
            self.pred_time_min[0] = predicted_activation_times
            self.pred_seg_min[0] = z_initial.copy()
            self.pred_velocity_min[0] = velocity_field.copy()
            
        return error_val

    # Instantiate the Eikonal solver with the initial velocity field
    def create_initial_velocity_field(self, sigma_h, sigma_s, gt_seg):
        # z = np.random.uniform(0, 1, len(segs))
        z = np.full(len(segs), 0.1)
        return z


    def optimize_velocity_field(self, type, velocity_field, bounds_val, *args):
        lower = np.ones_like(velocity_field)*bounds_val[0]
        upper = np.ones_like(velocity_field)*bounds_val[1]
        # bounds = [(low, up) for low, up in zip(lower, upper)]
        if type == 'segment':
            result = pybobyqa.solve(
                self.segment_objective_function, velocity_field,
                args=args,
                bounds=(lower,upper),
                # seek_global_minimum=True
                maxfun=500
            )
        else:
            result = pybobyqa.solve(
            self.objective_function, velocity_field,
            args=args,
            bounds=(lower,upper),
            seek_global_minimum=True,
            maxfun=500
            )
            
        return result.x

