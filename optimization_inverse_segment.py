import torch
import os
import numpy as np
import scipy.io
import random
import time
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from utils.utils import Params
import dataset_mesh as dataset_mesh
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pybobyqa
# from fimpy.solver import FIMPY
import utils as ut
import heapq

os.environ['CUDA_VISIBLE_DEVICES'] = '8'
json_name = 'test.json'
json_path = 'config/' + json_name
hparams = Params(json_path)
# ut.set_seed(100)
# exp_desc = hparams.exp_description
data_dir = 'data/volumetric_combined/dataset/'
# num_residual, num_data = hparams.training['num_residual'], hparams.training['num_data']
task_ids = {1:0, 3:1, 5:2, 13:3}
initial_ids = [i for i in range(0,1862,50)]
exp_desc = "check_segment_2"


seg_file = np.fromfile('data/volumetric/heart.seg', dtype=np.int32).flatten()
segs = [[0, 6, 1,7], [2,8,3,9], [4,10,5,11], [12,13,14,15,16]]
segs_dict = {0:[0, 6, 1,7], 1:[2,8,3,9], 2:[4,10,5,11], 3:[12,13,14,15,16]}
sigma_h, sigma_s = hparams.scar['sigma_h'],  hparams.scar['sigma_s']

 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
if not os.path.exists(f'results_{exp_desc}/'):
    os.makedirs(f'results_{exp_desc}/')

res_folder = f'results_{exp_desc}/'

ut.set_seed(hparams.training['seed'])

mat_contents = scipy.io.loadmat('data/volumetric/vheart.mat')
nodes = mat_contents['nodes']
triangs = mat_contents['edges'].T

class EikonalSolverPointCloud:
    def __init__(self, points, faces, velocity_field):
        self.points = points
        self.faces = faces
        self.velocity_field = velocity_field
    
    def get_vertex_neighbors(self, vertex_index):
        # Find neighboring vertices for a given node based on the mesh faces
        neighbors = set()
        for fi in np.where(self.faces == vertex_index)[0]:
            face = self.faces[fi]
            for vi in face:
                if vi != vertex_index:
                    neighbors.add(vi)
        return neighbors

    def solve(self, source_indices):
        # Initialize travel times with a large value
        travel_times = np.full(len(self.points), np.inf)
        
        # Set the travel time of source points to 0
        travel_times[source_indices] = 0
        
        # Use a min heap as the 'front' of the wave propagation
        front = [(0, src) for src in source_indices]
        heapq.heapify(front)
        
        while front:
            current_time, current_point = heapq.heappop(front)
            
            # Update the travel times for neighbors
            for neighbor in self.get_vertex_neighbors(current_point):
                edge_length = np.linalg.norm(self.points[current_point] - self.points[neighbor])
                new_time = current_time + edge_length / self.velocity_field[neighbor]
                
                # If the new time is less than the current estimate, update and push to the heap
                if new_time < travel_times[neighbor]:
                    travel_times[neighbor] = new_time
                    heapq.heappush(front, (new_time, neighbor))
                    
        return travel_times

# Load the heart mesh from the .mat file
class TaskDataset(Dataset):
    def __init__(self, cor, faces, activation_times, velocity_field, source_points, indices_residual, indices_data, initial_threshold, task_id, sample_id, device):
        # solver = dataset_mesh.IsotropicEikonalSolver(cor, faces, velocity_field)
        # activation_times = solver.solve(source_points)
        data = dataset_mesh.prepare_training_data_3d(cor,faces, activation_times, indices_residual, indices_data, initial_threshold, device)
        self.cor = data[10]
        self.faces = faces
        self.activation_times = activation_times
        self.velocity_field = velocity_field
        self.inputs_residual = data[0]
        self.outputs_residual = data[1]
        self.indices_residual = data[2]
        self.inputs_data = data[3]
        self.outputs_data = data[4]
        self.initial_graph = data[5]
        self.laplacian = data[6]
        self.index = (task_id, sample_id)
        self.indices_data = data[7]
        self.data_mask = data[8]
        # self.grad_operator = data[9]
        # weight_mask_residual = torch.ones((self.cor.shape[0]))*0.1
        # weight_mask_residual[self.indices_residual] = 1
        # self.weight_mask_residual = weight_mask_residual


    def __len__(self):
        # Assuming that all inputs have the same number of data points
        return len(self.inputs_residual)

# Create a DataLoader for each task
data_loaders = []
true_indices = {}

indices_residual = random.sample(range(len(nodes)), num_residual)
# indices_data = random.sample(range(len(nodes)), num_data)
indices_data = np.arange(220,477)
initial_threshold = hparams.dataset['initial_threshold']

for task_id in task_ids:
    sample_dataset = []
    initial_count = 0
    while len(sample_dataset) < 2:
        initial_id = initial_ids[initial_count]
        initial_count += 1
        file_path = os.path.join(data_dir, f'at_map_seg{task_id}_exc{initial_id}.pt')
        if os.path.exists(file_path):
            sample = torch.load(file_path) 

            # Normalize the matrix
            dataset = TaskDataset(sample.cor, sample.faces, sample.activation_times, sample.velocity_field, initial_id, indices_residual, indices_data, initial_threshold, task_id, initial_id, device)
            
            dataset.index = (task_ids[task_id], initial_id)
            sample_dataset.append(dataset)
            
        else:
            print(f"File {file_path} not found. Ensure the dataset is generated and saved correctly.")
    
    data_loaders.append(sample_dataset)     

pred_time_now = [0]
pred_time_min = [0]
pred_seg_min = [0]
pred_velocity_min = [0]
min_error = [10000]

def objective_function(z_initial, source_points, points, triangs, observed_activation_times):
    global min_error
    velocity_field = np.full(points.shape[0], sigma_h)
    for k in segs_dict.keys():
        mask = np.where(np.isin(seg_file, segs_dict[k]))[0]
        velocity_field[mask] = z_initial[k]

    solver = EikonalSolverPointCloud(points, triangs, velocity_field)
    predicted_activation_times = solver.solve([source_points,])

    pred_time_now[0] = predicted_activation_times

    error_val = np.sqrt(np.mean(observed_activation_times - predicted_activation_times) ** 2)
    
    print(f"error: {error_val}", flush=True)

    if error_val < min_error[0]:
        min_error[0] = error_val
        pred_time_min[0] = predicted_activation_times
        pred_seg_min[0] = z_initial.copy()
        pred_velocity_min[0] = velocity_field.copy()
        
    return error_val

# Instantiate the Eikonal solver with the initial velocity field
def create_initial_velocity_field(sigma_h, sigma_s, gt_seg):
    # z = np.random.uniform(0, 1, len(segs))
    z = np.full(len(segs), 0.1)
    # random_index = random.randint(0, len(segs))
    # z = [(sigma_h + 0.2) if i != gt_seg else (sigma_s+0.2) for i in range(len(segs))]
    return z

def add_noise_with_snr(signal, snr_db):
    """
    Add noise to the given signal such that the resultant signal-to-noise ratio is snr_db.

    Parameters:
    - signal (np.array): The original signal.
    - snr_db (float): Desired signal-to-noise ratio in decibels (dB).

    Returns:
    - np.array: Signal with added noise.
    """
    # Calculate the power of the original signal
    signal_power = np.mean(signal**2)

    # Calculate the required noise power based on the desired SNR
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = signal_power / snr_linear

    # Calculate standard deviation of the noise
    noise_std = np.sqrt(noise_power)

    # Generate noise with the calculated standard deviation
    noise = noise_std * np.random.randn(*signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise

    return noisy_signal

dataset = data_loaders[2][0]
source_point = np.where(dataset.activation_times < 0.001)
source_point = source_point[0]
data_output = dataset.activation_times
snr_db = 25
observed_activation_times = add_noise_with_snr(data_output, snr_db)
# noise = np.random.uniform(-noise_level/2, noise_level/2, data_output.shape)
# observed_activation_times = data_output + noise
gt_seg = dataset.index[0]
z_initial = create_initial_velocity_field(sigma_h, sigma_s, gt_seg)

# z_initial = z.copy()
initial_velocity_field = np.full(nodes.shape[0], sigma_h)
for k in segs_dict.keys():
    mask = np.where(np.isin(seg_file, segs_dict[k]))[0]
    initial_velocity_field[mask] = z_initial[k]


lower = np.ones_like(z_initial) * 0.001
upper = np.ones_like(z_initial)
# soln = pybobyqa.solve(objective_function, z_initial, args= (source_point, nodes, triangs, observed_activation_times), do_logging = True, print_progress = True, maxfun = 20000, bounds=(lower, upper), rhobeg = 0.2, rhoend = 1e-8)
soln = pybobyqa.solve(objective_function, z_initial, args= (source_point, nodes, triangs, observed_activation_times), do_logging = True, print_progress = True, seek_global_minimum=False, maxfun = 1000, bounds=(lower, upper), rhobeg = 1e-1, rhoend = 1e-16)

optimized_velocity_latent = soln.x
optimized_velocity_field = np.full(nodes.shape[0], sigma_h)
for k in segs_dict.keys():
    mask = np.where(np.isin(seg_file, segs_dict[k]))[0]
    optimized_velocity_field[mask] = optimized_velocity_latent[k]
velocity = dataset.velocity_field

ut.write_vtk_data_pointcloud(velocity, dataset.cor, res_folder+f'eval_velocity_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(initial_velocity_field, dataset.cor, res_folder+f'eval_initial_velocity_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(optimized_velocity_field, dataset.cor, res_folder+f'eval_pred_velocity_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(pred_velocity_min[0], dataset.cor, res_folder+f'eval_pred_velocity_mid_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(pred_time_min[0], dataset.cor, res_folder+f'eval_pred_at_mid_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(dataset.activation_times, dataset.cor, res_folder+f'eval_at_id_{0}_initial_{0}.vtk')
ut.write_vtk_data_pointcloud(pred_time_now[0], dataset.cor, res_folder+f'eval_at_pred_id_{0}_initial_{0}.vtk')