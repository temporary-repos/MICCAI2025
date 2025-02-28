import os
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

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



    def __len__(self):
        # Assuming that all inputs have the s
        return len(self.inputs_residual)

def load_geo(filepath, segpath):
    # return np.load(filepath)
    mat_contents = scipy.io.loadmat(filepath)
    segs = np.fromfile(segpath, dtype=np.int32).flatten()
    nodes = mat_contents['nodes']
    faces = mat_contents['edges'].T
    
    return nodes, faces, segs

def load_data(data_dir, geo, segment, initial, num_nodes, num_data, noise_level):
    # return np.load(filepath)
    file_path = os.path.join(data_dir, 'dataset', f'{geo}_at_map_seg{segment}_exc{initial}.pt')
    try:
        # Attempt to load the file
        sample = torch.load(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping to the next iteration...")
        return  None, None, None, None# Skip this iteration and return to the caller
    
    # sample = torch.load(file_path)
    activation_times_org, velocity_field_org = sample.activation_times, sample.velocity_field
    # activation_times_org = activation_times_org/1000
    
    mask_matrix = np.zeros_like(activation_times_org, dtype=np.float32)
    indices_data = random.sample(range(num_nodes), num_data)
    mask_matrix[indices_data] = 1
    
    # noise = np.random.uniform(-noise_level/2, noise_level/2, activation_times_org.shape)
    # at_noise = activation_times_org + noise
    at_noise = add_noise_with_snr(activation_times_org, noise_level)
    
    return activation_times_org, velocity_field_org, mask_matrix, at_noise


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