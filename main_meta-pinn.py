import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import logging
import scipy.io
from scipy.stats import linregress
from collections import defaultdict
import shutil
import psutil
import random
import time
from torch.optim import AdamW
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import Dataset, DataLoader
import utils.utils as ut
import torch.nn.init as init
import graph_network_arch4 as net
import matplotlib.pyplot as plt
from utils.utils import Params
import dataset_mesh as dataset_mesh
import lr
import pickle

# adjust experiment parameters
parser = argparse.ArgumentParser(description="Experiment parameters")
    
# Add arguments with default values
parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device to use (default: 0)")
parser.add_argument("--json_name", type=str, default="experiment1.json", help="Name of the JSON configuration file")
parser.add_argument("--generate_data", action="store_true", default=True, help="Flag to enable data generation (default: False)")
parser.add_argument("--model_type", type=str, default="meta_base", help="choose from meta-pinn, meta-base and opt")
seg_map = [[0,1,6,7], [4,5,10,11], [2,3,8,9], [12,13,14,15,16], [17]]

# Parse arguments
args = parser.parse_args()

# # Use parsed arguments
json_name = args.json_name
generate_data = args.generate_data
model_type = args.model_type

data_dir = "data/registered_5_updated/"
initial_ids = [i for i in range(0, 1862, 25)]

# read parameters from the config file
json_path = "config/" + json_name
hparams = Params(json_path)
is_train = hparams.train_stage
exp_desc = hparams.exp_description
learning_rate = hparams.training["lr"]
task_ids = hparams.dataset["task_ids"]
geo_ids = hparams.dataset["geo_ids"]
geo_ids_op = {0: "EC", 1: "AW", 2: "DC", 3: "pstov1", 4: "pstov2", 5: "pstov3"}
epochs = hparams.training["epochs"]
data_type = hparams.dataset["type"]
residual_perc, data_perc = (
    hparams.training["residual_perc"],
    hparams.training["data_perc"],
)
exp_desc = (
    exp_desc
    + "_id"
    + str(hparams.exp_id)
    + "_support_"
    + str(hparams.training["data_perc"])
    + "_epochs_"
    + str(epochs)
)
# num_samples = hparams.dataset["num_samples"]
gi_gsum_per_epoch = defaultdict(list)
probs_per_epoch = defaultdict(list)

device = torch.device('cuda:'+str(args.cuda_device) if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(8)

if not os.path.exists(f"Experiments/results_{exp_desc}/"):
    os.makedirs(f"Experiments/results_{exp_desc}/")

destination_file = os.path.join(f"Experiments/results_{exp_desc}/", os.path.basename(json_name))
shutil.copy(json_path, destination_file)
ut.set_seed(hparams.training["seed"])


class metaPINN(nn.Module):
    def __init__(self):
        super(metaPINN, self).__init__()

        # Main PINN network layers
        self.network = net.Base(hparams, "pinn").to(device)

        # Initialize main network layers weights
        for layer in self.network.net:
            if isinstance(layer, nn.Linear):
                # init.xavier_normal_(layer.weight)
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Scar/velocity network layers
        self.scar_network = net.Base(hparams, "scar").to(device)

        # Initialize scar network layers
        for layer in self.scar_network.net:
            if isinstance(layer, nn.Linear):
                # init.xavier_normal_(layer.weight)
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # PINN hypernetwork, Scar hypernetwork, AT encoder (for tissue embeddings/PINN embedding), initial_encoder for the initial activation
        self.hypernet_pinn = net.Hypernet(
            hparams=hparams, concat_initial=True, network_type="pinn"
        )
        self.hypernet_scar = net.Hypernet(
            hparams=hparams, concat_initial=False, network_type="scar"
        )
        self.encoder = net.Encoder(hparams)
        self.initial_encoder = net.Encoder(hparams)
        # self.AttentionAggregator = net.AttentionAggregator(hparams.encoder['out_dim'])
        # self.decoder = net.Decoder(hparams)

    # forward function --> generate both tissue and initial embeddings, aggregating to create the input for hypernetwork
    def forward(self, agg_embedding, data_loaders, init_id):
        init_embedding = self.initial_encoder(
            data_loaders[init_id].key,
            data_loaders[init_id].initial_graph,
            p_hier,
            g_hier,
            device,
        )
        pinn_embedding = torch.cat((agg_embedding, init_embedding), dim=1)
        net_weight, net_bias, net_bns = self.hypernet_pinn(pinn_embedding)
        self.network.set_params(net_weight, net_bias, net_bns)

    # one forward pass to outptut tissue embeddings
    def estimate_tissue_embedding(self, geo_name, encoder_input):
        return self.encoder(geo_name, encoder_input, p_hier, g_hier, device)

    # one forward pass to outptut tissue properties (at data space)
    def parameter_estimate(self, x):
        return self.scar_network(x)

    def prepare_encoder_input(self, data_loaders):
        sigmas = []
        if hparams.dataset["single"] == True:
            sigma_agg = self.estimate_tissue_embedding(
                dataset.key, data_loaders[0].pinn_graph
            )

        else:
            # preparing the tissue embeddings for the k samples in the k-shot setting, and aggregatng by taking the mean
            for i, dataset in enumerate(data_loaders[: hparams.dataset["k"]]):
                sigma = self.estimate_tissue_embedding(dataset.key, dataset.pinn_graph)
                sigmas.append(sigma)
            sigma_agg = torch.mean(torch.stack(sigmas), dim=0)

        # scar hypernetwork to output the tissue properties
        scar_net_weight, scar_net_bias, net_bns = self.hypernet_scar(sigma_agg)
        self.scar_network.set_params(scar_net_weight, scar_net_bias, net_bns)

        return sigma_agg

class metaBase(nn.Module):
    def __init__(self):
        super(metaBase, self).__init__()

        # Main PINN network layers
        self.network = net.Base(hparams, "pinn").to(device)

        # Initialize main network layers weights
        for layer in self.network.net:
            if isinstance(layer, nn.Linear):
                # init.xavier_normal_(layer.weight)
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # PINN hypernetwork, Scar hypernetwork, AT encoder (for tissue embeddings/PINN embedding), initial_encoder for the initial activation
        self.hypernet_pinn = net.Hypernet(
            hparams=hparams, concat_initial=True, network_type="pinn"
        )

        self.encoder = net.Encoder(hparams)
        self.initial_encoder = net.Encoder(hparams)


    # forward function --> generate both tissue and initial embeddings, aggregating to create the input for hypernetwork
    def forward(self, agg_embedding, data_loaders, init_id):
        init_embedding = self.initial_encoder(
            data_loaders[init_id].key,
            data_loaders[init_id].initial_graph,
            p_hier,
            g_hier,
            device,
        )
        pinn_embedding = torch.cat((agg_embedding, init_embedding), dim=1)
        net_weight, net_bias, net_bns = self.hypernet_pinn(pinn_embedding)
        self.network.set_params(net_weight, net_bias, net_bns)

    # one forward pass to outptut tissue embeddings
    def estimate_tissue_embedding(self, geo_name, encoder_input):
        return self.encoder(geo_name, encoder_input, p_hier, g_hier, device)

    def prepare_encoder_input(self, data_loaders):
        sigmas = []
        if hparams.dataset["single"] == True:
            sigma_agg = self.estimate_tissue_embedding(
                dataset.key, data_loaders[0].pinn_graph
            )

        else:
            # preparing the tissue embeddings for the k samples in the k-shot setting, and aggregatng by taking the mean
            for i, dataset in enumerate(data_loaders[: hparams.dataset["k"]]):
                sigma = self.estimate_tissue_embedding(dataset.key, dataset.pinn_graph)
                sigmas.append(sigma)
            sigma_agg = torch.mean(torch.stack(sigmas), dim=0)

        return sigma_agg

class VelocityNetwork(nn.Module):
    def __init__(self):
        super(VelocityNetwork, self).__init__()
        self.network = net.Base(hparams, "scar").to(device)
        
        for layer in self.network.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.network(x)

class PINN(nn.Module):
    def __init__(self, num_nodes, velocity_network):
        super(PINN, self).__init__()
        self.network = net.Base(hparams, "pinn").to(device)
        self.scar_network = velocity_network  # Shared across all initial conditions
        
        for layer in self.network.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        activation_time = self.network(x)
        velocity = self.scar_network(x)  # Shared velocity network
        return activation_time, velocity

class SinglePINN(nn.Module):
    def __init__(self, num_nodes):
        super(PINN, self).__init__()

        # PINN network for forward mapping (e.g., activation time)
        self.network = net.Base(hparams, "pinn").to(device)
        # Velocity network for forward mapping (e.g., activation time)
        self.scar_network = net.Base(hparams, "scar").to(device)

        # Initialize PINN weights
        for layer in self.network.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize PINN weights
        for layer in self.scar_network.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # # Inverse parameter (e.g., velocities) as a trainable vector
        # self.velocities = nn.Parameter(torch.randn(num_nodes, 1, device=device))

    def forward(self, x):
        # Forward pass for activation time
        activation_time = self.network(x)
        
        # Velocities are directly optimized parameters
        velocity = self.scar_network(x)
        
        return activation_time, velocity

# Load the dataset
class TaskDataset(Dataset):
    def __init__(
        self,
        key,
        segs,
        cor,
        faces,
        activation_times,
        velocity_field,
        source_points,
        indices_residual,
        indices_data,
        indices_loss_data,
        initial_threshold,
        task_id,
        sample_id,
        geo_id,
        device,
    ):
        graph_data = g_hier[key][0]
        
        data = dataset_mesh.prepare_training_data_3d(
            graph_data,
            cor,
            faces,
            activation_times,
            indices_residual,
            indices_data,
            initial_threshold,
            device,
        )
        self.key = key
        self.segs = segs
        self.cor_org = cor
        self.cor = data[10]
        self.faces = faces
        self.activation_times = activation_times
        self.velocity_field = velocity_field
        self.inputs_residual = data[0]
        self.outputs_residual = data[1]
        self.indices_residual = data[2]
        self.inputs_data = data[3]
        self.outputs_data = data[4]
        self.pinn_graph = data[5]
        self.initial_graph = data[6]
        self.laplacian = data[7]
        self.index = (task_id, sample_id, geo_id)
        self.indices_data = data[8]
        self.indices_loss_data = indices_loss_data
        self.data_mask = data[9]
        self.initial_indices = data[11]
        weight_mask_residual = torch.ones((self.cor.shape[0])) * 0.1
        weight_mask_residual[self.indices_residual] = 1
        self.weight_mask_residual = weight_mask_residual

    def __len__(self):
        # Assuming that all inputs have the same number of data points
        return len(self.inputs_residual)

    # def __getitem__(self, idx):
    #     # This method should return a tuple of your data points
    #     return (self.inputs_residual[idx], self.outputs_residual[idx], self.indices_residual[idx],
    #             self.inputs_data[idx], self.outputs_data[idx])


# prepare graph hierarchy
g_hier = {}
p_hier = {}
for geo_id in geo_ids:
    g_hier[geo_id] = []
    p_hier[geo_id] = []
    filename = data_dir + "hierarchy/" + geo_id + "_graclus_hier.pickle"
    with open(filename, "rb") as f:
        g = pickle.load(f)
        g_hier[geo_id].append(g)
        g1 = pickle.load(f)
        g_hier[geo_id].append(g1)
        g2 = pickle.load(f)
        g_hier[geo_id].append(g2)
        g3 = pickle.load(f)
        g_hier[geo_id].append(g3)
        g4 = pickle.load(f)
        g_hier[geo_id].append(g4)
        g5 = pickle.load(f)
        g_hier[geo_id].append(g5)
        g6 = pickle.load(f)
        g_hier[geo_id].append(g6)

        P01 = pickle.load(f)
        P01 = torch.tensor(P01, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P01)
        P12 = pickle.load(f)
        P12 = torch.tensor(P12, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P12)
        P23 = pickle.load(f)
        P23 = torch.tensor(P23, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P23)
        P34 = pickle.load(f)
        P34 = torch.tensor(P34, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P34)
        P45 = pickle.load(f)
        P45 = torch.tensor(P45, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P45)
        P56 = pickle.load(f)
        P56 = torch.tensor(P56, dtype=torch.float32, requires_grad=True).to(device)
        p_hier[geo_id].append(P56)


grad_operators = {}
for geo_id in geo_ids_op:
    geo_path = os.path.join(data_dir, f"geometry/{geo_ids_op[geo_id]}.mat")
    geometry = scipy.io.loadmat(geo_path)
    cor_org = geometry["nodes"]
    faces = geometry["edges"].transpose()
    gradient_operator, gradient_operator_basis = dataset_mesh.get_grad_operator(cor_org, faces, device)
    grad_operators[geo_id] = (gradient_operator, gradient_operator_basis)
    
    

data_loaders = []
datasets = []
data_iters = []
sample_iters = []
data_iters = []
exp_id = hparams.exp_id

if generate_data is True:
    initial_threshold = hparams.dataset["initial_threshold"]
    for i, geo in enumerate(geo_ids.keys()):
        geo_path = os.path.join(data_dir, f"geometry/{geo}.mat")
        geometry = scipy.io.loadmat(geo_path)
        indices_residual = random.sample(
            range(len(geometry["nodes"])), int(residual_perc * len(geometry["nodes"]))
        )
        indices_data,  indices_loss_data = dataset_mesh.get_data_indx(data_type, len(geometry["nodes"]), data_perc, data_dir, geo)
        for task_id in task_ids:
            sample_dataset = []
            initial_count = 0
            while (
                len(sample_dataset)
                < hparams.dataset["num_train_maps"] + hparams.dataset["num_test_maps"] + hparams.dataset["num_val_maps"]
            ):
                # while len(sample_dataset) < 1:
                initial_id = initial_ids[initial_count]
                initial_count += 1
                file_path = os.path.join(
                    data_dir, f"dataset/{geo}_at_map_seg{task_id}_exc{initial_id}.pt"
                )
                
                seg_path = os.path.join(
                    data_dir, f"geometry/{geo}.seg"
                )
                segs = np.fromfile(seg_path, dtype=np.int32).flatten()
                
                
                if os.path.exists(file_path):
                    sample = torch.load(file_path)

                    # Normalize the matrix
                    dataset = TaskDataset(
                        geo,
                        segs,
                        sample.cor,
                        sample.faces,
                        sample.activation_times,
                        sample.velocity_field,
                        initial_id,
                        indices_residual,
                        indices_data,
                        indices_loss_data,
                        initial_threshold,
                        task_id,
                        initial_id,
                        i,
                        device,
                    )

                    dataset.index = (task_id, initial_id, i)
                    sample_dataset.append(dataset)

                else:
                    print(
                        f"File {file_path} not found. Ensure the dataset is generated and saved correctly."
                    )

            data_loaders.append(sample_dataset)

    # Save the data using pickle
    with open(
        "data/registered_5_updated/" + str(exp_id) + "_data_loaders.pkl", "wb"
    ) as file:
        pickle.dump(data_loaders, file)
    train_loaders = []
    test_loaders = []
    val_loaders = []
    for j, tasks in enumerate(data_loaders):
        all_data = data_loaders[j]
        random.shuffle(all_data)
        if hparams.generalization["type"] == "initial":
            train_loaders.append(all_data[0 : hparams.dataset["num_train_maps"]])
            test_loaders.append(
                all_data[
                    hparams.dataset["num_train_maps"] : hparams.dataset["num_train_maps"]
                    + hparams.dataset["num_test_maps"]
                ]
            )
            val_loaders.append(all_data[hparams.dataset["num_train_maps"]+ hparams.dataset["num_test_maps"]: hparams.dataset["num_train_maps"]
                            + hparams.dataset["num_test_maps"] + hparams.dataset["num_val_maps"]
                            ])
        elif hparams.generalization["type"] == "geometry":
            if all_data[0].key == hparams.generalization["geometry"]:
                test_loaders.append(
                all_data[
                    hparams.dataset["num_train_maps"] : hparams.dataset["num_train_maps"]
                    + hparams.dataset["num_test_maps"]
                ]
                )
                val_loaders.append(all_data[hparams.dataset["num_train_maps"]+ hparams.dataset["num_test_maps"]: hparams.dataset["num_train_maps"]
                                + hparams.dataset["num_test_maps"] + hparams.dataset["num_val_maps"]
                                ])
            else:
                train_loaders.append(all_data[0 : hparams.dataset["num_train_maps"]])
            # for i in range(len(all_data)):
            #     if all_data[i].key == hparams.generalization["geometry"]:
            #         if i % 2 == 0 and num_test < 10:
            #             test_loaders.append(
            #                 all_data[i]
            #             )
            #             num_test += 1
            #         elif i % 2 == 1 and num_val < 10:
            #             val_loaders.append(all_data[hparams.dataset["num_train_maps"]+ hparams.dataset["num_test_maps"]: hparams.dataset["num_train_maps"]
            #                             + hparams.dataset["num_test_maps"] + hparams.dataset["num_val_maps"]
            #                             ])
            #             num_val += 1
                    
            #     else:
            #         if num_train < 10:
            #             train_loaders.append(all_data[0 : hparams.dataset["num_train_maps"]])
            #             num_train += 1
        
        elif hparams.generalization["type"] == "segment":
            # if all_data[i].index[0] == hparams.generalization["segment"]:
            if all_data[i].index[0] == all_data[i].index[2]:
                test_loaders.append(
                all_data[
                    hparams.dataset["num_train_maps"] : hparams.dataset["num_train_maps"]
                    + hparams.dataset["num_test_maps"]
                ]
                )
                val_loaders.append(all_data[hparams.dataset["num_train_maps"]+ hparams.dataset["num_test_maps"]: hparams.dataset["num_train_maps"]
                                + hparams.dataset["num_test_maps"] + hparams.dataset["num_val_maps"]
                                ])
            else:
                train_loaders.append(all_data[0 : hparams.dataset["num_train_maps"]])
            
        
        
    with open(
        "data/registered_5_updated/" +  str(exp_id) + "_train_loaders.pkl", "wb"
    ) as file:
        pickle.dump(train_loaders, file)
    with open(
        "data/registered_5_updated/" +  str(exp_id) + "_test_loaders.pkl", "wb"
    ) as file:
        pickle.dump(test_loaders, file)
    with open(
        "data/registered_5_updated/" +  str(exp_id) + "_val_loaders.pkl", "wb"
    ) as file:
        pickle.dump(val_loaders, file)
    data_loaders = train_loaders

else:
    with open(
        "data/registered_5_updated/" + str(exp_id) + "_train_loaders.pkl", "rb"
    ) as file:
        data_loaders = pickle.load(file)
    with open(
        "data/registered_5_updated/" + str(exp_id) + "_test_loaders.pkl", "rb"
    ) as file:
        test_loaders = pickle.load(file)
    with open(
        "data/registered_5_updated/" + str(exp_id) + "_val_loaders.pkl", "rb"
    ) as file:
        val_loaders = pickle.load(file)


def data_episodic_update(data_loaders):
    for data_loader in data_loaders:
        random.shuffle(data_loader)

    return data_loaders

def data_loss(model, points, act_time, data_loss_indices, weight_mask):
    predicted_values = model.network(points) * 1000
    # predicted_values = model.network(points)
    predicted_values = predicted_values.view(-1)
    full_data_loss = torch.abs(predicted_values - act_time)
    predicted_values = predicted_values[data_loss_indices]
    if torch.isnan(predicted_values).any():
        print("NaNs detected in predicted_values")
    return full_data_loss, torch.mean(torch.abs(weight_mask.squeeze().to(device)[data_loss_indices]*(predicted_values - act_time[data_loss_indices])))
    # return torch.mean(torch.abs(predicted_values - act_time[data_loss_indices]))

def exponential_scaling_with_rescaling(arr, p=2):
    """
    Applies exponential scaling and rescales the values back to the original range (PyTorch version).

    Parameters:
    - arr: torch.Tensor, input data to be scaled
    - p: float, exponent parameter (p > 1 stretches values towards extremes)

    Returns:
    - Scaled torch.Tensor with the same min-max range as the original.
    """
    # Step 1: Apply exponential scaling
    transformed = torch.sign(arr) * torch.abs(arr) ** p

    # Step 2: Normalize to original range
    orig_min, orig_max = torch.min(arr), torch.max(arr)
    trans_min, trans_max = torch.min(transformed), torch.max(transformed)

    # Avoid division by zero
    if trans_max - trans_min == 0:
        return arr

    # Rescale to the original range
    rescaled = orig_min + (transformed - trans_min) * (orig_max - orig_min) / (trans_max - trans_min)
    
    return rescaled

def pde_loss(model_type, at_map, cor, inputs, sigma, grad_operator, grad_operator_basis, laplacian, weight_mask):
    if model_type == "pinn" or model_type == "multi_pinn":
        output, sigma = model(inputs)
    elif model_type == "meta_pinn":
        output = model.network(inputs)
    # Scale pinn_prediction
    pinn_prediction = exponential_scaling_with_rescaling(output, p=3)
    pinn_prediction = output * 1000
    # Compute the gradient of activation times
    grad_activation_times = grad_operator @ pinn_prediction
    # Reshape the gradient to separate the x and y components in the tangent space
    grad_2D = grad_activation_times.reshape(-1, 2)

    # Project the 2D gradient into 3D using the tangent basis
    basis_x = torch.tensor(grad_operator_basis[1]).to(device)
    basis_y = torch.tensor(grad_operator_basis[2]).to(device)

    grad_3D = grad_2D[:, 0:1] * basis_x + grad_2D[:, 1:] * basis_y

    # Compute the magnitude of the gradient
    grad_magnitude = torch.norm(grad_3D, dim=1).view(-1, 1)
    # grad_magnitude = torch.clamp(torch.norm(grad_3D, dim=1), min=1e-7, max=1e3)

    # Compute the PDE loss
    full_loss = (grad_magnitude * sigma - 1) ** 2

    # Apply the weight mask
    masked_loss = torch.mean(weight_mask.squeeze().to(device) * full_loss)
    # masked_loss = weight_mask.squeeze().to(device) * full_loss
    pde_std = torch.std(sigma)
    # Compute Laplacian regularization loss
    # laplacian_U = torch.sparse.mm(laplacian.to(device), pinn_prediction)  # Compute ΔU
    laplacian_U = torch.sparse.mm(laplacian, pinn_prediction)  # Compute ΔU
    loss_laplacian = torch.mean(laplacian_U ** 2)  # Squared Laplacian loss
    
    
    return masked_loss, pde_std, full_loss, loss_laplacian

def gradient_check(at, grad_operator_basis):
    at = at.to(device)
    grad_operator = ut.scipy_to_torch_sparse(grad_operator_basis[0]).to(device)
    grad_activation_times = grad_operator @ at
    grad_2D = grad_activation_times.reshape(-1, 2)
    # Project the 2D gradient into 3D using the tangent basis
    basis_x = torch.tensor(grad_operator_basis[1]).to(device)
    basis_y = torch.tensor(grad_operator_basis[2]).to(device)

    grad_3D = grad_2D[:, 0:1] * basis_x + grad_2D[:, 1:] * basis_y

    # Compute the magnitude of the gradient in 3D (Euclidean norm)
    grad_magnitude = torch.norm(grad_3D, dim=1)

    return grad_magnitude

def evaluate(data_scaled, pinn_prediction, at_map, output_velocity, velocity, model_type):
    b = pinn_prediction.cpu().detach().numpy().squeeze()
    a = at_map
    # Calculate RMSE
    rmse = np.sqrt(np.mean((a - b) ** 2))
    # Calculate the range of a
    range_a = np.max(a) - np.min(a)
    # Calculate Normalized Relative RMSE
    data_normalized_relative_rmse = rmse / range_a
    # Calculate Correlation Coefficient
    data_corr_coef = np.corrcoef(a.flatten(), b.flatten())[0, 1]

    if model_type == 'meta_pinn':
        b = output_velocity.cpu().detach().numpy().squeeze()
        a = velocity
        # Calculate RMSE
        rmse = np.sqrt(np.mean((a - b) ** 2))
        # Calculate the range of a
        range_a = np.max(a) - np.min(a)
        # Calculate Normalized Relative RMSE
        velocity_normalized_relative_rmse = rmse / range_a
        # Calculate Correlation Coefficient
        velocity_corr_coef = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    elif model_type == 'meta_base':
        velocity_normalized_relative_rmse = None
        velocity_corr_coef = None

    return data_normalized_relative_rmse, velocity_normalized_relative_rmse, data_corr_coef, velocity_corr_coef

def check_nan_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name} parameters")
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in {name} gradients")
            exit()

def generate_velocity_map(
   result_folder, train_val, data_scaled, epoch, velocity, output_velocity, data_id, initial_id, geo_id, plot
):
    if plot:
        if epoch == 0:
            ut.write_vtk_data_pointcloud(
                velocity,
                data_scaled,
                f"{result_folder}/{train_val}_velocity_epoch_{epoch}_id_geo_{geo_id}_{data_id}_initial_{initial_id}.vtk",
            )
        ut.write_vtk_data_pointcloud(
            output_velocity,
            data_scaled,
            f"{result_folder}/{train_val}_velocity_prediction_epoch_{epoch}_id_geo_{geo_id}_{data_id}_initial_{initial_id}.vtk",
        )

    return output_velocity

# tsne to be used for visualizig the embeddings, testinh the disentangelment
def tsne(dataloaders, epoch, result_folder):
    # model.eval()
    embeddings = []
    task_labels = []  # Keep track of task labels for coloring

    # Define a list of colors or use a colormap
    colors = cm.rainbow(np.linspace(0, 1, len(dataloaders)))

    # get embeddings for all samples
    for d in range(len(dataloaders)):
        for i in range(len(dataloaders[d])):
            # sigma = model.estimate_tissue_embedding(dataloaders[d][i].initial_graph)
            # sigma = model.estimate_tissue_embedding(dataset.key, data_loaders[0].pinn_graph)
            task_id, _, _ = dataloaders[d][i].index
            sigma = model.estimate_tissue_embedding(
                dataloaders[d][i].key, dataloaders[d][i].pinn_graph
            )
            
            embeddings.append(sigma.detach().cpu().numpy())
            task_labels.extend(
                [task_id] * len(sigma)
            )  # Extend the task labels list by the batch size

    # Apply t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embeddings = np.concatenate(embeddings, axis=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Visualization
    plt.figure(figsize=(10, 5))

    # Scatter plot for each task with a different color
    for d in range(len(dataloaders)):
        task_indices = [i for i, x in enumerate(task_labels) if x == d]
        plt.scatter(
            reduced_embeddings[task_indices, 0],
            reduced_embeddings[task_indices, 1],
            label=f"Task {d}",
            color=colors[d],
            alpha=0.5,
        )

    plt.legend()
    plt.title("t-SNE visualization of velocity encoder")
    plt.savefig(f"{result_folder}/tsne_epoch_{epoch}.png")



if model_type == 'meta_pinn':
    #create results folder
    result_folder = f"Experiments/results_{exp_desc}/meta_pinn"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # Initialize the PINN
    model = metaPINN().to(device)
    # define optimizers and schedulers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    milestones = [i for i in range(1, epochs, 10)]
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.95
    )
    warmup_steps = 101
    scheduler_2 = lr.CosineAnnealingLR(optimizer, T_max=20, warmup_steps=20, eta_min=1e-7)
    scheduler = scheduler_2
    validation_interval = 10
    # stage 1 -- train loop
    if is_train == 1:
        start_time = time.time()
        pde_losses = []
        data_losses = []
        total_losses = []
        val_pde_losses = []
        val_data_losses = []
        val_losses = []
        best_loss = 10000
        num_tasks = len(data_loaders)        
        freeze = False
        for epoch in range(epochs):
            epoch_data_loss = 0
            epoch_total_loss = 0
            epoch_pde_loss = 0
            epoch_reg_loss = 0
            embeddings = []

            random.shuffle(data_loaders)
            
            # alpha = ut.update_alpha(epoch, start_epoch=1000, end_epoch=5000, start_alpha=1e-5, end_alpha=0.1)
            
            # if epoch < 500:
            #     alpha = 0
            #     freeze = False
            # else:
            #     alpha = ut.update_alpha(epoch, start_epoch=500, end_epoch=1500, start_alpha=1e-4, end_alpha=1)
            #     # alpha = ut.update_alpha(epoch, start_epoch=1000, end_epoch=1500, start_alpha=1e-4, end_alpha=1)
            #     if epoch < 2000:
            #         print("*************Freeze************")
            #         freeze = True
            #     else:
            #         freeze = False
            if epoch < 500:
                alpha = 0
                freeze = False
            else:
                alpha = ut.update_alpha(epoch, start_epoch=500, end_epoch=1500, start_alpha=1e-4, end_alpha=1)
                # alpha = ut.update_alpha(epoch, start_epoch=1000, end_epoch=1500, start_alpha=1e-4, end_alpha=1)
                if epoch < 2000:
                    print("*************Freeze************")
                    freeze = True
                else:
                    freeze = False
            
            # if epoch < 4000:
            #     alpha = 0
            #     freeze = False
            # else:
            #     alpha = ut.update_alpha(epoch, start_epoch=4000, end_epoch=6000, start_alpha=1e-4, end_alpha=1)
            #     # alpha = ut.update_alpha(epoch, start_epoch=1000, end_epoch=1500, start_alpha=1e-4, end_alpha=1)
            #     if epoch < 6000:
            #         print("*************Freeze************")
            #         freeze = True
            #     else:
            #         freeze = False
                            
            data_loaders = data_episodic_update(data_loaders)
            model.train()
            optimizer.zero_grad()

            embeddings = []
            num_tasks = len(data_loaders)
            for data_id in range(num_tasks):
                
                task_data_loss = 0
                task_total_loss = 0
                task_pde_loss = 0
                task_smooth_loss = 0
                task_reg_sigma_loss = 0
                task_reg_pred_loss = 0
                task_laplacian_loss = 0

                # get task embedding -- input to the hypernetwork
                agg_embedding = model.prepare_encoder_input(data_loaders[data_id])

                task_id =  data_loaders[data_id][0].index[0]
                    
                num_considered = len(
                    range(hparams.dataset["k"], len(data_loaders[data_id]))
                )
                
                for initial_id in range(hparams.dataset["k"], len(data_loaders[data_id])):
                    # print(torch.cuda.memory_summary())
                    data_set = data_loaders[data_id][initial_id]
                    cor_org = torch.tensor(
                        data_set.cor_org, dtype=torch.float32, requires_grad=True
                    ).to(device)
                    task_id, sample_id, geo_id = data_set.index
                    segs = data_set.segs
                    at_map = data_set.activation_times
                    true_range = (np.min(at_map), np.max(at_map))
                    at_map = torch.tensor(at_map, dtype=torch.float32).to(device)
                    velocity = data_set.velocity_field
                    laplacian = data_set.laplacian.to(device)
                    inputs_residual = data_set.inputs_residual
                    residual_indices = data_set.indices_residual
                    # input data -- for the data loss
                    inputs_data = data_set.inputs_data
                    outputs_data = data_set.outputs_data
                    
                    # all coodinates
                    data_scaled = torch.tensor(
                        data_set.cor, dtype=torch.float32, requires_grad=True
                    ).to(device)
                    data_mask = data_set.data_mask
                    weight_mask = data_set.weight_mask_residual
                    
                    grad_operator_basis, grad_operator = grad_operators[geo_id]

                    model(agg_embedding, data_loaders[data_id], initial_id)
                    pred_velocities = model.parameter_estimate(data_scaled)

                    loss_pde, loss_pde_std, full_loss, laplacian_loss = pde_loss("meta_pinn",
                        at_map,
                        cor_org,
                        data_scaled,
                        pred_velocities,
                        grad_operator_basis,
                        grad_operator,
                        laplacian,
                        weight_mask)

                    # compute the data loss
                    full_data_loss, loss_data = data_loss(
                        model, inputs_data, at_map, data_set.indices_loss_data, weight_mask
                    )
                    # full_data_loss, loss_data = data_loss(
                    #     model, inputs_data, outputs_data, data_set.indices_loss_data, weight_mask
                    # )
                    # get the output, spatially varying tissue properties/velocities
                    output_velocity = model.parameter_estimate(data_scaled)

                    # loss for regularization -- variance of the pde loss
                    reg_loss_sigma = loss_pde_std

                    # pinn_prediction_ is the output activation time map from the PINN model
                    pinn_prediction = model.network(data_scaled) * 1000
                    # pinn_prediction_ = model.network(data_scaled)
                    total_loss = alpha * loss_pde + loss_data 

                    task_data_loss += loss_data.item()
                    task_pde_loss += loss_pde.item()
                    task_reg_sigma_loss += reg_loss_sigma.item()
                    task_reg_pred_loss += loss_pde_std.item()
                    task_laplacian_loss += laplacian_loss.item()
                    task_total_loss += total_loss
                    

                    # saving intermediate results
                    if epoch == epochs - 1 or epoch % 1000 == 0:
                        with torch.no_grad():
                            # grad_norm = gradient_check(torch.tensor(pinn_prediction, dtype=torch.float32, requires_grad = True).to(device), grad_operator_basis)
                            # grad_norm = gradient_check(pinn_prediction, grad_operator_basis)
                            d_id = data_set.index[0]
                            s_id = data_set.index[1]
                            # ut.write_vtk_data_pointcloud(grad.detach().cpu().numpy(), data_set.cor, f'results_{exp_desc}/debug_epoch_{epoch}_at_grad_{d_id}_initial_{s_id}.vtk')
                            # ut.write_vtk_data_pointcloud(
                            #     grad_norm.detach().cpu().numpy(),
                            #     data_set.cor,
                            #     f"Experiments/results_{exp_desc}/debug_epoch_{epoch}_at_grad_norm_geo_{geo_id}_{d_id}_initial_{s_id}.vtk",
                            # )
                            ut.write_vtk_data_pointcloud(
                                1 / pred_velocities.detach().cpu().numpy(),
                                data_set.cor,
                                f"{result_folder}/1_dev_velocity_epoch_{epoch}_data_geo_{geo_id}_{d_id}_initial_{s_id}.vtk",
                            )
                            ut.write_vtk_data_pointcloud(
                                full_loss.detach().cpu().numpy(),
                                data_set.cor,
                                f"{result_folder}/full_loss__{epoch}_{d_id}_initial_geo_{geo_id}_{s_id}.vtk",
                            )
                            
                            ut.write_vtk_data_pointcloud(
                                full_data_loss.detach().cpu().numpy(),
                                data_set.cor,
                                f"{result_folder}/full_data_loss__{epoch}_{d_id}_initial_geo_{geo_id}_{s_id}.vtk",
                            )

                            
                            ut.AT_map_plot_3d(
                                data_set.cor,
                                epoch,
                                at_map.cpu().numpy(),
                                pinn_prediction.cpu().numpy(),
                                d_id,
                                s_id,
                                geo_id,
                                result_folder,
                                "train"
                            )
                            generate_velocity_map(
                                result_folder,
                                "train",
                                data_scaled.detach().cpu().numpy(),
                                epoch,
                                velocity,
                                output_velocity.detach().cpu().numpy(),
                                d_id,
                                s_id,
                                geo_id,
                                True,
                            )
                        
                           

                    print(
                        f"Sample {initial_id} , Epoch {epoch}, Loss: {total_loss.item()}, Data Loss: {loss_data.item()}, PDE Loss: {loss_pde.item()}"
                    )

                    check_nan_parameters(model)

                print(
                    f"Task {data_id} , Epoch {epoch}, Loss: {task_total_loss.item()/num_considered}, Data Loss: {task_data_loss/num_considered}, PDE Loss: {task_pde_loss/num_considered}, reg_sigma: {task_reg_sigma_loss/num_considered}, reg_at: {task_laplacian_loss/num_considered}"
                )

                epoch_data_loss += task_data_loss / num_considered
                epoch_pde_loss += task_pde_loss / num_considered
                epoch_total_loss += task_total_loss / num_considered
                epoch_reg_loss += task_reg_pred_loss / num_considered
            
            # epoch_total_loss += gama * (contrast_loss / (num_tasks*(num_tasks-1)))
            if epoch % validation_interval == 0:
                # validation loader
                model.eval()
                epoch_val_loss, epoch_val_data_loss, epoch_val_pde_loss = 0, 0, 0
                num_val_batches = 0

                with torch.no_grad():
                    for data_id in range(len(val_loaders)):
                        task_data_loss = 0
                        task_pde_loss = 0
                        task_total_loss = 0
                        # get task embedding -- input to the hypernetwork
                        agg_embedding = model.prepare_encoder_input(val_loaders[data_id])
                        # embeddings.append(agg_embedding)
                        task_id =  val_loaders[data_id][0].index[0]
                        
                        num_considered = len(
                            range(len(val_loaders[data_id]))
                        )
                        
                        for initial_id in range(num_considered):
                            # print(torch.cuda.memory_summary())
                            data_set = val_loaders[data_id][initial_id]
                            cor_org = torch.tensor(
                                data_set.cor_org, dtype=torch.float32, requires_grad=True
                            ).to(device)
                            task_id, sample_id, geo_id = data_set.index
                            at_map = data_set.activation_times
                            at_map = torch.tensor(at_map, dtype=torch.float32).to(device)
                            inputs_data = data_set.inputs_data
                            laplacian = data_set.laplacian.to(device)
                            # all coodinates
                            data_scaled = torch.tensor(
                                data_set.cor, dtype=torch.float32, requires_grad=True
                            ).to(device)
                            data_mask = data_set.data_mask
                            grad_operator_basis, grad_operator = grad_operators[geo_id]
                            weight_mask = data_set.weight_mask_residual

                            model(agg_embedding, data_loaders[data_id], initial_id)
                            pred_velocities = model.parameter_estimate(data_scaled)
                            
                            loss_pde, loss_pde_std, full_loss, laplacian_loss = pde_loss("meta_pinn",
                            at_map,
                            cor_org,
                            data_scaled,
                            pred_velocities,
                            grad_operator_basis,
                            grad_operator,
                            laplacian,
                            weight_mask)

                            # compute the data loss
                            full_data_loss, loss_data = data_loss(
                                model, inputs_data, at_map, data_set.indices_loss_data, weight_mask
                            )
        
                            # pinn_prediction_ is the output activation time map from the PINN model
                            pinn_prediction = model.network(data_scaled) * 1000
                            # pinn_prediction_ = model.network(data_scaled)
                            total_loss = alpha * loss_pde + loss_data 
                            task_data_loss += loss_data.item()
                            task_pde_loss += loss_pde.item()
                            task_total_loss += total_loss.item()

                            
                            if epoch == epochs - 1 or epoch in [500, 1000, 2500, 5000]:
                                ut.AT_map_plot_3d(
                                        data_set.cor,
                                        epoch,
                                        at_map.cpu().numpy(),
                                        pinn_prediction.cpu().numpy(),
                                        d_id,
                                        s_id,
                                        geo_id,
                                        result_folder,
                                        "validation"
                                    )
                            
                                generate_velocity_map(
                                    result_folder,
                                    "validation",
                                    data_scaled.detach().cpu().numpy(),
                                    epoch,
                                    velocity,
                                    pred_velocities.detach().cpu().numpy(),
                                    d_id,
                                    s_id,
                                    geo_id,
                                    True,
                                )
                        
                        epoch_val_loss += task_total_loss / num_considered
                        epoch_val_pde_loss += task_pde_loss / num_considered
                        epoch_val_data_loss += task_data_loss / num_considered

                # Average validation loss
                avg_val_loss = epoch_val_loss / num_tasks
                avg_pde_loss = epoch_val_pde_loss / num_tasks
                avg_data_loss = epoch_val_data_loss / num_tasks
                val_losses.append(avg_val_loss)
                val_data_losses.append(avg_data_loss)
                val_pde_losses.append(avg_pde_loss)

                print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
                with open(f"{result_folder}/log.txt", "a") as log_file:
                    log_file.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Data Loss: {avg_val_loss}, Validation PDE Loss: {avg_val_loss}\n")

            
            if epoch % 100 == 0:
                tsne(data_loaders, epoch, result_folder)
            
            epoch_total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            if freeze:
                for param in model.network.parameters():
                    param.grad = None 
                for param in model.hypernet_pinn.parameters():
                    param.grad = None 
                for param in model.encoder.parameters():
                    param.grad = None
                for param in model.initial_encoder.parameters():
                    param.grad = None
            
            optimizer.step()
            scheduler.step()
            average_pde_loss = epoch_pde_loss
            average_data_loss = epoch_data_loss
            average_total_loss = epoch_total_loss.item()
            average_reg_loss = epoch_reg_loss
            # average_smooth_loss = epoch_smooth_loss.item() / len(datasets)
            pde_losses.append(average_pde_loss / num_tasks)
            data_losses.append(average_data_loss/ num_tasks)
            total_losses.append(average_total_loss/ num_tasks)
            with open(f"{result_folder}/log.txt", "a") as log_file:
                print(
                    f"Average, Epoch {epoch}, Alpha: {alpha}, Loss: {average_total_loss/num_tasks}, Data Loss: {average_data_loss/num_tasks}, PDE Loss: {average_pde_loss/num_tasks}, reg Loss: {average_reg_loss/num_tasks}, freeze: {freeze}"
                )
                print(
                    "******************************************************************************************************"
                )
                log_file.write(
                    f"Average, Epoch {epoch}, Alpha: {alpha}, Loss: {average_total_loss/num_tasks}, Data Loss: {average_data_loss/num_tasks}, PDE Loss: {average_pde_loss/num_tasks}, reg Loss: {average_reg_loss/num_tasks}, freeze: {freeze}"
                )
            
            if epoch % 100 == 0 or epoch == epoch - 1:
                ut.save_checkpoint(
                    "meta_pinn",
                    f"{result_folder}/inverse_pinn_eikonal_model.pth",
                    model,
                    optimizer,
                    epoch,
                )
            # torch.save(model, f'results_{exp_desc}/inverse_pinn_eikonal_model.pth')
            #     torch.save({
            #     'main_model': model.state_dict(),
            #     'sub_network_1': model.sub_network_1.state_dict(),
            #     'sub_network_2': model.sub_network_2.state_dict(),
            # }, f'results_{exp_desc}/complete_model_state_dict.pth')

        # plot all losses
        ut.plot_losses(pde_losses, data_losses, total_losses, result_folder, "train")
        ut.plot_losses(val_pde_losses, val_data_losses, val_losses, result_folder, "val")
        # colors = plt.cm.tab10(np.linspace(0, 1, len(probs_per_epoch.keys())))
        # ut.plot_val_over_epoch(
        #     f"{result_folder}/",
        #     "prob_epoch.png",
        #     epochs,
        #     probs_per_epoch.keys(),
        #     probs_per_epoch,
        #     colors,
        #     "Epochs",
        #     "Probability values",
        #     "Probability values for each task across epochs",
        # )
        end_time = time.time()
        with open(f"{result_folder}/log.txt", "a") as log_file:
            log_file.write(f"Train time: {(end_time - start_time) / 60.0}\n")

    if is_train == 2:
        data_loaders = test_loaders
        eval_folder = f"{result_folder}/eval/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        model = ut.load_checkpoint(
            "meta_pinn", f"{result_folder}/inverse_pinn_eikonal_model.pth", model, optimizer
        )
        # Configure the logging
        logging.basicConfig(
            filename=f"{result_folder}/eval/error.log",  
            level=logging.INFO,     
            format="%(asctime)s - %(message)s",  
        )

        # Dictionaries to store losses for each geo_id and d_id
        geo_losses = {}
        geo_d_losses = {}

        num_tasks = len(data_loaders)
        for data_id in range(num_tasks):
            agg_embedding = model.prepare_encoder_input(data_loaders[data_id])
            num_considered = len(range(hparams.dataset["k"], len(data_loaders[data_id])))
            for initial_id in range(hparams.dataset["k"], len(data_loaders[data_id])):
                data_set = data_loaders[data_id][initial_id]
                cor_org = torch.tensor(
                    data_set.cor_org, dtype=torch.float32, requires_grad=True
                ).to(device)
                task_id, s_id, geo_id = data_set.index
                segs = data_set.segs
                at_map = data_set.activation_times
                true_range = (np.min(at_map), np.max(at_map))
                velocity = data_set.velocity_field
                laplacian = data_set.laplacian
                inputs_residual = data_set.inputs_residual
                residual_indices = data_set.indices_residual
                # input data -- for the data loss
                inputs_data = data_set.inputs_data
                outputs_data = data_set.outputs_data
                
                # all coodinates
                data_scaled = torch.tensor(
                    data_set.cor, dtype=torch.float32, requires_grad=True
                ).to(device)
                data_mask = data_set.data_mask
                weight_mask = data_set.weight_mask_residual

                model(agg_embedding, data_loaders[data_id], initial_id)
                pred_velocities = model.parameter_estimate(data_scaled)

                # get the output, spatially varying tissue properties/velocities
                output_velocity = model.parameter_estimate(data_scaled)



                    # pinn_prediction_ is the output activation time map from the PINN model
                pinn_prediction = model.network(data_scaled) * 1000
                data_loss, velocity_loss, data_cc, velocity_cc = evaluate(data_scaled, pinn_prediction, at_map, pred_velocities, velocity, 'meta_pinn')

                
                
                # data_set = data_loaders[data_id][initial_id]
                # cor_org = torch.tensor(
                #     data_set.cor_org, dtype=torch.float32, requires_grad=True
                # ).to(device)
                # task_id, sample_id, geo_id = data_set.index
                # at_map = data_set.activation_times
                # true_range = (np.min(at_map), np.max(at_map))
                # velocity = data_set.velocity_field
                # data_scaled = torch.tensor(
                #     data_set.cor, dtype=torch.float32, requires_grad=True
                # ).to(device)

                # d_id = data_set.index[0]
                # s_id = data_set.index[1]

                # # Model prediction and loss calculation
                # model(agg_embedding, data_loaders[data_id], initial_id)
                # output_velocity = model.parameter_estimate(data_scaled)
                # pinn_prediction_ = model.network(data_scaled)
                # c, d = torch.min(pinn_prediction_), torch.max(pinn_prediction_)
                # a, b = true_range[0], true_range[1]
                # pinn_prediction = (pinn_prediction_ - c) * (b - a) / (d - c) + a

                # # Evaluate data loss and velocity loss
                # data_loss, velocity_loss = evaluate(
                #     data_scaled, pinn_prediction, at_map, output_velocity, velocity
                # )

                # Store the losses in geo_id dictionary
                if geo_id not in geo_losses:
                    geo_losses[geo_id] = []
                geo_losses[geo_id].append((data_loss, velocity_loss, data_cc, velocity_cc))

                # Store the losses in geo_id, d_id dictionary
                if geo_id not in geo_d_losses:
                    geo_d_losses[geo_id] = {}
                if task_id not in geo_d_losses[geo_id]:
                    geo_d_losses[geo_id][task_id] = []
                geo_d_losses[geo_id][task_id].append((data_loss, velocity_loss, data_cc, velocity_cc))

                # Print individual evaluation results
                print(
                    f"geo_id: {geo_id} task_id: {task_id} initial_id: {s_id} data_loss: {data_loss} velocity_loss: {velocity_loss} data_cc: {data_cc:.4f} velocity_cc: {velocity_cc:.4f}"
                )
                
                logging.info(
                    f"geo_id: {geo_id} task_id: {task_id} initial_id: {s_id} data_loss: {data_loss} velocity_loss: {velocity_loss} data_cc: {data_cc:.4f} velocity_cc: {velocity_cc:.4f}"
                )
                

                # Write VTK files
                ut.write_vtk_data_pointcloud(
                    velocity,
                    data_set.cor,
                    eval_folder
                    + f"eval_velocity_geo_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    output_velocity.detach().cpu().numpy(),
                    data_set.cor,
                    eval_folder
                    + f"eval_pred_velocity_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    at_map,
                    data_set.cor,
                    eval_folder + f"eval_at_geo_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pinn_prediction.detach().cpu().numpy(),
                    data_set.cor,
                    eval_folder + f"eval_at_pred_geo_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )

        # Compute and print average and standard deviation of losses
        all_data_losses , all_data_ccs = [], []
        all_vel_losses , all_vel_ccs = [], []
        print("\nAggregated Loss Statistics:")
        for geo_id, losses in geo_losses.items():
            data_losses, velocity_losses, data_ccs, velocity_ccs = zip(*losses)
            mean_data_loss = np.mean(data_losses)
            std_data_loss = np.std(data_losses)
            mean_velocity_loss = np.mean(velocity_losses)
            std_velocity_loss = np.std(velocity_losses)
            mean_data_cc = np.mean(data_ccs)
            std_data_cc = np.std(data_ccs)
            mean_velocity_cc = np.mean(velocity_ccs)
            std_velocity_cc = np.std(velocity_ccs)
            
            # Append values to lists for aggregated statistics
            all_data_losses.extend(data_losses)
            all_data_ccs.extend(data_ccs)
            all_vel_losses.extend(velocity_losses)
            all_vel_ccs.extend(velocity_ccs)
            
            print(
                f"geo_id: {geo_id} - Data Loss: mean={mean_data_loss:.4f}, std={std_data_loss:.4f} | Velocity Loss: mean={mean_velocity_loss:.4f}, std={std_velocity_loss:.4f} | Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f} Velocity Loss: mean={mean_velocity_cc:.4f}, std={std_velocity_cc:.4f}"
            )

            # Log the information
            logging.info(
                f"geo_id: {geo_id} - Data Loss: mean={mean_data_loss:.4f}, std={std_data_loss:.4f} | Velocity Loss: mean={mean_velocity_loss:.4f}, std={std_velocity_loss:.4f} | Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f} Velocity Loss: mean={mean_velocity_cc:.4f}, std={std_velocity_cc:.4f}"
            )    
            # Compute per d_id statistics
            for d_id, d_losses in geo_d_losses[geo_id].items():
                d_data_losses, d_velocity_losses, data_ccs, velocity_ccs = zip(*d_losses)
                d_mean_data_loss = np.mean(d_data_losses)
                d_std_data_loss = np.std(d_data_losses)
                d_mean_velocity_loss = np.mean(d_velocity_losses)
                d_std_velocity_loss = np.std(d_velocity_losses)
                mean_data_cc = np.mean(data_ccs)
                std_data_cc = np.std(data_ccs)
                mean_velocity_cc = np.mean(velocity_ccs)
                std_velocity_cc = np.std(velocity_ccs)
                print(
                    f"geo_id: {geo_id}, d_id: {d_id} - Data Loss: mean={d_mean_data_loss:.4f}, std={d_std_data_loss:.4f} | Velocity Loss: mean={d_mean_velocity_loss:.4f}, std={d_std_velocity_loss:.4f} | Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f} Velocity Loss: mean={mean_velocity_cc:.4f}, std={std_velocity_cc:.4f}"
                )
                # Log the per d_id statistics
                logging.info(
                    f"geo_id: {geo_id}, d_id: {d_id} - Data Loss: mean={d_mean_data_loss:.4f}, std={d_std_data_loss:.4f} | Velocity Loss: mean={d_mean_velocity_loss:.4f}, std={d_std_velocity_loss:.4f} | Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f} Velocity Loss: mean={mean_velocity_cc:.4f}, std={std_velocity_cc:.4f}"
                )

        # Compute aggregated statistics across all cases
        agg_mean_data_loss = np.mean(all_data_losses)
        agg_std_data_loss = np.std(all_data_losses)
        agg_mean_data_cc = np.mean(all_data_ccs)
        agg_std_data_cc = np.std(all_data_ccs)
        agg_mean_vel_loss = np.mean(all_vel_losses)
        agg_std_vel_loss = np.std(all_vel_losses)
        agg_mean_vel_cc = np.mean(all_vel_ccs)
        agg_std_vel_cc = np.std(all_vel_ccs)
        # Print and log aggregated statistics
        print(
            f"Aggregated Statistics - Data Loss: mean={agg_mean_data_loss:.4f}, std={agg_std_data_loss:.4f}, Data CC: mean={agg_mean_data_cc:.4f}, std={agg_std_data_cc:.4f}"
        )
        print(
            f"Aggregated Statistics - Vel Loss: mean={agg_mean_vel_loss:.4f}, std={agg_mean_vel_loss:.4f}, Vel CC: mean={agg_mean_vel_cc:.4f}, std={agg_mean_vel_cc:.4f}"
        )
        logging.info(
            f"Aggregated Statistics - Data Loss: mean={agg_mean_data_loss:.4f}, std={agg_std_data_loss:.4f}, Data CC: mean={agg_mean_data_cc:.4f}, std={agg_std_data_cc:.4f}"
        )
        logging.info(
            f"Aggregated Statistics - Vel Loss: mean={agg_mean_vel_loss:.4f}, std={agg_mean_vel_loss:.4f}, Vel CC: mean={agg_mean_vel_cc:.4f}, std={agg_mean_vel_cc:.4f}"
        )
        # Plotting the results

        # Plot 1: Box plots for each geometry
        geo_ids = list(geo_losses.keys())
        data_errors_geo = [list(zip(*geo_losses[geo_id]))[0] for geo_id in geo_ids]
        velocity_errors_geo = [list(zip(*geo_losses[geo_id]))[1] for geo_id in geo_ids]
        data_cc_geo = [list(zip(*geo_losses[geo_id]))[2] for geo_id in geo_ids]
        velocity_cc_geo = [list(zip(*geo_losses[geo_id]))[3] for geo_id in geo_ids]

        ut.errorplots(data_errors_geo, geo_ids, eval_folder, "Error", "AT Error")
        ut.errorplots(velocity_errors_geo, geo_ids, eval_folder, "Error", "Velocity Error")
        ut.errorplots(data_cc_geo, geo_ids, eval_folder, "CC", "AT CC")
        ut.errorplots(velocity_cc_geo, geo_ids, eval_folder, "CC", "Velocity CC")
        
        
        # Plot 2: Box plots for tasks within each geometry
        for geo_id in geo_ids:
            d_ids = list(geo_d_losses[geo_id].keys())
            data_errors_tasks = [
                list(zip(*geo_d_losses[geo_id][d_id]))[0] for d_id in d_ids
            ]

            plt.figure(figsize=(10, 6))
            plt.boxplot(
                data_errors_tasks,
                positions=np.arange(len(d_ids)),
                widths=0.4,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen"),
            )
            plt.scatter(
                np.arange(len(d_ids)),
                [np.mean(errors) for errors in data_errors_tasks],
                color="red",
                label="Mean",
                s=50,
            )
            plt.errorbar(
                np.arange(len(d_ids)),
                [np.mean(errors) for errors in data_errors_tasks],
                yerr=[np.std(errors) for errors in data_errors_tasks],
                fmt="o",
                color="blue",
                label="Std Dev",
            )

            plt.xticks(np.arange(len(d_ids)), [f"Task {d_id}" for d_id in d_ids])
            plt.title(f"Errors across Tasks for Geometry {geo_id}")
            plt.ylabel("Error")
            plt.legend()
            # plt.show()
            plt.savefig(eval_folder + f"Errors_Tasks_Geometry {geo_id}.png")

    elif is_train == 3:
        eval_folder = f"Experiments/results_{exp_desc}/eval_3/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        model.load_state_dict(
            torch.load(f"Experiments/results_{exp_desc}/inverse_pinn_eikonal_model.pth"), strict=False
        )
        model.train()

        for d_sample in range(hparams.dataset["num_samples"]):
            num_initials = hparams.dataset["num_at_maps"]
            agg_embedding = model.prepare_encoder_input(data_loaders[d_sample])
            for at_id in range(num_initials):
                data_set = data_loaders[d_sample][at_id]
                data_id = data_set.index[0]
                initial_id = data_set.index[1]
                at_map = data_set.activation_times
                velocity = data_set.velocity_field
                laplacian = data_set.laplacian
                inputs_residual = data_set.inputs_residual
                residual_indices = data_set.indices_residual
                inputs_data = data_set.inputs_data
                outputs_data = data_set.outputs_data
                data_scaled = torch.tensor(
                    dataset.cor, dtype=torch.float32, requires_grad=True
                ).to(device)
                init_encoder_input = data_set.initial_graph.to(device)
                grad_operator = data_set.grad_operator
                # _ = model(data_loaders[data_id], initial_id)
                model(agg_embedding, data_loaders[d_sample], at_id)
                # pred_velocities =  model.parameter_estimate(inputs_residual)

                # loss_data = data_loss(model, inputs_data, outputs_data)
                # velocities = generate_velocity_map(data_scaled, epoch, velocity, scar_mask, sigma_latent, data_id, initial_id, False)
                pred_velocities = model.parameter_estimate(data_scaled)
                true_range = (np.min(at_map), np.max(at_map))
                pinn_prediction_ = model.network(data_scaled)

                c, d = torch.min(pinn_prediction_), torch.max(
                    pinn_prediction_
                )  # Min and max of original data
                a, b = true_range[0], true_range[1]  # Target range

                # Scale pinn_prediction
                pinn_prediction = (pinn_prediction_ - c) * (b - a) / (d - c) + a

                # pinn_prediction = pinn_prediction * (true_range[1] - true_range[0]) + true_range[0]
                gradient_gt, _ = gradient_check(
                    torch.tensor(at_map, dtype=torch.float32, requires_grad=True).to(
                        device
                    ),
                    grad_operator,
                )
                gradient_pred, _ = gradient_check(pinn_prediction, grad_operator)
                # grt = outputs_data

                ut.write_vtk_data_pointcloud(
                    velocity,
                    dataset.cor,
                    eval_folder + f"eval_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pred_velocities.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder
                    + f"eval_pred_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    1 / pred_velocities.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder
                    + f"eval_dev1_pred_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                # ut.write_vtk_data_pointcloud(full_loss.detach().cpu().numpy(), dataset.cor, eval_folder+f'eval_pde_id_{data_id}_initial_{initial_id}.vtk')
                ut.write_vtk_data_pointcloud(
                    at_map,
                    dataset.cor,
                    eval_folder + f"eval_at_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pinn_prediction.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_at_pred_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    gradient_gt.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_grads_gt_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    gradient_pred.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_grad_pred_id_{data_id}_initial_{initial_id}.vtk",
                )


elif model_type == 'meta_base':
    
    #create results folder
    result_folder = f"Experiments/results_{exp_desc}/meta_base"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    destination_file = os.path.join(result_folder, os.path.basename(json_name))
    shutil.copy(json_path, destination_file)
    ut.set_seed(hparams.training["seed"])
    
    # Initialize the PINN
    model = metaBase().to(device)
    # define optimizers and schedulers
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    milestones = [i for i in range(1, epochs, 10)]
    warmup_steps = 101
    scheduler = lr.CosineAnnealingLR(optimizer, T_max=20, warmup_steps=20, eta_min=1e-7)
    # stage 1 -- train loop
    if is_train == 1:
        start_time = time.time()
        data_losses = []
        pde_losses = []
        total_losses = []
        val_losses = []
        best_val_loss = float('inf')
        ema_alpha = 0.1
        ema_loss = None
        ema_losses = []
        trend_window = 20
        validation_interval = 5
        num_tasks = len(data_loaders)
        stop = False
        for epoch in range(epochs):
            if stop == True:
                break
            epoch_data_loss = 0
            epoch_total_loss = 0
            epoch_pde_loss = 0
            epoch_reg_loss = 0
            embeddings = []

            random.shuffle(data_loaders)
            
            data_loaders = data_episodic_update(data_loaders)
            model.train()
            optimizer.zero_grad()

            embeddings = []
            num_tasks = len(data_loaders)
            for data_id in range(num_tasks):
                
                task_data_loss, task_total_loss = 0, 0

                # get task embedding -- input to the hypernetwork
                agg_embedding = model.prepare_encoder_input(data_loaders[data_id])
                embeddings.append(agg_embedding)
                task_id =  data_loaders[data_id][0].index[0]
                
                num_considered = len(
                    range(hparams.dataset["k"], len(data_loaders[data_id]))
                )
                
                for initial_id in range(hparams.dataset["k"], len(data_loaders[data_id])):
                    # print(torch.cuda.memory_summary())
                    data_set = data_loaders[data_id][initial_id]
                    cor_org = torch.tensor(
                        data_set.cor_org, dtype=torch.float32, requires_grad=True
                    ).to(device)
                    task_id, sample_id, geo_id = data_set.index
                    segs = data_set.segs
                    at_map = data_set.activation_times
                    true_range = (np.min(at_map), np.max(at_map))
                    at_map = torch.tensor(at_map, dtype=torch.float32).to(device)
                    velocity = data_set.velocity_field
                    laplacian = data_set.laplacian
                    inputs_residual = data_set.inputs_residual
                    residual_indices = data_set.indices_residual
                    # input data -- for the data loss
                    inputs_data = data_set.inputs_data
                    outputs_data = data_set.outputs_data
                    
                    # all coodinates
                    data_scaled = torch.tensor(
                        data_set.cor, dtype=torch.float32, requires_grad=True
                    ).to(device)
                    data_mask = data_set.data_mask
                    weight_mask = data_set.weight_mask_residual

                    model(agg_embedding, data_loaders[data_id], initial_id)
                    # compute the data loss
                    full_data_loss, loss_data = data_loss(
                        model, inputs_data, at_map, data_set.indices_loss_data, weight_mask
                    )

                    # pinn_prediction_ is the output activation time map from the PINN model
                    pinn_prediction = model.network(data_scaled) * 1000
                    # pinn_prediction_ = model.network(data_scaled)
                    total_loss = loss_data 

                    task_data_loss += loss_data.item()
                    task_total_loss += total_loss
                    

                    # saving intermediate results
                    if epoch == epochs - 1 or epoch in [0, 500, 1000, 2500, 5000]:
                        with torch.no_grad():
                            d_id = data_set.index[0]
                            s_id = data_set.index[1]
                             
                            ut.write_vtk_data_pointcloud(
                                full_data_loss.detach().cpu().numpy(),
                                data_set.cor,
                                f"{result_folder}/full_data_loss__{epoch}_{d_id}_initial_geo_{geo_id}_{s_id}.vtk",
                            )
                            
                            ut.AT_map_plot_3d(
                                data_set.cor,
                                epoch,
                                at_map.cpu().numpy(),
                                pinn_prediction.cpu().numpy(),
                                d_id,
                                s_id,
                                geo_id,
                                result_folder,
                                "train"
                            )
                           
                    print(
                        f"Sample {initial_id} , Epoch {epoch}, Loss: {total_loss.item()}"
                    )

                    check_nan_parameters(model)

                print(
                    f"Task {data_id} , Epoch {epoch}, Loss: {task_total_loss.item()/num_considered}"
                )

                epoch_data_loss += task_data_loss / num_considered
                epoch_total_loss += task_total_loss / num_considered
            # epoch_smooth_loss += (task_smooth_loss/num_initials)
            # epoch_smooth_loss += (task_smooth_loss/num_initials)
            if epoch % 100 == 0:
                tsne(data_loaders, epoch, result_folder)
            
            epoch_total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
           
            
            optimizer.step()
            scheduler.step()
            average_data_loss = epoch_data_loss
            average_total_loss = epoch_total_loss.item()
           
            data_losses.append(average_data_loss/ num_tasks)
            total_losses.append(average_total_loss/ num_tasks)
            with open(f"{result_folder}/log.txt", "a") as log_file:
                print(
                    f"Average, Epoch {epoch},  Loss: {average_total_loss/num_tasks}"
                )
                log_file.write(
                    f"Average, Epoch {epoch}, Loss: {average_total_loss/num_tasks}\n"
                )

            ut.save_checkpoint(
                model_type,
                f"{result_folder}/inverse_pinn_eikonal_model.pth",
                model,
                optimizer,
                epoch,
            )

            if epoch % validation_interval == 0:
                # validation loader
                model.eval()
                epoch_val_loss = 0
                num_val_batches = 0

                with torch.no_grad():
                    for data_id in range(len(val_loaders)):
                        task_data_loss = 0
                        task_total_loss = 0
                        # get task embedding -- input to the hypernetwork
                        agg_embedding = model.prepare_encoder_input(val_loaders[data_id])
                        embeddings.append(agg_embedding)
                        task_id =  val_loaders[data_id][0].index[0]
                        
                        num_considered = len(
                            range(len(val_loaders[data_id]))
                        )
                        
                        for initial_id in range(num_considered):
                            # print(torch.cuda.memory_summary())
                            data_set = val_loaders[data_id][initial_id]
                            cor_org = torch.tensor(
                                data_set.cor_org, dtype=torch.float32, requires_grad=True
                            ).to(device)
                            task_id, sample_id, geo_id = data_set.index
                            at_map = data_set.activation_times
                            at_map = torch.tensor(at_map, dtype=torch.float32).to(device)
                            inputs_data = data_set.inputs_data
                            
                            # all coodinates
                            data_scaled = torch.tensor(
                                data_set.cor, dtype=torch.float32, requires_grad=True
                            ).to(device)
                            data_mask = data_set.data_mask
                            # grad_operator_basis = data_set.grad_operator
                            weight_mask = data_set.weight_mask_residual

                            model(agg_embedding, data_loaders[data_id], initial_id)
                            # compute the data loss
                            full_data_loss, val_loss = data_loss(
                                model, inputs_data, at_map, data_set.indices_loss_data, weight_mask
                            )

                            # pinn_prediction_ is the output activation time map from the PINN model
                            pinn_prediction = model.network(data_scaled) * 1000
                            # pinn_prediction_ = model.network(data_scaled)
                            total_loss = val_loss 

                            task_data_loss += val_loss.item()
                            
                            if epoch == epochs - 1 or epoch in [500, 1000, 2500, 5000]:
                                ut.AT_map_plot_3d(
                                        data_set.cor,
                                        epoch,
                                        at_map.cpu().numpy(),
                                        pinn_prediction.cpu().numpy(),
                                        d_id,
                                        s_id,
                                        geo_id,
                                        result_folder,
                                        "validation"
                                    )
                            
                        
                        epoch_val_loss += task_data_loss / num_considered

                # Average validation loss
                avg_val_loss = epoch_val_loss / num_tasks
                val_losses.append(avg_val_loss)

                print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
                with open(f"{result_folder}/log.txt", "a") as log_file:
                    log_file.write(
                            f"Epoch {epoch}, Validation Loss: {avg_val_loss}\n"
                        )

                # Update EMA
                if ema_loss is None:
                    ema_loss = avg_val_loss
                else:
                    ema_loss = ema_alpha * avg_val_loss + (1 - ema_alpha) * ema_loss
                ema_losses.append(ema_loss)

                with open(f"{result_folder}/log.txt", "a") as log_file:
                    log_file.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}\n")

                # # Check EMA trend for early stopping
                # if epoch > 500 and len(ema_losses) >= trend_window:
                #     recent_ema = ema_losses[-trend_window:]
                #     x = np.arange(trend_window)
                #     slope, _, _, _, _ = linregress(x, recent_ema)

                #     if slope > 0:
                #         print(f"Early stopping at epoch {epoch}, EMA slope: {slope:.5f}")
                #         stop = True
                #         break
            print(
                    "******************************************************************************************************"
                )
        
        ut.plot_loss(total_losses, data_losses, "data loss", result_folder)
        colors = plt.cm.tab10(np.linspace(0, 1, len(probs_per_epoch.keys())))
        end_time = time.time()
        with open(f"{result_folder}/log.txt", "a") as log_file:
            log_file.write(f"Train time: {(end_time - start_time) / 60.0}\n")

    if is_train == 2:
        data_loaders = test_loaders
        eval_folder = f"{result_folder}/eval/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        model = ut.load_checkpoint(
            "meta_base", f"{result_folder}/inverse_pinn_eikonal_model.pth", model, optimizer
        )
        # Configure the logging
        logging.basicConfig(
            filename=f"{result_folder}/eval/error.log",  
            level=logging.INFO,     
            format="%(asctime)s - %(message)s",  
        )

        # Dictionaries to store losses for each geo_id and d_id
        geo_losses = {}
        geo_d_losses = {}

        num_tasks = len(data_loaders)
        for data_id in range(num_tasks):
            agg_embedding = model.prepare_encoder_input(data_loaders[data_id])
            num_considered = len(range(hparams.dataset["k"], len(data_loaders[data_id])))
        
            for initial_id in range(hparams.dataset["k"], len(data_loaders[data_id])):
                # print(torch.cuda.memory_summary())
                data_set = data_loaders[data_id][initial_id]
                cor_org = torch.tensor(
                    data_set.cor_org, dtype=torch.float32, requires_grad=True
                ).to(device)
                task_id, s_id, geo_id = data_set.index
                segs = data_set.segs
                at_map = data_set.activation_times
                true_range = (np.min(at_map), np.max(at_map))
                # at_map = torch.tensor(at_map, dtype=torch.float32).to(device)
                velocity = data_set.velocity_field
                laplacian = data_set.laplacian
                inputs_residual = data_set.inputs_residual
                residual_indices = data_set.indices_residual
                # input data -- for the data loss
                inputs_data = data_set.inputs_data
                outputs_data = data_set.outputs_data
                
                # all coodinates
                data_scaled = torch.tensor(
                    data_set.cor, dtype=torch.float32, requires_grad=True
                ).to(device)
                data_mask = data_set.data_mask
                weight_mask = data_set.weight_mask_residual

                model(agg_embedding, data_loaders[data_id], initial_id)
                pinn_prediction = model.network(data_scaled) * 1000                
                data_loss, velocity_loss, data_cc, _ = evaluate(data_scaled, pinn_prediction, at_map, None, None, 'meta_base')

    
                # Store the losses and correlation coefficients in geo_id dictionary
                if geo_id not in geo_losses:
                    geo_losses[geo_id] = []
                geo_losses[geo_id].append((data_loss, velocity_loss, data_cc))

                # Store the losses and correlation coefficients in geo_id, d_id dictionary
                if geo_id not in geo_d_losses:
                    geo_d_losses[geo_id] = {}
                if task_id not in geo_d_losses[geo_id]:
                    geo_d_losses[geo_id][task_id] = []
                geo_d_losses[geo_id][task_id].append((data_loss, velocity_loss, data_cc))

                # Print individual evaluation results
                print(
                    f"geo_id: {geo_id} task_id: {task_id} initial_id: {s_id} data_loss: {data_loss} data_cc: {data_cc:.4f}"
                ) 
                
                logging.info(
                    f"geo_id: {geo_id} task_id: {task_id} initial_id: {s_id} data_loss: {data_loss} data_cc: {data_cc:.4f}"
                )
                

                ut.write_vtk_data_pointcloud(
                    at_map,
                    data_set.cor,
                    eval_folder + f"eval_at_geo_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pinn_prediction.detach().cpu().numpy(),
                    data_set.cor,
                    eval_folder + f"eval_at_pred_geo_{geo_id}_id_{task_id}_initial_{s_id}.vtk",
                )

        # Compute and print average and standard deviation of losses
        all_data_losses, all_data_ccs = [], []
        print("\nAggregated Loss Statistics:")
        for geo_id, losses in geo_losses.items():
            data_losses, velocity_losses, data_ccs = zip(*losses)
            mean_data_loss = np.mean(data_losses)
            std_data_loss = np.std(data_losses)
            mean_data_cc = np.mean(data_ccs)
            std_data_cc = np.std(data_ccs)
            print(
                f"geo_id: {geo_id} - Data Loss: mean={mean_data_loss:.4f}, std={std_data_loss:.4f}, Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f}"
            )

            # Log the information
            logging.info(
                f"geo_id: {geo_id} - Data Loss: mean={mean_data_loss:.4f}, std={std_data_loss:.4f}, Data CC: mean={mean_data_cc:.4f}, std={std_data_cc:.4f}"
            )
            # Compute per d_id statistics
            for d_id, d_losses in geo_d_losses[geo_id].items():
                d_data_losses, d_velocity_losses, d_data_ccs = zip(*d_losses)
                d_mean_data_loss = np.mean(d_data_losses)
                d_std_data_loss = np.std(d_data_losses)
                d_mean_data_cc = np.mean(d_data_ccs)
                d_std_data_cc = np.std(d_data_ccs)
                print(
                    f"geo_id: {geo_id}, d_id: {d_id} - Data Loss: mean={d_mean_data_loss:.4f}, std={d_std_data_loss:.4f}, Data CC: mean={d_mean_data_cc:.4f}, std={d_std_data_cc:.4f}"
                )
                logging.info(
                    f"geo_id: {geo_id}, d_id: {d_id} - Data Loss: mean={d_mean_data_loss:.4f}, std={d_std_data_loss:.4f}, Data CC: mean={d_mean_data_cc:.4f}, std={d_std_data_cc:.4f}"
                )
                # Append values to lists for aggregated statistics
                all_data_losses.extend(d_data_losses)
                all_data_ccs.extend(d_data_ccs)

        agg_mean_data_loss = np.mean(all_data_losses)
        agg_std_data_loss = np.std(all_data_losses)
        agg_mean_data_cc = np.mean(all_data_ccs)
        agg_std_data_cc = np.std(all_data_ccs)

        # Print and log aggregated statistics
        print(
            f"Aggregated Statistics - Data Loss: mean={agg_mean_data_loss:.4f}, std={agg_std_data_loss:.4f}, Data CC: mean={agg_mean_data_cc:.4f}, std={agg_std_data_cc:.4f}"
        )
        logging.info(
            f"Aggregated Statistics - Data Loss: mean={agg_mean_data_loss:.4f}, std={agg_std_data_loss:.4f}, Data CC: mean={agg_mean_data_cc:.4f}, std={agg_std_data_cc:.4f}"
        )

        # Plot 1: Box plots for each geometry
        geo_ids = list(geo_losses.keys())
        data_errors_geo = [list(zip(*geo_losses[geo_id]))[0] for geo_id in geo_ids]
        data_cc_geo = [list(zip(*geo_losses[geo_id]))[2] for geo_id in geo_ids]
       
        ut.errorplots(data_errors_geo, geo_ids, eval_folder, "Error", "AT Error")
        ut.errorplots(data_cc_geo, geo_ids, eval_folder, "CC", "AT CC")

        # Plot 2: Box plots for tasks within each geometry
        for geo_id in geo_ids:
            d_ids = list(geo_d_losses[geo_id].keys())
            data_errors_tasks = [
                list(zip(*geo_d_losses[geo_id][d_id]))[0] for d_id in d_ids
            ]

            plt.figure(figsize=(10, 6))
            plt.boxplot(
                data_errors_tasks,
                positions=np.arange(len(d_ids)),
                widths=0.4,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen"),
            )
            plt.scatter(
                np.arange(len(d_ids)),
                [np.mean(errors) for errors in data_errors_tasks],
                color="red",
                label="Mean",
                s=50,
            )
            plt.errorbar(
                np.arange(len(d_ids)),
                [np.mean(errors) for errors in data_errors_tasks],
                yerr=[np.std(errors) for errors in data_errors_tasks],
                fmt="o",
                color="blue",
                label="Std Dev",
            )

            plt.xticks(np.arange(len(d_ids)), [f"Task {d_id}" for d_id in d_ids])
            plt.title(f"Errors across Tasks for Geometry {geo_id}")
            plt.ylabel("Error")
            plt.legend()
            # plt.show()
            plt.savefig(eval_folder + f"Errors_Tasks_Geometry {geo_id}.png")

    elif is_train == 3:
        eval_folder = f"Experiments/results_{exp_desc}/eval_3/"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        model.load_state_dict(
            torch.load(f"Experiments/results_{exp_desc}/inverse_pinn_eikonal_model.pth"), strict=False
        )
        model.train()

        for d_sample in range(hparams.dataset["num_samples"]):
            num_initials = hparams.dataset["num_at_maps"]
            agg_embedding = model.prepare_encoder_input(data_loaders[d_sample])
            for at_id in range(num_initials):
                data_set = data_loaders[d_sample][at_id]
                data_id = data_set.index[0]
                initial_id = data_set.index[1]
                at_map = data_set.activation_times
                velocity = data_set.velocity_field
                laplacian = data_set.laplacian
                inputs_residual = data_set.inputs_residual
                residual_indices = data_set.indices_residual
                inputs_data = data_set.inputs_data
                outputs_data = data_set.outputs_data
                data_scaled = torch.tensor(
                    dataset.cor, dtype=torch.float32, requires_grad=True
                ).to(device)
                init_encoder_input = data_set.initial_graph.to(device)
                grad_operator = data_set.grad_operator
                # _ = model(data_loaders[data_id], initial_id)
                model(agg_embedding, data_loaders[d_sample], at_id)
                # pred_velocities =  model.parameter_estimate(inputs_residual)

                # loss_data = data_loss(model, inputs_data, outputs_data)
                # velocities = generate_velocity_map(data_scaled, epoch, velocity, scar_mask, sigma_latent, data_id, initial_id, False)
                pred_velocities = model.parameter_estimate(data_scaled)
                true_range = (np.min(at_map), np.max(at_map))
                pinn_prediction_ = model.network(data_scaled)

                c, d = torch.min(pinn_prediction_), torch.max(
                    pinn_prediction_
                )  # Min and max of original data
                a, b = true_range[0], true_range[1]  # Target range

                # Scale pinn_prediction
                pinn_prediction = (pinn_prediction_ - c) * (b - a) / (d - c) + a

                # pinn_prediction = pinn_prediction * (true_range[1] - true_range[0]) + true_range[0]
                gradient_gt, _ = gradient_check(
                    torch.tensor(at_map, dtype=torch.float32, requires_grad=True).to(
                        device
                    ),
                    grad_operator,
                )
                gradient_pred, _ = gradient_check(pinn_prediction, grad_operator)
                # grt = outputs_data

                ut.write_vtk_data_pointcloud(
                    velocity,
                    dataset.cor,
                    eval_folder + f"eval_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pred_velocities.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder
                    + f"eval_pred_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    1 / pred_velocities.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder
                    + f"eval_dev1_pred_velocity_id_{data_id}_initial_{initial_id}.vtk",
                )
                # ut.write_vtk_data_pointcloud(full_loss.detach().cpu().numpy(), dataset.cor, eval_folder+f'eval_pde_id_{data_id}_initial_{initial_id}.vtk')
                ut.write_vtk_data_pointcloud(
                    at_map,
                    dataset.cor,
                    eval_folder + f"eval_at_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    pinn_prediction.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_at_pred_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    gradient_gt.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_grads_gt_id_{data_id}_initial_{initial_id}.vtk",
                )
                ut.write_vtk_data_pointcloud(
                    gradient_pred.detach().cpu().numpy(),
                    dataset.cor,
                    eval_folder + f"eval_grad_pred_id_{data_id}_initial_{initial_id}.vtk",
                )

elif model_type == "pinn":
    # Create results folder
    result_base_folder = f"Experiments/results_{exp_desc}/pinn"
    if not os.path.exists(result_base_folder):
        os.makedirs(result_base_folder)

    destination_file = os.path.join(result_base_folder, os.path.basename(json_name))
    shutil.copy(json_path, destination_file)
    ut.set_seed(hparams.training["seed"])

    num_tasks = len(test_loaders)

    for data_id in range(num_tasks):
        data_loader = test_loaders[data_id]

        for initial_id in range(len(data_loader)):    
            data_set = data_loaders[data_id][initial_id]
            cor_org = torch.tensor(data_set.cor_org, dtype=torch.float32, requires_grad=True, device=device)
            data_scaled = torch.tensor(data_set.cor, dtype=torch.float32, requires_grad=True, device=device)
            at_map = torch.tensor(data_set.activation_times, dtype=torch.float32, device=device)

            task_id, sample_id, geo_id = data_set.index
            segs = data_set.segs
            velocity = data_set.velocity_field
            laplacian = data_set.laplacian.to(device)
            inputs_residual = data_set.inputs_residual
            residual_indices = data_set.indices_residual
            # input data -- for the data loss
            inputs_data = data_set.inputs_data
            outputs_data = data_set.outputs_data
                
            data_mask = data_set.data_mask
            grad_operator_basis = data_set.grad_operator
            grad_operator = ut.scipy_to_torch_sparse(grad_operator_basis[0]).to(device)
            weight_mask = data_set.weight_mask_residual
            
            # Initialize a new PINN for each initial condition
            model = PINN(len(cor_org)).to(device)
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = lr.CosineAnnealingLR(optimizer, T_max=20, warmup_steps=20, eta_min=1e-7)

            start_time = time.time()
            
            result_folder = f"Experiments/results_{exp_desc}/pinn/geo_{geo_id}_task_{task_id}_initial_{sample_id}"
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            data_losses = []
            total_losses = []
            pde_losses = []
            best_val_loss = float('inf')
            early_stopping_counter = 0
            patience = 20  # Early stopping patience

            data_set = data_loader[initial_id]
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                alpha = ut.update_alpha(epoch, start_epoch=500, end_epoch=2000, start_alpha=1e-4, end_alpha=1)
                
                # compute the data loss
                full_data_loss, loss_data = data_loss(
                    model, inputs_data, at_map, data_set.indices_loss_data, weight_mask
                )

                loss_pde, _, _, _ = pde_loss("pinn", at_map,
                    cor_org,
                    data_scaled,
                    None,
                    grad_operator,
                    grad_operator_basis,
                    laplacian,
                    weight_mask)
                
                pinn_prediction, pred_velocities = model(data_scaled)
                pinn_prediction = pinn_prediction * 1000
                # pinn_prediction_ = model.network(data_scaled)
                total_loss = alpha * loss_pde + loss_data 
               

                total_loss = loss_data
                total_loss.backward()

                optimizer.step()
                scheduler.step()

                average_data_loss = loss_data.item()
                average_total_loss = total_loss.item()
                average_pde_loss = loss_pde.item()

                data_losses.append(average_data_loss)
                total_losses.append(average_total_loss)
                pde_losses.append(average_pde_loss)

                print(f"geo {geo_id} task {task_id}, initial {sample_id}, Epoch {epoch}, Loss: {average_total_loss} Data Loss : {average_data_loss} PDE Loss : {average_pde_loss}")
                
                with open(f"{result_folder}/log.txt", "a") as log_file:
                    print(f"geo {geo_id} task {task_id}, initial {sample_id}, Epoch {epoch}, Loss: {average_total_loss} Data Loss : {average_data_loss} PDE Loss : {average_pde_loss}")
                    log_file.write(
                        f"geo {geo_id} task {task_id}, initial {sample_id}, Epoch {epoch}, Loss: {average_total_loss} Data Loss : {average_data_loss} PDE Loss : {average_pde_loss}"
                    )

                
                
                if epoch % 1000 == 0 or epoch == epoch - 1:
                    ut.AT_map_plot_3d(
                        data_set.cor,
                        epoch,
                        at_map.cpu().numpy(),
                        pinn_prediction.detach().cpu().numpy(),
                        task_id,
                        sample_id,
                        geo_id,
                        result_folder,
                        "train"
                    )
                    generate_velocity_map(
                        result_folder,
                        data_scaled.detach().cpu().numpy(),
                        epoch,
                        velocity,
                        pred_velocities.detach().cpu().numpy(),
                        task_id,
                        sample_id,
                        geo_id,
                        True,
                    )
                
                
                # Early stopping check after 20 epochs
                if epoch > 20:
                    recent_losses = total_losses[-20:]
                    if max(recent_losses) - min(recent_losses) < 1e-4:
                        print(f"Early stopping triggered at epoch {epoch} for sample {data_id}, initial {initial_id}")
                        break

            # Save results and losses for each initial condition
            ut.save_checkpoint(
                model_type,
                f"{result_folder}/inverse_pinn_sample_{data_id}_initial_{initial_id}.pth",
                model,
                optimizer,
                epoch,
            )

            ut.plot_losses(pde_losses, total_losses, data_losses, result_folder)

            end_time = time.time()
            with open(f"{result_folder}/log.txt", "a") as log_file:
                log_file.write(f"Train time for sample {data_id} initial {initial_id}: {(end_time - start_time) / 60.0}\n")

elif model_type == "multi_pinn":
    result_base_folder = f"Experiments/results_{exp_desc}/pinn"
    os.makedirs(result_base_folder, exist_ok=True)

    destination_file = os.path.join(result_base_folder, os.path.basename(json_name))
    shutil.copy(json_path, destination_file)
    ut.set_seed(hparams.training["seed"])

    num_tasks = len(test_loaders)

    for data_id in range(num_tasks):        
        data_loader = test_loaders[data_id]

        velocity_network = VelocityNetwork().to(device)
        velocity_optimizer = AdamW(velocity_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        models = [PINN(len(data_loader[i].cor_org), velocity_network).to(device) for i in range(len(data_loader))]
        optimizers = [AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) for model in models]
        schedulers = [lr.CosineAnnealingLR(opt, T_max=20, warmup_steps=20, eta_min=1e-7) for opt in optimizers]
        start_time = time.time()
        
        total_losses, data_losses, pde_losses = [], [], []
        
        
        for epoch in range(epochs):            
            velocity_network.train()
            for model in models:
                model.train()
                
            velocity_optimizer.zero_grad()

            task_total_loss = 0
            task_data_loss = 0
            task_pde_loss = 0

            alpha = ut.update_alpha(epoch, start_epoch=500, end_epoch=2000, start_alpha=1e-4, end_alpha=1)
            
            for initial_id in range(len(data_loader)):    
                data_set = data_loaders[data_id][initial_id]
                task_id, sample_id, geo_id = data_set.index
                
                
                result_folder = f"Experiments/results_{exp_desc}/pinn/geo_{geo_id}_task_{task_id}_initial_{sample_id}"
                os.makedirs(result_folder, exist_ok=True)
                
                
                cor_org = torch.tensor(data_set.cor_org, dtype=torch.float32, device=device)
                data_scaled = torch.tensor(data_set.cor, dtype=torch.float32, device=device)
                at_map = torch.tensor(data_set.activation_times, dtype=torch.float32, device=device)

                laplacian = data_set.laplacian.to(device)
                grad_operator_basis, grad_operator = grad_operators[geo_id]
                weight_mask = data_set.weight_mask_residual.to(device)
                
                pinn_model = models[initial_id]
                optimizer = optimizers[initial_id]
                optimizer.zero_grad()
                
                pinn_prediction, pred_velocities = pinn_model(data_scaled)
                pinn_prediction = pinn_prediction * 1000
                
                loss_pde, loss_pde_std, full_loss, laplacian_loss = pde_loss("multi_pinn",
                        at_map,
                        cor_org,
                        data_scaled,
                        pred_velocities,
                        grad_operator_basis,
                        grad_operator,
                        laplacian,
                        weight_mask)
                
                full_data_loss, loss_data = data_loss(pinn_model, data_set.inputs_data, at_map, np.union1d(data_set.indices_data, data_set.initial_indices), weight_mask)
                
                total_loss = alpha * loss_pde + loss_data
                task_total_loss += total_loss
                task_data_loss += loss_data
                task_pde_loss += loss_pde
                
                print(f"Epoch {epoch} geo {geo_id} task {task_id}, initial {sample_id}, Loss: {total_loss.item()} Data Loss: {loss_data.item()} PDE Loss: {loss_pde.item()}")
                with open(f"{result_folder}/log.txt", "a") as log_file:
                    log_file.write(
                        f"Epoch {epoch} geo {geo_id} task {task_id}, initial {sample_id}, Loss: {total_loss.item()} Data Loss: {loss_data.item()} PDE Loss: {loss_pde.item()}\n"
                        )
                    
                if epoch % 1000 == 0 or epoch == epochs- 1:
                    ut.AT_map_plot_3d(
                        data_set.cor, epoch, at_map.cpu().numpy(), pinn_prediction.detach().cpu().numpy(), task_id, sample_id, geo_id, result_folder, "train"
                    )
                    generate_velocity_map(
                        result_folder, "train", data_scaled.detach().cpu().numpy(), epoch, data_set.velocity_field, pred_velocities.detach().cpu().numpy(), task_id, sample_id, geo_id, True,
                    )
                

                total_loss.backward(retain_graph=True)  # Retain velocity_net computation graph
                optimizer.step()  # Update only this PINN
                
            
            velocity_optimizer.step()  # Update velocity network after all PINNs
            
            data_losses.append(task_data_loss.item()/len(data_loader))
            total_losses.append(task_total_loss.item()/len(data_loader))
            pde_losses.append(task_pde_loss.item()/len(data_loader))
            
            for scheduler in schedulers:
                scheduler.step()
            
            if epoch % 100 == 0 or epoch == epoch - 1:    
                for initial_id, (pinn_model, optimizer) in enumerate(zip(models, optimizers)):
                    data_set = data_loaders[data_id][initial_id]
                    task_id, sample_id, geo_id = data_set.index

                    result_folder = f"Experiments/results_{exp_desc}/pinn/geo_{geo_id}_task_{task_id}_initial_{sample_id}"
                    os.makedirs(result_folder, exist_ok=True)
                    
                    ut.save_checkpoint(
                        model_type, f"{result_folder}/inverse_pinn_sample_geo_{geo_id}_task_{task_id}_initial_{sample_id}.pth",
                        pinn_model, (optimizer, velocity_optimizer), epoch
                    )

                    print(f"Checkpoint saved for geo {geo_id}, task {task_id}, initial {sample_id} at epoch {epoch}")
            
            # if epoch > 20 and max(total_losses[-20:]) - min(total_losses[-20:]) < 1e-4:
            #     print(f"Early stopping triggered at epoch {epoch} for sample {data_id}, initial {initial_id}")
            #     break
            # if epoch % 50 == 0 or epoch == epoch - 1:
            #     ut.save_checkpoint(
            #         model_type, f"{result_folder}/inverse_pinn_sample_geo_{geo_id}_task_{task_id}_initial_{sample_id}.pth", model, optimizer, epoch
            #     )
                
        ut.plot_losses(pde_losses, total_losses, data_losses, result_folder, "train")

        with open(f"{result_folder}/log.txt", "a") as log_file:
            log_file.write(f"Train time for geo {geo_id} and task {data_id}: {(time.time() - start_time) / 60.0}\n")

# elif model_type == "multi_pinn":
#     result_base_folder = f"Experiments/results_{exp_desc}/pinn"
#     os.makedirs(result_base_folder, exist_ok=True)

#     destination_file = os.path.join(result_base_folder, os.path.basename(json_name))
#     shutil.copy(json_path, destination_file)
#     ut.set_seed(hparams.training["seed"])

#     num_tasks = len(test_loaders)

#     for data_id in range(num_tasks):
#         data_loader = test_loaders[data_id]

#         velocity_network = VelocityNetwork().to(device)
#         velocity_optimizer = AdamW(velocity_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
#         models = [PINN(len(data_loader[i].cor_org), velocity_network).to(device) for i in range(len(data_loader))]
#         optimizers = [AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) for model in models]
#         schedulers = [lr.CosineAnnealingLR(opt, T_max=20, warmup_steps=20, eta_min=1e-7) for opt in optimizers]
#         start_time = time.time()
        
#         total_losses, data_losses, pde_losses = [], [], []

#         for epoch in range(epochs):
#             velocity_network.train()
#             for model in models:
#                 model.train()

#             # ✅ Reset velocity network optimizer
#             velocity_optimizer.zero_grad()

#             task_total_loss = 0
#             task_data_loss = 0
#             task_pde_loss = 0

#             alpha = ut.update_alpha(epoch, start_epoch=500, end_epoch=2000, start_alpha=1e-4, end_alpha=1)

#             for initial_id in range(len(data_loader)):    
#                 data_set = data_loaders[data_id][initial_id]
#                 task_id, sample_id, geo_id = data_set.index

#                 result_folder = f"Experiments/results_{exp_desc}/pinn/geo_{geo_id}_task_{task_id}_initial_{sample_id}"
#                 os.makedirs(result_folder, exist_ok=True)

#                 cor_org = torch.tensor(data_set.cor_org, dtype=torch.float32, device=device)
#                 data_scaled = torch.tensor(data_set.cor, dtype=torch.float32, device=device)
#                 at_map = torch.tensor(data_set.activation_times, dtype=torch.float32, device=device)

#                 laplacian = data_set.laplacian.to(device)
#                 grad_operator_basis, grad_operator = grad_operators[geo_id]
#                 weight_mask = data_set.weight_mask_residual.to(device)

#                 pinn_model = models[initial_id]
#                 optimizer = optimizers[initial_id]
                
#                 optimizer_i.zero_grad()

#                 pinn_prediction, pred_velocities = pinn_model(data_scaled)
#                 pinn_prediction = pinn_prediction * 1000

#                 loss_pde, loss_pde_std, full_loss, laplacian_loss = pde_loss(
#                     "multi_pinn", at_map, cor_org, data_scaled, pred_velocities,
#                     grad_operator_basis, grad_operator, laplacian, weight_mask
#                 )

#                 full_data_loss, loss_data = data_loss(
#                     pinn_model, data_set.inputs_data, at_map, 
#                     np.union1d(data_set.indices_data, data_set.initial_indices), weight_mask
#                 )

#                 total_loss = alpha * loss_pde + loss_data
#                 task_total_loss += total_loss
#                 task_data_loss += loss_data
#                 task_pde_loss += loss_pde

#                 print(f"Epoch {epoch} geo {geo_id} task {task_id}, initial {sample_id}, Loss: {total_loss.item()} Data Loss: {loss_data.item()} PDE Loss: {loss_pde.item()}")

#                 with open(f"{result_folder}/log.txt", "a") as log_file:
#                     log_file.write(
#                         f"Epoch {epoch} geo {geo_id} task {task_id}, initial {sample_id}, "
#                         f"Loss: {total_loss.item()} Data Loss: {loss_data.item()} PDE Loss: {loss_pde.item()}\n"
#                     )

#                 # velocity_loss_accum = velocity_loss_accum + total_loss.detach()  # ✅ Correct loss accumulation
#                 total_loss.backward(retain_graph=True)  # ✅ Retain computation graph for velocity_net
#                 optimizer.step()  # ✅ Update only this PINN

#             velocity_optimizer.step()  # ✅ Update velocity network

#             data_losses.append(task_data_loss.item() / len(data_loader))
#             total_losses.append(task_total_loss.item() / len(data_loader))
#             pde_losses.append(task_pde_loss.item() / len(data_loader))

#             for scheduler in schedulers:
#                 scheduler.step()

#             # ✅ Save checkpoints correctly
#             if epoch % 100 == 0 or epoch == epochs - 1:    
#                 for initial_id, (pinn_model, optimizer) in enumerate(zip(models, optimizers)):
#                     data_set = data_loaders[data_id][initial_id]
#                     task_id, sample_id, geo_id = data_set.index

#                     result_folder = f"Experiments/results_{exp_desc}/pinn/geo_{geo_id}_task_{task_id}_initial_{sample_id}"
#                     os.makedirs(result_folder, exist_ok=True)

#                     ut.save_checkpoint(
#                         model_type, f"{result_folder}/inverse_pinn_sample_geo_{geo_id}_task_{task_id}_initial_{sample_id}.pth",
#                         pinn_model, (optimizer, velocity_optimizer), epoch
#                     )

#                     print(f"Checkpoint saved for geo {geo_id}, task {task_id}, initial {sample_id} at epoch {epoch}")

#         ut.plot_losses(pde_losses, total_losses, data_losses, result_folder, "train")

#         with open(f"{result_folder}/log.txt", "a") as log_file:
#             log_file.write(f"Train time for geo {geo_id} and task {data_id}: {(time.time() - start_time) / 60.0}\n")

elif model_type == 'optimization':
    print("to be implemented")