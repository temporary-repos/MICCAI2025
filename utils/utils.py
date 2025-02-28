import torch
import numpy as np
import random
import json
import json
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support

# from vtk.util.numpy_support import numpy_to_vtk


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


def set_seed(seed=42):
    """
    Set the seed for reproducibility in all operations in PyTorch.
    """
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# incomplete -- DATS sampler to be added for smart weightening or resource allocation for the meta-model
def DATS(kl_type, data_loaders):
    gradients_i = {}
    gradients_j = {}
    gradient_training = {}
    task_losses_i = {}
    task_losses_j = {}
    probs = {}
    task_u_losses = {}
    task_f_losses = {}

    for data_id in range(hparams.dataset["num_samples"]):
        agg_embedding = model.prepare_encoder_input(data_loaders[data_id])
        for initial_id in range(len(data_loaders[data_id])):
            optimizer.zero_grad()
            data_set = data_loaders[data_id][initial_id]
            travel_time = data_set.activation_times
            true_range = (np.min(travel_time), np.max(travel_time))
            residual_indices = data_set.indices_residual
            inputs_data = data_set.inputs_data
            outputs_data = data_set.outputs_data
            data_scaled = torch.tensor(
                data_set.cor, dtype=torch.float32, requires_grad=True
            ).to(device)
            data_mask = data_set.data_mask
            grad_operator = data_set.grad_operator
            weight_mask = torch.zeros((data_scaled.shape[0], 1)).to(device)
            weight_mask[residual_indices[0 : len(residual_indices) // 2]] = 1
            task_id, sample_id, geo_id = data_set.index
            model(agg_embedding, data_loaders[data_id], initial_id)
            pred_velocities = model.parameter_estimate(data_scaled)
            loss_pde, loss_pde_std, _, _ = pde_loss(
                data_scaled, pred_velocities, true_range, grad_operator, weight_mask
            )
            loss_data = data_loss(
                model, inputs_data, outputs_data, data_set.indices_data, data_mask
            )

            total_loss = alpha * loss_pde + loss_data
            total_loss.backward(retain_graph=True)
            gradient_ = [param.grad.detach().clone() for param in model.parameters()]
            gradient_training[(geo_id, task_id, sample_id)] = gradient_
            task_losses_i[(geo_id, task_id, sample_id)] = total_loss.detach()
            task_u_losses[(geo_id, task_id, sample_id)] = loss_data.detach()
            task_f_losses[(geo_id, task_id, sample_id)] = loss_pde.detach()

            gradient = []
            for g in gradient_:
                if g is not None:
                    gradient.append(g.detach().view(-1))  # <-- detach added here
            gradient = torch.cat(gradient)
            gradients_i[geo_id, task_id, sample_id] = gradient

            optimizer.zero_grad()
            weight_mask = torch.zeros_like(pred_velocities).to(device)
            weight_mask[residual_indices[len(residual_indices) // 2] :] = 1
            loss_pde, loss_pde_std, full_loss, _ = pde_loss(
                data_scaled, pred_velocities, true_range, grad_operator, weight_mask
            )
            total_loss = alpha * loss_pde + loss_data
            total_loss.backward(retain_graph=True)
            gradient_ = [param.grad.detach().clone() for param in model.parameters()]
            task_losses_j[(geo_id, task_id, sample_id)] = total_loss.detach()
            gradient = []
            for g in gradient_:
                if g is not None:
                    gradient.append(g.detach().view(-1))  # <-- detach added here
            gradient = torch.cat(gradient)
            gradients_j[(geo_id, task_id, sample_id)] = gradient

    for key in task_losses_i:
        if kl_type == "uniform":
            gigsum = torch.sum(
                torch.stack(
                    [
                        torch.sqrt(task_losses_i[key] * task_losses_j[task2])
                        * torch.nn.functional.cosine_similarity(
                            gradients_i[key], gradients_j[task2], dim=0
                        )
                        for j, task2 in enumerate(task_losses_i.keys())
                    ]
                )
            )
            gi_gsum_per_epoch[key].append(gigsum.item())
            probs[key] = 1 / len(task_losses_i.keys()) * torch.exp(eta * gigsum)
        elif kl_type == "consecutive":
            gigsum = torch.sum(
                torch.stack(
                    [
                        torch.sqrt(task_losses_i[key] * task_losses_j[task2])
                        * torch.nn.functional.cosine_similarity(
                            gradients_i[key], gradients_j[task2], dim=0
                        )
                        for j, task2 in enumerate(task_losses_i.keys())
                    ]
                )
            )
            gi_gsum_per_epoch[key].append(gigsum.item())
            probs[key] = probs[key].to(device) * torch.exp(eta * gigsum)
        else:
            print("KL option is not implemented")
    prob_sum = sum(torch.sum(p) for p in probs.values() if isinstance(p, torch.Tensor))
    for task in probs.keys():
        probs[task] = probs[task] / prob_sum
        probs_per_epoch[task].append(probs[task])
    keys = list(probs.keys())
    random.shuffle(keys)
    for k in keys:
        probs[k] = probs[k]
    gradients_i = gradients_i
    task_losses_i = task_losses_i
    task_f_losses = task_f_losses
    task_u_losses = task_u_losses

    return probs


def generate_rectangular_scar_mask(
    grid_x, grid_y, grid_z=None, scar_region=None, scar_desc="block"
):
    if scar_desc == "block":
        if grid_z is not None:  # 3D case
            scar_mask = (
                (scar_region["xmin"] <= grid_x)
                & (grid_x <= scar_region["xmax"])
                & (scar_region["ymin"] <= grid_y)
                & (grid_y <= scar_region["ymax"])
                & (scar_region["zmin"] <= grid_z)
                & (grid_z <= scar_region["zmax"])
            )
        else:  # 2D case
            scar_mask = (
                (scar_region["xmin"] <= grid_x)
                & (grid_x <= scar_region["xmax"])
                & (scar_region["ymin"] <= grid_y)
                & (grid_y <= scar_region["ymax"])
            )
    elif scar_desc == "half":
        if grid_z is not None:  # 3D case
            # Define your 3D 'half' condition here
            scar_mask = 0 > grid_x + grid_y + grid_z
        else:  # 2D case
            scar_mask = 0 > grid_x + grid_y

    return scar_mask


def generate_scar_mask(
    grid_x, grid_y, grid_z=None, scar_region=None, scar_desc="block"
):
    scar_center = scar_region[0]
    scar_radius = scar_region[1]
    # Check if the scar description is 'round' for a circular/spherical mask

    if grid_z is not None:  # 3D case
        # Calculate the distance from the center for each point and compare with the radius
        scar_mask = (
            (grid_x - scar_center["x"]) ** 2
            + (grid_y - scar_center["y"]) ** 2
            + (grid_z - scar_center["z"]) ** 2
        ) <= scar_radius**2
    else:  # 2D case
        # Calculate the distance from the center for each point and compare with the radius
        scar_mask = (
            (grid_x - scar_center["x"]) ** 2 + (grid_y - scar_center["y"]) ** 2
        ) <= scar_radius**2

    return scar_mask


def update_alpha(epoch, start_epoch=50, end_epoch=400, start_alpha=0.1, end_alpha=2):
    if epoch <= start_epoch:
        return start_alpha
    elif epoch > start_epoch and epoch <= end_epoch:
        delta_alpha = end_alpha - start_alpha  # Total change required
        delta_epoch = end_epoch - start_epoch  # Total epochs over which change occurs
        increment_per_epoch = delta_alpha / delta_epoch  # Change per epoch
        # Calculate the updated alpha for the current epoch
        updated_alpha = start_alpha + (epoch - start_epoch) * increment_per_epoch
        return updated_alpha
    else:
        return end_alpha


def Velocity_map_plot_2d(
    grid_x,
    grid_y,
    grid_z,
    epoch,
    speed,
    scar_mask,
    output_velocity,
    data_id,
    initial_id,
    exp_desc,
):
    # vmin = speed.min()
    # vmax = speed.max()

    plt.figure(figsize=(8, 6))

    # Ground Truth
    plt.subplot(1, 2, 1)
    # plt.contourf(grid_x, grid_y, speed.reshape(grid_x.shape), levels=50, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contourf(grid_x, grid_y, speed, levels=50, cmap="rainbow")
    plt.colorbar(label="Velocity")
    plt.contour(grid_x, grid_y, scar_mask, levels=[0], colors=("red",), linewidths=(2,))
    plt.title("Ground Truth")

    # PINN Prediction
    plt.subplot(1, 2, 2)
    plt.contourf(
        grid_x,
        grid_y,
        output_velocity.reshape(grid_x.shape[0], grid_y.shape[0]),
        levels=50,
        cmap="rainbow",
    )
    plt.colorbar(label="TVelocity")
    plt.contour(grid_x, grid_y, scar_mask, levels=[0], colors=("red",), linewidths=(2,))
    plt.title("Prediction")
    plt.savefig(
        f"Experiments/results_{exp_desc}/velocity_map_inverse_scar_epoch_{epoch}_id_{data_id}_initial_{initial_id}.png"
    )


def Velocity_map_plot_3d(
    grid_x,
    grid_y,
    grid_z,
    epoch,
    speed,
    scar_mask,
    output_velocity,
    data_id,
    initial_id,
    exp_desc,
):
    coordinates = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    # if epoch == 0 or epoch == 100:
    write_vtk_data_pointcloud(
        speed,
        coordinates,
        f"Experiments/results_{exp_desc}/velocity_epoch_{epoch}_id_{data_id}_initial_{initial_id}.vtk",
    )
    write_vtk_data_pointcloud(
        output_velocity,
        coordinates,
        f"Experiments/results_{exp_desc}/velocity_prediction_epoch_{epoch}_id_{data_id}_initial_{initial_id}.vtk",
    )



def AT_map_plot_2d(
    grid_x, grid_y, epoch, travel_time, pinn_prediction, scar_mask, data_id, exp_desc
):
    # Determine the range of travel_time for consistent color scaling
    # vmin = travel_time.min()
    # vmax = travel_time.max()

    plt.figure(figsize=(8, 6))

    # Ground Truth
    plt.subplot(1, 2, 1)
    # plt.contourf(grid_x, grid_y, travel_time.reshape(grid_x.shape), levels=50, cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.contourf(
        grid_x,
        grid_y,
        travel_time.reshape(grid_x.shape[0], grid_y.shape[0]),
        levels=50,
        cmap="rainbow",
    )
    plt.colorbar(label="Travel Time")
    plt.contour(grid_x, grid_y, scar_mask, levels=[0], colors=("red",), linewidths=(2,))
    plt.title("Ground Truth")

    # PINN Prediction
    plt.subplot(1, 2, 2)
    plt.contourf(grid_x, grid_y, pinn_prediction, levels=50, cmap="rainbow")
    plt.colorbar(label="Travel Time")
    plt.contour(grid_x, grid_y, scar_mask, levels=[0], colors=("red",), linewidths=(2,))
    plt.title("Prediction")
    plt.savefig(
        f"Experiments/results_{exp_desc}/training_result_inverse_scar_epoch_{epoch}_id_{data_id}.png"
    )


def write_vtk_data(data, coordinates, filename):
    # Create a VTK points object
    points = vtk.vtkPoints()

    # Transfer grid coordinates to the VTK points structure
    for i in range(coordinates[0].shape[0]):
        for j in range(coordinates[1].shape[0]):
            for k in range(coordinates[2].shape[0]):
                points.InsertNextPoint(
                    coordinates[0][i, j, k],
                    coordinates[1][i, j, k],
                    coordinates[2][i, j, k],
                )

    # Convert numpy data to VTK data format
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order="F").copy(), deep=True, array_type=vtk.VTK_FLOAT
    )
    # vtk_data = numpy_support.numpy_to_vtk(num_array=data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    # Create the grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(coordinates[0].shape)
    grid.SetPoints(points)
    grid.GetPointData().SetScalars(vtk_data)

    # Write the VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()  # or use SetFileTypeToASCII() for ASCII files
    writer.Write()


def write_vtk_data_pointcloud(data, coordinates, filename):
    points = vtk.vtkPoints()
    # Transfer grid coordinates to the VTK points structure
    for i in range(len(coordinates)):
        points.InsertNextPoint(coordinates[i])
    # for i in range(coordinates[0].shape[0]):
    #     for j in range(coordinates[1].shape[0]):
    #         for k in range(coordinates[2].shape[0]):
    #             points.InsertNextPoint(coordinates[0][i, j, k],
    #                                    coordinates[1][i, j, k],
    #    coordinates[2][i, j, k])

    # Create a point cloud
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)

    # Create a vertex for each point
    vertices = vtk.vtkCellArray()
    for i in range(polyData.GetNumberOfPoints()):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)

    # Add the vertices to the polydata
    polyData.SetVerts(vertices)

    # Add activation times
    activation_times_vtk = numpy_support.numpy_to_vtk(
        data.ravel(order="F").copy(), deep=True, array_type=vtk.VTK_FLOAT
    )
    activation_times_vtk.SetName("ActivationTimes")
    polyData.GetPointData().SetScalars(activation_times_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()


def AT_map_plot_3d(
    cor, epoch, travel_time, pinn_prediction, data_id, initial_id, geo_id, result_folder, description
):
    if epoch == 0:
        write_vtk_data_pointcloud(
            travel_time,
            cor,
            f"{result_folder}/{description}_travel_time_epoch_{epoch}_geo_{geo_id}_id_{data_id}_initial_{initial_id}.vtk",
        )
    write_vtk_data_pointcloud(
        pinn_prediction,
        cor,
        f"{result_folder}/{description}_pinn_prediction_epoch_{epoch}_geo_{geo_id}_id_{data_id}_initial_{initial_id}.vtk",
    )


def plot_losses(pde_losses, data_losses, total_losses, result_folder, train_val):
    plt.figure(figsize=(10, 6))  # Set the figure size
    epochs = range(len(total_losses))  # Assuming all lists are of the same length

    # Plot each of the losses with different colors and labels
    plt.plot(epochs, pde_losses, color="blue", label="PDE Loss")
    plt.plot(epochs, data_losses, color="red", label="Data Loss")
    plt.plot(epochs, total_losses, color="green", label="Total Loss")

    plt.title("Training Losses Over Epochs")  # Title of the plot
    plt.xlabel("Epochs")  # X-axis label
    plt.ylabel("Loss")  # Y-axis label
    plt.legend()  # Add legend to the plot
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust the layout to fit all elements

    plt.savefig(f"{result_folder}/losses_{train_val}.png")  # Display the plot


def plot_loss(pde_losses, losses, name, result_folder):
    plt.figure(figsize=(10, 6))  # Set the figure size
    epochs = range(len(pde_losses))  # Assuming all lists are of the same length

    # Plot each of the losses with different colors and labels
    plt.plot(epochs, losses, color="blue", label=name)

    plt.title(name)  # Title of the plot
    plt.xlabel("Epochs")  # X-axis label
    plt.ylabel("Loss")  # Y-axis label
    plt.legend()  # Add legend to the plot
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()  # Adjust the layout to fit all elements

    plt.savefig(f"{result_folder}/{name}.png")  # Display the plot


def initial_map_plot(
    x, y, z, epoch, mask_input, initial_mask_recon, data_id, initial_id, exp_desc
):
    coordinates = np.meshgrid(x, y, z, indexing="ij")

    # Write VTK files for ground truth and prediction
    write_vtk_data_pointcloud(
        mask_input,
        coordinates,
        f"Experiments/results_{exp_desc}/initial_data_epoch_{epoch}_id_{data_id}_initial_{initial_id}.vtk",
    )
    write_vtk_data_pointcloud(
        initial_mask_recon,
        coordinates,
        f"Experiments/results_{exp_desc}/initial_prediction_epoch_{epoch}_id_{data_id}_initial_{initial_id}..vtk",
    )


def plot_val_over_epoch(
    output_path,
    plot_name,
    epochs,
    tasks,
    task_probs,
    colors,
    x_title,
    y_title,
    plot_title,
):

    # colors = plt.cm.jet(np.linspace(0, 1, len(tasks)))

    fig = plt.figure(figsize=(10, 8))

    for i, task in enumerate(tasks):
        plt.plot(
            range(len(task_probs[task])),
            task_probs[task].cpu().numpy(),
            label="Task " + str(task),
            color=colors[i],
        )

    plt.title(plot_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # plt.legend(loc='best')
    # add legend to the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.subplots_adjust(right=0.7)
    plt.savefig(output_path + "/" + plot_name + ".png")


def scipy_to_torch_sparse(scipy_sparse_matrix):
    coo = scipy_sparse_matrix.tocoo()  # Convert to COO format (coordinate format)
    indices = np.vstack((coo.row, coo.col))  # Stack row and column indices
    values = coo.data  # Non-zero values in the sparse matrix
    shape = coo.shape  # Shape of the sparse matrix

    i = torch.tensor(
        indices, dtype=torch.long
    ).cuda()  # Convert indices to torch and move to GPU
    v = torch.tensor(
        values, dtype=torch.float32
    ).cuda()  # Convert values to torch and move to GPU
    shape = torch.Size(shape)  # Size of the sparse matrix

    return torch.sparse_coo_tensor(i, v, shape)


def gaussian_kernel(distances, sigma=1):
    return torch.exp(-(distances**2) / (2 * sigma**2))


def smoothing(node_org, values, sigma=1):
    # pairwise distances between nodes
    distances = torch.cdist(node_org, node_org, p=2)

    # Apply Gaussian kernel to distances
    kernel = gaussian_kernel(distances, sigma)

    # Normalize the kernel so that the weights for each node sum to 1
    kernel = kernel / kernel.sum(dim=1, keepdim=True)

    smoothed_values = torch.matmul(kernel, values)

    return smoothed_values


def save_checkpoint(model_type, save_path, model, optimizer=None, epoch=None):
    # Create a dictionary of state dictionaries for all components
    if model_type == 'meta_pinn':
        checkpoint = {
            "hypernet_pinn_state_dict": model.hypernet_pinn.state_dict(),
            "hypernet_scar_state_dict": model.hypernet_scar.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "initial_encoder_state_dict": model.initial_encoder.state_dict(),
            # 'net_state_dict': net.state_dict(),
            # 'scar_net_state_dict': scar_net.state_dict(),
        }

        # Save optimizer state and other training information if available
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
    elif model_type == 'meta_base':
        checkpoint = {
            "hypernet_pinn_state_dict": model.hypernet_pinn.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "initial_encoder_state_dict": model.initial_encoder.state_dict()
        }

        # Save optimizer state and other training information if available
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
            
    elif model_type == 'pinn':
        checkpoint = {
            "pinn_state_dict": model.network.state_dict(),
        }

        # Save optimizer state and other training information if available
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
            
    elif model_type == 'multi_pinn':        
        checkpoint = {
            "pinn_state_dict": model.network.state_dict(),  # Save PINN parameters
            "velocity_state_dict": model.scar_network.state_dict(),  # Save shared velocity network parameters
            "pinn_optimizer_state_dict": optimizer[0].state_dict(),  # Save PINN optimizer
            "velocity_optimizer_state_dict": optimizer[1].state_dict(),  # Save velocity optimizer
            "epoch": epoch,  # Save epoch number
            }

        

    # Save the checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(model_type, load_path, model, optimizer=None):
    # Load the checkpoint
    checkpoint = torch.load(load_path)
    
    if model_type == 'meta_pinn':
        # Load state dictionaries into respective models
        model.hypernet_pinn.load_state_dict(checkpoint["hypernet_pinn_state_dict"])
        model.hypernet_scar.load_state_dict(checkpoint["hypernet_scar_state_dict"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.initial_encoder.load_state_dict(checkpoint["initial_encoder_state_dict"])
        # net.load_state_dict(checkpoint['net_state_dict'])
        # scar_net.load_state_dict(checkpoint['scar_net_state_dict'])
    elif model_type == 'meta_base':
        model.hypernet_pinn.load_state_dict(checkpoint["hypernet_pinn_state_dict"])
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.initial_encoder.load_state_dict(checkpoint["initial_encoder_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load additional training information if available
    epoch = checkpoint.get("epoch", None)
    # loss = checkpoint.get('loss', None)

    print(f"Checkpoint loaded from {load_path}.")
    return model  # Return epoch and loss if you want to use them

def errorplots(data_errors_geo, geo_ids, eval_folder, y_title, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data_errors_geo,
        positions=np.arange(len(geo_ids)),
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue"),
    )
    plt.scatter(
        np.arange(len(geo_ids)),
        [np.mean(errors) for errors in data_errors_geo],
        color="red",
        label="Mean",
        s=50,
    )
    plt.errorbar(
        np.arange(len(geo_ids)),
        [np.mean(errors) for errors in data_errors_geo],
        yerr=[np.std(errors) for errors in data_errors_geo],
        fmt="o",
        color="green",
        label="Std Dev",
    )

    plt.xticks(np.arange(len(geo_ids)), [f"Geo {geo_id}" for geo_id in geo_ids])
    plt.title(f"{title}")
    plt.ylabel(y_title)
    plt.legend()
    # plt.show()
    plt.savefig(eval_folder + f"{title}.png")
