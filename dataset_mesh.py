import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
from heapq import heappush, heappop
import vtk
import scipy

# from fimpy.solver import FIMPY
import scipy.io
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk
import random
from collections import deque
import numpy as np
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian
import pickle
from mesh2graph import prepare_gcn_input
import robust_laplacian
from plyfile import PlyData
import polyscope as ps

import utils.utils as ut


class Graph:
    def __init__(self, node_org, nodes, faces):
        self.nodes = nodes
        self.node_org = node_org
        self.faces = faces.T
        self.edges = faces.T
        self.adjacancy_matrix = self.compute_adjacency_matrix(self.edges, len(nodes))

    def mesh_to_edges(self, faces):
        # Create edges from faces by taking unique pairs of vertices from each triangle
        edges = set()
        for face in faces:
            for i in range(3):
                # Create an edge for each pair of vertices in the face
                edge = (
                    min(face[i], face[(i + 1) % 3]),
                    max(face[i], face[(i + 1) % 3]),
                )
                edges.add(edge)
        return np.array(list(edges)).T  # Transpose to get 2xN shape

    def create_graph_data(self, graph_data, features, mask, device):
        # Convert numpy arrays to torch tensors

        nodes = torch.tensor(self.nodes, dtype=torch.float32).to(device)
        x = torch.cat(
            (features.view(features.shape[0], -1), mask.view(features.shape[0], -1)),
            dim=1,
        )
        # x = torch.cat((features.view(features.shape[0], -1), nodes, mask.view(features.shape[0], -1),  self.initial_mask.view(features.shape[0], -1)), dim=1)
        # x = features
        edge_index = torch.tensor(self.edges.astype(np.int64), dtype=torch.int64)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            pos=nodes,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            batch=batch,
        )
        return data

    def compute_adjacency_matrix(self, edges, num_nodes):
        # Initialize the adjacency matrix with zeros
        adjacency_matrix = torch.zeros((num_nodes, num_nodes))

        # Fill in the adjacency matrix based on edges
        for i in range(edges.shape[1]):  # edges.shape[1] is the number of edges
            source, target = edges[:, i]
            adjacency_matrix[source, target] = 1
            adjacency_matrix[target, source] = 1  # Assuming undirected graph

        return adjacency_matrix

    def compute_laplacian_matrix(self, nodes):
        L, M = robust_laplacian.point_cloud_laplacian(nodes)
        L = L.tocoo()  # Ensure it's in COOrdinate format (row, col, value)
    
        indices = torch.tensor([L.row, L.col], dtype=torch.long)  # Indices of nonzero elements
        values = torch.tensor(L.data, dtype=torch.float32)  # Values of nonzero elements
        
        sparse_L = torch.sparse_coo_tensor(indices, values, L.shape, dtype=torch.float32)
        return sparse_L
        # # Compute the degree matrix
        # degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))

        # # Compute the Laplacian matrix
        # laplacian_matrix = degree_matrix - adjacency_matrix
        # return laplacian_matrix

    def compute_edge_lengths(self, nodes, edges):
        # Calculate the Euclidean distance between nodes for each edge
        node_coords = torch.tensor(nodes[edges], dtype=torch.float32)
        edge_lengths = torch.sqrt(
            torch.sum((node_coords[0] - node_coords[1]) ** 2, dim=1)
        )
        return edge_lengths

    def edge_length(self, node1, node2):
        return np.linalg.norm(node1 - node2)

    def compute_grad_operator(self):
        edge_index = knn_graph(self.node_org, 10)
        normal, x_basis, y_basis = estimate_basis(self.node_org, edge_index)
        # Build gradient and divergence operators (Scipy sparse matrices)
        grad, _ = build_grad_div(self.node_org, normal, x_basis, y_basis, edge_index)

        grad_and_basis = [grad, x_basis, y_basis]

        return grad_and_basis

    def compute_laplacian(self, device):
        # Adjust indexing if necessary (MATLAB to Python)
        adjacency_matrix = self.compute_adjacency_matrix(self.edges, len(self.nodes))
        laplacian_matrix = self.compute_laplacian_matrix(self.nodes)

        # grad_operator = self.compute_grad_operator(edges, len(self.nodes))

        # You can now use the laplacian_matrix and grad_operator for further processing or regularization
        return laplacian_matrix


class IsotropicEikonalSolver:
    def __init__(self, points, triangs, velocity_field):
        self.points = points
        self.triangs = triangs
        self.velocity_field = (
            velocity_field  # This is a scalar field corresponding to each point
        )

        # Compute the average velocity for each triangle
        tri_velocities = np.mean(velocity_field[triangs], axis=1)

        # Since it's isotropic, the tensor simplifies to a scalar velocity field
        # Create one isotropic metric tensor for each triangle
        D_iso = np.array([np.diag([v, v, v]) for v in tri_velocities])

        # Initialize the Fast Iterative Method solver
        self.fim = FIMPY.create_fim_solver(
            self.points, self.triangs, D_iso, device="cpu", use_active_list=False
        )

    def solve(self, source_points):
        # Compute the Fast Marching activation times
        activation_times = self.fim.comp_fim(
            source_points, np.zeros_like(source_points)
        )
        return activation_times


def generate_slow_conduction_blocks(
    points, num_blocks, block_radius, normal_conduction_value, slow_conduction_value
):
    """
    Generate random blocks of slow conduction regions within the mesh.

    :param points: numpy array of point coordinates in the mesh.
    :param num_blocks: number of slow conduction blocks to generate.
    :param block_radius: radius around each block's central point that defines the block size.
    :param slow_conduction_value: the velocity value to assign within the slow conduction blocks.
    :return: velocity_field (numpy array with a velocity value for each point in the mesh).
    """
    # Initialize the velocity field with a default value (assuming normal conduction everywhere initially)
    velocity_field = np.full(
        points.shape[0], normal_conduction_value, dtype=float
    )  # Set your default conduction velocity here

    # Build a KDTree for efficient spatial queries
    kdtree = cKDTree(points)

    for _ in range(num_blocks):
        # Randomly select a central point for the block
        block_center_idx = np.random.randint(len(points))
        block_center = points[block_center_idx]

        # Find all points within the block_radius of the block_center
        indices = kdtree.query_ball_point(block_center, block_radius)

        # Set the slow conduction value for points within the block
        velocity_field[indices] = slow_conduction_value

    return velocity_field, indices


def generate_initial_condition(points, block_radius, scar_indices):
    kdtree = cKDTree(points)

    # Initialize variables
    indices = []
    valid = False

    # Loop until a valid block center is found
    while not valid:
        block_center_idx = np.random.randint(len(points))
        block_center = points[block_center_idx]

        # Find all points within the block_radius of the block_center
        indices = kdtree.query_ball_point(block_center, block_radius)

        # Check if any of the indices are in scar_indices
        if not any(idx in scar_indices for idx in indices):
            valid = True  # Found a valid set of indices
        else:
            # If invalid, clear indices and try again
            indices = []

    return indices


def save_as_vtk(vertices, faces, activation_times, filename):
    # Create a vtkPoints object and insert the vertices
    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(v)

    # Create the polygons (triangles)
    triangles = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, f[0])
        triangle.GetPointIds().SetId(1, f[1])
        triangle.GetPointIds().SetId(2, f[2])
        triangles.InsertNextCell(triangle)

    # Create a vtkPolyData object and set the points and polygons
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(triangles)

    # Convert the activation times to a vtkFloatArray
    activation_times_vtk = numpy_to_vtk(
        activation_times, deep=True, array_type=vtk.VTK_FLOAT
    )
    activation_times_vtk.SetName("ActivationTimes")
    polyData.GetPointData().SetScalars(activation_times_vtk)

    # Write the polydata object to a file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()


def save_scar_area_as_vtk(points, scar_indices, filename):
    """
    Save the scar area as a VTK file that can be visualized as a point cloud with adjustable size.

    :param points: numpy array of all point coordinates in the mesh.
    :param scar_indices: list or numpy array of indices that define the scar area.
    :param filename: path to the output VTK file.
    """
    # Create a vtkPoints object and insert the scar points
    vtk_points = vtk.vtkPoints()
    for idx in scar_indices:
        vtk_points.InsertNextPoint(points[idx])

    # Create a vtkCellArray object and insert a vertex for each scar point
    vertices = vtk.vtkCellArray()
    for i in range(len(scar_indices)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)

    # Create a vtkPolyData object and set the points and vertices
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)
    polyData.SetVerts(vertices)

    # Write the polydata object to a file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()


def prepare_training_data_3d(
    graph_data,
    cor_org,
    faces,
    activation_times_org,
    indices_residual,
    indices_data,
    initial_threshold,
    device,
):
    # min_val = cor_org.min()
    # max_val = cor_org.max()
    # cor = (cor_org - min_val) / (max_val - min_val)
    cor = cor_org / 1000
    # cor = cor + abs(cor.min())
    activation_times = activation_times_org.copy()
    activation_times = activation_times / 1000
    # act_min = activation_times.min()
    # act_max = activation_times.max()

    # activation_times = (activation_times - act_min) / (act_max - act_min)
    # noise_level = 0.2
    # noise = np.random.uniform(-noise_level/2, noise_level/2, activation_times.shape)

    # Add noise to the original data
    # activation_times = activation_times + noise

    # t = activation_times.ravel().copy()

    # cor_len = len(cor)

    # Sample indices for residual and data points

    # Convert selected samples to tensors
    
    # initial_mask = -1 * np.ones_like(activation_times, dtype=np.float32)
    # initial_mask[activation_times < initial_threshold] = activation_times[
    #     activation_times < initial_threshold
    # ]
    # initial_activation_times = activation_times
    # initial_activation_times[initial_mask == 0] = -
    
    num_keep = int(len(activation_times) * initial_threshold)  # 20% of indices
    threshold_value = np.partition(activation_times, num_keep)[num_keep]  # Get cutoff value

    # Get indices where activation time is below threshold
    initial_indices = np.where(activation_times <= threshold_value)[0]

    # Create mask matrix: 1 at selected indices, 0 elsewhere
    initial_mask = np.zeros_like(activation_times)
    initial_mask[initial_indices] = 1

    # Create activation_time_masked: -1 everywhere except for selected indices
    initial_activation_times = np.full_like(activation_times, -1)
    initial_activation_times[initial_indices] = activation_times[initial_indices]
    
    inputs_residual = torch.tensor(cor, dtype=torch.float32, requires_grad=True).to(
        device
    )
    outputs_residual = torch.tensor(activation_times, dtype=torch.float32).to(device)
    # inputs_data = torch.tensor(cor[indices_data], dtype=torch.float32).to(device)
    # outputs_data = torch.tensor(activation_times[indices_data], dtype=torch.float32).to(device)
    inputs_data = torch.tensor(cor, dtype=torch.float32).to(device)
    outputs_data = torch.tensor(activation_times, dtype=torch.float32).to(device)

    # Create and reshape the mask matrix to 3D
    mask_matrix = np.zeros_like(activation_times, dtype=np.float32)
    mask_matrix[indices_data] = 1

    activation_times[mask_matrix == 0] = -1

    # t_min = t.min()
    # t_max = t.max()

    # # Perform min-max scaling
    # t_normalized = (t - t_min) / (t_max - t_min)


    at = torch.tensor(activation_times, dtype=torch.float32).to(device)
    mask_matrix = torch.tensor(mask_matrix, dtype=torch.float32).to(device)
    initial_mask = torch.tensor(initial_mask, dtype=torch.float32).to(device)
    
    graph = Graph(cor_org, cor, faces)
    # geo_name, num_meshfree, features, mask, device
    initial_graph = graph.create_graph_data(
        graph_data,
        torch.tensor(
            initial_activation_times, dtype=torch.float32, requires_grad=True
        ).to(device),
        initial_mask,
        device,
    )
    pinn_graph = graph.create_graph_data(graph_data, at, mask_matrix, device)
    # pinn_graph = graph.create_graph_data(at, mask_matrix, device)
    # gradient_operator = graph.compute_grad_operator()
    laplacian = graph.compute_laplacian(device)

    return (
        inputs_residual,
        outputs_residual,
        indices_residual,
        inputs_data,
        outputs_data,
        pinn_graph,
        initial_graph,
        laplacian,
        indices_data,
        mask_matrix,
        cor,
        initial_indices
    )


def get_grad_operator(cor_org, faces, device):
    cor = cor_org / 1000
    graph = Graph(cor_org, cor, faces)
    gradient_operator_basis = graph.compute_grad_operator()
    grad_operator = ut.scipy_to_torch_sparse(gradient_operator_basis[0]).to(device)
    return grad_operator, gradient_operator_basis
    
    

def get_data_indx(data_type, num_nodes, data_perc, data_dir, geo_name):
    idx_file = f"{data_dir}geometry/{geo_name}.idx"
    idx = np.fromfile(idx_file, dtype=np.int32)
    idx = idx.reshape((-1, 1))
    if data_type == "random":
        target_indices = random.sample(range(num_nodes), int(data_perc * num_nodes))
        loss_indices = target_indices
    elif data_type == "endo":
        target_indices = np.where((idx == 1) | (idx == 2))[0]
        # loss_indices = target_indices
        loss_indices = random.sample(range(num_nodes), int(data_perc * num_nodes))
    elif data_type == "epi":
        target_indices = np.where((idx == 3))[0]
        # loss_indices = target_indices
        loss_indices = random.sample(range(num_nodes), int(data_perc * num_nodes))
    else:
        raise NotImplementedError("The specified data_type is not implemented.")
    
    return target_indices, loss_indices