import torch
import os
import torch.nn as nn
import numpy as np
import scipy.io
import shutil
import psutil
import random
import time
from torch.optim import AdamW
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import MessagePassing
# from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import utils as ut
import torch.nn.init as init
import graph_network_arch4 as net
import matplotlib.pyplot as plt
# from utils import Params
import scipy.io as sio
import torch_geometric.transforms as T
import dataset_mesh as dataset_mesh
import lr
import heapq
from vtk.util.numpy_support import numpy_to_vtk
import vtk

# Load the heart mesh from the .mat file
class TaskDataset(Dataset):
    def __init__(self, cor, faces, velocity_field, activation_times, task_id, sample_id):
        self.cor = cor
        self.faces = faces
        self.activation_times = activation_times
        self.velocity_field = velocity_field
        self.index = (task_id, sample_id)

    def __len__(self):
        return len(self.activation_times)

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

def read_matrix(file_name):
    return np.fromfile(file_name)
        
def save_as_vtk(vertices, faces, activation_times, filename):
    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(v)

    triangles = vtk.vtkCellArray()
    for f in faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, f[0])
        triangle.GetPointIds().SetId(1, f[1])
        triangle.GetPointIds().SetId(2, f[2])
        triangles.InsertNextCell(triangle)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(triangles)

    activation_times_vtk = numpy_to_vtk(activation_times, deep=True, array_type=vtk.VTK_FLOAT)
    activation_times_vtk.SetName("ActivationTimes")
    polyData.GetPointData().SetScalars(activation_times_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()

def save_scar_area_as_vtk(points, scar_indices, filename):
    vtk_points = vtk.vtkPoints()
    for idx in scar_indices:
        vtk_points.InsertNextPoint(points[idx])

    vertices = vtk.vtkCellArray()
    for i in range(len(scar_indices)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)
    polyData.SetVerts(vertices)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)
    writer.Write()

def save_point_cloud_as_vtk(points, activation_times, filename):
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point.tolist())  # Convert each point to a list before insertion

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)

    # Convert activation times to a vtk array and associate it with the points
    activation_times_vtk = vtk.vtkFloatArray()
    activation_times_vtk.SetName("ActivationTimes")
    for time in activation_times:
        activation_times_vtk.InsertNextValue(time)
    polyData.GetPointData().SetScalars(activation_times_vtk)

    # Apply vtkVertexGlyphFilter to convert points to vertices
    glyphFilter = vtk.vtkVertexGlyphFilter()
    glyphFilter.SetInputData(polyData)
    glyphFilter.Update()

    # Update the mapper and writer to use the output from the glyph filter
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyphFilter.GetOutput())

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(glyphFilter.GetOutput())
    writer.Write()

    # If needed, you can adjust point size in ParaView through the 'Point Size' option in the 'Properties' panel.


normal_conduction_value = 0.6
slow_conduction_value = 0.1  # The slowed down conduction velocity
# Define a scalar velocity field (for example, 0.8 everywhere and 0.1 in a specified region)

# Load the heart mesh from the .mat file
name = "AW"
# mat_contents = scipy.io.loadmat("data/registered_5/geometry/" + name + ".mat")
# nodes = mat_contents['nodes']
# faces = mat_contents['edges'].T



device = 'cuda:0'
nodes = read_matrix("data/registered_5_updated/geometry/" + name + ".cor")
nodes = nodes.reshape((-1, 3))

# Generate the velocity field with random slow conduction blocks
seg_file = np.fromfile('data/registered_5_updated/geometry/' + name + ".seg", dtype=np.int32).flatten()

def visualize_segments(seg_file):
    save_point_cloud_as_vtk(nodes, seg_file, 'data/registered_5_updated/segments/'+ f'geo{name}_segments.vtk')

visualize_segments(seg_file)
#pair different segments for larger segments
# segs = [[0,1,6,7], [4,5,10,11], [2,3,8,9], [12,13,14,15,16], [17]] #DC, AW, EC
segs = [[0,1,6,7,12], [4,5,10,11,15], [2,3,8,9,13,14], [16], [17]] #DC, AW, EC
# segs = [[1,2,7,8], [4,5,10,11,15], [3,8,9,13, 14], [15,16], [17]] #DC, AW, EC
# segs = [[1,4], [3,6,8], [2,5], [7,9], [10,11,12,13,14]] #PSTOV3
# segs = [[1,4], [3,6], [2,5], [7,8], [9,10]] #PSTOV1,2
# segs = [[9,10]]
# segs = [[0]]

# #17 is right ventricle and segments start from 0 so we subtract 1 from all
# for i, seg in enumerate(segs):
#     for j, s in enumerate(seg):
#         segs[i][j] = segs[i][j] - 1
t_nodes = torch.from_numpy(nodes)
graph = Data(x=torch.zeros(nodes.shape[0], 1), y=torch.zeros(2), pos=t_nodes)
pre_transform = T.KNNGraph(k=5)
graph = pre_transform(graph)
faces = graph.edge_index.cpu().numpy().T
sio.savemat("data/registered_5_updated/geometry/" + name + ".mat", {'nodes': nodes, 'edges': graph.edge_index.cpu().numpy()})
    
# 17 segmentations
for i in range(len(segs)):
    velocity_field = np.full(nodes.shape[0], normal_conduction_value)
    mask = np.where(np.isin(seg_file, segs[i]))[0]
    velocity_field[mask] = slow_conduction_value
    solver = EikonalSolverPointCloud(nodes, faces, velocity_field)

    for j in range(0, nodes.shape[0], 25):
    # for j in range(0, 1):
        if j in mask:
            continue
        activation_times = solver.solve([j,])
        id = segs[i][0]+1
        dataset = TaskDataset(nodes, faces, velocity_field, activation_times, id, j)
        # TODO: save the activation time map to the path you want
        # filename = 'at_map_seg{}_exc{}'.format(i, j)
        filename = os.path.join('data/registered_5_updated/dataset/', f'{name}_at_map_seg{i}_exc{j}.pt')
        torch.save(dataset, filename)
        save_point_cloud_as_vtk(nodes, activation_times, 'data/registered_5_updated/at_vtks/'+ f'{name}_at_map_seg{i}_exc{j}.vtk')
        save_point_cloud_as_vtk(nodes, dataset.velocity_field, 'data/registered_5_updated/velocity_vtks/'+ f'{name}_vel_map_seg{i}_exc{j}.vtk')