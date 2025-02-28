import torch
import numpy as np
from torch_geometric.data import Data
from heapq import heappush, heappop
import vtk
import scipy.io
from vtk.util.numpy_support import numpy_to_vtk
import random
from sklearn.neighbors import NearestNeighbors
import heapq

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

def save_point_cloud_as_vtk(points, activation_times, filename, point_size=5.0):
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point.tolist())  # Convert each point to a list before insertion

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)

    activation_times_vtk = numpy_to_vtk(activation_times, deep=True, array_type=vtk.VTK_FLOAT)
    activation_times_vtk.SetName("ActivationTimes")
    polyData.GetPointData().SetScalars(activation_times_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polyData)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    writer.Write()


# Load the heart mesh from the .mat file
mat_contents = scipy.io.loadmat('/data/Halifax/EC/Final/vheart.mat')
nodes = mat_contents['nodes']
faces = mat_contents['edges'].T
num_blocks = 1  # Number of blocks of slow conduction you want to create
block_radius = 20  # Radius around the block center that defines the block size

normal_conduction_value = 0.5
slow_conduction_value = 0.01  # The slowed down conduction velocity
# Define a scalar velocity field (for example, 0.8 everywhere and 0.1 in a specified region)

# Generate the velocity field with random slow conduction blocks
seg = np.fromfile('/data/Halifax/EC/Final/heart.seg', dtype=np.int32)

# 17 segmentations
for i in range(17):
    velocity_field = np.full(nodes.shape[0], normal_conduction_value)
    mask = np.where(seg == i)[0]
    velocity_field[mask] = slow_conduction_value
    solver = EikonalSolverPointCloud(nodes, faces, velocity_field)

    for j in range(nodes.shape[0]):
        if j in mask:
            continue
        activation_times = solver.solve([j,])

        # TODO: save the activation time map to the path you want
        save('/path/to/output/at_map_seg{}_exc{}.mat'.format(i, j), {'at_map': activation_times})
        filename = '3d_hetero_pcloud.vtk'
        save_point_cloud_as_vtk(nodes, activation_times, filename, point_size=0.5)
        scar_filename = 'scar_area_pcloud.vtk'
        save_scar_area_as_vtk(nodes, scar_indices, scar_filename)