import vtk
from vtk.util import numpy_support

# def save_as_vtk(vertices, faces, activation_times, filename):
#     # Code for saving as VTK
#     pass

# def save_scar_area_as_vtk(points, scar_indices, filename):
#     # Code for saving scar area
#     pass

# def save_point_cloud_as_vtk(points, activation_times, filename):
#     # Code for saving point cloud
#     pass

def write_vtk_data_pointcloud(data, coordinates, filename):
    points = vtk.vtkPoints()
    # Transfer grid coordinates to the VTK points structure
    for i in range(len(coordinates)):
        points.InsertNextPoint(coordinates[i])
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