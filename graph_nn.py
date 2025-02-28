import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.io as sio
import torch_geometric.transforms as T
from torch_geometric.data import Data


def read_matrix(file_name):
    return np.fromfile(file_name)


device = 'cuda:0'
nodes = read_matrix("/home/mt6129/Eikonal/data/volumetric_combined/geometry/EC.cor")
nodes = nodes.reshape((-1, 3))
t_nodes = torch.from_numpy(nodes)

graph = Data(x=torch.zeros(nodes.shape[0], 1), y=torch.zeros(2), pos=t_nodes)

pre_transform = T.KNNGraph(k=12)
graph = pre_transform(graph)

# a = kneighbors_graph(nodes, 3)
# arr = a.toarray()

# points = np.where(arr == 1)
# edges = np.reshape(points, (2, -1))

# import ipdb; ipdb.set_trace()
sio.savemat("/home/mt6129/Eikonal/data/volumetric_combined_large_neighbor/geometry/EC.mat", {'nodes': nodes, 'edges': graph.edge_index.cpu().numpy()})
