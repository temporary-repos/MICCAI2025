import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SplineConv,
    global_max_pool,
    global_mean_pool,
    graclus,
    max_pool,
)
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, coalesce, degree
import time


class CustomSigmoid(nn.Module):
    def __init__(self, steepness=1, sigma_s=0, sigma_h=1):
        """
        Initialize the custom sigmoid function.
        :param steepness: Coefficient to make the sigmoid function steeper.
        :param sigma_s: The lower bound of the output range.
        :param sigma_h: The upper bound of the output range.
        """
        super(CustomSigmoid, self).__init__()
        self.steepness = steepness
        self.sigma_s = sigma_s
        self.sigma_h = sigma_h

    def forward(self, x):
        """
        Forward pass through the custom sigmoid function.
        :param x: Input tensor.
        :return: Scaled and shifted sigmoid output.
        """
        # Make sigmoid steeper
        # steepened_x = self.steepness * x

        # # Apply the standard sigmoid function
        # sigmoid_output = 1 / (1 + torch.exp(-steepened_x))
        sigmoid_output = torch.sigmoid(self.steepness * x)

        # Scale and shift the output to the desired range [sigma_s, sigma_h]
        scaled_output = sigmoid_output * (self.sigma_h - self.sigma_s) + self.sigma_s

        return scaled_output


# class Encoder_attention(nn.Module):
#     def __init__(self, hparams):
#         super(Encoder_attention, self).__init__()
#         self.num_node = hparams.dataset["num_nodes"]
#         self.num_node_features = 5
#         self.num_layers = hparams.encoder["num_layers"]
#         self.out_dim = hparams.encoder["out_dim"]

#         # Replace GCNConv with GATConv
#         self.convs = nn.ModuleList()
#         channels = self.num_node_features
#         out_channels = [8, 16, 32, 64, 256]
#         for i in range(self.num_layers):
#             heads = 1 if i == self.num_layers - 1 else 8  # Last layer uses 1 head
#             self.convs.append(GATConv(channels, out_channels[i], heads=heads))
#             channels = out_channels[i] * heads

#         # Linear layers remain similar
#         self.fc1 = nn.Linear(self.num_node * out_channels[-1], 256)
#         self.fc3 = nn.Linear(256, self.out_dim)

#         self.activation_fn = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, graph, device):
#         x = graph.x[:, 0:5].to(device)
#         edge_index = graph.edge_index.to(device)

#         for conv in self.convs[:-1]:
#             x = self.activation_fn(conv(x, edge_index))
#             x = self.dropout(x)

#         x = self.convs[-1](x, edge_index)

#         x = self.fc1(x.flatten())
#         x = self.activation_fn(x)
#         graph_embedding = self.fc3(x).view(1, -1)

#         return graph_embedding


# class Encoder(nn.Module):
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#         self.batch_size = 1
#         self.num_node_features = 2  # Node input features (e.g., activation time)
#         self.num_layers = hparams.encoder["num_layers"]
#         self.latent_dim = hparams.encoder["out_dim"]
#         self.nf = hparams.encoder["nf"]

#         # Graph Attention Layers (Replaces Some SplineConv Layers)
#         self.conv1 = GATConv(self.num_node_features, self.nf[1], heads=4, concat=False)
#         self.conv2 = GATConv(self.nf[1], self.nf[2], heads=4, concat=False)
#         self.conv3 = GATConv(self.nf[2], self.nf[3], heads=4, concat=False)
#         self.conv4 = SplineConv(self.nf[3], self.nf[4], dim=3, kernel_size=5)  # Keep Spline for deeper layers
#         self.conv5 = SplineConv(self.nf[4], self.nf[5], dim=3, kernel_size=5)
#         self.conv6 = SplineConv(self.nf[5], self.nf[6], dim=3, kernel_size=5)

#         self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)

#         self.fce1 = torch.nn.Conv2d(self.nf[-1], self.nf[-1], 1)
#         self.fce21 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)
#         self.fce22 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)

#         self.activation_fn = nn.ReLU()

#     def compute_edge_attr(self, x, edge_index):
#         """Assign higher weights to edges connected to nodes with late activation times."""
#         src, dst = edge_index  # Get source and destination nodes
#         activation_times = x[:, 0]  # Assuming activation time is the first feature

#         # Compute edge weights based on later activation nodes
#         edge_weights = torch.exp(activation_times[src])  # Exponential emphasis
#         edge_weights = edge_weights / torch.max(edge_weights)  # Normalize to [0,1]

#         return edge_weights.unsqueeze(1)  # Ensure correct shape for edge_attr

#     def forward(self, geo_name, data, ps, gs, device):
#         """Graph convolutional encoder with attention biasing towards late activation nodes."""
#         data = data.to(device)
#         gParams = gs[geo_name]
#         pParams = ps[geo_name]

#         # Graph hierarchy
#         bg1, bg2, bg3, bg4, bg5 = [gParams[i].to(device) for i in range(1, 6)]
#         P01, P12, P23, P34, P45, P56 = [pParams[i].to(device) for i in range(6)]

#         # Preprocess node features
#         x, edge_index, edge_attr = data.x.clone(), data.edge_index, data.edge_attr  # Clone to avoid modification
#         x[:, 0] = x[:, 0] / torch.max(x[:, 0])  # Normalize activation times (0 to 1)
#         x[:, 0] = x[:, 0] ** 2  # Square activation times to emphasize late activation

#         # Layer 1 (GAT Attention)
#         edge_attr = self.compute_edge_attr(x, edge_index)
#         x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P01.T, x)
#         x = x.view(self.batch_size, -1, self.nf[1])

#         # Layer 2 (GAT Attention)
#         edge_attr = self.compute_edge_attr(x.view(-1, self.nf[1]), bg1.edge_index)
#         x = F.elu(self.conv2(x.view(-1, self.nf[1]), bg1.edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P12.T, x)
#         x = x.view(self.batch_size, -1, self.nf[2])

#         # Layer 3 (GAT Attention)
#         edge_attr = self.compute_edge_attr(x.view(-1, self.nf[2]), bg2.edge_index)
#         x = F.elu(self.conv3(x.view(-1, self.nf[2]), bg2.edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P23.T, x)
#         x = x.view(self.batch_size, -1, self.nf[3])

#         # Layer 4 (SplineConv)
#         edge_attr = self.compute_edge_attr(x.view(-1, self.nf[3]), bg3.edge_index)
#         x = F.elu(self.conv4(x.view(-1, self.nf[3]), bg3.edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P34.T, x)
#         x = x.view(self.batch_size, -1, self.nf[4])

#         # Layer 5 (SplineConv)
#         edge_attr = self.compute_edge_attr(x.view(-1, self.nf[4]), bg4.edge_index)
#         x = F.elu(self.conv5(x.view(-1, self.nf[4]), bg4.edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P45.T, x)
#         x = x.view(self.batch_size, -1, self.nf[5])

#         # Layer 6 (SplineConv)
#         edge_attr = self.compute_edge_attr(x.view(-1, self.nf[5]), bg5.edge_index)
#         x = F.elu(self.conv6(x.view(-1, self.nf[5]), bg5.edge_index, edge_attr=edge_attr))
#         x = torch.matmul(P56.T, x)
#         x = x.view(self.batch_size, -1, self.nf[6])

#         # Global pooling
#         x = self.pooling_layer(x.transpose(1, 2))
#         embedding = x.squeeze(-1)

#         return embedding

# Contrastive Loss Function (Optional)
def contrastive_loss(anchor, positive, negative, margin=1.0):
    """Encourage embeddings of different activation distributions to be distinct."""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return torch.clamp(margin + pos_dist - neg_dist, min=0).mean()

class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.batch_size = 1
        self.num_node_features = 2  # Node input features (e.g., activation time)
        self.num_layers = hparams.encoder["num_layers"]
        self.latent_dim = hparams.encoder["out_dim"]
        self.nf = hparams.encoder["nf"]

        # Define output channels per layer (based on the diagram)
        out_channels = [8, 16, 32, 64, 128]  # Channels per layer

        # SplineConv layers
        self.conv1 = SplineConv(
            self.num_node_features, self.nf[1], dim=3, kernel_size=5
        )
        self.conv2 = SplineConv(self.nf[1], self.nf[2], dim=3, kernel_size=5)
        self.conv3 = SplineConv(self.nf[2], self.nf[3], dim=3, kernel_size=5)
        self.conv4 = SplineConv(self.nf[3], self.nf[4], dim=3, kernel_size=5)
        self.conv5 = SplineConv(self.nf[4], self.nf[5], dim=3, kernel_size=5)
        self.conv6 = SplineConv(self.nf[5], self.nf[6], dim=3, kernel_size=5)

        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)

        self.fce1 = torch.nn.Conv2d(self.nf[-1], self.nf[-1], 1)
        self.fce21 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)
        self.fce22 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)

        self.activation_fn = nn.ReLU()

    def compute_edge_attr(self, pos, edge_index):
        """For testing: Return a tensor of all ones with shape [num_edges, 3]."""
        num_edges = edge_index.size(1)
        edge_attr = torch.ones(num_edges, 3).to(
            pos.device
        )  # Shape: [num_edges, 3] (3 for dim=3)
        return edge_attr

    def forward(self, geo_name, data, ps, gs, device):
        """graph convolutional encoder"""
        data = data.to(device)
        gParams = gs[geo_name]
        pParams = ps[geo_name]
        bg = gParams[0].to(device)
        bg1 = gParams[1].to(device)
        bg2 = gParams[2].to(device)
        bg3 = gParams[3].to(device)
        bg4 = gParams[4].to(device)
        bg5 = gParams[5].to(device)

        P01 = pParams[0].to(device)
        P12 = pParams[1].to(device)
        P23 = pParams[2].to(device)
        P34 = pParams[3].to(device)
        P45 = pParams[4].to(device)
        P56 = pParams[5].to(device)

        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr,
        )  # (1230*bs) X f[0]
        x = F.elu(self.conv1(x, edge_index, edge_attr))  # (1230*bs) X f[1]
        x = torch.matmul(P01.T, x)  # bs X 648 X f[1]
        x = x.view(self.batch_size, -1, self.nf[1])  # bs X 1230 X f[1]
        # layer 2
        x, edge_index, edge_attr = x.view(-1, self.nf[1]), bg1.edge_index, bg1.edge_attr
        x = F.elu(self.conv2(x, edge_index, edge_attr))  # 648*bs X f[2]
        x = torch.matmul(P12.T, x)  # bs X 347 X f[2]
        x = x.view(self.batch_size, -1, self.nf[2])  # bs X 648 X f[2]

        # layer 3
        x, edge_index, edge_attr = x.view(-1, self.nf[2]), bg2.edge_index, bg2.edge_attr
        x = F.elu(self.conv3(x, edge_index, edge_attr))  # 347*bs X f[3]
        x = torch.matmul(P23.T, x)  # bs X 184 X f[3]
        x = x.view(self.batch_size, -1, self.nf[3])  # bs X 347 X f[3]

        # layer 4
        x, edge_index, edge_attr = x.view(-1, self.nf[3]), bg3.edge_index, bg3.edge_attr
        x = F.elu(self.conv4(x, edge_index, edge_attr))  # 184*bs X f[4]
        x = torch.matmul(P34.T, x)  # bs X 97 X f[4]
        x = x.view(self.batch_size, -1, self.nf[4])  # bs X 184 X f[4]
        # layer 5
        x, edge_index, edge_attr = x.view(-1, self.nf[4]), bg4.edge_index, bg4.edge_attr
        x = F.elu(self.conv5(x, edge_index, edge_attr))  # 184*bs X f[5]
        x = torch.matmul(P45.T, x)  # bs X 1 X f[5]
        x = x.view(self.batch_size, -1, self.nf[5])  # bs X 97 X f[5]

        # layer 6
        x, edge_index, edge_attr = x.view(-1, self.nf[5]), bg5.edge_index, bg5.edge_attr
        x = F.elu(self.conv6(x, edge_index, edge_attr))  # 184*bs X f[5]
        x = torch.matmul(P56.T, x)  # bs X 1 X f[5]
        x = x.view(self.batch_size, -1, self.nf[6])  # bs X 97 X f[5]

        # Reshape for max pooling (batch size, num_nodes, num_features)
        # Apply global max pooling over the nodes
        # This pools the node features into a single feature vector for each graph in the batch
        # x = F.adaptive_max_pool1d(x.transpose(1, 2), output_size=self.latent_dim)  # Shape: [batch_size, 128, 1]
        x = self.pooling_layer(x.transpose(1, 2))
        embedding = x.squeeze(-1)
        # Remove the extra dimension
        # x = x.view(self.batch_size, -1)  # Shape: [batch_size, 128]

        return embedding


class Base(nn.Module):
    def __init__(self, hparams, network_type):
        super(Base, self).__init__()
        # parameters
        self.network_type = network_type
        if self.network_type == "pinn":
            self.in_dim = hparams.network["in_dim"]
            self.out_dim = hparams.network["out_dim"]
            self.num_layers = hparams.network["num_layers"]
            self.hidden_dim = hparams.network["hidden_dim"]
            self.activation_fn = nn.ReLU()
            # self.activation_fn = nn.Tanh()
            self.activation_last_layer = nn.Sigmoid()
        else:
            self.in_dim = hparams.scar_network["in_dim"]
            self.out_dim = hparams.scar_network["out_dim"]
            self.num_layers = hparams.scar_network["num_layers"]
            self.hidden_dim = hparams.scar_network["hidden_dim"]
            # self.activation_fn = nn.ReLU()
            self.activation_fn = nn.LeakyReLU(negative_slope=0.01)
            # self.activation_fn = nn.Tanh()
            # self.activation_last_layer = nn.Sigmoid()
            self.activation_last_layer = CustomSigmoid(
                steepness=hparams.scar["steep"],
                sigma_s=hparams.scar["sigma_s"],
                sigma_h=hparams.scar["sigma_h"],
            )

        self.batch_norms = nn.ModuleList([])
        # if network_type == 'pinn':
        #     self.activation_fn = nn.ReLU()
        #     self.activation_last_layer = nn.Sigmoid()
        # elif network_type == 'scar':
        #     self.activation_fn = nn.ReLU()
        #     self.activation_last_layer = nn.Sigmoid()
        #     # self.activation_last_layer = CustomNormalizationActivation()
        self.net = nn.ModuleList([])
        self.net.append(torch.nn.Linear(self.in_dim, self.hidden_dim))
        for i in range(self.num_layers - 2):
            self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))

        self._initialize_weights()

    def binary_step(self, x, sigma_h, sigma_s):
        # Custom binary step function for the output layer
        # Here, x > 0 is a condition that results in a tensor of the same shape as x,
        # filled with True or False. Multiplying by self.sigma_h or self.sigma_s
        # applies the mapping based on the condition.
        return torch.where(
            x > 0.5, torch.full_like(x, sigma_h), torch.full_like(x, sigma_s)
        )

    def _initialize_weights(self):
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                if self.network_type == "pinn":
                    nn.init.kaiming_normal_(layer.weight)
                    # Xavier initialization for Tanh activation
                    # nn.init.xavier_normal_(layer.weight)
                elif self.network_type == "scar":
                    nn.init.kaiming_normal_(layer.weight)
                    # Kaiming initialization for Tanh activation
                    # nn.init.xavier_normal_(layer.weight, mode='fan_out', nonlinearity='Tanh')
                    # nn.init.xavier_normal_(layer.weight)
                # Bias initialized to zero
                nn.init.constant_(layer.bias, 0)
        # No activation function for the last layer (output layer)
        # nn.init.xavier_normal_(self.net[-1].weight)
        nn.init.kaiming_normal_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0)

    def set_params(self, net_weight, net_bias, bn_params):
        weights_pointer = 0
        bias_pointer = 0
        bn_pointer = 0  # Pointer for BatchNorm parameters

        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                # Assign weights to linear layers
                num_weights = layer.weight.data.numel()
                layer_weight_shape = layer.weight.shape
                del layer.weight
                self.net[i].weight = net_weight[
                    weights_pointer : weights_pointer + num_weights
                ].view(layer_weight_shape)
                weights_pointer += num_weights

                # Assign biases to linear layers
                num_biases = layer.bias.data.numel()
                layer_bias_shape = layer.bias.shape
                del layer.bias
                self.net[i].bias = net_bias[
                    bias_pointer : bias_pointer + num_biases
                ].view(layer_bias_shape)
                bias_pointer += num_biases

        for i, bn_layer in enumerate(self.batch_norms):
            if isinstance(bn_layer, nn.BatchNorm1d):
                # Assign weights (gamma) to BatchNorm layers
                num_bn_weights = bn_layer.weight.data.numel()
                layer_weight_shape = bn_layer.weight.shape
                del bn_layer.weight
                self.batch_norms[i].weight = bn_params[
                    bn_pointer : bn_pointer + num_bn_weights
                ].view(layer_weight_shape)
                bn_pointer += num_bn_weights

                # Assign biases (beta) to BatchNorm layers
                num_bn_biases = bn_layer.bias.data.numel()
                layer_bias_shape = bn_layer.bias.shape
                del bn_layer.bias
                self.batch_norms[i].bias = bn_params[
                    bn_pointer : bn_pointer + num_bn_biases
                ].view(layer_bias_shape)
                bn_pointer += num_bn_biases

        
    def forward(self, x):

        x = self.net[0](x)
        x = self.activation_fn(x)
        for i in range(self.num_layers - 2):
            x = self.net[i + 1](x)
            x = self.batch_norms[i](x)
            x = self.activation_fn(x)
        out = self.net[self.num_layers - 1](x)
        if self.network_type == "scar":
            out = self.activation_last_layer(out)

        return out


class Hypernet(nn.Module):
    def __init__(self, hparams, concat_initial=True, network_type="pinn"):
        super(Hypernet, self).__init__()
        self.network_type = network_type
        if self.network_type == "pinn":
            self.in_dim = hparams.network["in_dim"]
            self.out_dim = hparams.network["out_dim"]
            self.num_layers = hparams.network["num_layers"]
            self.hidden_dim = hparams.network["hidden_dim"]
        else:
            self.in_dim = hparams.scar_network["in_dim"]
            self.out_dim = hparams.scar_network["out_dim"]
            self.num_layers = hparams.scar_network["num_layers"]
            self.hidden_dim = hparams.scar_network["hidden_dim"]

        self.total_net_weight = (
            self.in_dim * self.hidden_dim
            + (self.num_layers - 2) * self.hidden_dim * self.hidden_dim
            + self.hidden_dim * self.out_dim
        )
        self.total_net_bias = (
            self.hidden_dim + (self.num_layers - 2) * self.hidden_dim + self.out_dim
        )
        self.total_bn_wb = (self.num_layers - 2) * self.hidden_dim * 2
        if concat_initial:
            embedding_dim = hparams.encoder["out_dim"] + hparams.encoder["init_out_dim"]
        else:
            embedding_dim = hparams.encoder["out_dim"]
       
        # if self.network_type == "pinn":
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            # nn.Tanh(),
            # # nn.Dropout(0.2),
            nn.Linear(256, 256),
            # # nn.Tanh(),
            nn.ReLU(),
            # nn.Tanh(),
            # # # # nn.Dropout(0.2),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            # nn.Tanh(),
            # # # nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            # # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(
                512, self.total_net_weight + self.total_net_bias + self.total_bn_wb
            ),
        )
        # else:
        #     self.net = nn.Sequential(
        #         nn.Linear(embedding_dim, 256),
        #         nn.ReLU(),
        #         # nn.Tanh(),
        #         # # nn.Dropout(0.2),
        #         nn.Linear(256, 256),
        #         # # nn.Tanh(),
        #         nn.ReLU(),
        #         # nn.Tanh(),
        #         # # # # nn.Dropout(0.2),
        #         nn.Linear(256, 256),
        #         # nn.Tanh(),
        #         nn.ReLU(),
        #         # # # nn.Dropout(0.2),
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         # nn.Tanh(),
        #         nn.Linear(256, 256),
        #         # nn.Tanh(),
        #         nn.ReLU(),
        #         # # # nn.Dropout(0.2),
        #         nn.Linear(256, 512),
        #         # nn.Tanh(),
        #         nn.ReLU(),
        #         nn.Linear(
        #             512, self.total_net_weight + self.total_net_bias + self.total_bn_wb
        #         ),
        #     )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                if self.network_type == "pinn":
                    nn.init.kaiming_normal_(layer.weight)
                elif self.network_type == "scar":
                    nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        nn.init.kaiming_normal_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.01)

    def forward(self, x):
        params = self.net(x).flatten()
        net_weight = params[0 : self.total_net_weight]
        net_bias = params[
            self.total_net_weight : self.total_net_weight + self.total_net_bias
        ]
        bn_params = params[self.total_net_weight + self.total_net_bias :]

        return net_weight, net_bias, bn_params


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()

        self.num_layers = hparams.decoder[
            "num_layers"
        ]  # Assuming this is the desired number of FC layers
        self.input_dim = hparams.encoder["out_dim"]  # Dimension of the input embedding
        self.num_nodes = hparams.dataset["num_nodes"]  # Number of nodes to reconstruct
        self.heart_feature_dim = 1  # Feature dimension of the heart (node features)

        # Calculate the dimension of the tensor after the last FC layer to match the 'heart' size
        self.fc_output_dim = self.num_nodes * self.heart_feature_dim

        # Initial fully connected layers to upsample the embedding
        self.fc_layers = nn.ModuleList()
        current_dim = self.input_dim
        for i in range(
            self.num_layers - 1
        ):  # Subtract 1 because the last layer is defined separately
            next_dim = current_dim * 4  # Example upscaling, adjust as needed
            self.fc_layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim

        # The last FC layer to match the 'heart' size
        self.fc_layers.append(nn.Linear(current_dim, self.fc_output_dim))

        # Define one or more graph convolutional layers with the output feature dimension being 1
        # Adjust the number and dimensions as needed
        self.gconv_layers = nn.ModuleList()
        self.gconv_layers.append(
            GCNConv(self.heart_feature_dim, 1)
        )  # Output 1 feature per node

        # self.activation_fn = nn.Tanh()
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, embedding, graph):
        x = embedding
        # Pass through the fully connected layers
        for fc in self.fc_layers:
            x = self.dropout(self.activation_fn(fc(x)))

        # Reshape the output to fit into graph convolutional layers
        # The tensor should be of shape [num_nodes, heart_feature_dim]
        x = x.view(self.num_nodes, self.heart_feature_dim)

        # Pass through graph convolutional layers
        for gconv in self.gconv_layers[:-1]:
            x = self.activation_fn(gconv(x, graph.edge_index))
        x = self.gconv_layers[-1](x, graph.edge_index)
        # The output `x` is the reconstructed node features matrix with 1 feature per node
        return x
