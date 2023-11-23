"""
Author: Shuning Han

This module defines a Functional connectom（FC）based（Graph Neural Network (GNN) model for fMRI analysis using PyTorch Geometric.

"""

import numpy as np

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, TopKPooling, BatchNorm
from torch_geometric.nn import GraphConv

class FCbasedGCN(torch.nn.Module):
    def __init__(self, node_num, class_num, hidden_channels):
        """
        Initializes the FCbasedGCN model.

         Parameters:
            - node_num (int): Number of nodes in the graph, also interpreted as feature channels.
            - class_num (int): Number of classes for classification.
            - hidden_channels (int): Number of hidden channels in the model.

        """
        super(FCbasedGCN, self).__init__()
        torch.manual_seed(12345)

        # Graph Convolutional Layers
        self.conv1 = GraphConv(node_num, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(4 * hidden_channels, hidden_channels)

        # Batch Normalization Layer
        self.batch_norm5 = BatchNorm(hidden_channels)

        # Fully Connected Layer
        self.lin = Linear(hidden_channels, class_num)

    def forward(self, x, edge_index, batch):
        """
        Defines the forward pass of the FCbasedGCN model.

        Parameters:
            - x (Tensor): Node features.
            - edge_index (LongTensor): Graph edge indices.
            - batch (LongTensor): Batch vector.

        Returns:
            - x (Tensor): Output tensor after the forward pass.

        """
        # Graph Convolutional Layers with ReLU Activation
        x1 = self.conv1(x, edge_index)
        x1 = x1.relu()

        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()

        x3 = self.conv3(x2, edge_index)
        x3 = x3.relu()

        x4 = self.conv4(x3, edge_index)
        x4 = x4.relu()

        # Concatenate the outputs of all layers
        x = torch.cat((x1, x2, x3, x4), 1)

        # Additional Graph Convolutional Layer
        x5 = self.conv5(x, edge_index)

        # Batch Normalization
        x5 = self.batch_norm5(x5)

        # Global Mean Pooling
        x = global_mean_pool(x5, batch)

        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x