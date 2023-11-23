# FC-based-GCN


## Overview
This repository contains the implementation of the FCbasedGCN model using PyTorch Geometric. The model is specifically designed for fMRI analysis and is applied in the paper titled "**Early Prediction of Dementia using fMRI Data with a Graph Convolutional Network Approach**" For detailed information, please refer to the paper.

## Paper Reference
If you use or refer to this FCbasedGCN model in your work, please cite the following paper:
[Paper Title]
"Early Prediction of Dementia using fMRI Data with a Graph Convolutional Network Approach"

## Requirements
PyTorch Geometric,
NumPy

## Example Usage
```python
# Importing the FCbasedGCN model
from fc_based_gcn import FCbasedGCN

# Creating an instance of the FCbasedGCN model
model = FCbasedGCN(node_num=..., class_num=..., hidden_channels=...)

...

# Forward pass
output = model(x, edge_index, batch)
