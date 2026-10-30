"""
modules.py - Model components for Facebook Page-Page node classification.

Two models are compared on the identical split:
    1. MLPBaseline: uses ONLY each page's own 128-dim features (ignores edges).
    2. GCN: a 2-layer Graph Convolutional Network (Kipf & Welling, 2017) that
       also aggregates information from each page's neighbours.

Pure PyTorch / PyTorch Geometric only - no NumPy (per the task spec).

References:
    T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph
    Convolutional Networks," ICLR 2017.
    PyG GCNConv docs: https://pytorch-geometric.readthedocs.io/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLPBaseline(nn.Module):
    """Features-only baseline: Linear -> ReLU -> Dropout -> Linear.

    It never looks at edge_index, so any improvement the GCN shows over
    this model can be attributed to using the graph structure.
    """

    def __init__(self, in_dim=128, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_embedding=False):
        # edge_index is accepted (and ignored) so both models share one interface
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.lin2(h)
        return (logits, h) if return_embedding else logits


class GCN(nn.Module):
    """Two-layer GCN.

    Each GCNConv layer computes, for every node i:
        h_i' = W * sum_{j in N(i) ∪ {i}}  h_j / sqrt(deg(i) * deg(j))
    i.e. a degree-normalised average of the node's own features and its
    neighbours' features, followed by a learned linear map W.

    Two layers = each node sees information from up to 2 hops away.
    More layers risk over-smoothing (all node embeddings becoming similar).
    """

    def __init__(self, in_dim=128, hidden_dim=64, num_classes=4, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.conv1(x, edge_index))      # hidden embedding (used for t-SNE)
        h_drop = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h_drop, edge_index)   # raw class scores (softmax applied in the loss)
        return (logits, h) if return_embedding else logits


def count_parameters(model):
    """Number of trainable parameters (for the resource-profiling table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)