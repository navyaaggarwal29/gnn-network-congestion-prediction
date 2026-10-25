"""Temporal graph and non-graph sequence models for link utilization."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


def batch_edge_index(edge_index: torch.Tensor, graph_count: int, node_count: int) -> torch.Tensor:
    """Replicate a directed edge index for disjoint graphs in a batch."""
    offsets = torch.arange(graph_count, device=edge_index.device, dtype=edge_index.dtype) * node_count
    return (edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)


class TemporalGNN(nn.Module):
    """Two directed GCN layers per time step followed by a shared per-node GRU.

    Input is ``[batch, time, nodes, features]`` and output is one bounded
    utilization estimate per node ``[batch, nodes]``. The graph edge index is
    passed through unchanged: messages flow only along its source-to-target
    edges and no reverse edges are added.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, gru_hidden_dim: int | None = None,
                 dropout: float = 0.2):
        super().__init__()
        gru_hidden_dim = gru_hidden_dim or hidden_dim
        self.gcn1 = GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.head = nn.Linear(gru_hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, time, nodes, features]")
        batch, steps, nodes, features = x.shape
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, graph_edges]")
        graph_batch = batch_edge_index(edge_index, batch * steps, nodes)
        spatial = x.reshape(batch * steps * nodes, features)
        spatial = self.dropout(torch.relu(self.gcn1(spatial, graph_batch)))
        spatial = self.dropout(torch.relu(self.gcn2(spatial, graph_batch)))
        spatial = spatial.reshape(batch, steps, nodes, -1).permute(0, 2, 1, 3)
        temporal = spatial.reshape(batch * nodes, steps, -1)
        _, hidden = self.gru(temporal)
        prediction = torch.sigmoid(self.head(hidden[-1])).reshape(batch, nodes)
        return prediction


class LinkGRU(nn.Module):
    """Per-link GRU baseline sharing temporal weights, without graph messages."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, time, nodes, features]")
        batch, steps, nodes, features = x.shape
        temporal = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features)
        _, hidden = self.gru(temporal)
        prediction = torch.sigmoid(self.head(self.dropout(hidden[-1]))).reshape(batch, nodes)
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def regression_loss(prediction: torch.Tensor, target: torch.Tensor,
                    positive_weight: float | None = None, threshold: float = 0.85) -> torch.Tensor:
    """Mean square error, optionally upweighting positives using a train-only weight."""
    squared_error = (prediction - target).square()
    if positive_weight is None:
        return squared_error.mean()
    weights = 1.0 + (positive_weight - 1.0) * (target >= threshold).to(target.dtype)
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)
