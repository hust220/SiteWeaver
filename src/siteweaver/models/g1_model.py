from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..egnn import GNN

from . import g1_const as const


class FPocketPocketNodeRanker(nn.Module):
    """Final 16-layer pocket-node ranker used by the SiteWeaver release."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.backbone = GNN(
            c_h=int(const.HIDDEN_NF),
            n_layers=int(const.N_LAYERS),
            in_node_dim=const.NODE_FEATURE_DIM,
            in_edge_dim=const.EDGE_FEATURE_DIM,
            out_node_dim=int(const.HIDDEN_NF),
            out_edge_dim=1,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(const.HIDDEN_NF)),
            nn.Linear(int(const.HIDDEN_NF), int(const.HIDDEN_NF)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(const.HIDDEN_NF), int(const.EMBEDDING_DIM)),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(int(const.EMBEDDING_DIM)),
            nn.Linear(int(const.EMBEDDING_DIM), int(const.HIDDEN_NF)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(const.HIDDEN_NF), 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        graph = batch["graph"]
        node_h, _ = self.backbone(graph.ndata["h"], graph.edge_index, graph.edata["e"])
        node_z = self.projection(node_h)
        pocket_mask = graph.ndata["pocket_node_mask"]
        pocket_z = node_z[pocket_mask]
        score = self.score_head(pocket_z).squeeze(-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return score * scale, scale
