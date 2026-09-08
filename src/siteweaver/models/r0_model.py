from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..egnn import GNN
from .r0_variants import get_variant

from . import r0_const as const


class HeavyAtomR0KnownActive(nn.Module):
    """16-layer residue ranker operating on heavy-atom graphs."""

    def __init__(self, hidden_nf: int = 64, embedding_dim: int = 64, n_layers: int = 16):
        super().__init__()
        self.backbone = GNN(
            c_h=int(hidden_nf),
            n_layers=int(n_layers),
            in_node_dim=const.MODEL_NODE_FEATURE_DIM,
            in_edge_dim=const.EDGE_FEATURE_DIM,
            out_node_dim=int(hidden_nf),
            out_edge_dim=1,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(hidden_nf)),
            nn.Linear(int(hidden_nf), int(hidden_nf)),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(int(hidden_nf), int(embedding_dim)),
        )
        score_dim = int(embedding_dim) * 3 + 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, int(hidden_nf)),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(int(hidden_nf), 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
        self.spec = get_variant("R0")

    def residue_embeddings(self, graph):
        node_h, _ = self.backbone(graph.ndata["h"], graph.edge_index, graph.edata["e"])
        node_z = self.projection(node_h)
        residue_index = graph.ndata["global_residue_index"].long()
        n_residues = int(residue_index.max().item()) + 1 if residue_index.numel() else 0
        residue_z = node_z.new_zeros((n_residues, node_z.size(-1)))
        counts = node_z.new_zeros((n_residues, 1))
        residue_z.index_add_(0, residue_index, node_z)
        counts.index_add_(
            0,
            residue_index,
            torch.ones((node_z.size(0), 1), dtype=node_z.dtype, device=node_z.device),
        )
        return F.normalize(residue_z / counts.clamp_min(1.0), dim=-1)

    def residue_scores(self, batch):
        residue_z = self.residue_embeddings(batch["graph"])
        sample_ids = batch["residue_sample_ids"].to(residue_z.device)
        probability = batch["active_site_probability"].to(residue_z.device).clamp(0.0, 1.0)
        context = residue_z.new_zeros((int(batch["num_residues_per_sample"].numel()), residue_z.size(-1)))
        for sample_idx in range(context.size(0)):
            mask = sample_ids == sample_idx
            p = probability[mask]
            if p.sum() > 1e-6:
                context[sample_idx] = (residue_z[mask] * p[:, None]).sum(0) / p.sum()
        ctx = context[sample_ids]
        score_input = torch.cat([residue_z, ctx, residue_z * ctx, probability[:, None]], dim=-1)
        score = self.score_head(score_input).squeeze(-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return score * scale, scale

    def forward(self, batch):
        return self.residue_scores(batch)


__all__ = ["HeavyAtomR0KnownActive"]
