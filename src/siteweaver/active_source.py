from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import const
from .egnn import GNN


def aggregate_nodes_to_residues(node_values: torch.Tensor, residue_index: torch.Tensor, n_residues: int) -> torch.Tensor:
    out = node_values.new_zeros((int(n_residues), node_values.shape[-1]))
    counts = node_values.new_zeros((int(n_residues), 1))
    out.index_add_(0, residue_index.long(), node_values)
    counts.index_add_(0, residue_index.long(), torch.ones((node_values.shape[0], 1), device=node_values.device, dtype=node_values.dtype))
    return out / counts.clamp_min(1.0)


class SiteWeaverActiveSiteModel(nn.Module):
    def __init__(self, hidden_nf: int = 64, n_layers: int = 8, dropout: float = 0.0, node_input_dim: int = 94):
        super().__init__()
        self.backbone = GNN(
            c_h=hidden_nf,
            n_layers=n_layers,
            in_node_dim=int(node_input_dim),
            in_edge_dim=const.EDGE_FEATURE_DIM,
            out_node_dim=hidden_nf,
            out_edge_dim=1,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_nf, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        graph = batch["graph"]
        node_h, _ = self.backbone(graph.ndata["h"], graph.edge_index, graph.edata["e"])
        node_logits = self.head(node_h)
        residue_index = graph.ndata["global_residue_index"].long()
        n_residues = int(batch["num_residues_per_sample"].sum().item())
        residue_logits = aggregate_nodes_to_residues(node_logits, residue_index, n_residues).squeeze(-1)
        return residue_logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, pos_weight_max: float = 100.0, dice_weight: float = 0.2):
        positives = targets.sum()
        negatives = targets.numel() - positives
        pos_weight = (negatives / (positives + 1e-6)).clamp(min=1.0, max=float(pos_weight_max))
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + targets.sum() + 1.0)
        loss = bce + float(dice_weight) * dice
        return loss, {"bce": bce.detach(), "dice": dice.detach(), "pos_weight": pos_weight.detach()}
