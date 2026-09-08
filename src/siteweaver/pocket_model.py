from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import const
from .egnn import GNN
from .fused_gnn import FusedGNN


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice


class LigandFreeYuelPocket(nn.Module):
    """Protein-only residue pocket predictor using YuelPocket's GNN backbone."""

    def __init__(
        self,
        hidden_nf: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        gnn_backend: str = "reference",
    ):
        super().__init__()
        self.gnn_backend = str(gnn_backend)
        gnn_kwargs = {
            "c_h": hidden_nf,
            "n_layers": n_layers,
            "in_node_dim": const.NODE_FEATURE_DIM,
            "in_edge_dim": const.EDGE_FEATURE_DIM,
            "out_node_dim": hidden_nf,
            "out_edge_dim": 1,
        }
        if self.gnn_backend == "reference":
            self.gnn = GNN(**gnn_kwargs)
        elif self.gnn_backend == "fused":
            self.gnn = FusedGNN(**gnn_kwargs)
        else:
            raise ValueError(f"Unknown gnn_backend={self.gnn_backend!r}")
        self.pocket_head = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_nf, 1),
        )

    def copy_backbone_from_reference(self, reference_model: "LigandFreeYuelPocket") -> None:
        """Copy reference weights into a fused model for checkpoint parity tests."""
        if not isinstance(self.gnn, FusedGNN) or not isinstance(reference_model.gnn, GNN):
            raise TypeError("copy_backbone_from_reference requires fused and reference models")
        self.gnn.copy_from_reference(reference_model.gnn)
        self.pocket_head.load_state_dict(reference_model.pocket_head.state_dict())

    def load_reference_state_dict(self, state_dict, strict: bool = True):
        """Load a reference model state dict into the fused backend."""
        if not isinstance(self.gnn, FusedGNN):
            raise TypeError("load_reference_state_dict requires gnn_backend='fused'")
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
        hidden_nf = int(self.gnn.layers[0].layer_norm_h.normalized_shape[0])
        reference = LigandFreeYuelPocket(
            hidden_nf=hidden_nf,
            n_layers=len(self.gnn.layers),
            dropout=float(self.pocket_head[3].p),
            gnn_backend="reference",
        )
        result = reference.load_state_dict(state_dict, strict=strict)
        self.copy_backbone_from_reference(reference)
        return result

    def forward(self, g):
        h = g.ndata["h"]
        e = g.edata["e"]
        h_out, _ = self.gnn(h, g.edge_index, e)
        return self.pocket_head(h_out).squeeze(-1)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight_max: float = 100.0,
        dice_weight: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positives = targets.sum()
        negatives = targets.numel() - positives
        pos_weight = (negatives / (positives + 1e-6)).clamp(min=1.0, max=float(pos_weight_max))
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice = dice_loss(logits, targets)
        loss = bce + float(dice_weight) * dice
        return loss, {"bce": bce.detach(), "dice": dice.detach(), "pos_weight": pos_weight.detach()}

    @torch.no_grad()
    def sample_chain(self, g):
        return torch.sigmoid(self.forward(g))
