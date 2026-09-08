from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import const
from .egnn import GNN

from .variants import VariantSpec


def _sample_from(mask: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0 or n <= 0:
        return idx[:0]
    return idx[torch.randint(0, idx.numel(), (int(n),), device=idx.device)]


def _score_pairs(scores, pos_idx, neg_idx, repeats):
    pos = scores[pos_idx]
    neg = scores[neg_idx]
    if pos.numel() == 0 or neg.numel() == 0:
        return pos[:0], neg[:0]
    repeated_pos = pos.repeat_interleave(max(1, int(repeats)))
    if repeated_pos.numel() < neg.numel():
        repeated_pos = pos.repeat((neg.numel() + pos.numel() - 1) // pos.numel())
    return repeated_pos[: neg.numel()], neg


class SiteWeaverAblationModel(nn.Module):
    def __init__(self, spec: VariantSpec, hidden_nf=64, embedding_dim=64, dropout=0.0):
        super().__init__()
        self.spec = spec
        self.backbone = GNN(
            c_h=int(hidden_nf),
            n_layers=int(spec.n_layers),
            in_node_dim=int(spec.node_input_dim),
            in_edge_dim=const.EDGE_FEATURE_DIM,
            out_node_dim=int(hidden_nf),
            out_edge_dim=1,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(int(hidden_nf)),
            nn.Linear(int(hidden_nf), int(hidden_nf)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_nf), int(embedding_dim)),
        )
        score_dim = int(embedding_dim)
        if spec.use_active_context:
            score_dim *= 3
        if spec.use_active_node_probability:
            score_dim += 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, int(hidden_nf)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_nf), 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))

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
        parts = [residue_z]
        if self.spec.use_active_context:
            probability = batch["active_site_probability"].to(residue_z.device).clamp(0.0, 1.0)
            context = residue_z.new_zeros(
                (int(batch["num_residues_per_sample"].numel()), residue_z.size(-1))
            )
            for sample_idx in range(context.size(0)):
                mask = sample_ids == sample_idx
                p = probability[mask]
                if p.sum() > 1e-6:
                    context[sample_idx] = (residue_z[mask] * p[:, None]).sum(0) / p.sum()
            ctx = context[sample_ids]
            parts.extend([ctx, residue_z * ctx])
        if self.spec.use_active_node_probability:
            parts.append(
                batch["active_site_probability"].to(residue_z.device).clamp(0.0, 1.0)[:, None]
            )
        score = self.score_head(torch.cat(parts, dim=-1)).squeeze(-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return score * scale, scale

    def forward(self, batch):
        return self.residue_scores(batch)


def allosteric_rank_loss(
    residue_scores,
    score_scale,
    active_mask,
    allosteric_mask,
    residue_sample_ids,
    spec: VariantSpec,
    margin=0.2,
    negatives_per_positive=32,
    cross_protein_negatives_per_positive=16,
):
    device = residue_scores.device
    active_mask = active_mask.to(device)
    allosteric_mask = allosteric_mask.to(device)
    residue_sample_ids = residue_sample_ids.to(device)
    same_losses, cross_losses = [], []
    pos_scores_all, same_neg_all, cross_neg_all = [], [], []
    margins = []
    n_pos = n_same = n_cross = 0
    for sample_id in torch.unique(residue_sample_ids).tolist():
        in_sample = residue_sample_ids == int(sample_id)
        pos_idx = torch.nonzero(in_sample & allosteric_mask, as_tuple=False).flatten()
        if not pos_idx.numel():
            continue
        pos_scores = residue_scores[pos_idx]
        pos_scores_all.append(pos_scores.detach())
        n_pos += int(pos_scores.numel())
        if spec.negative_mode in {"both", "same"} and spec.same_weight > 0:
            same_mask = in_sample & ~allosteric_mask & ~active_mask
            same_idx = _sample_from(
                same_mask, int(pos_scores.numel()) * int(negatives_per_positive)
            )
            if same_idx.numel():
                pos, neg = _score_pairs(residue_scores, pos_idx, same_idx, negatives_per_positive)
                same_losses.append(F.softplus(float(margin) - pos + neg).mean())
                same_neg_all.append(neg.detach())
                margins.append((pos.detach() - neg.detach()) > float(margin))
                n_same += int(neg.numel())
        if spec.negative_mode in {"both", "cross"} and spec.cross_weight > 0:
            cross_mask = ~in_sample & ~allosteric_mask & ~active_mask
            cross_idx = _sample_from(
                cross_mask, int(pos_scores.numel()) * int(cross_protein_negatives_per_positive)
            )
            if cross_idx.numel():
                pos, neg = _score_pairs(
                    residue_scores, pos_idx, cross_idx, cross_protein_negatives_per_positive
                )
                cross_losses.append(F.softplus(float(margin) - pos + neg).mean())
                cross_neg_all.append(neg.detach())
                margins.append((pos.detach() - neg.detach()) > float(margin))
                n_cross += int(neg.numel())
    # Keep an empty-negative batch differentiable. This can occur for the
    # cross-only ablation when a batch has no eligible cross-protein residue.
    zero = residue_scores.sum() * 0.0
    same_loss = torch.stack(same_losses).mean() if same_losses else zero
    cross_loss = torch.stack(cross_losses).mean() if cross_losses else zero
    loss = float(spec.same_weight) * same_loss + float(spec.cross_weight) * cross_loss
    pos = torch.cat(pos_scores_all) if pos_scores_all else residue_scores[:0]
    same = torch.cat(same_neg_all) if same_neg_all else residue_scores[:0]
    cross = torch.cat(cross_neg_all) if cross_neg_all else residue_scores[:0]
    neg = torch.cat([part for part in (same, cross) if part.numel()]) if (same.numel() or cross.numel()) else residue_scores[:0]
    margin_hits = torch.cat(margins).float() if margins else residue_scores[:0]
    return loss, {
        "loss": loss.detach(),
        "same_loss": same_loss.detach(),
        "cross_loss": cross_loss.detach(),
        "num_positive_residues": torch.tensor(float(n_pos), device=device),
        "num_same_negatives": torch.tensor(float(n_same), device=device),
        "num_cross_negatives": torch.tensor(float(n_cross), device=device),
        "score_scale": score_scale.detach(),
        "pos_score_mean": pos.mean() if pos.numel() else zero,
        "same_neg_score_mean": same.mean() if same.numel() else zero,
        "cross_neg_score_mean": cross.mean() if cross.numel() else zero,
        "neg_score_mean": neg.mean() if neg.numel() else zero,
        "margin_satisfied_frac": margin_hits.mean() if margin_hits.numel() else zero,
    }
