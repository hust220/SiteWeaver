from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class RankLossSpec(Protocol):
    negative_mode: str
    same_weight: float
    cross_weight: float


def _sample_from(mask: torch.Tensor, n: int) -> torch.Tensor:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0 or n <= 0:
        return indices[:0]
    return indices[torch.randint(0, indices.numel(), (int(n),), device=indices.device)]


def _score_pairs(scores, positive_indices, negative_indices, repeats):
    positive = scores[positive_indices]
    negative = scores[negative_indices]
    if positive.numel() == 0 or negative.numel() == 0:
        return positive[:0], negative[:0]
    repeated = positive.repeat_interleave(max(1, int(repeats)))
    if repeated.numel() < negative.numel():
        repeated = positive.repeat((negative.numel() + positive.numel() - 1) // positive.numel())
    return repeated[: negative.numel()], negative


def allosteric_rank_loss(
    residue_scores,
    score_scale,
    active_mask,
    allosteric_mask,
    residue_sample_ids,
    spec: RankLossSpec,
    margin=0.2,
    negatives_per_positive=32,
    cross_protein_negatives_per_positive=16,
):
    device = residue_scores.device
    active_mask = active_mask.to(device)
    allosteric_mask = allosteric_mask.to(device)
    residue_sample_ids = residue_sample_ids.to(device)
    same_losses, cross_losses = [], []
    positive_scores, same_negative_scores, cross_negative_scores = [], [], []
    margin_hits = []
    n_positive = n_same = n_cross = 0
    for sample_id in torch.unique(residue_sample_ids).tolist():
        in_sample = residue_sample_ids == int(sample_id)
        pos_idx = torch.nonzero(in_sample & allosteric_mask, as_tuple=False).flatten()
        if not pos_idx.numel():
            continue
        pos_scores = residue_scores[pos_idx]
        positive_scores.append(pos_scores.detach())
        n_positive += int(pos_scores.numel())
        if spec.negative_mode in {"both", "same"} and spec.same_weight > 0:
            same_mask = in_sample & ~allosteric_mask & ~active_mask
            same_idx = _sample_from(same_mask, int(pos_scores.numel()) * int(negatives_per_positive))
            if same_idx.numel():
                pos, neg = _score_pairs(residue_scores, pos_idx, same_idx, negatives_per_positive)
                same_losses.append(F.softplus(float(margin) - pos + neg).mean())
                same_negative_scores.append(neg.detach())
                margin_hits.append((pos.detach() - neg.detach()) > float(margin))
                n_same += int(neg.numel())
        if spec.negative_mode in {"both", "cross"} and spec.cross_weight > 0:
            cross_mask = ~in_sample & ~allosteric_mask & ~active_mask
            cross_idx = _sample_from(cross_mask, int(pos_scores.numel()) * int(cross_protein_negatives_per_positive))
            if cross_idx.numel():
                pos, neg = _score_pairs(residue_scores, pos_idx, cross_idx, cross_protein_negatives_per_positive)
                cross_losses.append(F.softplus(float(margin) - pos + neg).mean())
                cross_negative_scores.append(neg.detach())
                margin_hits.append((pos.detach() - neg.detach()) > float(margin))
                n_cross += int(neg.numel())
    zero = residue_scores.sum() * 0.0
    same_loss = torch.stack(same_losses).mean() if same_losses else zero
    cross_loss = torch.stack(cross_losses).mean() if cross_losses else zero
    loss = float(spec.same_weight) * same_loss + float(spec.cross_weight) * cross_loss
    pos = torch.cat(positive_scores) if positive_scores else residue_scores[:0]
    same = torch.cat(same_negative_scores) if same_negative_scores else residue_scores[:0]
    cross = torch.cat(cross_negative_scores) if cross_negative_scores else residue_scores[:0]
    neg = torch.cat([part for part in (same, cross) if part.numel()]) if (same.numel() or cross.numel()) else residue_scores[:0]
    hits = torch.cat(margin_hits).float() if margin_hits else residue_scores[:0]
    return loss, {
        "loss": loss.detach(),
        "same_loss": same_loss.detach(),
        "cross_loss": cross_loss.detach(),
        "num_positive_residues": torch.tensor(float(n_positive), device=device),
        "num_same_negatives": torch.tensor(float(n_same), device=device),
        "num_cross_negatives": torch.tensor(float(n_cross), device=device),
        "score_scale": score_scale.detach(),
        "pos_score_mean": pos.mean() if pos.numel() else zero,
        "same_neg_score_mean": same.mean() if same.numel() else zero,
        "cross_neg_score_mean": cross.mean() if cross.numel() else zero,
        "neg_score_mean": neg.mean() if neg.numel() else zero,
        "margin_satisfied_frac": hits.mean() if hits.numel() else zero,
    }
