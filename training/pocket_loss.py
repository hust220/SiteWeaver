from __future__ import annotations

import torch
import torch.nn.functional as F


def _protein_loss(scores, qualities, quality_temperature, score_temperature, margin, hard_negative_count):
    quality_target = F.softmax(qualities / float(quality_temperature), dim=0)
    prediction = F.log_softmax(scores / float(score_temperature), dim=0)
    listwise = -(quality_target * prediction).sum()
    best_quality = qualities.max()
    best_mask = qualities >= best_quality - 1e-7
    negative_mask = ~best_mask
    if best_quality <= 0.0 or not negative_mask.any():
        zero = scores.sum() * 0.0
        return listwise, zero, zero, int(best_mask.sum().item())
    negative_indices = torch.nonzero(negative_mask, as_tuple=False).flatten()
    count = min(int(hard_negative_count), int(negative_indices.numel()))
    top_negative = negative_indices[torch.topk(scores[negative_indices], count).indices]
    gaps = scores[best_mask, None] - scores[top_negative][None, :]
    hard = F.softplus(float(margin) - gaps).mean()
    satisfied = (gaps >= float(margin)).float().mean()
    return listwise, hard, satisfied, int(best_mask.sum().item())


def listwise_hard_negative_loss(
    scores,
    qualities,
    sample_ids,
    quality_temperature=0.10,
    score_temperature=1.0,
    margin=0.2,
    hard_negative_count=16,
    hard_negative_weight=0.5,
):
    qualities = qualities.to(scores.device, dtype=torch.float32)
    sample_ids = sample_ids.to(scores.device)
    listwise_losses = []
    hard_losses = []
    satisfied = []
    best_counts = 0
    protein_count = 0
    for sample_id in torch.unique(sample_ids).tolist():
        mask = sample_ids == int(sample_id)
        listwise, hard, fraction, best_count = _protein_loss(
            scores[mask], qualities[mask], quality_temperature,
            score_temperature, margin, hard_negative_count,
        )
        listwise_losses.append(listwise)
        if hard.requires_grad:
            hard_losses.append(hard)
            satisfied.append(fraction)
        best_counts += best_count
        protein_count += 1
    zero = scores.sum() * 0.0
    listwise_loss = torch.stack(listwise_losses).mean() if listwise_losses else zero
    hard_loss = torch.stack(hard_losses).mean() if hard_losses else zero
    satisfied_fraction = torch.stack(satisfied).mean() if satisfied else zero
    return listwise_loss + float(hard_negative_weight) * hard_loss, {
        "listwise_loss": listwise_loss.detach(),
        "hard_negative_loss": hard_loss.detach(),
        "margin_satisfied_frac": satisfied_fraction.detach(),
        "best_candidate_count": scores.new_tensor(float(best_counts)),
        "protein_count": scores.new_tensor(float(protein_count)),
    }
