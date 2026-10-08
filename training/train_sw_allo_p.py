#!/usr/bin/env python3
"""Train the SW-Allo-P/G1 FPocket pocket-node ranker.

The G1 cache must already contain FPocket pocket nodes, pocket quality
targets, and all node/edge features. Active-site-derived pocket-node features
are therefore part of cache preprocessing; changing active-site probabilities
requires rebuilding that cache.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from siteweaver.cache.g1_cache_io import load_samples
from siteweaver.models.g1_model import FPocketPocketNodeRanker

from _common import (
    collate_g1,
    load_checkpoint,
    move_batch,
    save_checkpoint,
    set_seed,
    split_samples,
    state_dict_from_checkpoint,
    write_summary,
)
from pocket_loss import listwise_hard_negative_loss


class CacheDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _metrics(chunks):
    if not chunks:
        return {}
    scores = np.concatenate([item["scores"] for item in chunks])
    targets = np.concatenate([item["targets"] for item in chunks])
    qualities = np.concatenate([item["qualities"] for item in chunks])
    sample_ids = np.concatenate([
        item["sample_ids"] + sum(len(part["complex_ids"]) for part in chunks[:index])
        for index, item in enumerate(chunks)
    ])
    complex_ids = [value for item in chunks for value in item["complex_ids"]]
    macro_auprc, macro_auroc, top1, top5, reciprocal_rank = [], [], [], [], []
    for sample_index, _ in enumerate(complex_ids):
        mask = sample_ids == sample_index
        local_scores = scores[mask]
        local_targets = targets[mask]
        local_qualities = qualities[mask]
        valid = local_targets >= 0
        if valid.any() and np.unique(local_targets[valid]).size > 1:
            macro_auprc.append(float(average_precision_score(local_targets[valid], local_scores[valid])))
            macro_auroc.append(float(roc_auc_score(local_targets[valid], local_scores[valid])))
        order = np.argsort(-local_scores, kind="stable")
        if local_qualities.size:
            top1.append(float(local_qualities[order[0]]))
            top5.append(float(local_qualities[order[:5]].max()))
        positive_positions = np.flatnonzero(local_targets[order] == 1)
        reciprocal_rank.append(float(1.0 / (positive_positions[0] + 1)) if positive_positions.size else 0.0)
    result = {
        "top1_jaccard": float(np.mean(top1)) if top1 else math.nan,
        "top5_jaccard": float(np.mean(top5)) if top5 else math.nan,
        "reciprocal_rank": float(np.mean(reciprocal_rank)) if reciprocal_rank else math.nan,
    }
    if macro_auprc:
        result["macro_auprc"] = float(np.mean(macro_auprc))
        result["macro_auroc"] = float(np.mean(macro_auroc))
    return result


def _run_epoch(model, loader, device, settings, train):
    model.train(train)
    chunks = []
    losses = []
    for batch in loader:
        if batch is None:
            continue
        batch = move_batch(batch, device, ("pocket_sample_ids", "pocket_targets", "pocket_jaccards"))
        with torch.set_grad_enabled(train):
            scores, _ = model(batch)
            loss, _ = listwise_hard_negative_loss(
                scores,
                batch["pocket_jaccards"],
                batch["pocket_sample_ids"],
                quality_temperature=settings["quality_temperature"],
                score_temperature=settings["score_temperature"],
                margin=settings["margin"],
                hard_negative_count=settings["hard_negative_count"],
                hard_negative_weight=settings["hard_negative_weight"],
            )
            if train:
                optimizer = settings["optimizer"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if settings["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        chunks.append({
            "scores": scores.detach().float().cpu().numpy(),
            "targets": batch["pocket_targets"].detach().cpu().numpy(),
            "qualities": batch["pocket_jaccards"].detach().float().cpu().numpy(),
            "sample_ids": batch["pocket_sample_ids"].detach().cpu().numpy(),
            "complex_ids": list(batch["complex_ids"]),
        })
    result = {"loss": float(np.mean(losses)) if losses else math.nan}
    result.update(_metrics(chunks))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    args = parser.parse_args(argv)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if int(manifest.get("node_feature_dim", 25)) != 25 or int(manifest.get("edge_feature_dim", 22)) != 22:
        raise ValueError("SW-Allo-P expects node_feature_dim=25 and edge_feature_dim=22")
    samples = load_samples(args.cache_dir)
    split_map = {split: split_samples(samples, args.split_file, split) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(
            CacheDataset(split_map[split]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if split == "train" else 0,
            collate_fn=collate_g1,
            pin_memory=device.type == "cuda" and split == "train",
        )
        for split in ("train", "val", "test")
    }
    settings = {
        "hidden_nf": 64,
        "embedding_dim": 64,
        "n_layers": 16,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "quality_temperature": 0.10,
        "score_temperature": 1.0,
        "margin": 0.2,
        "hard_negative_count": 16,
        "hard_negative_weight": 0.5,
        "gradient_clip": 0.0,
    }
    model = FPocketPocketNodeRanker(dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    start_epoch = 0
    best_value = -float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_value = float(checkpoint.get("best_val_metric", -float("inf")))
    settings["optimizer"] = optimizer

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must" if args.wandb_run_id else None,
            name=args.out_dir.name,
            config={k: v for k, v in settings.items() if k != "optimizer"}
            | {"batch_size": args.batch_size, "shuffle": False},
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.max_epochs):
        epoch_metrics = {"epoch": epoch}
        for split in ("train", "val"):
            values = _run_epoch(model, loaders[split], device, settings, split == "train")
            epoch_metrics.update({f"{split}/{key}": value for key, value in values.items()})
        val_value = epoch_metrics.get("val/top1_jaccard", -float("inf"))
        improved = val_value > best_value
        if improved:
            best_value = val_value
        checkpoint_settings = {k: v for k, v in settings.items() if k != "optimizer"}
        for name in ("newest_completed.ckpt", "last.ckpt"):
            save_checkpoint(
                args.out_dir / name, model, optimizer, epoch, args,
                checkpoint_settings, epoch_metrics, best_value,
            )
        if improved:
            save_checkpoint(
                args.out_dir / "best.ckpt", model, optimizer, epoch, args,
                checkpoint_settings, epoch_metrics, best_value,
            )
        if wandb_run:
            wandb_run.log(epoch_metrics, step=epoch)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    test_metrics = _run_epoch(model, loaders["test"], device, settings, False)
    summary = {
        "model": "SW-Allo-P",
        "variant": "G1 pocket-node listwise hard-negative",
        "uses_prs": False,
        "cache_dir": str(args.cache_dir),
        "split_file": str(args.split_file),
        "split_sizes": {key: len(value) for key, value in split_map.items()},
        "device": str(device),
        "test": test_metrics,
        "best_val_top1_jaccard": best_value,
    }
    write_summary(args.out_dir / "run_summary.json", summary)
    if wandb_run:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
