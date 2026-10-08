#!/usr/bin/env python3
"""Train the no-PRS SW-Allo-R/R0 residue ranker.

The cache must contain 14-dimensional R0 heavy-atom node features, 21-
dimensional edge features, active/allosteric masks, and active-site
probabilities. PRS channels are intentionally absent so collaborators can
add and test their own PRS implementation without receiving a PRS-trained
checkpoint.
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

from siteweaver.cache.r0_cache_io import load_all_cached_samples
from siteweaver.models.r0_model import HeavyAtomR0KnownActive
from siteweaver.ranker_model import allosteric_rank_loss
from siteweaver.models.r0_variants import R0

from _common import (
    apply_active_probabilities,
    collate_r0,
    load_checkpoint,
    move_batch,
    save_checkpoint,
    set_seed,
    split_samples,
    state_dict_from_checkpoint,
    write_summary,
)


class CacheDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _metrics(records):
    auprc, auroc, top10, enrichment = [], [], [], []
    for labels, scores in records:
        if np.unique(labels).size > 1:
            auprc.append(float(average_precision_score(labels, scores)))
            auroc.append(float(roc_auc_score(labels, scores)))
        positives = int(labels.sum())
        if positives:
            k = max(1, int(round(labels.size * 0.10)))
            chosen = np.argsort(-scores, kind="stable")[:k]
            top10.append(float(labels[chosen].sum() / positives))
            prevalence = float(labels.mean())
            enrichment.append(float(labels[chosen].mean() / prevalence) if prevalence else 0.0)
    result = {}
    if auprc:
        result["macro_auprc"] = float(np.mean(auprc))
        result["macro_auroc"] = float(np.mean(auroc))
    if top10:
        result["macro_top10pct_recall"] = float(np.mean(top10))
        result["macro_top10pct_enrichment"] = float(np.mean(enrichment))
    return result


def _run_epoch(model, loader, device, settings, train):
    model.train(train)
    records = []
    losses = []
    for batch in loader:
        if batch is None:
            continue
        batch = move_batch(
            batch,
            device,
            (
                "active_site_mask",
                "active_site_probability",
                "allosteric_site_mask",
                "residue_sample_ids",
                "num_residues_per_sample",
            ),
        )
        with torch.set_grad_enabled(train):
            scores, scale = model(batch)
            loss, _ = allosteric_rank_loss(
                scores,
                scale,
                batch["active_site_mask"],
                batch["allosteric_site_mask"],
                batch["residue_sample_ids"],
                R0,
                margin=settings["margin"],
                negatives_per_positive=settings["negatives_per_positive"],
                cross_protein_negatives_per_positive=settings["cross_negatives_per_positive"],
            )
            if train:
                optimizer = settings["optimizer"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if settings["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels = batch["allosteric_site_mask"].detach().cpu().numpy().astype(np.int64)
        sample_ids = batch["residue_sample_ids"].detach().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy()
        for sample_id in np.unique(sample_ids):
            mask = sample_ids == sample_id
            records.append((labels[mask], scores_np[mask]))
    result = {"loss": float(np.mean(losses)) if losses else math.nan}
    result.update(_metrics(records))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--active-probabilities", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
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
    if int(manifest.get("node_feature_dim", 14)) != 14 or int(manifest.get("edge_feature_dim", 21)) != 21:
        raise ValueError("SW-Allo-R expects node_feature_dim=14 and edge_feature_dim=21")
    samples = load_all_cached_samples(args.cache_dir)
    if args.active_probabilities:
        apply_active_probabilities(samples, args.active_probabilities)
    split_map = {split: split_samples(samples, args.split_file, split) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(
            CacheDataset(split_map[split]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if split == "train" else 0,
            collate_fn=collate_r0,
            pin_memory=device.type == "cuda" and split == "train",
        )
        for split in ("train", "val", "test")
    }
    settings = {
        "hidden_nf": 64,
        "embedding_dim": 64,
        "n_layers": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "margin": 0.2,
        "negatives_per_positive": 32,
        "cross_negatives_per_positive": 16,
        "gradient_clip": 0.0,
    }
    model = HeavyAtomR0KnownActive(
        hidden_nf=settings["hidden_nf"],
        embedding_dim=settings["embedding_dim"],
        n_layers=settings["n_layers"],
    ).to(device)
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
        val_value = epoch_metrics.get("val/macro_auprc", -float("inf"))
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
        "model": "SW-Allo-R",
        "variant": "R0",
        "uses_prs": False,
        "cache_dir": str(args.cache_dir),
        "split_file": str(args.split_file),
        "split_sizes": {key: len(value) for key, value in split_map.items()},
        "active_probability_file": str(args.active_probabilities) if args.active_probabilities else "cache field",
        "device": str(device),
        "test": test_metrics,
        "best_val_macro_auprc": best_value,
    }
    write_summary(args.out_dir / "run_summary.json", summary)
    if wandb_run:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
