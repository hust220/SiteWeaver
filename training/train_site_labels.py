#!/usr/bin/env python3
"""Train the released AS_PROB or CR_PROB active/cryptic architecture."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from siteweaver.active_source import SiteWeaverActiveSiteModel
from siteweaver.cryptic_source import CrypticPocketModel

from site_label_cache import SiteLabelDataset, collate_site_labels


SPLITS = {
    "active": {"train": ("train",), "val": ("val",), "test": ("test",)},
    "cryptic": {
        "train": ("train-0", "train-1", "train-2"),
        "val": ("train-3",),
        "test": ("test",),
    },
}


class SiteLabelModule(pl.LightningModule):
    def __init__(self, cache_dir: str, task: str, batch_size: int = 2, num_workers: int = 2):
        super().__init__()
        self.save_hyperparameters()
        model_cls = SiteWeaverActiveSiteModel if task == "active" else CrypticPocketModel
        self.model = model_cls(hidden_nf=64, n_layers=8, dropout=0.0, node_input_dim=30)
        self._outputs = {"train": [], "val": [], "test": []}

    def setup(self, stage: str | None = None) -> None:
        target = "active_site_mask" if self.hparams.task == "active" else "allosteric_site_mask"
        splits = SPLITS[self.hparams.task]
        if stage in (None, "fit"):
            self.train_set = SiteLabelDataset(Path(self.hparams.cache_dir), splits["train"], target)
            self.val_set = SiteLabelDataset(Path(self.hparams.cache_dir), splits["val"], target)
        if stage in (None, "test"):
            self.test_set = SiteLabelDataset(Path(self.hparams.cache_dir), splits["test"], target)

    def _loader(self, data, workers: int = 0):
        return DataLoader(
            data, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=workers, collate_fn=collate_site_labels,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._loader(self.train_set, self.hparams.num_workers)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch, split: str):
        batch["graph"] = batch["graph"].to(self.device)
        target = batch["target"].to(self.device)
        logits = self(batch)
        loss, parts = self.model.compute_loss(logits, target, 100.0, 0.2)
        self.log(f"loss/{split}", loss, batch_size=self.hparams.batch_size)
        self.log(f"bce/{split}", parts["bce"], batch_size=self.hparams.batch_size)
        self.log(f"dice/{split}", parts["dice"], batch_size=self.hparams.batch_size)
        self._outputs[split].append((target.detach().float().cpu(), torch.sigmoid(logits).detach().float().cpu()))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _epoch_metrics(self, split: str):
        pieces = self._outputs[split]
        self._outputs[split] = []
        if not pieces:
            return
        labels = torch.cat([item[0] for item in pieces]).numpy().astype(np.int64)
        scores = torch.cat([item[1] for item in pieces]).numpy()
        prefix = "active" if self.hparams.task == "active" else "cryptic"
        if np.unique(labels).size > 1:
            self.log(f"{prefix}/{split}_auprc", float(average_precision_score(labels, scores)))
            self.log(f"{prefix}/{split}_auroc", float(roc_auc_score(labels, scores)))

    def on_train_epoch_end(self):
        self._epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._epoch_metrics("val")

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)


class SaveFinalCheckpoint(Callback):
    def __init__(self, path: Path):
        self.path = path

    def on_train_end(self, trainer, module):
        trainer.save_checkpoint(str(self.path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("active", "cryptic"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_gpu = torch.cuda.is_available() and args.device != "cpu"
    module = SiteLabelModule(str(args.cache_dir), args.task, args.batch_size, args.num_workers)
    if args.dry_run:
        module.setup("test")
        batch = collate_site_labels([module.test_set[0]])
        logits = module.model(batch)
        loss, _ = module.model.compute_loss(logits, batch["target"])
        loss.backward()
        if not all(torch.isfinite(param.grad).all() for param in module.model.parameters() if param.grad is not None):
            raise RuntimeError("Dry-run produced non-finite gradients")
        print(f"task={args.task} complex={batch['complex_ids'][0]} residues={len(logits)} loss={loss.item():.6f}")
        return
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(args.out_dir), name="logs")
    if args.wandb_project:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, save_dir=str(args.out_dir))
    monitor = f"{'active' if args.task == 'active' else 'cryptic'}/val_auprc"
    checkpoint = ModelCheckpoint(
        dirpath=str(args.out_dir), filename="{epoch}", monitor=monitor,
        mode="max", save_top_k=2, save_last=True,
    )
    if use_gpu:
        torch.set_float32_matmul_precision("high")
    precision = 32 if args.task == "active" else (
        "bf16-mixed" if use_gpu and torch.cuda.get_device_capability()[0] >= 8
        else ("16-mixed" if use_gpu else 32)
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, logger=logger,
        callbacks=[checkpoint, SaveFinalCheckpoint(args.out_dir / "final.ckpt")],
        accelerator="gpu" if use_gpu else "cpu", devices=1, precision=precision,
        log_every_n_steps=10,
    )
    trainer.fit(module, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
