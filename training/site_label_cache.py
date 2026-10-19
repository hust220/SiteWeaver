"""Load the frozen active-site and cryptic-site packed training caches."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from siteweaver.cache.r0_cache_io import load_samples
from siteweaver.graph import Graph, batch as batch_graphs


class SiteLabelDataset(Dataset):
    def __init__(self, cache_dir: Path, splits: tuple[str, ...], target_key: str):
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("feature_mode") != "probability_only" or manifest.get("node_input_dim") != 30:
            raise ValueError(f"Expected a probability-only, 30-feature cache: {manifest_path}")
        self.target_key = target_key
        self.samples = []
        for split in splits:
            item = manifest["splits"][split]
            self.samples.extend(load_samples(cache_dir / item["path"], item["format"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        node_features = torch.as_tensor(sample["node_features"], dtype=torch.float32)
        edge_features = torch.as_tensor(sample["edge_features"], dtype=torch.float32)
        if node_features.shape[1] != 30 or edge_features.shape[1] != 5:
            raise ValueError(f"Unexpected graph feature shape for {sample.get('complex_id')}")
        graph = Graph(torch.as_tensor(sample["edge_index"], dtype=torch.long), len(node_features))
        graph.ndata["h"] = node_features
        graph.ndata["residue_index"] = torch.as_tensor(sample["residue_indices"], dtype=torch.long)
        graph.edata["e"] = edge_features
        target = torch.as_tensor(sample[self.target_key], dtype=torch.float32)
        if len(target) != int(sample["num_residues"]):
            raise ValueError(f"Target length disagrees with residues for {sample.get('complex_id')}")
        return {
            "graph": graph,
            "target": target,
            "num_residues": int(sample["num_residues"]),
            "complex_id": sample["complex_id"],
        }


def collate_site_labels(samples: list[dict]) -> dict:
    graph = batch_graphs([item["graph"] for item in samples])
    offsets = []
    total = 0
    for item in samples:
        offsets.append(total)
        total += item["num_residues"]
    graph.ndata["global_residue_index"] = graph.ndata["residue_index"] + torch.repeat_interleave(
        torch.as_tensor(offsets, dtype=torch.long),
        torch.as_tensor(graph.batch_num_nodes, dtype=torch.long),
    )
    return {
        "graph": graph,
        "target": torch.cat([item["target"] for item in samples]),
        "num_residues_per_sample": torch.as_tensor([item["num_residues"] for item in samples]),
        "complex_ids": [item["complex_id"] for item in samples],
    }
