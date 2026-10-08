from __future__ import annotations

import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np
import torch

from siteweaver.cache.r0_cache_io import load_split_registry
from siteweaver.graph import Graph, batch as graph_batch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_key(sample: dict) -> str:
    value = str(sample.get("complex_id", "")).strip().upper()
    if not value:
        raise ValueError("Every cache sample must contain complex_id")
    return value


def split_samples(samples: list[dict], split_file: str | Path, split: str) -> list[dict]:
    registry = load_split_registry(split_file)
    sample_keys = {sample_key(sample) for sample in samples}
    if sample_keys != set(registry):
        missing = sorted(sample_keys - set(registry))[:5]
        extra = sorted(set(registry) - sample_keys)[:5]
        raise ValueError(f"Cache and split registry differ; missing={missing}, extra={extra}")
    selected = [sample for sample in samples if registry[sample_key(sample)] == str(split).lower()]
    if not selected:
        raise ValueError(f"Split {split!r} is empty")
    return selected


def sample_graph(sample: dict, pocket_nodes: bool = False) -> Graph:
    graph = Graph(
        torch.as_tensor(sample["edge_index"], dtype=torch.long),
        int(sample["node_features"].shape[0]),
    )
    graph.ndata["h"] = torch.as_tensor(sample["node_features"], dtype=torch.float32)
    graph.ndata["residue_index"] = torch.as_tensor(sample["residue_indices"], dtype=torch.long)
    if pocket_nodes:
        graph.ndata["pocket_node_mask"] = torch.as_tensor(
            sample["pocket_node_mask"], dtype=torch.bool
        )
    graph.edata["e"] = torch.as_tensor(sample["edge_features"], dtype=torch.float32)
    return graph


def collate_r0(items: list[dict] | None) -> dict | None:
    items = [item for item in (items or []) if item is not None]
    if not items:
        return None
    graph = graph_batch([sample_graph(item) for item in items])
    residue_offsets = []
    residue_ids = []
    node_ids = []
    offset = 0
    for sample_id, item in enumerate(items):
        n_res = int(item["num_residues"])
        residue_offsets.append(offset)
        offset += n_res
        residue_ids.append(torch.full((n_res,), sample_id, dtype=torch.long))
        node_ids.append(torch.full((item["node_features"].shape[0],), sample_id, dtype=torch.long))
    residue_offsets = torch.as_tensor(residue_offsets, dtype=torch.long)
    graph.ndata["global_residue_index"] = graph.ndata["residue_index"] + residue_offsets[torch.cat(node_ids)]
    return {
        "graph": graph,
        "active_site_mask": torch.cat([
            torch.as_tensor(item["active_site_mask"], dtype=torch.bool) for item in items
        ]),
        "active_site_probability": torch.cat([
            torch.as_tensor(item["active_site_probability"], dtype=torch.float32) for item in items
        ]),
        "allosteric_site_mask": torch.cat([
            torch.as_tensor(item["allosteric_site_mask"], dtype=torch.bool) for item in items
        ]),
        "residue_sample_ids": torch.cat(residue_ids),
        "num_residues_per_sample": torch.as_tensor(
            [item["num_residues"] for item in items], dtype=torch.long
        ),
        "complex_ids": [sample_key(item) for item in items],
    }


def collate_g1(items: list[dict] | None) -> dict | None:
    items = [item for item in (items or []) if item is not None]
    if not items:
        return None
    graph = graph_batch([sample_graph(item, pocket_nodes=True) for item in items])
    node_ids = []
    residue_offsets = []
    residue_ids = []
    pocket_ids = []
    residue_offset = 0
    for sample_id, item in enumerate(items):
        n_nodes = int(item["node_features"].shape[0])
        n_res = int(item["num_residues"])
        node_ids.append(torch.full((n_nodes,), sample_id, dtype=torch.long))
        residue_offsets.append(residue_offset)
        residue_offset += n_res
        residue_ids.append(torch.full((n_res,), sample_id, dtype=torch.long))
        n_pockets = int(torch.as_tensor(item["pocket_targets"]).numel())
        pocket_ids.append(torch.full((n_pockets,), sample_id, dtype=torch.long))
    node_ids = torch.cat(node_ids)
    offsets = torch.as_tensor(residue_offsets, dtype=torch.long)
    residue_index = graph.ndata["residue_index"].clone()
    atom_mask = ~graph.ndata["pocket_node_mask"]
    residue_index[atom_mask] = residue_index[atom_mask] + offsets[node_ids[atom_mask]]
    graph.ndata["global_residue_index"] = residue_index
    return {
        "graph": graph,
        "pocket_sample_ids": torch.cat(pocket_ids),
        "pocket_targets": torch.cat([
            torch.as_tensor(item["pocket_targets"], dtype=torch.int8) for item in items
        ]),
        "pocket_jaccards": torch.cat([
            torch.as_tensor(item["pocket_jaccards"], dtype=torch.float32) for item in items
        ]),
        "complex_ids": [sample_key(item) for item in items],
    }


def move_batch(batch: dict, device: torch.device, keys: tuple[str, ...]) -> dict:
    batch["graph"] = batch["graph"].to(device)
    for key in keys:
        batch[key] = batch[key].to(device)
    return batch


def load_active_probability_table(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    values: dict[str, list[tuple[int, float]]] = {}
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"complex_id", "residue_index", "active_site_probability"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Active-probability file must contain {sorted(required)}")
        for row in reader:
            key = str(row["complex_id"]).strip().upper()
            values.setdefault(key, []).append(
                (int(row["residue_index"]), float(row["active_site_probability"]))
            )
    result = {}
    for key, rows in values.items():
        rows.sort(key=lambda item: item[0])
        indices = [index for index, _ in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"Residue indices for {key} are not contiguous from zero")
        result[key] = np.asarray([value for _, value in rows], dtype=np.float32)
    return result


def apply_active_probabilities(samples: list[dict], path: str | Path) -> None:
    table = load_active_probability_table(path)
    for sample in samples:
        key = sample_key(sample)
        if key not in table:
            raise ValueError(f"Active-probability file is missing {key}")
        values = table[key]
        expected = int(sample["num_residues"])
        if values.size != expected:
            raise ValueError(f"{key}: probability rows={values.size}, cache residues={expected}")
        sample["active_site_probability"] = values


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def state_dict_from_checkpoint(checkpoint: dict) -> dict:
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    return {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, settings: dict, metrics: dict, best_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "settings": settings,
            "metrics": metrics,
            "best_val_metric": float(best_value),
        },
        path,
    )


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
