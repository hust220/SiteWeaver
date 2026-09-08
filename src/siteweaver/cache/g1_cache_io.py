from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import torch
import zstandard as zstd


def _load(path: Path):
    raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(io.BytesIO(raw), map_location="cpu")


def _save(path: Path, payload: dict) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(zstd.ZstdCompressor(level=3).compress(buffer.getvalue()))


def pack_samples(samples: list[dict]) -> dict:
    node_offsets = [0]
    edge_offsets = [0]
    residue_offsets = [0]
    pocket_offsets = [0]
    node_features = []
    edge_index = []
    edge_features = []
    residue_indices = []
    node_pocket_masks = []
    active_masks = []
    active_probabilities = []
    allosteric_masks = []
    pocket_jaccards = []
    pocket_targets = []
    meta = []
    for sample in samples:
        nf = torch.as_tensor(sample["node_features"], dtype=torch.float32)
        ei = torch.as_tensor(sample["edge_index"], dtype=torch.long)
        ef = torch.as_tensor(sample["edge_features"], dtype=torch.float32)
        ri = torch.as_tensor(sample["residue_indices"], dtype=torch.long)
        pm = torch.as_tensor(sample["pocket_node_mask"], dtype=torch.bool)
        active = torch.as_tensor(sample["active_site_mask"], dtype=torch.bool)
        active_probability = torch.as_tensor(sample["active_site_probability"], dtype=torch.float32)
        allosteric = torch.as_tensor(sample["allosteric_site_mask"], dtype=torch.bool)
        qualities = torch.as_tensor([record["max_jaccard"] for record in sample["pocket_records"]], dtype=torch.float32)
        targets = torch.as_tensor(
            [record["target"] for record in sample["pocket_records"]],
            dtype=torch.int8,
        )
        node_features.append(nf)
        edge_index.append(ei)
        edge_features.append(ef)
        residue_indices.append(ri)
        node_pocket_masks.append(pm)
        active_masks.append(active)
        active_probabilities.append(active_probability)
        allosteric_masks.append(allosteric)
        pocket_jaccards.append(qualities)
        pocket_targets.append(targets)
        node_offsets.append(node_offsets[-1] + int(nf.shape[0]))
        edge_offsets.append(edge_offsets[-1] + int(ei.shape[1]))
        residue_offsets.append(residue_offsets[-1] + int(active.shape[0]))
        pocket_offsets.append(pocket_offsets[-1] + int(targets.numel()))
        meta.append({
            "complex_id": str(sample.get("complex_id", "")),
            "pdb_id": str(sample.get("pdb_id", "")),
            "pdb_path": str(sample.get("pdb_path", "")),
            "num_residues": int(sample["num_residues"]),
            "residue_meta": sample.get("residue_meta", []),
            "allosteric_groups": sample.get("allosteric_groups", []),
            "pocket_records": sample["pocket_records"],
        })
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_index, dim=1) if edge_index else torch.zeros((2, 0), dtype=torch.long),
        "edge_features": torch.cat(edge_features, dim=0),
        "residue_indices": torch.cat(residue_indices, dim=0),
        "pocket_node_mask": torch.cat(node_pocket_masks, dim=0),
        "active_site_mask": torch.cat(active_masks, dim=0),
        "active_site_probability": torch.cat(active_probabilities, dim=0),
        "allosteric_site_mask": torch.cat(allosteric_masks, dim=0),
        "pocket_jaccards": torch.cat(pocket_jaccards, dim=0) if pocket_jaccards else torch.zeros((0,), dtype=torch.float32),
        "pocket_targets": torch.cat(pocket_targets, dim=0) if pocket_targets else torch.zeros((0,), dtype=torch.int8),
        "node_offsets": torch.as_tensor(node_offsets, dtype=torch.long),
        "edge_offsets": torch.as_tensor(edge_offsets, dtype=torch.long),
        "residue_offsets": torch.as_tensor(residue_offsets, dtype=torch.long),
        "pocket_offsets": torch.as_tensor(pocket_offsets, dtype=torch.long),
        "meta": meta,
    }


def save_cache(cache_dir: Path, samples: list[dict], manifest: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "data.packed.ptzst"
    _save(data_path, pack_samples(samples))
    manifest = dict(manifest)
    manifest["data_file"] = data_path.name
    manifest["data_size_bytes"] = data_path.stat().st_size
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _load_samples(cache_dir: str) -> list[dict]:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    payload = _load(cache_dir / manifest["data_file"])
    samples = []
    for index, sample_meta in enumerate(payload["meta"]):
        n0, n1 = int(payload["node_offsets"][index]), int(payload["node_offsets"][index + 1])
        e0, e1 = int(payload["edge_offsets"][index]), int(payload["edge_offsets"][index + 1])
        r0, r1 = int(payload["residue_offsets"][index]), int(payload["residue_offsets"][index + 1])
        p0, p1 = int(payload["pocket_offsets"][index]), int(payload["pocket_offsets"][index + 1])
        samples.append({
            "node_features": payload["node_features"][n0:n1],
            "edge_index": payload["edge_index"][:, e0:e1],
            "edge_features": payload["edge_features"][e0:e1],
            "residue_indices": payload["residue_indices"][n0:n1],
            "pocket_node_mask": payload["pocket_node_mask"][n0:n1],
            "active_site_mask": payload["active_site_mask"][r0:r1],
            "active_site_probability": payload["active_site_probability"][r0:r1],
            "allosteric_site_mask": payload["allosteric_site_mask"][r0:r1],
            "pocket_jaccards": payload["pocket_jaccards"][p0:p1],
            "pocket_targets": payload["pocket_targets"][p0:p1],
            **sample_meta,
        })
    return samples


@lru_cache(maxsize=2)
def _load_samples_cached(cache_dir: str) -> list[dict]:
    return _load_samples(cache_dir)


def load_samples(cache_dir: str | Path) -> list[dict]:
    """Load one packed cache once per process and reuse its decoded samples."""
    return _load_samples_cached(str(Path(cache_dir).resolve()))
