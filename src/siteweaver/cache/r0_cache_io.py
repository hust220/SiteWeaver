from __future__ import annotations

import io
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd


TORCH_ZSTD = "torch_zstd"
TORCH_PT = "torch_pt"
TORCH_PACKED_ZSTD = "torch_packed_zstd"
TORCH_PACKED_PT = "torch_packed_pt"
NPZ = "npz"
NPZ_COMPRESSED = "npz_compressed"

FORMATS = {TORCH_ZSTD, TORCH_PT, TORCH_PACKED_ZSTD, TORCH_PACKED_PT, NPZ, NPZ_COMPRESSED}
VALID_SPLITS = {"train", "val", "test"}


def suffix_for_format(cache_format: str) -> str:
    if cache_format == TORCH_ZSTD:
        return ".ptzst"
    if cache_format == TORCH_PT:
        return ".pt"
    if cache_format == TORCH_PACKED_ZSTD:
        return ".packed.ptzst"
    if cache_format == TORCH_PACKED_PT:
        return ".packed.pt"
    if cache_format in {NPZ, NPZ_COMPRESSED}:
        return ".npz"
    raise ValueError(f"Unknown cache format: {cache_format}")


def _as_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def save_samples(path: Path, samples: list[dict], cache_format: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == TORCH_ZSTD:
        buffer = io.BytesIO()
        torch.save(samples, buffer)
        path.write_bytes(zstd.ZstdCompressor(level=3).compress(buffer.getvalue()))
        return
    if cache_format == TORCH_PT:
        torch.save(samples, path)
        return
    if cache_format in {TORCH_PACKED_ZSTD, TORCH_PACKED_PT}:
        payload = pack_samples_torch(samples)
        if cache_format == TORCH_PACKED_ZSTD:
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            path.write_bytes(zstd.ZstdCompressor(level=3).compress(buffer.getvalue()))
        else:
            torch.save(payload, path)
        return
    if cache_format in {NPZ, NPZ_COMPRESSED}:
        save_packed_npz(path, samples, compressed=(cache_format == NPZ_COMPRESSED))
        return
    raise ValueError(f"Unknown cache format: {cache_format}")


def load_samples(path: Path, cache_format: str | None = None) -> list[dict]:
    path = Path(path)
    if cache_format is None:
        name = path.name
        if name.endswith(".packed.ptzst"):
            cache_format = TORCH_PACKED_ZSTD
        elif name.endswith(".packed.pt"):
            cache_format = TORCH_PACKED_PT
        elif path.suffix == ".ptzst":
            cache_format = TORCH_ZSTD
        elif path.suffix == ".pt":
            cache_format = TORCH_PT
        elif path.suffix == ".npz":
            cache_format = NPZ
        else:
            raise ValueError(f"Cannot infer cache format from {path}")
    if cache_format == TORCH_ZSTD:
        raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
        try:
            return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(io.BytesIO(raw), map_location="cpu")
    if cache_format == TORCH_PT:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    if cache_format in {TORCH_PACKED_ZSTD, TORCH_PACKED_PT}:
        if cache_format == TORCH_PACKED_ZSTD:
            raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
            source = io.BytesIO(raw)
        else:
            source = path
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        return unpack_samples_torch(payload)
    if cache_format in {NPZ, NPZ_COMPRESSED}:
        return load_packed_npz(path)
    raise ValueError(f"Unknown cache format: {cache_format}")


def _sample_key(sample: dict) -> str:
    complex_id = str(sample.get("complex_id", "")).strip()
    if not complex_id:
        raise ValueError("Cached sample is missing a non-empty complex_id")
    return complex_id.upper()


@lru_cache(maxsize=8)
def _load_all_cached_samples_cached(cache_dir: str) -> list[dict]:
    """Load a legacy split cache or a merged v2 cache without changing samples."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if "data_file" in manifest:
        samples = load_samples(cache_dir / manifest["data_file"], manifest.get("cache_format"))
    else:
        split_meta = manifest.get("splits")
        if not isinstance(split_meta, dict):
            raise ValueError(f"Manifest has neither data_file nor splits: {manifest_path}")
        samples = []
        for split in ("train", "val", "test"):
            meta = split_meta.get(split)
            if meta is None:
                continue
            samples.extend(load_samples(cache_dir / meta["path"], meta.get("format")))

    seen = set()
    duplicates = []
    for sample in samples:
        key = _sample_key(sample)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Cache contains duplicate complex_id values: {duplicates[:5]}")
    return samples


def load_all_cached_samples(cache_dir: str | Path) -> list[dict]:
    """Load all samples, reusing the decoded cache within one process."""
    return _load_all_cached_samples_cached(str(Path(cache_dir).resolve()))


def load_split_registry(path: str | Path) -> dict[str, str]:
    """Read a two-column ``complex_id,split`` registry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split registry not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["complex_id", "split"]:
            raise ValueError(
                f"Split registry must have exactly complex_id,split columns: {path}"
            )
        registry = {}
        for row_number, row in enumerate(reader, start=2):
            complex_id = str(row["complex_id"]).strip()
            split = str(row["split"]).strip().lower()
            if not complex_id or split not in VALID_SPLITS:
                raise ValueError(f"Invalid registry row {row_number} in {path}: {row}")
            key = complex_id.upper()
            if key in registry:
                raise ValueError(f"Duplicate complex_id {complex_id!r} in {path}")
            registry[key] = split
    return registry


def load_split_samples(
    cache_dir: str | Path,
    split: str,
    split_file: str | Path | None = None,
) -> list[dict]:
    """Load one split, optionally selected by an external split registry.

    With no ``split_file`` this preserves the legacy manifest behavior. When a
    registry is supplied, the complete cache is loaded once and filtered by the
    registry, so the same raw cache can support multiple independent splits.
    """
    split = str(split).strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected train, val, or test")
    cache_dir = Path(cache_dir)

    if split_file is None:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        split_meta = manifest.get("splits", {}).get(split)
        if split_meta is None:
            raise ValueError(
                f"Cache {cache_dir} has no embedded split {split!r}; pass split_file"
            )
        return load_samples(cache_dir / split_meta["path"], split_meta.get("format"))

    registry = load_split_registry(split_file)
    samples = load_all_cached_samples(cache_dir)
    sample_keys = {_sample_key(sample) for sample in samples}
    registry_keys = set(registry)
    missing = sorted(sample_keys - registry_keys)
    extra = sorted(registry_keys - sample_keys)
    if missing or extra:
        raise ValueError(
            f"Split registry/cache mismatch for {split_file}: "
            f"missing_registry_rows={missing[:5]} extra_registry_rows={extra[:5]}"
        )
    return [sample for sample in samples if registry[_sample_key(sample)] == split]


def pack_samples_torch(samples: list[dict]) -> dict:
    node_offsets = [0]
    edge_offsets = [0]
    residue_offsets = [0]
    node_features = []
    edge_index = []
    edge_features = []
    residue_indices = []
    active_masks = []
    allosteric_masks = []
    active_probabilities = []
    has_active_probability = bool(samples) and all("active_site_probability" in sample for sample in samples)
    meta = []

    for sample in samples:
        nf = torch.as_tensor(sample["node_features"], dtype=torch.float32)
        ei = torch.as_tensor(sample["edge_index"], dtype=torch.long)
        ef = torch.as_tensor(sample["edge_features"], dtype=torch.float32)
        ri = torch.as_tensor(sample["residue_indices"], dtype=torch.long)
        active = torch.as_tensor(sample["active_site_mask"], dtype=torch.bool)
        allosteric = torch.as_tensor(sample["allosteric_site_mask"], dtype=torch.bool)
        node_features.append(nf)
        edge_index.append(ei)
        edge_features.append(ef)
        residue_indices.append(ri)
        active_masks.append(active)
        allosteric_masks.append(allosteric)
        if has_active_probability:
            active_probabilities.append(torch.as_tensor(sample["active_site_probability"], dtype=torch.float32))
        node_offsets.append(node_offsets[-1] + int(nf.shape[0]))
        edge_offsets.append(edge_offsets[-1] + int(ei.shape[1]))
        residue_offsets.append(residue_offsets[-1] + int(active.shape[0]))
        meta.append({
            "pdb_id": sample.get("pdb_id", ""),
            "complex_id": sample.get("complex_id", ""),
            "pdb_path": sample.get("pdb_path", ""),
            "split": sample.get("split", ""),
            "num_residues": int(sample["num_residues"]),
            "residue_meta": sample.get("residue_meta", []),
            "allosteric_groups": sample.get("allosteric_groups", []),
        })

    n_samples = len(samples)
    payload = {
        "node_features": torch.cat(node_features, dim=0) if n_samples else torch.zeros((0, 0), dtype=torch.float32),
        "edge_index": torch.cat(edge_index, dim=1) if n_samples else torch.zeros((2, 0), dtype=torch.long),
        "edge_features": torch.cat(edge_features, dim=0) if n_samples else torch.zeros((0, 0), dtype=torch.float32),
        "residue_indices": torch.cat(residue_indices, dim=0) if n_samples else torch.zeros((0,), dtype=torch.long),
        "active_site_mask": torch.cat(active_masks, dim=0) if n_samples else torch.zeros((0,), dtype=torch.bool),
        "allosteric_site_mask": torch.cat(allosteric_masks, dim=0) if n_samples else torch.zeros((0,), dtype=torch.bool),
        "node_offsets": torch.as_tensor(node_offsets, dtype=torch.long),
        "edge_offsets": torch.as_tensor(edge_offsets, dtype=torch.long),
        "residue_offsets": torch.as_tensor(residue_offsets, dtype=torch.long),
        "meta": meta,
    }
    if has_active_probability:
        payload["active_site_probability"] = torch.cat(active_probabilities, dim=0)
    return payload


def unpack_samples_torch(payload: dict) -> list[dict]:
    meta = payload["meta"]
    node_offsets = payload["node_offsets"]
    edge_offsets = payload["edge_offsets"]
    residue_offsets = payload["residue_offsets"]
    samples = []
    for idx, sample_meta in enumerate(meta):
        n0, n1 = int(node_offsets[idx]), int(node_offsets[idx + 1])
        e0, e1 = int(edge_offsets[idx]), int(edge_offsets[idx + 1])
        r0, r1 = int(residue_offsets[idx]), int(residue_offsets[idx + 1])
        sample = {
            "node_features": payload["node_features"][n0:n1],
            "edge_index": payload["edge_index"][:, e0:e1],
            "edge_features": payload["edge_features"][e0:e1],
            "residue_indices": payload["residue_indices"][n0:n1],
            "active_site_mask": payload["active_site_mask"][r0:r1],
            "allosteric_site_mask": payload["allosteric_site_mask"][r0:r1],
            **sample_meta,
        }
        if "active_site_probability" in payload:
            sample["active_site_probability"] = payload["active_site_probability"][r0:r1]
        samples.append(sample)
    return samples


def save_packed_npz(path: Path, samples: list[dict], compressed: bool) -> None:
    n_samples = len(samples)
    node_offsets = [0]
    edge_offsets = [0]
    residue_offsets = [0]
    node_features = []
    edge_index = []
    edge_features = []
    residue_indices = []
    active_masks = []
    allosteric_masks = []
    active_probabilities = []
    has_active_probability = bool(samples) and all("active_site_probability" in sample for sample in samples)
    meta = []

    for sample in samples:
        nf = _as_numpy(sample["node_features"], np.float32)
        ei = _as_numpy(sample["edge_index"], np.int64)
        ef = _as_numpy(sample["edge_features"], np.float32)
        ri = _as_numpy(sample["residue_indices"], np.int64)
        active = _as_numpy(sample["active_site_mask"], np.bool_)
        allosteric = _as_numpy(sample["allosteric_site_mask"], np.bool_)
        node_features.append(nf)
        edge_index.append(ei)
        edge_features.append(ef)
        residue_indices.append(ri)
        active_masks.append(active)
        allosteric_masks.append(allosteric)
        if has_active_probability:
            active_probabilities.append(_as_numpy(sample["active_site_probability"], np.float32))
        node_offsets.append(node_offsets[-1] + int(nf.shape[0]))
        edge_offsets.append(edge_offsets[-1] + int(ei.shape[1]))
        residue_offsets.append(residue_offsets[-1] + int(active.shape[0]))
        meta.append({
            "pdb_id": sample.get("pdb_id", ""),
            "complex_id": sample.get("complex_id", ""),
            "pdb_path": sample.get("pdb_path", ""),
            "split": sample.get("split", ""),
            "num_residues": int(sample["num_residues"]),
            "residue_meta": sample.get("residue_meta", []),
            "allosteric_groups": sample.get("allosteric_groups", []),
        })

    payload = {
        "node_features": np.concatenate(node_features, axis=0) if n_samples else np.zeros((0, 0), dtype=np.float32),
        "edge_index": np.concatenate(edge_index, axis=1) if n_samples else np.zeros((2, 0), dtype=np.int64),
        "edge_features": np.concatenate(edge_features, axis=0) if n_samples else np.zeros((0, 0), dtype=np.float32),
        "residue_indices": np.concatenate(residue_indices, axis=0) if n_samples else np.zeros((0,), dtype=np.int64),
        "active_site_mask": np.concatenate(active_masks, axis=0) if n_samples else np.zeros((0,), dtype=np.bool_),
        "allosteric_site_mask": np.concatenate(allosteric_masks, axis=0) if n_samples else np.zeros((0,), dtype=np.bool_),
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "residue_offsets": np.asarray(residue_offsets, dtype=np.int64),
        "meta_json": np.asarray(json.dumps(meta)),
    }
    if has_active_probability:
        payload["active_site_probability"] = np.concatenate(active_probabilities, axis=0)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def load_packed_npz(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    node_offsets = data["node_offsets"]
    edge_offsets = data["edge_offsets"]
    residue_offsets = data["residue_offsets"]
    samples = []
    for idx, sample_meta in enumerate(meta):
        n0, n1 = int(node_offsets[idx]), int(node_offsets[idx + 1])
        e0, e1 = int(edge_offsets[idx]), int(edge_offsets[idx + 1])
        r0, r1 = int(residue_offsets[idx]), int(residue_offsets[idx + 1])
        sample = {
            "node_features": data["node_features"][n0:n1],
            "edge_index": data["edge_index"][:, e0:e1],
            "edge_features": data["edge_features"][e0:e1],
            "residue_indices": data["residue_indices"][n0:n1],
            "active_site_mask": data["active_site_mask"][r0:r1],
            "allosteric_site_mask": data["allosteric_site_mask"][r0:r1],
            **sample_meta,
        }
        if "active_site_probability" in data:
            sample["active_site_probability"] = data["active_site_probability"][r0:r1]
        samples.append(sample)
    return samples
