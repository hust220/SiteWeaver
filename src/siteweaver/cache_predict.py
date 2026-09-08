from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .cache.g1_cache_io import load_samples as load_g1_samples
from .cache.r0_cache_io import load_all_cached_samples as load_r0_samples
from .graph import Graph
from .models.g1_model import FPocketPocketNodeRanker
from .models.r0_model import HeavyAtomR0KnownActive


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    state = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    return checkpoint, state


def _graph_from_sample(sample: dict, device: torch.device) -> Graph:
    edge_index = torch.as_tensor(sample["edge_index"], dtype=torch.long, device=device)
    graph = Graph(edge_index, int(sample["node_features"].shape[0]))
    graph.ndata["h"] = torch.as_tensor(sample["node_features"], dtype=torch.float32, device=device)
    graph.ndata["residue_index"] = torch.as_tensor(sample["residue_indices"], dtype=torch.long, device=device)
    graph.ndata["global_residue_index"] = graph.ndata["residue_index"].clone()
    if "pocket_node_mask" in sample:
        graph.ndata["pocket_node_mask"] = torch.as_tensor(
            sample["pocket_node_mask"], dtype=torch.bool, device=device
        )
    graph.edata["e"] = torch.as_tensor(sample["edge_features"], dtype=torch.float32, device=device)
    return graph


def _percentile(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return torch.full_like(scores, 100.0)
    order = torch.argsort(scores, stable=True)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=torch.float32)
    return 100.0 * ranks / float(scores.numel() - 1)


def _sample_id(sample: dict) -> str:
    value = str(sample.get("complex_id", "")).strip()
    if not value:
        raise ValueError("Each cached sample must contain complex_id")
    return value.upper()


def _find_sample(samples: list[dict], complex_id: str) -> dict:
    wanted = str(complex_id).strip().upper()
    for sample in samples:
        if _sample_id(sample) == wanted:
            return sample
    raise KeyError(f"complex_id={complex_id!r} is not present in the supplied cache")


def _default_weights_dir() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "weights",
        Path(sys.prefix) / "siteweaver" / "weights",
        Path(__file__).resolve().parents[4] / "siteweaver" / "weights",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _write_residue_scores(sample: dict, scores: torch.Tensor, out_dir: Path) -> dict:
    score_array = scores.detach().cpu().numpy().astype("float32")
    percentiles = _percentile(scores).detach().cpu().numpy().astype("float32")
    metadata = list(sample.get("residue_meta", []))
    if len(metadata) != score_array.size:
        raise ValueError(
            f"Cache metadata has {len(metadata)} residues but model returned {score_array.size} scores"
        )
    path = out_dir / f"{_sample_id(sample)}_residue_rank_scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["residue_index", "chain_id", "residue_number", "insertion_code", "residue_name", "score", "rank_percentile"],
        )
        writer.writeheader()
        for index, (meta, score, percentile) in enumerate(zip(metadata, score_array, percentiles)):
            writer.writerow({
                "residue_index": index,
                "chain_id": meta.get("chain_id", ""),
                "residue_number": meta.get("res_num", ""),
                "insertion_code": meta.get("ins_code", ""),
                "residue_name": meta.get("res_name", ""),
                "score": f"{float(score):.8g}",
                "rank_percentile": f"{float(percentile):.6g}",
            })
    return {"csv": str(path), "score_min": float(score_array.min()), "score_max": float(score_array.max())}


def _write_pocket_scores(sample: dict, scores: torch.Tensor, out_dir: Path) -> dict:
    score_array = scores.detach().cpu().numpy().astype("float32")
    percentiles = _percentile(scores).detach().cpu().numpy().astype("float32")
    records = list(sample.get("pocket_records", []))
    if len(records) != score_array.size:
        raise ValueError(
            f"Cache has {len(records)} pocket records but model returned {score_array.size} scores"
        )
    path = out_dir / f"{_sample_id(sample)}_pocket_rank_scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pocket_number", "score", "rank_percentile", "nearby_residue_indices"],
        )
        writer.writeheader()
        for record, score, percentile in zip(records, score_array, percentiles):
            writer.writerow({
                "pocket_number": record.get("pocket_number", ""),
                "score": f"{float(score):.8g}",
                "rank_percentile": f"{float(percentile):.6g}",
                "nearby_residue_indices": ";".join(str(v) for v in record.get("nearby_residue_indices", [])),
            })
    top_index = int(score_array.argmax()) if score_array.size else -1
    selected = {
        "complex_id": _sample_id(sample),
        "selected_pocket_number": records[top_index].get("pocket_number") if top_index >= 0 else None,
        "selected_pocket_residue_indices": records[top_index].get("nearby_residue_indices", []) if top_index >= 0 else [],
        "selection_rule": "highest SiteWeaver pocket-node score",
    }
    selected_path = out_dir / f"{_sample_id(sample)}_selected_pocket.json"
    selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    return {
        "csv": str(path),
        "selected_pocket": str(selected_path),
        "score_min": float(score_array.min()) if score_array.size else None,
        "score_max": float(score_array.max()) if score_array.size else None,
    }


@torch.inference_mode()
def predict_cached(
    cache_dir: str | Path,
    complex_id: str,
    out_dir: str | Path,
    mode: str = "pocket",
    checkpoint: str | Path | None = None,
    weights_dir: str | Path | None = None,
    device: str = "auto",
) -> dict:
    mode = str(mode).lower()
    if mode not in {"pocket", "residue"}:
        raise ValueError("mode must be pocket or residue")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    target = torch.device(resolved_device)
    weights = Path(weights_dir) if weights_dir else _default_weights_dir()
    if checkpoint is None:
        checkpoint = weights / ("allosteric_pocket_rank_final.ckpt" if mode == "pocket" else "allosteric_residue_rank_final.ckpt")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    samples = load_g1_samples(cache_dir) if mode == "pocket" else load_r0_samples(cache_dir)
    sample = _find_sample(samples, complex_id)
    graph = _graph_from_sample(sample, target)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_data, state = _load_checkpoint(checkpoint_path, target)
    if mode == "pocket":
        model = FPocketPocketNodeRanker().to(target)
    else:
        model = HeavyAtomR0KnownActive().to(target)
        n_residues = int(sample["num_residues"])
        batch = {
            "graph": graph,
            "active_site_probability": torch.as_tensor(sample["active_site_probability"], dtype=torch.float32, device=target),
            "residue_sample_ids": torch.zeros(n_residues, dtype=torch.long, device=target),
            "num_residues_per_sample": torch.tensor([n_residues], dtype=torch.long, device=target),
        }
    model.load_state_dict(state, strict=True)
    model.eval()
    if mode == "pocket":
        scores, scale = model({"graph": graph})
        result = _write_pocket_scores(sample, scores, out_dir)
    else:
        scores, scale = model(batch)
        result = _write_residue_scores(sample, scores, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "package": "SiteWeaver",
        "mode": mode,
        "complex_id": _sample_id(sample),
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint_path.expanduser().resolve()),
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "device": str(target),
        "score_scale": float(scale.detach().cpu().item()),
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_residues": int(sample.get("num_residues", 0)),
        "result": result,
    }
    metadata_path = out_dir / f"{_sample_id(sample)}_{mode}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["metadata_json"] = str(metadata_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SiteWeaver allosteric rankers on an external packed cache.")
    parser.add_argument("--cache-dir", required=True, help="External cache directory containing manifest.json.")
    parser.add_argument("--complex-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["pocket", "residue"], default="pocket")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    result = predict_cached(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
