from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..pdb_utils import parse_pdb_residues
from ..prs_features import build_prs_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_active_probabilities(path: Path, residue_meta: list[dict], column: str) -> np.ndarray:
    values: dict[tuple[str, int, str], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chain_id", "residue_number", "insertion_code", column}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Active-score CSV must contain {sorted(required)}: {path}")
        for row in reader:
            key = (
                str(row["chain_id"]).strip(),
                int(row["residue_number"]),
                str(row.get("insertion_code", "")).strip(),
            )
            values[key] = float(row[column])
    output = np.zeros(len(residue_meta), dtype=np.float32)
    missing = []
    for index, meta in enumerate(residue_meta):
        key = (str(meta.get("chain_id", "")).strip(), int(meta.get("res_num", 0)), str(meta.get("ins_code", "")).strip())
        if key not in values:
            missing.append(key)
        else:
            output[index] = values[key]
    if missing:
        raise ValueError(f"Active-score CSV is missing {len(missing)} PDB residues; first missing key: {missing[0]}")
    return output


def run(
    pdb: str | Path,
    active_scores: str | Path,
    out_dir: str | Path,
    active_column: str = "probability",
    cutoff: float = 15.0,
    gamma: float = 1.0,
    n_modes: int = 20,
    dense_max_residues: int = 1200,
    chains: tuple[str, ...] | None = None,
) -> dict:
    """Generate the three directional CA-only PRS channels used by SiteWeaver."""
    pdb_path = Path(pdb).expanduser().resolve()
    active_path = Path(active_scores).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    residue_meta = parse_pdb_residues(pdb_path)
    if not residue_meta:
        raise ValueError(f"No protein residues found in {pdb_path}")
    probability = _read_active_probabilities(active_path, residue_meta, active_column)
    features, stats = build_prs_features(
        pdb_path,
        residue_meta,
        probability,
        cutoff=float(cutoff),
        gamma=float(gamma),
        n_modes=int(n_modes),
        dense_max_residues=int(dense_max_residues),
        chain_ids=chains,
    )
    csv_path = output / "prs_features.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["residue_index", "chain_id", "residue_number", "insertion_code", "residue_name", "active_probability", "active_to_residue", "residue_to_active", "directional_difference"],
        )
        writer.writeheader()
        for index, (meta, active, row) in enumerate(zip(residue_meta, probability, features)):
            writer.writerow({
                "residue_index": index,
                "chain_id": meta.get("chain_id", ""),
                "residue_number": meta.get("res_num", ""),
                "insertion_code": meta.get("ins_code", ""),
                "residue_name": meta.get("res_name", ""),
                "active_probability": f"{float(active):.8g}",
                "active_to_residue": f"{float(row[0]):.8g}",
                "residue_to_active": f"{float(row[1]):.8g}",
                "directional_difference": f"{float(row[2]):.8g}",
            })
    npy_path = output / "prs_features.npy"
    np.save(npy_path, features.astype(np.float32, copy=False))
    metadata = {
        "tool": "ProDy PRS/ANM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdb": str(pdb_path),
        "pdb_sha256": _sha256(pdb_path),
        "active_scores": str(active_path),
        "active_scores_sha256": _sha256(active_path),
        "active_column": active_column,
        "selection": "protein and name CA",
        "channels": ["active_to_residue", "residue_to_active", "directional_difference"],
        "parameters": {
            "cutoff_angstrom": float(cutoff),
            "gamma": float(gamma),
            "n_modes": int(n_modes),
            "dense_max_residues": int(dense_max_residues),
            "chains": list(chains) if chains else "all",
        },
        "prody_stats": stats,
        "outputs": {"csv": str(csv_path), "npy": str(npy_path)},
    }
    metadata_path = output / "prs_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["metadata_json"] = str(metadata_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SiteWeaver's three CA-only ProDy PRS feature channels.")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--active-scores", required=True, help="SiteWeaver active-site CSV or equivalent residue table.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--active-column", default="probability")
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--dense-max-residues", type=int, default=1200)
    parser.add_argument("--chain", nargs="+", default=None)
    args = parser.parse_args(argv)
    result = run(
        args.pdb,
        args.active_scores,
        args.out_dir,
        args.active_column,
        args.cutoff,
        args.gamma,
        args.n_modes,
        args.dense_max_residues,
        tuple(args.chain) if args.chain else None,
    )
    print(json.dumps(result["prody_stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
