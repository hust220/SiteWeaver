from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .models import g1_const as const
from .models import r0_const
from .r0_features import _parse_heavy_atoms, build_r0_graph


_POCKET_LABEL = re.compile(r"Pocket\s+(\d+)\s*:")
_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def write_protein_only_pdb(source: Path, destination: Path, residue_meta: list[dict]) -> None:
    allowed = {
        (str(meta.get("chain_id", "")).strip(), int(meta["res_num"]), str(meta.get("ins_code", "")).strip())
        for meta in residue_meta
    }
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    destination.parent.mkdir(parents=True, exist_ok=True)
    with opener(source, "rt", errors="ignore") as reader, destination.open("w") as writer:
        for line in reader:
            if not line.startswith("ATOM") or line[16:17].strip() not in {"", "A"}:
                continue
            try:
                key = (line[21].strip(), int(line[22:26]), line[26].strip())
            except ValueError:
                continue
            if key not in allowed:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                token = "".join(ch for ch in atom_name.upper() if ch.isalpha())
                element = token[:2] if token.startswith(("CL", "BR")) else token[:1]
            if r0_const.normalize_element(element) == "H":
                continue
            writer.write(line)
        writer.write("END\n")


def load_frozen_stats(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    if path is None:
        candidates = (
            Path(__file__).resolve().parents[2] / "configs" / "g1_feature_stats.json",
            Path(sys.prefix) / "siteweaver" / "configs" / "g1_feature_stats.json",
        )
        path = next((item for item in candidates if item.is_file()), candidates[0])
    payload = json.loads(Path(path).read_text())
    stats = payload["pocket_feature_statistics"]
    if set(stats) != const.STANDARDIZED_POCKET_FEATURES:
        raise ValueError("Frozen G1 feature statistics do not match the model schema")
    return stats


def _parse_info(path: Path) -> dict[int, dict[str, float]]:
    key_map = {
        "score": "native_score",
        "druggability score": "druggability_score",
        "volume": "volume",
        "mean alpha sphere radius": "mean_alpha_sphere_radius",
        "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_accessibility",
    }
    output: dict[int, dict[str, float]] = {}
    current = None
    if not path.is_file():
        return output
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _POCKET_LABEL.search(line)
            if match:
                current = int(match.group(1))
                output.setdefault(current, {})
                continue
            if current is None or ":" not in line:
                continue
            key = key_map.get(line.split(":", 1)[0].strip().lower())
            value = _FLOAT.search(line.split(":", 1)[1])
            if key is not None and value is not None:
                output[current][key] = float(value.group())
    return output


def _vertex_centers(path: Path) -> np.ndarray:
    centers = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line[:6].strip().upper() not in {"ATOM", "HETATM"}:
                continue
            centers.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not centers:
        raise ValueError(f"No alpha spheres in {path}")
    return np.asarray(centers, dtype=np.float32)


def _pocket_number(path: Path) -> int:
    match = re.search(r"pocket(\d+)_vert\.pqr$", path.name)
    if not match:
        raise ValueError(f"Unexpected FPocket vertex file: {path}")
    return int(match.group(1))


def _records(
    structure_path: Path,
    fpocket_output_dir: Path,
    residue_meta: list[dict],
    base: dict,
    active_mask: np.ndarray,
) -> list[dict]:
    parsed = _parse_heavy_atoms(structure_path)
    atoms = []
    by_residue = []
    for meta in residue_meta:
        key = (str(meta["chain_id"]), int(meta["res_num"]), str(meta.get("ins_code", "")))
        item = parsed.get(key)
        if item is None or not item["atoms"]:
            raise ValueError(f"Missing heavy atoms for {key}")
        indices = list(range(len(atoms), len(atoms) + len(item["atoms"])))
        atoms.extend(item["atoms"])
        by_residue.append(indices)
    residue_indices = np.asarray(base["residue_indices"], dtype=np.int64)
    if len(atoms) != len(residue_indices):
        raise ValueError("R0 graph and FPocket structure have different atom counts")
    coords = np.asarray([atom["coord"] for atom in atoms], dtype=np.float32)
    tree = cKDTree(coords)
    active_ids = np.flatnonzero(active_mask[residue_indices])
    active_tree = cKDTree(coords[active_ids])
    pocket_probability = np.asarray(base["node_features"])[:, r0_const.N_ELEMENT_FEATURES + r0_const.N_ROLE_FEATURES]
    info_paths = list(fpocket_output_dir.glob("*_info.txt"))
    info = _parse_info(info_paths[0]) if len(info_paths) == 1 else {}
    records = []
    for path in sorted((fpocket_output_dir / "pockets").glob("pocket*_vert.pqr"), key=_pocket_number):
        centers = _vertex_centers(path)
        hits = tree.query_ball_point(centers, r=const.POCKET_NEIGHBOR_DISTANCE)
        nearby = sorted({int(residue_indices[atom_id]) for group in hits for atom_id in group})
        anchors = []
        distances = []
        for residue_id in nearby:
            ids = by_residue[residue_id]
            ca = [index for index in ids if atoms[index]["name"] == "CA"]
            residue_coords = coords[ids]
            matrix = np.linalg.norm(residue_coords[:, None, :] - centers[None, :, :], axis=2)
            anchors.append(int(ca[0] if ca else ids[int(np.argmin(matrix.min(axis=1)))]))
            distances.append(float(matrix.min()))
        active_distances = np.asarray(active_tree.query(centers, k=1)[0], dtype=np.float32)
        native = info.get(_pocket_number(path), {})
        records.append({
            "pocket_number": _pocket_number(path),
            "num_alpha_spheres": int(centers.shape[0]),
            "nearby_residue_indices": nearby,
            "anchor_atom_indices": anchors,
            "pocket_residue_distances": distances,
            "volume": float(native.get("volume", 0.0)),
            "mean_alpha_sphere_radius": float(native.get("mean_alpha_sphere_radius", 0.0)),
            "mean_alpha_sphere_solvent_accessibility": float(native.get("mean_alpha_sphere_solvent_accessibility", 0.0)),
            "mean_ligand_free_pocket_probability": float(pocket_probability[[by_residue[i][0] for i in nearby]].mean()) if nearby else 0.0,
            "max_ligand_free_pocket_probability": float(pocket_probability[[by_residue[i][0] for i in nearby]].max()) if nearby else 0.0,
            "active_residue_fraction": float(active_mask[nearby].mean()) if nearby else 0.0,
            "min_active_distance": float(active_distances.min()),
            "mean_active_distance": float(active_distances.mean()),
        })
    if not records:
        raise ValueError(f"FPocket returned no candidate pockets in {fpocket_output_dir}")
    return records


def _pocket_features(record: dict, stats: dict) -> np.ndarray:
    raw = {
        "log1p_alpha_sphere_count": float(np.log1p(record["num_alpha_spheres"])),
        "log1p_nearby_residue_count": float(np.log1p(len(record["nearby_residue_indices"]))),
        "mean_alpha_sphere_radius": record["mean_alpha_sphere_radius"],
        "log1p_volume": float(np.log1p(max(0.0, record["volume"]))),
        "mean_alpha_sphere_solvent_accessibility": record["mean_alpha_sphere_solvent_accessibility"],
        "min_active_distance": record["min_active_distance"],
        "mean_active_distance": record["mean_active_distance"],
    }
    values = np.zeros(const.POCKET_NODE_EXTRA_DIM, dtype=np.float32)
    values[0] = 1.0
    for index, name in enumerate(const.POCKET_FEATURE_NAMES[1:], start=1):
        value = raw[name] if name in const.STANDARDIZED_POCKET_FEATURES else record[name]
        if name in const.STANDARDIZED_POCKET_FEATURES:
            value = (value - stats[name]["mean"]) / stats[name]["std"]
        values[index] = float(value)
    return values


def build_g1_graph(
    structure_path: Path,
    fpocket_output_dir: Path,
    residue_meta: list[dict],
    pocket_probability: np.ndarray,
    active_mask: np.ndarray,
    stats: dict | None = None,
) -> dict:
    stats = stats or load_frozen_stats()
    base = build_r0_graph(structure_path, residue_meta, pocket_probability, active_mask.astype(np.float32))
    records = _records(structure_path, fpocket_output_dir, residue_meta, base, active_mask)
    n_atoms = int(base["node_features"].shape[0])
    pocket_nf = np.zeros((len(records), const.NODE_FEATURE_DIM), dtype=np.float32)
    links = []
    link_features = []
    centers = np.linspace(0.0, const.POCKET_EDGE_RBF_CUTOFF, const.RBF_BASIS, dtype=np.float32)
    width = max(const.POCKET_EDGE_RBF_CUTOFF / const.RBF_BASIS, 1e-3)
    for index, record in enumerate(records):
        pocket_nf[index, const.BASE_NODE_FEATURE_DIM:] = _pocket_features(record, stats)
        for anchor, distance in zip(record["anchor_atom_indices"], record["pocket_residue_distances"]):
            links.append([int(anchor), n_atoms + index])
            rbf = np.exp(-((float(distance) - centers) / width) ** 2).astype(np.float32)
            link_features.append(np.concatenate([rbf, np.zeros(const.BASE_EDGE_FEATURE_DIM - const.RBF_BASIS, dtype=np.float32), [1.0]]))
    node_features = np.concatenate([
        np.concatenate([base["node_features"], np.zeros((n_atoms, const.POCKET_NODE_EXTRA_DIM), dtype=np.float32)], axis=1),
        pocket_nf,
    ])
    edge_index = np.concatenate([base["edge_index"], np.asarray(links, dtype=np.int64).reshape(-1, 2).T], axis=1)
    edge_features = np.concatenate([
        np.concatenate([base["edge_features"], np.zeros((base["edge_features"].shape[0], 1), dtype=np.float32)], axis=1),
        np.asarray(link_features, dtype=np.float32).reshape(-1, const.EDGE_FEATURE_DIM),
    ])
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "residue_indices": np.concatenate([base["residue_indices"], np.full(len(records), -1, dtype=np.int64)]),
        "pocket_node_mask": np.concatenate([np.zeros(n_atoms, dtype=bool), np.ones(len(records), dtype=bool)]),
        "pocket_records": records,
    }
