from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from .models import r0_const as const
from . import const as residue_const


def _element_from_pdb(line: str, atom_name: str) -> str:
    field = line[76:78].strip() if len(line) >= 78 else ""
    if not field:
        token = "".join(ch for ch in atom_name.upper() if ch.isalpha())
        field = token[:2] if token.startswith(("CL", "BR")) else token[:1]
    return const.normalize_element(field)


def _key(chain_id: str, residue_number: int, insertion_code: str) -> tuple[str, int, str]:
    return str(chain_id), int(residue_number), str(insertion_code or "")


def _parse_heavy_atoms(path: Path) -> dict[tuple[str, int, str], dict]:
    residues: dict[tuple[str, int, str], dict] = {}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if line[16].strip() not in {"", "A"}:
                continue
            element = _element_from_pdb(line, atom_name)
            if element == "H":
                continue
            try:
                chain_id = line[21].strip()
                residue_number = int(line[22:26])
                insertion_code = line[26].strip()
                coord = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except (ValueError, IndexError):
                continue
            residue_name = line[17:20].strip().upper() or "UNK"
            if residue_name not in residue_const.RESIDUE2IDX:
                continue
            key = _key(chain_id, residue_number, insertion_code)
            residues.setdefault(key, {"atoms": []})["atoms"].append(
                {"name": atom_name, "element": element, "coord": coord, "chain_id": chain_id}
            )
    return residues


def _is_covalent(element_i: str, element_j: str, distance: float) -> bool:
    radius_i = const.COVALENT_RADII.get(element_i)
    radius_j = const.COVALENT_RADII.get(element_j)
    return (
        radius_i is not None
        and radius_j is not None
        and 0.8 <= distance <= radius_i + radius_j + 0.45
    )


def _rbf(distance: float, cutoff: float) -> np.ndarray:
    centers = np.linspace(0.0, float(cutoff), const.RBF_BASIS, dtype=np.float32)
    width = max(float(cutoff) / const.RBF_BASIS, 1e-3)
    return np.exp(-((float(distance) - centers) / width) ** 2).astype(np.float32)


def build_r0_graph(
    pdb_path: str | Path,
    residue_meta: list[dict],
    pocket_probability: np.ndarray,
    active_probability: np.ndarray,
    contact_cutoff: float = const.CONTACT_CUTOFF,
    max_neighbors: int = const.MAX_NEIGHBORS,
):
    """Build the full-atom graph expected by the frozen R0 checkpoint."""
    residue_meta = list(residue_meta)
    pocket_probability = np.asarray(pocket_probability, dtype=np.float32)
    active_probability = np.asarray(active_probability, dtype=np.float32)
    if len(residue_meta) != pocket_probability.size or len(residue_meta) != active_probability.size:
        raise ValueError("R0 feature channels and residue metadata have different lengths")
    parsed = _parse_heavy_atoms(Path(pdb_path))
    atoms = []
    residue_atom_indices = []
    for residue_index, meta in enumerate(residue_meta):
        key = _key(meta.get("chain_id", ""), meta.get("res_num", 0), meta.get("ins_code", ""))
        residue = parsed.get(key)
        if residue is None or not residue["atoms"]:
            raise ValueError(f"Missing heavy atoms for residue {key} in {pdb_path}")
        indices = []
        for atom in residue["atoms"]:
            indices.append(len(atoms))
            atoms.append({**atom, "residue_index": residue_index})
        residue_atom_indices.append(indices)

    coords = np.asarray([atom["coord"] for atom in atoms], dtype=np.float32)
    residue_indices = np.asarray([atom["residue_index"] for atom in atoms], dtype=np.int64)
    features = np.zeros((len(atoms), const.NODE_FEATURE_DIM), dtype=np.float32)
    for index, atom in enumerate(atoms):
        feature = const.element_feature(atom["element"])
        name = str(atom["name"]).upper()
        feature[const.N_ELEMENT_FEATURES] = float(name in const.BACKBONE_ATOMS)
        feature[const.N_ELEMENT_FEATURES + 1] = float(name == "CA")
        feature[const.N_ELEMENT_FEATURES + 2] = float(name == "OXT")
        features[index] = feature

    edge_pairs: set[tuple[int, int]] = set()
    distances: dict[tuple[int, int], float] = {}
    tree = KDTree(coords)
    for i in range(len(atoms)):
        neighbors = tree.query_ball_point(coords[i], r=float(contact_cutoff))
        neighbors = [j for j in neighbors if j != i]
        neighbors.sort(key=lambda j: (float(np.linalg.norm(coords[i] - coords[j])), int(j)))
        for j in neighbors[: int(max_neighbors)]:
            pair = (min(i, j), max(i, j))
            edge_pairs.add(pair)
            distances[pair] = float(np.linalg.norm(coords[i] - coords[j]))

    covalent_pairs: set[tuple[int, int]] = set()
    peptide_pairs: set[tuple[int, int]] = set()
    for atom_ids in residue_atom_indices:
        for offset, i in enumerate(atom_ids):
            for j in atom_ids[offset + 1:]:
                distance = float(np.linalg.norm(coords[i] - coords[j]))
                if _is_covalent(atoms[i]["element"], atoms[j]["element"], distance):
                    pair = (min(i, j), max(i, j))
                    covalent_pairs.add(pair)
                    edge_pairs.add(pair)
                    distances[pair] = distance
    for residue_index in range(len(residue_atom_indices) - 1):
        left = residue_meta[residue_index]
        right = residue_meta[residue_index + 1]
        if left.get("chain_id", "") != right.get("chain_id", ""):
            continue
        left_c = next((i for i in residue_atom_indices[residue_index] if atoms[i]["name"] == "C"), None)
        right_n = next((i for i in residue_atom_indices[residue_index + 1] if atoms[i]["name"] == "N"), None)
        if left_c is None or right_n is None:
            continue
        distance = float(np.linalg.norm(coords[left_c] - coords[right_n]))
        if distance <= 2.0:
            pair = (min(left_c, right_n), max(left_c, right_n))
            covalent_pairs.add(pair)
            peptide_pairs.add(pair)
            edge_pairs.add(pair)
            distances[pair] = distance

    sorted_pairs = sorted(edge_pairs)
    edge_features = []
    for i, j in sorted_pairs:
        distance = distances.get((i, j), float(np.linalg.norm(coords[i] - coords[j])))
        ri, rj = int(residue_indices[i]), int(residue_indices[j])
        same_chain = float(atoms[i]["chain_id"] == atoms[j]["chain_id"])
        sequential = float(same_chain and abs(ri - rj) == 1)
        edge_features.append(
            np.concatenate([
                _rbf(distance, contact_cutoff),
                np.asarray([
                    float(ri == rj),
                    same_chain,
                    float((i, j) in covalent_pairs),
                    float((i, j) in peptide_pairs),
                    sequential,
                ], dtype=np.float32),
            ])
        )
    node_features = np.concatenate([
        features,
        pocket_probability[residue_indices, None],
        active_probability[residue_indices, None],
    ], axis=1).astype(np.float32, copy=False)
    return {
        "node_features": node_features,
        "edge_index": np.asarray(sorted_pairs, dtype=np.int64).T if sorted_pairs else np.zeros((2, 0), dtype=np.int64),
        "edge_features": np.asarray(edge_features, dtype=np.float32).reshape(-1, const.EDGE_FEATURE_DIM),
        "residue_indices": residue_indices,
        "num_residues": len(residue_meta),
    }
