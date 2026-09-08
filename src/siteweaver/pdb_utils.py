from __future__ import annotations

import re
import gzip
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from . import const

BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "OXT"}


def atom_coords(residue: dict) -> np.ndarray:
    atoms = residue.get("atoms", []) or []
    coords = [atom.get("coord") for atom in atoms if atom.get("coord") is not None]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32).reshape(-1, 3)


def residue_node_coords(residue: dict) -> tuple[np.ndarray, np.ndarray]:
    coords_dict = residue.get("coords_dict", {}) or {}
    ca = coords_dict.get("CA")
    all_coords = atom_coords(residue)
    if ca is None:
        ca = all_coords.mean(axis=0) if len(all_coords) else np.zeros(3, dtype=np.float32)
    ca = np.asarray(ca, dtype=np.float32)

    sidechain = []
    for atom in residue.get("atoms", []) or []:
        if str(atom.get("name", "")).strip() not in BACKBONE_ATOMS:
            sidechain.append(atom.get("coord"))
    if sidechain:
        sc = np.asarray(sidechain, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    else:
        sc = ca.copy()
    return ca, sc.astype(np.float32)


def node_feature(res_name: str, is_bb: bool) -> np.ndarray:
    feature = np.zeros(const.NODE_FEATURE_DIM, dtype=np.float32)
    residue_token = "BB" if is_bb else res_name
    feature[const.residue_name_to_index(residue_token)] = 1.0
    feature[const.N_RESIDUE_TYPES] = 1.0 if is_bb else 0.0
    feature[const.N_RESIDUE_TYPES + 1] = 0.0 if is_bb else 1.0
    return feature


def parse_pdb_residues(path: Path) -> list[dict]:
    residues = []
    current_key = None
    current = None
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            res_name = line[17:20].strip().upper() or "UNK"
            chain_id = line[21].strip()
            try:
                res_num = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            # ATOM records can also contain nucleic acids in a complex. The
            # bundled models are protein-only, so retain recognized residue
            # names and ignore non-protein ATOM records.
            if res_name not in const.RESIDUE2IDX:
                continue
            ins_code = line[26].strip()
            key = (chain_id, res_num, ins_code)
            if key != current_key:
                if current is not None:
                    residues.append(current)
                current_key = key
                current = {
                    "chain_id": chain_id,
                    "res_num": res_num,
                    "ins_code": ins_code,
                    "res_name": res_name,
                    "protein_canonical_res_name": res_name if res_name in const.RESIDUE2IDX else "UNK",
                    "atoms": [],
                    "coords_dict": {},
                }
            coord = np.asarray([x, y, z], dtype=np.float32)
            current["atoms"].append({"name": atom_name, "coord": coord})
            current["coords_dict"][atom_name] = coord
    if current is not None:
        residues.append(current)
    return residues


def build_protein_only_graph_arrays(protein_residues: list[dict], contact_cutoff: float) -> dict | None:
    if not protein_residues:
        return None

    node_coords = []
    node_features = []
    residue_indices = []
    residue_meta = []

    for ridx, residue in enumerate(protein_residues):
        res_name = str(residue.get("protein_canonical_res_name") or residue.get("res_name") or "UNK").upper()
        ca, sc = residue_node_coords(residue)
        residue_meta.append({
            "chain_id": residue.get("chain_id", ""),
            "res_num": int(residue.get("res_num", 0)),
            "ins_code": residue.get("ins_code", ""),
            "res_name": res_name,
        })
        for is_bb, coord in ((True, ca), (False, sc)):
            node_coords.append(coord)
            node_features.append(node_feature(res_name, is_bb=is_bb))
            residue_indices.append(ridx)

    coords = np.asarray(node_coords, dtype=np.float32)
    features = np.asarray(node_features, dtype=np.float32)
    residue_indices_np = np.asarray(residue_indices, dtype=np.int64)
    edge_feature_by_pair = {}

    def add_edge(i: int, j: int, flags: tuple[float, float, float]):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_feature_by_pair:
            return
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        bb_sc, sequential, same_chain = flags
        edge_feature_by_pair[(a, b)] = [min(dist / contact_cutoff, 4.0), 1.0, bb_sc, sequential, same_chain]

    tree = KDTree(coords)
    for i, j in tree.query_pairs(r=contact_cutoff):
        ri = int(residue_indices_np[i])
        rj = int(residue_indices_np[j])
        same_chain = float(residue_meta[ri]["chain_id"] == residue_meta[rj]["chain_id"])
        add_edge(i, j, (float(ri == rj), float(abs(ri - rj) == 1), same_chain))

    for ridx in range(len(protein_residues)):
        add_edge(2 * ridx, 2 * ridx + 1, (1.0, 0.0, 1.0))
        if ridx + 1 < len(protein_residues):
            same_chain = float(residue_meta[ridx]["chain_id"] == residue_meta[ridx + 1]["chain_id"])
            if same_chain:
                add_edge(2 * ridx, 2 * (ridx + 1), (0.0, 1.0, 1.0))

    if edge_feature_by_pair:
        sorted_edges = sorted(edge_feature_by_pair)
        edge_index = np.asarray(sorted_edges, dtype=np.int64).T
        edge_features = np.asarray([edge_feature_by_pair[pair] for pair in sorted_edges], dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, const.EDGE_FEATURE_DIM), dtype=np.float32)

    return {
        "node_features": features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "residue_indices": residue_indices_np,
        "residue_meta": residue_meta,
        "num_residues": len(protein_residues),
    }


def parse_active_site_tokens(tokens: list[str]) -> set[tuple[str, int, str]]:
    out = set()
    for token in tokens:
        token = str(token).strip().strip("'\"[](){}")
        if not token:
            continue
        parts = token.split("-")
        if len(parts) < 2:
            continue
        chain = parts[0].strip()
        res_token = parts[-1].strip()
        match = re.match(r"(-?\d+)([A-Za-z]?)$", res_token)
        if not match:
            continue
        res_num = int(match.group(1))
        ins_code = match.group(2) or ""
        out.add((chain, res_num, ins_code))
    return out


def residue_mask(residue_meta: list[dict], tokens: set[tuple[str, int, str]]) -> np.ndarray:
    mask = np.zeros(len(residue_meta), dtype=np.bool_)
    for idx, meta in enumerate(residue_meta):
        key = (str(meta.get("chain_id", "")), int(meta.get("res_num", 0)), str(meta.get("ins_code", "") or ""))
        if key in tokens:
            mask[idx] = True
    return mask
