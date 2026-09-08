from __future__ import annotations

from pathlib import Path

import numpy as np


def _sparse_prs_features(coords, probability, cutoff, gamma, n_modes):
    """Compute only the two weighted PRS profiles without an N x N matrix.

    ProDy's dense covariance path is exact but becomes impractical for large
    proteins. The sparse path uses the same ANM spring Hessian and a low-rank
    eigenspace, then contracts the PRS matrix directly against the active
    probability vector. This keeps memory O(N * modes^2), not O(N^2).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    pairs = np.asarray(
        list(cKDTree(coords).query_pairs(float(cutoff))), dtype=np.int64
    )
    if pairs.size == 0:
        raise ValueError("Sparse ANM has no contacts")
    pairs = pairs.reshape(-1, 2)
    row_parts = []
    col_parts = []
    data_parts = []
    diagonal = np.zeros((n, 3, 3), dtype=np.float64)
    for i, j in pairs:
        delta = coords[i] - coords[j]
        dist2 = float(np.dot(delta, delta))
        if dist2 <= 1e-10:
            continue
        block = float(gamma) * np.outer(delta, delta) / dist2
        diagonal[i] += block
        diagonal[j] += block
        ii = 3 * int(i) + np.arange(3)
        jj = 3 * int(j) + np.arange(3)
        grid_i, grid_j = np.meshgrid(ii, ii, indexing="ij")
        row_parts.append(grid_i.reshape(-1))
        col_parts.append(grid_j.reshape(-1))
        data_parts.append(block.reshape(-1))
        grid_i, grid_j = np.meshgrid(jj, jj, indexing="ij")
        row_parts.append(grid_i.reshape(-1))
        col_parts.append(grid_j.reshape(-1))
        data_parts.append(block.reshape(-1))
        grid_i, grid_j = np.meshgrid(ii, jj, indexing="ij")
        row_parts.append(grid_i.reshape(-1))
        col_parts.append(grid_j.reshape(-1))
        data_parts.append((-block).reshape(-1))
        grid_i, grid_j = np.meshgrid(jj, ii, indexing="ij")
        row_parts.append(grid_i.reshape(-1))
        col_parts.append(grid_j.reshape(-1))
        data_parts.append((-block).reshape(-1))
    if not row_parts:
        raise ValueError("Sparse ANM could not build any Hessian entries")
    hessian = coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(3 * n, 3 * n),
    ).tocsr()
    requested = min(int(n_modes) + 8, max(2, 3 * n - 1))
    eigenvalues, eigenvectors = eigsh(
        hessian,
        k=requested,
        sigma=1e-6,
        which="LM",
        tol=1e-4,
        maxiter=max(1000, 20 * requested),
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(1e-8, float(eigenvalues.max()) * 1e-8)
    positive_indices = np.flatnonzero(positive)[: int(n_modes)]
    if positive_indices.size == 0:
        raise ValueError("Sparse ANM returned no positive normal modes")
    modes = eigenvectors[:, positive_indices] / np.sqrt(eigenvalues[positive_indices])[None, :]
    modes = modes.reshape(n, 3, -1)
    gram = np.einsum("nac,nad->ncd", modes, modes).reshape(n, -1)
    self_response = np.einsum("nd,nd->n", gram, gram).clip(min=1e-12)
    active_to_residue = ((probability / self_response) @ gram) @ gram.T
    aggregate_target = probability @ gram
    residue_to_active = (gram @ aggregate_target) / self_response
    return active_to_residue.astype(np.float32), residue_to_active.astype(np.float32), int(positive_indices.size)


def _icode(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _pdb_keys(atoms) -> list[tuple[str, int, str]]:
    chains = atoms.getChids()
    numbers = atoms.getResnums()
    icodes = atoms.getIcodes()
    return [
        (str(chain).strip(), int(number), _icode(icode))
        for chain, number, icode in zip(chains, numbers, icodes)
    ]


def build_prs_features(
    pdb_path: str | Path,
    residue_meta: list[dict],
    active_probability,
    cutoff: float = 15.0,
    gamma: float = 1.0,
    n_modes: int = 20,
    dense_max_residues: int = 1200,
    chain_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict]:
    """Build three residue features from CA-only ANM/PRS.

    The returned columns are active-to-residue response, residue-to-active
    response, and their directional difference. Only predicted active-site
    probabilities are used; allosteric labels never enter this calculation.
    """
    from prody import ANM, calcPerturbResponse, parsePDB

    parsed = parsePDB(str(pdb_path))
    selection = "protein and name CA"
    if chain_ids:
        selection += " and chain " + " ".join(str(chain).strip() for chain in chain_ids)
    atoms = parsed.select(selection) if parsed is not None else None
    if atoms is None or atoms.numAtoms() < 4:
        raise ValueError(f"Could not obtain at least four CA atoms from {pdb_path}")

    keys = _pdb_keys(atoms)
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    cache_to_prs = []
    missing = []
    for idx, meta in enumerate(residue_meta):
        key = (
            str(meta.get("chain_id", "")).strip(),
            int(meta.get("res_num", 0)),
            _icode(meta.get("ins_code", "")),
        )
        if key not in key_to_index:
            missing.append((idx, key))
        else:
            cache_to_prs.append((idx, key_to_index[key]))
    probability = np.asarray(active_probability, dtype=np.float32).reshape(-1)
    if probability.shape[0] != len(residue_meta):
        raise ValueError(
            f"Active probability length mismatch for {pdb_path}: "
            f"{probability.shape[0]} != {len(residue_meta)}"
        )
    n = atoms.numAtoms()
    p = np.zeros(n, dtype=np.float32)
    for cache_idx, prs_idx in cache_to_prs:
        p[prs_idx] = max(float(probability[cache_idx]), 0.0)
    if float(p.sum()) <= 1e-8:
        p[:] = 1.0 / float(n)
    else:
        p /= p.sum()

    modes = min(int(n_modes), max(1, n - 1))
    if n <= int(dense_max_residues):
        anm = ANM(f"PRS:{Path(pdb_path).stem}")
        anm.buildHessian(atoms.getCoords(), cutoff=float(cutoff), gamma=float(gamma))
        anm.calcModes(n_modes=modes, zeros=False, turbo=True)
        prs_matrix, _, _ = calcPerturbResponse(anm, n_modes=modes)
        prs_matrix = np.asarray(prs_matrix, dtype=np.float32)
        if prs_matrix.shape != (n, n):
            raise ValueError(f"Unexpected PRS matrix shape {prs_matrix.shape} for {pdb_path}")
        active_to_residue = p @ prs_matrix
        residue_to_active = prs_matrix @ p
        path = "prody_dense"
    else:
        active_to_residue, residue_to_active, modes = _sparse_prs_features(
            atoms.getCoords(), p, float(cutoff), float(gamma), modes
        )
        path = "sparse_low_rank"
    direction = active_to_residue - residue_to_active
    prs_features = np.stack(
        [active_to_residue, residue_to_active, direction], axis=1
    ).astype(np.float32, copy=False)

    aligned = np.zeros((len(residue_meta), 3), dtype=np.float32)
    for cache_idx, prs_idx in cache_to_prs:
        aligned[cache_idx] = prs_features[prs_idx]
    stats = {
        "pdb_path": str(pdb_path),
        "n_ca": int(n),
        "n_modes": int(modes),
        "method": path,
        "missing_residue_count": int(len(missing)),
        "prs_min": float(aligned.min()),
        "prs_max": float(aligned.max()),
        "prs_mean": float(aligned.mean()),
    }
    return aligned, stats
