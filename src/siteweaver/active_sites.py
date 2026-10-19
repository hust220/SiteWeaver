from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_active_site_mask(path: str | Path, residue_meta: list[dict]) -> np.ndarray:
    """Map an annotated active-site residue list onto the supplied PDB graph."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    lookup = {
        (str(meta.get("chain_id", "")).strip(), int(meta["res_num"]), str(meta.get("ins_code", "")).strip()): index
        for index, meta in enumerate(residue_meta)
    }
    if len(lookup) != len(residue_meta):
        raise ValueError("The input structure contains duplicate residue identifiers")
    mask = np.zeros(len(residue_meta), dtype=bool)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chain_id", "residue_number", "insertion_code"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Active-site CSV must contain chain_id,residue_number,insertion_code")
        count = 0
        for row in reader:
            key = (
                str(row["chain_id"] or "").strip(),
                int(row["residue_number"]),
                str(row["insertion_code"] or "").strip(),
            )
            if key not in lookup:
                raise ValueError(f"Annotated active-site residue {key} is absent from the input structure")
            if mask[lookup[key]]:
                raise ValueError(f"Duplicate active-site residue {key}")
            mask[lookup[key]] = True
            count += 1
    if not count:
        raise ValueError("Active-site CSV contains no residues")
    return mask
