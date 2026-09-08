from __future__ import annotations

from pathlib import Path
import gzip


def format_bfactor(value: float) -> str:
    return f"{value:6.2f}"[:6]


def write_bfactor_pdb(pdb_path: Path, out_path: Path, residue_scores: list[float], residue_meta: list[dict], scale_to_100: bool = False) -> dict:
    residue_lookup = {}
    for idx, meta in enumerate(residue_meta):
        key = (
            str(meta.get("chain_id", "")).strip(),
            int(meta.get("res_num", 0)),
            str(meta.get("ins_code", "") or "").strip(),
        )
        residue_lookup[key] = float(residue_scores[idx])

    values = list(residue_lookup.values())
    if scale_to_100 and values:
        vmin = min(values)
        vmax = max(values)
        if vmax > vmin:
            residue_lookup = {k: 100.0 * (v - vmin) / (vmax - vmin) for k, v in residue_lookup.items()}
        else:
            residue_lookup = {k: 0.0 for k in residue_lookup}

    lines_out = []
    mapped_atoms = 0
    seen_residues = set()
    pdb_path = Path(pdb_path)
    opener = gzip.open if pdb_path.suffix.lower() == ".gz" else open
    with opener(pdb_path, "rt") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                lines_out.append(line)
                continue
            chain = line[21].strip()
            resseq_raw = line[22:26].strip()
            if not resseq_raw:
                lines_out.append(line)
                continue
            try:
                resseq = int(resseq_raw)
            except ValueError:
                lines_out.append(line)
                continue
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in residue_lookup:
                score = residue_lookup[key]
                mapped_atoms += 1
                seen_residues.add(key)
                line = f"{line[:60]}{format_bfactor(score)}{line[66:]}"
            lines_out.append(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines_out))
    return {
        "pdb_path": str(pdb_path),
        "out_path": str(out_path),
        "mapped_residues": len(seen_residues),
        "mapped_atoms": mapped_atoms,
        "scale_to_100": bool(scale_to_100),
    }
