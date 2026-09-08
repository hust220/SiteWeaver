from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_binary(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    found = shutil.which(str(value))
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(f"FPocket executable not found: {value}")


def _probe_binary(binary: Path) -> str:
    for args in (("--version",), ("-h",)):
        completed = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, check=False, timeout=30
        )
        text = (completed.stdout + "\n" + completed.stderr).strip()
        if text:
            return text[:4000]
    return ""


@dataclass
class RunRecord:
    complex_id: str
    input_pdb: str
    output_dir: str
    command: list[str]
    returncode: int
    ok: bool
    reused: bool
    input_sha256: str
    binary: str
    binary_probe: str
    started_at: str
    finished_at: str
    log_file: str
    error: str = ""


def _run_one(
    binary: Path,
    pdb_path: Path,
    output_root: Path,
    complex_id: str | None = None,
    reuse: bool = True,
) -> RunRecord:
    pdb_path = pdb_path.expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)
    if pdb_path.suffix.lower() != ".pdb":
        raise ValueError(f"FPocket input must be an uncompressed .pdb file: {pdb_path}")
    identifier = str(complex_id or pdb_path.stem).strip()
    if not identifier:
        raise ValueError(f"Empty complex identifier for {pdb_path}")
    work_dir = output_root / identifier
    local_pdb = work_dir / f"{identifier}.pdb"
    expected = work_dir / f"{identifier}_out"
    log_file = work_dir / "run.log"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    probe = _probe_binary(binary)
    if reuse and (expected / "pockets").is_dir():
        return RunRecord(
            identifier, str(pdb_path), str(work_dir), [str(binary), "-f", local_pdb.name],
            0, True, True, _sha256(pdb_path), str(binary), probe, started, _utc_now(), str(log_file)
        )
    shutil.copy2(pdb_path, local_pdb)
    command = [str(binary), "-f", local_pdb.name]
    completed = subprocess.run(command, cwd=work_dir, capture_output=True, text=True, check=False)
    log_file.write_text(completed.stdout + "\n" + completed.stderr)
    ok = completed.returncode == 0 and (expected / "pockets").is_dir()
    error = "" if ok else (completed.stderr[-1000:] or "missing FPocket output directory")
    return RunRecord(
        identifier, str(pdb_path), str(work_dir), command, completed.returncode, ok, False,
        _sha256(pdb_path), str(binary), probe, started, _utc_now(), str(log_file), error
    )


def run(
    fpocket_bin: str | Path,
    pdb: str | Path | None,
    out_dir: str | Path,
    manifest: str | Path | None = None,
    workers: int = 8,
    reuse: bool = True,
) -> dict:
    """Run FPocket using the same local-copy and ``fpocket -f file`` policy as training."""
    binary = _resolve_binary(fpocket_bin)
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str | None, Path]] = []
    if pdb is not None:
        jobs.append((None, Path(pdb)))
    if manifest is not None:
        with Path(manifest).expanduser().open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No rows found in manifest: {manifest}")
        fields = set(rows[0])
        if not {"complex_id", "pdb_path"}.issubset(fields):
            raise ValueError("FPocket manifest must contain complex_id,pdb_path columns")
        jobs.extend((str(row["complex_id"]), Path(row["pdb_path"])) for row in rows)
    if not jobs:
        raise ValueError("Pass --pdb or --manifest")
    records: list[RunRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_run_one, binary, path, output_root, identifier, reuse) for identifier, path in jobs]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record.complex_id)
    report = {
        "tool": "fpocket",
        "binary": str(binary),
        "binary_probe": records[0].binary_probe if records else "",
        "workers": int(max(1, workers)),
        "output_root": str(output_root),
        "manifest": str(Path(manifest).expanduser().resolve()) if manifest else None,
        "created_at": _utc_now(),
        "num_jobs": len(records),
        "num_success": sum(record.ok for record in records),
        "runs": [asdict(record) for record in records],
    }
    report_path = output_root / "fpocket_run.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["num_success"] != report["num_jobs"]:
        raise RuntimeError(f"FPocket failed for {report['num_jobs'] - report['num_success']} job(s); see {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run external FPocket with SiteWeaver-compatible output layout.")
    parser.add_argument("--fpocket-bin", required=True, help="FPocket executable or PATH name.")
    parser.add_argument("--pdb", default=None, help="One uncompressed protein PDB file.")
    parser.add_argument("--manifest", default=None, help="CSV with exactly the required columns complex_id,pdb_path plus optional columns.")
    parser.add_argument("--out-dir", required=True, help="Output root; one subdirectory is created per complex.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-reuse", action="store_true", help="Run FPocket even when the expected output already exists.")
    args = parser.parse_args(argv)
    report = run(args.fpocket_bin, args.pdb, args.out_dir, args.manifest, args.workers, reuse=not args.no_reuse)
    print(json.dumps({key: report[key] for key in ("tool", "output_root", "num_jobs", "num_success")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
