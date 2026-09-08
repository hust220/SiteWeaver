#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from siteweaver.inference import SiteWeaverPredictor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bundled SiteWeaver residue-level predictors on a protein PDB."
    )
    parser.add_argument("--pdb", required=True, help="Input protein PDB or PDB.GZ file.")
    parser.add_argument(
        "--out-dir", required=True, help="Directory for CSV, B-factor PDB, and metadata outputs."
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "pocket", "active_site", "allosteric", "cryptic"],
        help="Prediction task. 'all' runs all four models (default).",
    )
    parser.add_argument(
        "--chain",
        nargs="+",
        default=None,
        help="Optional chain IDs. Omit this option to use every protein chain.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cpu, cuda, or cuda:1 (default: auto).",
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Optional directory containing the bundled checkpoint filenames.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictor = SiteWeaverPredictor(
        weights_dir=args.weights_dir,
        device=args.device,
        chains=args.chain,
    )
    metadata = predictor.predict(args.pdb, args.out_dir, task=args.task)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
