#!/usr/bin/env python3
"""Source-tree launcher for SiteWeaver PDB inference."""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from siteweaver.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
