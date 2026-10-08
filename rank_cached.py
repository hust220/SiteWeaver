#!/usr/bin/env python3
"""Source-tree launcher for cached allosteric ranking."""

from pathlib import Path
import sys


SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from siteweaver.cache_predict import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
