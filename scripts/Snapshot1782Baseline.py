#!/usr/bin/env python3
"""
Snapshot the current 1782_discovery filenames into a timestamped JSON list.

Usage:
    python snapshot_1782_baseline.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    source_dir = Path("data/case_law/1782_discovery")
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    filenames = sorted(p.name for p in source_dir.glob("*.json"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = Path("data/case_law/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    output_path = baseline_dir / f"1782_discovery_baseline_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filenames, f, indent=2, ensure_ascii=False)

    print(f"Baseline snapshot saved to {output_path} ({len(filenames)} files)")


if __name__ == "__main__":
    main()
