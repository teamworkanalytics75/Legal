#!/usr/bin/env python3
"""
Export top SHAP features for §1782 to reports/ml/targets_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

INPUT = Path("case_law_data/analysis/section_1782_discovery_SHAP_SUMMARY.json")
OUTPUT = Path("reports/ml/targets_summary.json")


def main() -> int:
    if not INPUT.exists():
        print(f"Input SHAP summary not found: {INPUT}")
        return 0
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    top = data.get("top_features", [])
    # Keep top 20
    top = top[:20]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"top_features": top}, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

