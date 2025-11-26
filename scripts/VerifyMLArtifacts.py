#!/usr/bin/env python3
"""
Verify presence of key ML artifacts (non-blocking summary).
"""

from __future__ import annotations

from pathlib import Path
import json


def list_files(glob: str) -> list[str]:
    return [str(p) for p in Path().glob(glob)]


def main() -> int:
    models = list_files("case_law_data/models/*.cbm")
    fi_csvs = list_files("case_law_data/models/*_feature_importance.csv")
    shap_json = Path("case_law_data/analysis/section_1782_discovery_SHAP_SUMMARY.json")

    result = {
        "models_count": len(models),
        "feature_importances_count": len(fi_csvs),
        "shap_summary_present": shap_json.exists(),
        "examples": {
            "model": models[0] if models else None,
            "feature_importance": fi_csvs[0] if fi_csvs else None,
            "shap_summary": str(shap_json) if shap_json.exists() else None,
        }
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

