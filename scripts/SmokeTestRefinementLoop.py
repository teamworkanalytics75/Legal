#!/usr/bin/env python3
"""
Smoke test: initialize RefinementLoop and confirm SHAP targets are loaded.

Finds a CatBoost model in priority order and constructs RefinementLoop with no plugins.
Prints a short JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def find_model() -> str | None:
    base = Path("case_law_data/models")
    order = [
        "catboost_outline_unified.cbm",
        "catboost_outline_both_interactions.cbm",
        "catboost_motion_seal_pseudonym.cbm",
        "catboost_motion.cbm",
        "section_1782_discovery_model.cbm",
    ]
    for name in order:
        p = base / name
        if p.exists():
            return str(p)
    # Fallback to any .cbm
    for p in base.glob("*.cbm"):
        return str(p)
    return None


def main() -> int:
    model_path = find_model()
    if not model_path:
        print(json.dumps({"ok": False, "error": "No CatBoost model found"}))
        return 0

    # Import after to avoid heavy imports at module load
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "writer_agents"))
    sys.path.insert(0, str(root))
    # Stub semantic_kernel for import-only purposes
    import types
    if 'semantic_kernel' not in sys.modules:
        sk_stub = types.SimpleNamespace(Kernel=object)
        sys.modules['semantic_kernel'] = sk_stub
        sys.modules['semantic_kernel.connectors'] = types.SimpleNamespace()
        sys.modules['semantic_kernel.core_plugins'] = types.SimpleNamespace()
        sys.modules['semantic_kernel.memory'] = types.SimpleNamespace()
        # functions stubs
        sys.modules['semantic_kernel.functions'] = types.SimpleNamespace(KernelFunction=object)
        sys.modules['semantic_kernel.functions.kernel_function_decorator'] = types.SimpleNamespace(kernel_function=lambda f: f)
    from code.sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop

    rl = RefinementLoop(plugins={}, model_path=Path(model_path), debug_mode=False)

    summary = {
        "ok": True,
        "model_path": model_path,
        "targets_count": len(rl.feature_targets or {}),
        "shap_loaded": bool(rl.shap_importance),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
