#!/usr/bin/env python3
"""
Verify local-first LLM configuration (no heavy deps required).

Checks:
- Env toggles (MATRIX_USE_LOCAL, MATRIX_LOCAL_MODEL, MATRIX_LOCAL_URL)
- Agents default (ModelConfig.use_local == True) by static source read
- SK default path: confirms fallback block exists by static pattern

Outputs a JSON-ish summary to stdout and exits 0 regardless (non-blocking).
"""

from __future__ import annotations

import os
from pathlib import Path
import json


def contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    agents_py = repo / "writer_agents" / "code" / "agents.py"
    sk_config_py = repo / "writer_agents" / "code" / "sk_config.py"

    env = {
        "MATRIX_USE_LOCAL": os.environ.get("MATRIX_USE_LOCAL"),
        "MATRIX_LOCAL_MODEL": os.environ.get("MATRIX_LOCAL_MODEL"),
        "MATRIX_LOCAL_URL": os.environ.get("MATRIX_LOCAL_URL"),
        "OPENAI_API_KEY_present": bool(os.environ.get("OPENAI_API_KEY")),
    }

    checks = {
        "agents_default_local": contains(agents_py, "use_local: bool = True"),
        "agents_env_override": contains(agents_py, "MATRIX_USE_LOCAL"),
        "agents_fallback_present": contains(agents_py, "fallback to OpenAI"),
        "sk_env_override": contains(sk_config_py, "MATRIX_USE_LOCAL"),
        "sk_fallback_present": contains(sk_config_py, "falling back to OpenAI for SK chat service"),
    }

    result = {
        "env": env,
        "checks": checks,
        "status": "ok" if all(checks.values()) else "partial",
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

