#!/usr/bin/env python3
"""
Initialize focus on the personal legal corpus → AutoGen + Semantic Kernel refinement loop
for a Motion to Seal and to Proceed Under Pseudonym.

This script does not run heavy jobs automatically. It:
- Checks corpus pipeline completeness
- Detects likely personal files (HK statement, OGC emails)
- Prints a compact "Do this next" with verified copy-paste commands
- Surfaces local LLM env toggles (Ollama-first) per repo policy

Usage:
    python scripts/init_personal_corpus_focus.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any


REPO = Path(__file__).resolve().parents[1]
CASE_LAW = REPO / "case_law_data"
WRITER_SCRIPTS = REPO / "writer_agents" / "scripts"


def _load_focus_profile() -> Dict[str, Any]:
    """Load optional focus profile with sane defaults if present."""
    profile = REPO / "writer_agents" / "focus_profiles" / "pseudonym_seal.json"
    if profile.exists():
        try:
            return json.loads(profile.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    # Defaults if profile is missing
    return {
        "jurisdiction": "D. Massachusetts",
        "target_confidence": 0.75,
        "max_iterations": 5,
        "autogen_model": os.environ.get("MATRIX_LOCAL_MODEL", "qwen2.5:14b"),
        "sk_use_local": True,
    }


def _detect_personal_files() -> Dict[str, str | None]:
    """Suggest likely HK Statement and OGC emails from corpus directory if present."""
    corpus_dir = CASE_LAW / "tmp_corpus"
    hk = None
    ogc = None
    if corpus_dir.exists():
        # Heuristics for filenames
        for p in sorted(corpus_dir.glob("*")):
            name = p.name.lower()
            if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
                if hk is None and ("hk" in name or "statement" in name):
                    hk = str(p)
                if ogc is None and ("ogc" in name or "email" in name):
                    ogc = str(p)
            if hk and ogc:
                break
    return {"hk_statement": hk, "ogc_emails": ogc}


def _corpus_completeness() -> Dict[str, bool]:
    """Use the same checks as writer_agents/scripts/refresh_personal_corpus.py."""
    try:
        import sys
        # Ensure writer_agents/scripts is importable
        scripts_dir = WRITER_SCRIPTS
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from refresh_personal_corpus import check_corpus_completeness  # type: ignore
    except Exception:
        # Fallback minimalistic check
        check_corpus_completeness = None  # type: ignore

    corpus_dir = CASE_LAW / "tmp_corpus"
    results_dir = CASE_LAW / "results"
    if check_corpus_completeness:
        return check_corpus_completeness(corpus_dir, results_dir)  # type: ignore

    # Fallback checks
    return {
        "corpus_dir_exists": corpus_dir.exists() and any(corpus_dir.iterdir()),
        "results_dir_exists": results_dir.exists(),
        "features_csv": (results_dir / "personal_corpus_features.csv").exists(),
        "aggregated_statistics": (results_dir / "personal_corpus_aggregated_statistics.json").exists(),
        "embeddings_db": (results_dir / "personal_corpus_embeddings.db").exists(),
        "faiss_index": (results_dir / "personal_corpus_embeddings.faiss").exists(),
        "master_draft": any(results_dir.glob("master_draft*.txt")),
    }


def main() -> int:
    profile = _load_focus_profile()
    detected = _detect_personal_files()
    checks = _corpus_completeness()

    # Compact summary
    print("Personal Corpus Focus — Motion to Seal / Pseudonym")
    print("- Jurisdiction:", profile["jurisdiction"])
    print("- Target confidence:", profile["target_confidence"])
    print("- Max iterations:", profile["max_iterations"])
    print("- AutoGen model:", profile["autogen_model"], "(local)")
    print("- SK local:", profile["sk_use_local"]) 

    print("\nCorpus status (key outputs):")
    for k, v in checks.items():
        print(f"- {k}:", "OK" if v else "MISSING")
    if not (checks.get("features_csv") and checks.get("aggregated_statistics")):
        print("WARNING: run writer_agents/scripts/refresh_personal_corpus.py to unlock personalized CatBoost features.")

    hk_hint = detected.get("hk_statement") or "<path-to-hk-statement.txt>"
    ogc_hint = detected.get("ogc_emails") or "<path-to-ogc-emails.txt>"

    print("\nEnvironment toggles (local-first):")
    print("- MATRIX_USE_LOCAL=true")
    print("- MATRIX_LOCAL_MODEL=qwen2.5:14b   # or phi3:mini for SK")
    print("- MATRIX_LOCAL_URL=http://localhost:11434")
    print("- MATRIX_ENABLE_NETWORK_MODELS=true  # unlock semantic retrieval + personal CatBoost deltas")

    # Action box: copy-paste commands
    print("\nDo this next (copy/paste):")
    print("# 1) Verify local LLM defaults")
    print("python scripts/VerifyLocalLLMSetup.py")

    print("\n# 2) Refresh personal corpus (auto-detects source)")
    print("python writer_agents/scripts/refresh_personal_corpus.py")

    print("\n# 3) Generate master motion with full refinement loop (local models)")
    print(
        "python writer_agents/scripts/generate_master_motion_with_refinement.py "
        f"--hk-statement '{hk_hint}' --ogc-emails '{ogc_hint}' "
        f"--target-confidence {profile['target_confidence']} --max-iterations {profile['max_iterations']} "
        "--autogen-model 'qwen2.5:14b'"
    )

    print("\nOptional: set env for local models (add to .env)")
    print("MATRIX_USE_LOCAL=true\nMATRIX_LOCAL_MODEL=qwen2.5:14b\nMATRIX_LOCAL_URL=http://localhost:11434")

    # Non-blocking success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
