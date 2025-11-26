#!/usr/bin/env python3
"""
Generate a compliance checklist and QA notes for a motion to seal.

This is a light‑weight stub that:
- Tries to extract text (docx if python-docx is available, else treats as plain text)
- Runs heuristic checks and emits:
  - reports/motion_to_seal/compliance_checklist.json
  - reports/motion_to_seal/qa_notes.md

Usage:
  python scripts/CheckSealMotionCompliance.py \
      --path reports/motion_to_seal/motion_to_seal_v2_clean.docx

Optional flags:
  --output-dir reports/motion_to_seal
  --motion-type non_dispositive|dispositive
  --jurisdiction "D. Massachusetts"
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def read_text_from_file(path: Path) -> str:
    """Best-effort text extraction. Supports .txt/.md; .docx if python-docx is available."""
    if not path.exists():
        return ""

    if path.suffix.lower() in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return path.read_text(errors="ignore")

    if path.suffix.lower() == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            # Fallback: no text, leave content-based checks as unknown
            return ""

    # Fallback for unknown types
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def basic_checks(text: str, motion_type: Optional[str]) -> Tuple[Dict[str, Any], list]:
    """Run simple heuristics and return (checklist fields, qa_notes)."""
    notes = []
    wc = word_count(text) if text else 0

    # Heuristic presence checks
    has_cert_service = bool(re.search(r"certificate of service", text, flags=re.I)) if text else False
    has_cert_compliance = bool(re.search(r"certif(icate)? of compliance", text, flags=re.I)) if text else False
    mentions_factors = bool(re.search(r"balance|balancing test|intel factor|narrowly tailored", text, flags=re.I)) if text else False
    mentions_alt = bool(re.search(r"alternative(s)? considered|less restrictive", text, flags=re.I)) if text else False
    mentions_exhibits = bool(re.search(r"exhibit\s+[A-Z0-9]+|redact", text, flags=re.I)) if text else False
    mentions_local_rules = bool(re.search(r"local rule|LR \d|L\.R\.", text, flags=re.I)) if text else False

    # Standard correctness and dispositive match cannot be decided reliably here
    standard_correct = None
    motion_type_detected = motion_type or "unknown"

    # Word/page limit: compute words; limit evaluation is jurisdiction-specific (unknown here)
    within_limit = None

    if not mentions_factors:
        notes.append("Add explicit balancing/factor mapping and narrowly tailored analysis.")
    if not mentions_alt:
        notes.append("Document alternatives considered (e.g., redactions, protective orders).")
    if not mentions_exhibits:
        notes.append("Map exhibits with redaction instructions and durations.")
    if not has_cert_service:
        notes.append("Add Certificate of Service.")
    if not has_cert_compliance:
        notes.append("Add Certificate of Compliance (if required).")
    if not mentions_local_rules:
        notes.append("Cite and apply applicable local rules.")

    checklist = {
        "version": "0.1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "standard_correct": standard_correct,  # unknown
        "motion_type": motion_type_detected,
        "declarations_attached": None,  # unknown at this layer
        "narrowly_tailored": True if re.search(r"narrowly tailored", text, flags=re.I) else None,
        "alternatives_considered": True if mentions_alt else None,
        "exhibits_mapped": True if mentions_exhibits else None,
        "local_rules_applied": True if mentions_local_rules else None,
        "page_or_word_limits": {
            "word_count": wc,
            "within_limit": within_limit,
        },
        "certificate_blocks_present": {
            "service": has_cert_service,
            "compliance": has_cert_compliance,
        },
        "notes": notes,
    }

    return checklist, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate motion-to-seal compliance checklist")
    parser.add_argument("--path", required=True, help="Path to motion (.docx/.txt/.md)")
    parser.add_argument("--output-dir", default="reports/motion_to_seal", help="Output directory")
    parser.add_argument("--motion-type", choices=["dispositive", "non_dispositive"], help="Motion type")
    parser.add_argument("--jurisdiction", help="Jurisdiction (e.g., D. Massachusetts)")
    args = parser.parse_args()

    src = Path(args.path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = read_text_from_file(src)
    checklist, notes = basic_checks(text, args.motion_type)
    checklist["source"] = str(src)
    if args.jurisdiction:
        checklist["jurisdiction"] = args.jurisdiction

    # Write JSON checklist
    checklist_path = out_dir / "compliance_checklist.json"
    checklist_path.write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    # Write QA notes
    qa_notes = ["# QA Notes", "", f"Source: {src}", f"Generated: {checklist['generated_at']}", ""]
    if notes:
        qa_notes.append("## Findings / TODOs")
        qa_notes.extend([f"- {n}" for n in notes])
    else:
        qa_notes.append("All checks passed heuristics. Perform human review for legal sufficiency.")

    (out_dir / "qa_notes.md").write_text("\n".join(qa_notes), encoding="utf-8")

    print(str(checklist_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

