#!/usr/bin/env python3
"""Export truth-table facts into a sealing-label template with metadata."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli_style import compact_parser

TRUTH_TABLE_DEFAULT = Path("case_law_data/facts_truth_table_v2.csv")
TEMPLATE_DEFAULT = Path("case_law_data/facts_labels_sealing_template.csv")
SUMMARY_DEFAULT = Path("reports/analysis_outputs/sealing_label_summary.md")

COMMUNICATION_KEYWORDS = {
    "email",
    "wrote",
    "write",
    "message",
    "call",
    "text",
    "messaged",
    "responded",
    "reply",
    "communication",
    "notified",
    "said",
    "told",
}

PII_KEYWORDS = {
    "address",
    "phone",
    "telephone",
    "cell",
    "mobile",
    "ssn",
    "social security",
    "passport",
    "license",
    "dob",
    "date of birth",
    "email",
    "contact",
}

HEALTH_KEYWORDS = {
    "medical",
    "diagnosis",
    "treatment",
    "injury",
    "hospital",
    "clinic",
    "health",
}

FINANCIAL_KEYWORDS = {
    "bank",
    "account",
    "wire",
    "payment",
    "transfer",
    "financial",
    "compensation",
    "salary",
    "fee",
}


def read_truth_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def detect_fact_type(row: dict[str, str]) -> str:
    proposition = (row.get("Proposition") or "").lower()
    event_type = (row.get("EventType") or "").strip()
    if event_type:
        return event_type
    if any(keyword in proposition for keyword in COMMUNICATION_KEYWORDS):
        return "Communication"
    if any(keyword in proposition for keyword in PII_KEYWORDS):
        return "PII"
    if any(keyword in proposition for keyword in HEALTH_KEYWORDS):
        return "Health"
    if any(keyword in proposition for keyword in FINANCIAL_KEYWORDS):
        return "Financial"
    return "General"


def importance_and_reasons(row: dict[str, str]) -> tuple[float, list[str]]:
    score = 0.35
    reasons: list[str] = []

    risk = (row.get("SafetyRisk") or "").strip().lower()
    exposure = (row.get("PublicExposure") or "").strip().lower()
    proposition = (row.get("Proposition") or "").lower()
    source_document = (row.get("SourceDocument") or "").lower()

    if risk in {"extreme", "high"}:
        score += 0.4
        reasons.append("high safety risk")
    elif risk == "medium":
        score += 0.25
        reasons.append("medium safety risk")
    elif risk == "low":
        score += 0.1
        reasons.append("low safety risk")

    if "not_public" in exposure or "non-public" in exposure or "private" in exposure:
        score += 0.15
        reasons.append("not public")
    elif "public" in exposure:
        score -= 0.1
        reasons.append("already public")

    sensitive_hits = sum(1 for keyword in PII_KEYWORDS if keyword in proposition)
    if sensitive_hits:
        score += 0.07 * min(sensitive_hits, 3)
        reasons.append("PII keyword match")

    health_hits = sum(1 for keyword in HEALTH_KEYWORDS if keyword in proposition)
    if health_hits:
        score += 0.08 * min(health_hits, 2)
        reasons.append("health context")

    financial_hits = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in proposition)
    if financial_hits:
        score += 0.06 * min(financial_hits, 2)
        reasons.append("financial detail")

    if any(keyword in proposition for keyword in COMMUNICATION_KEYWORDS):
        score += 0.05
        reasons.append("communication fact")

    if source_document.endswith(".email") or "email" in source_document:
        score += 0.05
        reasons.append("email source")

    score = max(0.0, min(score, 1.0))
    return round(score, 2), reasons


def recommended_action(score: float, fact_type: str, risk: str) -> str:
    if risk in {"extreme", "high"} or score >= 0.75 or fact_type in {"PII", "Health"}:
        return "seal"
    if score >= 0.5:
        return "review"
    return "leave_unsealed"


def export_template(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, str]]:
    enriched_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        fact_id = (row.get("FactID") or "").strip() or f"AUTO_{index:05d}"
        fact_type = detect_fact_type(row)
        score, reasons = importance_and_reasons(row)
        risk = (row.get("SafetyRisk") or "").strip().lower()
        action = recommended_action(score, fact_type, risk)
        reason_summary = "; ".join(dict.fromkeys(reasons)) or "baseline importance"

        enriched_rows.append(
            {
                "fact_id": fact_id,
                "proposition": (row.get("Proposition") or "").strip(),
                "fact_type": fact_type,
                "source_document": (row.get("SourceDocument") or "").strip(),
                "event_date": (row.get("EventDate") or "").strip(),
                "importance_score": f"{score:.2f}",
                "safety_risk": (row.get("SafetyRisk") or "").strip(),
                "public_exposure": (row.get("PublicExposure") or "").strip(),
                "recommended_action": action,
                "recommendation_reason": reason_summary,
                "label_sealing_required": "",
                "label_sealing_critical": "",
                "label_notes": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(enriched_rows[0].keys()) if enriched_rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for enriched in enriched_rows:
            writer.writerow(enriched)
    return enriched_rows


def write_summary(
    rows: list[dict[str, str]],
    summary_path: Path,
    template_path: Path,
    truth_table: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    action_counts = Counter(row["recommended_action"] for row in rows)
    seal_candidates = [
        row for row in rows if row["recommended_action"] == "seal"
    ]
    seal_candidates.sort(key=lambda row: row["importance_score"], reverse=True)
    top_candidates = seal_candidates[:10]

    def format_candidate(row: dict[str, str]) -> str:
        proposition = row["proposition"]
        if len(proposition) > 120:
            proposition = proposition[:117].rstrip() + "..."
        return (
            f"- {row['fact_id']} — score {row['importance_score']} — "
            f"{row['source_document'] or 'Unknown source'} — {proposition}"
        )

    total = len(rows)
    seal_count = action_counts.get("seal", 0)
    review_count = action_counts.get("review", 0)
    leave_count = action_counts.get("leave_unsealed", 0)

    lines: list[str] = [
        "# Sealing Label Export Summary",
        "",
        "## Status",
        f"- Facts exported: {total}",
        f"- Template path: {template_path}",
        f"- Truth table source: {truth_table}",
        "",
        "## Recommendations",
        f"- Seal: {seal_count}",
        f"- Review: {review_count}",
        f"- Leave unsealed: {leave_count}",
        "",
        "## High-Priority Facts",
    ]

    if top_candidates:
        lines.extend(format_candidate(row) for row in top_candidates)
    else:
        lines.append("- No high-priority sealing candidates identified.")

    lines.extend(
        [
            "",
            "## Do This Next",
            "```",
            f"$ python scripts/export_facts_for_sealing_labeling.py --input {truth_table} "
            f"--output {template_path}",
            f"$ open {template_path}",
            "```",
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = compact_parser(
        prog="export_facts_for_sealing_labeling",
        description="Export truth-table facts into a sealing template with metadata.",
        commands=[
            "export_facts_for_sealing_labeling --input case_law_data/facts_truth_table_v2.csv "
            "--output case_law_data/facts_labels_sealing_template.csv",
            "export_facts_for_sealing_labeling --summary reports/analysis_outputs/sealing_label_summary.md",
        ],
    )
    parser.add_argument(
        "--input",
        default=str(TRUTH_TABLE_DEFAULT),
        help="Path to the fact truth-table CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(TEMPLATE_DEFAULT),
        help="Path to write the sealing label template CSV.",
    )
    parser.add_argument(
        "--summary",
        default=str(SUMMARY_DEFAULT),
        help="Optional Markdown summary output path.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    truth_table_path = Path(args.input)
    template_path = Path(args.output)
    summary_path = Path(args.summary)

    if not truth_table_path.exists():
        print(f"[error] Missing truth table: {truth_table_path}", file=sys.stderr)
        return 1

    rows = read_truth_table(truth_table_path)
    if not rows:
        print("[error] No facts found in truth table.", file=sys.stderr)
        return 1

    enriched = export_template(rows, template_path)
    write_summary(enriched, summary_path, template_path, truth_table_path)
    print(
        f"[ok] Exported {len(enriched)} facts to {template_path} and summary to {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
