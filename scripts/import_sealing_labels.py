#!/usr/bin/env python3
"""Import sealing labels, validate coverage, and enrich the truth table."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cli_style import compact_parser

TEMPLATE_DEFAULT = Path("case_law_data/facts_labels_sealing_template.csv")
TRUTH_TABLE_DEFAULT = Path("case_law_data/facts_truth_table_v2.csv")
OUTPUT_DEFAULT = Path("case_law_data/facts_truth_table_with_sealing_labels.csv")
REPORT_DEFAULT = Path("reports/analysis_outputs/sealing_label_validation.md")

POSITIVE_LABELS = {"yes", "y", "true", "1", "seal", "sealed", "required"}
NEGATIVE_LABELS = {"no", "n", "false", "0", "public", "leave_unsealed"}
REVIEW_LABELS = {"review", "maybe", "needs_review", "flag"}


@dataclass
class LabelDecision:
    fact_id: str
    decision: str
    critical: bool
    notes: str
    recommended_action: str
    importance_score: float


def read_template(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_decision(row: dict[str, str]) -> tuple[LabelDecision | None, bool]:
    fact_id = (row.get("fact_id") or row.get("FactID") or "").strip()
    decision_raw = (row.get("label_sealing_required") or "").strip().lower()
    critical_raw = (row.get("label_sealing_critical") or "").strip().lower()
    notes = (row.get("label_notes") or "").strip()

    if not fact_id:
        return None, True

    if decision_raw in POSITIVE_LABELS:
        decision = "seal"
    elif decision_raw in NEGATIVE_LABELS:
        decision = "no_seal"
    elif decision_raw in REVIEW_LABELS:
        decision = "review"
    else:
        return LabelDecision(
            fact_id=fact_id,
            decision="",
            critical=False,
            notes=notes,
            recommended_action=(row.get("recommended_action") or "").strip(),
            importance_score=_to_float(row.get("importance_score")),
        ), True

    critical = critical_raw in POSITIVE_LABELS
    return (
        LabelDecision(
            fact_id=fact_id,
            decision=decision,
            critical=critical,
            notes=notes,
            recommended_action=(row.get("recommended_action") or "").strip(),
            importance_score=_to_float(row.get("importance_score")),
        ),
        False,
    )


def _to_float(value: str | None) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def parse_template(rows: list[dict[str, str]]) -> tuple[dict[str, LabelDecision], list[str]]:
    decisions: dict[str, LabelDecision] = {}
    missing: list[str] = []
    for row in rows:
        decision, is_missing = normalize_decision(row)
        if decision is None:
            continue
        decisions[decision.fact_id] = decision
        if is_missing:
            missing.append(decision.fact_id)
    return decisions, missing


def read_truth_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_augmented_truth_table(
    truth_rows: list[dict[str, str]],
    decisions: dict[str, LabelDecision],
    output_path: Path,
) -> None:
    augmented_rows: list[dict[str, str]] = []
    augmented_headers: list[str] = []
    for row in truth_rows:
        fact_id = (row.get("FactID") or "").strip()
        decision = decisions.get(fact_id)
        augmented = dict(row)
        augmented["SealingLabelDecision"] = decision.decision if decision else ""
        augmented["SealingLabelCritical"] = "1" if decision and decision.critical else "0"
        augmented["SealingLabelNotes"] = decision.notes if decision else ""
        augmented["SealingRecommendedAction"] = (
            decision.recommended_action if decision else ""
        )
        augmented["SealingImportanceScore"] = (
            f"{decision.importance_score:.2f}" if decision else ""
        )
        augmented_rows.append(augmented)

    if truth_rows:
        augmented_headers = list(truth_rows[0].keys()) + [
            "SealingLabelDecision",
            "SealingLabelCritical",
            "SealingLabelNotes",
            "SealingRecommendedAction",
            "SealingImportanceScore",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=augmented_headers)
        writer.writeheader()
        for row in augmented_rows:
            writer.writerow(row)


def build_report(
    decisions: dict[str, LabelDecision],
    missing: list[str],
    template_path: Path,
    report_path: Path,
    output_path: Path | None,
) -> None:
    counts = {
        "seal": 0,
        "review": 0,
        "no_seal": 0,
    }
    for decision in decisions.values():
        if decision.decision in counts:
            counts[decision.decision] += 1

    mismatches = [
        decision
        for decision in decisions.values()
        if decision.decision == "no_seal" and decision.recommended_action == "seal"
    ]

    report_lines = [
        "# Sealing Label Validation",
        "",
        "## Status",
        f"- Template path: {template_path}",
        f"- Output path: {output_path or 'not generated'}",
        f"- Labeled facts: {len(decisions)}",
        "",
        "## Label Totals",
        f"- Seal: {counts['seal']}",
        f"- Review: {counts['review']}",
        f"- No seal: {counts['no_seal']}",
    ]

    if missing:
        sample = ", ".join(missing[:10])
        report_lines.extend(
            [
                "",
                "## Missing Labels",
                f"- Count: {len(missing)}",
                f"- Sample: {sample}",
            ]
        )

    if mismatches:
        report_lines.append("")
        report_lines.append("## Conflicts (Recommended Seal but Labeled No)")
        for decision in mismatches[:10]:
            report_lines.append(
                f"- {decision.fact_id} — importance {decision.importance_score:.2f}"
            )

    report_lines.extend(
        [
            "",
            "## Do This Next",
            "```",
            f"$ python scripts/import_sealing_labels.py --template {template_path} "
            f"--truth-table {TRUTH_TABLE_DEFAULT} --output {OUTPUT_DEFAULT}",
            "```",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = compact_parser(
        prog="import_sealing_labels",
        description="Validate sealing labels and merge into the truth table.",
        commands=[
            "import_sealing_labels --template case_law_data/facts_labels_sealing_template.csv "
            "--truth-table case_law_data/facts_truth_table_v2.csv "
            "--output case_law_data/facts_truth_table_with_sealing_labels.csv",
            "import_sealing_labels --report reports/analysis_outputs/sealing_label_validation.md",
        ],
    )
    parser.add_argument(
        "--template",
        default=str(TEMPLATE_DEFAULT),
        help="Path to the labeled sealing template.",
    )
    parser.add_argument(
        "--truth-table",
        default=str(TRUTH_TABLE_DEFAULT),
        help="Path to the fact truth table CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help="Path for the augmented truth table with labels.",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_DEFAULT),
        help="Markdown report path for validation results.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing labels without failing validation.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    template_path = Path(args.template)
    truth_table_path = Path(args.truth_table)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not template_path.exists():
        print(f"[error] Missing template file: {template_path}", file=sys.stderr)
        return 1

    template_rows = read_template(template_path)
    decisions, missing = parse_template(template_rows)
    if missing and not args.allow_missing:
        build_report(decisions, missing, template_path, report_path, output_path)
        print(
            f"[error] Found {len(missing)} facts without labels. "
            "Use --allow-missing to override.",
            file=sys.stderr,
        )
        return 1

    truth_rows = read_truth_table(truth_table_path) if truth_table_path.exists() else []
    if truth_rows:
        write_augmented_truth_table(truth_rows, decisions, output_path)

    build_report(decisions, missing, template_path, report_path, output_path)
    print(
        f"[ok] Validated {len(decisions)} labels. "
        f"Report: {report_path} "
        f"{'and updated truth table.' if truth_rows else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
