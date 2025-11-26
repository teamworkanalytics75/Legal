#!/usr/bin/env python3
"""Validate outputs of the ML fact extraction pipeline."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "case_law_data" / "lawsuit_facts_database.db"
DEFAULT_TRUTH_TABLE = REPO_ROOT / "case_law_data" / "facts_truth_table_v2.csv"
DEFAULT_REPORT = REPO_ROOT / "reports" / "analysis_outputs" / "fact_extraction_validation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fact extraction outputs for Agent 2 pipeline rerun."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="Path to lawsuit_facts_database.db")
    parser.add_argument(
        "--truth-table",
        type=Path,
        default=DEFAULT_TRUTH_TABLE,
        help="Path to facts_truth_table CSV (default: facts_truth_table_v2.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Optional markdown report output path",
    )
    parser.add_argument(
        "--min-facts",
        type=int,
        default=2000,
        help="Minimum expected fact count (default: 2,000)",
    )
    parser.add_argument(
        "--min-communications",
        type=int,
        default=25,
        help="Minimum expected number of communication facts in truth table",
    )
    return parser.parse_args()


def load_fact_registry_stats(db_path: Path) -> Dict[str, any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT fact_type, COUNT(*) FROM fact_registry GROUP BY fact_type")
    per_type = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.execute("SELECT source_doc FROM fact_registry WHERE source_doc IS NOT NULL AND source_doc != ''")
    source_entries = [row[0] for row in cursor.fetchall()]
    conn.close()

    unique_docs: set[str] = set()
    extensions = Counter()
    for entry in source_entries:
        for candidate in entry.split(";"):
            candidate = candidate.strip()
            if not candidate:
                continue
            unique_docs.add(candidate)
            suffix = Path(candidate).suffix.lower()
            if suffix:
                extensions[suffix] += 1
            else:
                extensions["<none>"] += 1

    return {
        "total_facts": sum(per_type.values()),
        "per_type": per_type,
        "unique_docs": len(unique_docs),
        "extensions": extensions,
    }


def parse_truth_table(truth_path: Path) -> Dict[str, any]:
    totals = 0
    event_counts = Counter()
    ogc_ack_flags: List[Tuple[int, str]] = []

    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            totals += 1
            event_counts[row.get("EventType", "").strip()] += 1
            prop = row.get("Proposition", "") or ""
            prop_lower = prop.lower()
            if "ogc" in prop_lower and "acknowledg" in prop_lower:
                if not any(neg in prop_lower for neg in ("not", "never", "no ", "failed", "did not")):
                    ogc_ack_flags.append((idx, prop))

    return {
        "total_rows": totals,
        "event_counts": event_counts,
        "ogc_ack_flags": ogc_ack_flags,
    }


def format_report(
    db_stats: Dict[str, any],
    truth_stats: Dict[str, any],
    expectations: Dict[str, int],
) -> str:
    lines: List[str] = []
    lines.append("# Fact Extraction Validation Report")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Fact registry rows | {db_stats['total_facts']:,} |")
    lines.append(f"| Unique source documents | {db_stats['unique_docs']:,} |")
    lines.append(f"| Truth table rows | {truth_stats['total_rows']:,} |")
    lines.append(f"| Communication facts (truth table) | {truth_stats['event_counts'].get('Communication', 0):,} |")
    lines.append("")

    lines.append("## File type coverage")
    lines.append("")
    ext_counts = db_stats["extensions"]
    for ext in sorted(ext_counts):
        lines.append(f"- {ext or '<none>'}: {ext_counts[ext]:,} facts")
    lines.append("")

    lines.append("## Fact types (database)")
    lines.append("")
    for fact_type, count in sorted(db_stats["per_type"].items()):
        lines.append(f"- {fact_type}: {count:,}")
    lines.append("")

    if truth_stats["ogc_ack_flags"]:
        lines.append("## ⚠️ Potential OGC acknowledgments detected")
        for idx, text in truth_stats["ogc_ack_flags"]:
            lines.append(f"- Row {idx}: {text}")
    else:
        lines.append("## ✅ OGC acknowledgment check")
        lines.append("- No positive acknowledgement statements found for Harvard OGC.")
    lines.append("")

    lines.append("## Expectations check")
    lines.append("")
    fact_goal = expectations["min_facts"]
    comm_goal = expectations["min_communications"]
    fact_pass = db_stats["total_facts"] >= fact_goal
    comm_pass = truth_stats["event_counts"].get("Communication", 0) >= comm_goal
    lines.append(f"- Fact count threshold ({fact_goal:,}) -> {'PASS' if fact_pass else 'FAIL'}")
    lines.append(f"- Communication facts threshold ({comm_goal}) -> {'PASS' if comm_pass else 'FAIL'}")

    required_exts = [".txt", ".docx", ".pdf", ".md", ".html"]
    coverage = {ext: ext_counts.get(ext, 0) > 0 for ext in required_exts}
    for ext, present in coverage.items():
        status = "PASS" if present else "MISS"
        lines.append(f"- Coverage {ext}: {status}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.database.exists():
        raise SystemExit(f"Database not found: {args.database}")
    if not args.truth_table.exists():
        raise SystemExit(f"Truth table not found: {args.truth_table}")

    db_stats = load_fact_registry_stats(args.database)
    truth_stats = parse_truth_table(args.truth_table)

    report_md = format_report(
        db_stats,
        truth_stats,
        {"min_facts": args.min_facts, "min_communications": args.min_communications},
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
