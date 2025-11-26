"""
Summarize quality metrics for all STORM runs.
Generates a markdown table summarizing quality score, word count, citations, and appendix paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def gather_results(output_dir: Path) -> List[Dict[str, Any]]:
    entries = []
    for json_path in sorted(output_dir.glob("*_STORM_Inspired_Results.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            quality = metadata.get("quality")
            entry = {
                "topic": data.get("topic", json_path.stem),
                "quality": quality if isinstance(quality, (int, float)) else None,
                "word_count": metadata.get("quality", {}).get("details", {}).get("word_count")
                if isinstance(metadata.get("quality"), dict)
                else None,
                "citations": metadata.get("quality", {}).get("details", {}).get("citations")
                if isinstance(metadata.get("quality"), dict)
                else None,
                "primary_sources": metadata.get("quality", {}).get("details", {}).get("primary_sources")
                if isinstance(metadata.get("quality"), dict)
                else None,
                "report_file": json_path.with_name(json_path.name.replace("_Results.json", "_Report.md")).as_posix(),
                "appendix_file": json_path.with_name(json_path.name.replace("_Results.json", "_Mechanism_Appendix.md")).as_posix(),
            }
            entries.append(entry)
        except Exception:
            continue
    return entries


def format_table(entries: List[Dict[str, Any]]) -> str:
    header = "| Topic | Quality | Word Count | Citations | Primary Sources | Report | Appendix |\n"
    header += "|-------|---------|------------|-----------|-----------------|--------|----------|\n"
    rows = []
    for entry in entries:
        rows.append(
            "| {topic} | {quality} | {word_count} | {citations} | {primary} | {report} | {appendix} |".format(
                topic=entry["topic"],
                quality=entry["quality"] or "-",
                word_count=entry["word_count"] or "-",
                citations=entry["citations"] or "-",
                primary=entry["primary_sources"] or "-",
                report=entry["report_file"],
                appendix=entry["appendix_file"],
            )
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    output_dir = Path("storm_inspired_outputs")
    entries = gather_results(output_dir)
    if not entries:
        print("No STORM results found.")
        return

    table = format_table(entries)
    summary_path = output_dir / "STORM_Quality_Summary.md"
    summary_path.write_text(table, encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
