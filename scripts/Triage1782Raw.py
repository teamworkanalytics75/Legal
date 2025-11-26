#!/usr/bin/env python3
"""
Triage the wide-net § 1782 corpus.

Reads JSON opinions from a source directory (default: data/case_law/1782_discovery_raw),
applies the `is_actual_1782_case` heuristic, and writes high-confidence matches to a
destination directory. Skipped opinions are listed in a JSON log for later review.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set

import importlib.util

ROOT = Path(__file__).parent.parent

spec = importlib.util.spec_from_file_location(
    "filters",
    ROOT / "document_ingestion" / "filters.py",
)
filters_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters_module)
is_actual_1782_case = filters_module.is_actual_1782_case

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def triage(
    source_dir: Path,
    destination_dir: Path,
    skip_log: Path,
) -> Dict[str, int]:
    ensure_directory(destination_dir)
    ensure_directory(skip_log.parent)

    kept_hashes: Set[str] = set()
    skipped: list[Dict[str, str]] = []
    kept = 0
    total = 0

    for json_path in sorted(source_dir.glob("*.json")):
        total += 1
        try:
            data = load_json(json_path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", json_path, exc)
            skipped.append({"file": json_path.name, "reason": f"read_error:{exc}"})
            continue

        case_hash = data.get("_matrix_case_hash")
        if not case_hash:
            skipped.append({"file": json_path.name, "reason": "missing_case_hash"})
            continue

        if not is_actual_1782_case(data):
            skipped.append({"file": json_path.name, "reason": "filter_reject"})
            continue

        if case_hash in kept_hashes:
            skipped.append({"file": json_path.name, "reason": "duplicate_hash"})
            continue

        destination_path = destination_dir / json_path.name
        destination_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        kept_hashes.add(case_hash)
        kept += 1

    skip_log.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "total": total,
        "kept": kept,
        "skipped": len(skipped),
        "unique_hashes": len(kept_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage raw §1782 downloads into a filtered corpus.")
    parser.add_argument(
        "--source",
        default=ROOT / "data" / "case_law" / "1782_discovery_raw",
        type=Path,
        help="Path to the raw JSON directory.",
    )
    parser.add_argument(
        "--destination",
        default=ROOT / "data" / "case_law" / "1782_discovery_filtered",
        type=Path,
        help="Where to save filtered opinions.",
    )
    parser.add_argument(
        "--skip-log",
        default=ROOT / "data" / "case_law" / "logs" / "1782_raw_skipped.json",
        type=Path,
        help="JSON file describing skipped opinions.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Source: %s", args.source)
    logger.info("Destination: %s", args.destination)
    logger.info("Skip log: %s", args.skip_log)

    stats = triage(args.source, args.destination, args.skip_log)

    logger.info("Triage complete.")
    logger.info("Total opinions processed: %d", stats["total"])
    logger.info("Kept (high-confidence §1782): %d", stats["kept"])
    logger.info("Skipped: %d", stats["skipped"])
    logger.info("Unique hashes kept: %d", stats["unique_hashes"])


if __name__ == "__main__":
    main()
