#!/usr/bin/env python3
"""
Filter Harvard CAP bulk downloads for 28 U.S.C. § 1782 opinions.

Usage:
    py scripts\\triage_cap_directory.py --source <path-to-extracted-cap-bulk> \
        --destination data\\case_law\\1782_discovery_cap
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
FILTERS_PATH = ROOT / "document_ingestion" / "filters.py"


def load_is_actual_1782_case():
    spec = importlib.util.spec_from_file_location("triage_filters", FILTERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.is_actual_1782_case


is_actual_1782_case = load_is_actual_1782_case()
logger = logging.getLogger(__name__)


def iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def extract_case_text(case: Dict) -> str:
    texts: List[str] = []
    body = case.get("casebody")

    if isinstance(body, dict):
        for key in ("text", "preface", "syllabus"):
            value = body.get(key)
            if isinstance(value, str):
                texts.append(value)

        data = body.get("data")
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str):
                        texts.append(text)

        opinions = body.get("opinions")
        if isinstance(opinions, list):
            for opinion in opinions:
                if isinstance(opinion, dict):
                    text = opinion.get("text")
                    if isinstance(text, str):
                        texts.append(text)

    combined = "\n".join(t.strip() for t in texts if t)
    if combined:
        return combined

    if isinstance(case.get("casebody"), str):
        return case["casebody"]

    return json.dumps(case, ensure_ascii=False)


def build_normalized_opinion(path: Path, case: Dict, text: str) -> Dict:
    court = case.get("court") or {}
    jurisdiction = case.get("jurisdiction") or {}

    return {
        "source": "CAP",
        "absolute_url": case.get("frontend_url"),
        "cluster_id": case.get("id"),
        "caseName": case.get("name_abbreviation") or case.get("name"),
        "caseNameFull": case.get("name"),
        "decision_date": case.get("decision_date"),
        "citations": case.get("citations"),
        "court": court.get("name"),
        "court_id": court.get("slug"),
        "jurisdiction": jurisdiction.get("name"),
        "jurisdiction_id": jurisdiction.get("slug"),
        "opinions": [
            {
                "type": "opinion",
                "plain_text": text,
            }
        ],
        "_source_file": str(path),
    }


def triage_cap_directory(source: Path, destination: Path) -> Dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    kept = 0
    processed = 0

    for json_path in iter_json_files(source):
        processed += 1
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            cases = payload if isinstance(payload, list) else [payload]
        except Exception as exc:
            logger.warning("Skipping %s (read error: %s)", json_path, exc)
            continue

        for case in cases:
            try:
                text = extract_case_text(case)
            except Exception as exc:
                logger.debug("Skipping %s entry (%s)", json_path, exc)
                continue

            if "1782" not in text:
                continue

            normalized = build_normalized_opinion(json_path, case, text)

            if not is_actual_1782_case(normalized):
                continue

            case_id = case.get("id") or json_path.stem
            outfile = destination / f"cap_{case_id}.json"
            outfile.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
            kept += 1

    return {"processed": processed, "kept": kept}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Harvard CAP bulk downloads for § 1782 opinions.")
    parser.add_argument("--source", required=True, type=Path, help="Directory containing extracted CAP bulk JSON files.")
    parser.add_argument(
        "--destination",
        default=Path("data/case_law/1782_discovery_cap"),
        type=Path,
        help="Destination directory for matching CAP cases.",
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

    if not args.source.exists():
        raise SystemExit(f"Source directory not found: {args.source}")

    logger.info("Scanning CAP directory: %s", args.source)
    stats = triage_cap_directory(args.source, args.destination)
    logger.info("Processed %d JSON files", stats["processed"])
    logger.info("Kept %d potential §1782 opinions", stats["kept"])
    logger.info("Output directory: %s", args.destination)


if __name__ == "__main__":
    main()
