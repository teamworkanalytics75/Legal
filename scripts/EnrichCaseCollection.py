#!/usr/bin/env python3
"""
Enrich a directory of CourtListener opinion JSON files with docket metadata and
RECAP document information so we can build richer analytical datasets
for multiple workflows (e.g., 28 U.S.C. § 1782 discovery, MA motion practice).

Usage example:
    python scripts/enrich_case_collection.py \
        --input case_law_data/ma_precedential \
        --output case_law_data/enriched \
        --download-pdfs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MODULE_ROOT = Path(__file__).resolve().parents[1]
HARVESTER_PATH = MODULE_ROOT / "document_ingestion" / "case_harvester.py"
_spec = importlib.util.spec_from_file_location("case_harvester", HARVESTER_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive guard
    raise ImportError(f"Cannot load case_harvester module from {HARVESTER_PATH}")
case_harvester_module = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = case_harvester_module
_spec.loader.exec_module(case_harvester_module)

CourtListenerCaseHarvester = case_harvester_module.CourtListenerCaseHarvester  # type: ignore[attr-defined]

logger = logging.getLogger("case_enricher")


@dataclass
class HarvestStats:
    processed: int = 0
    skipped_no_docket: int = 0
    skipped_errors: int = 0
    docket_metadata_saved: int = 0
    recap_documents_saved: int = 0
    recap_documents_downloaded: int = 0
    topics: Dict[str, int] = field(default_factory=dict)

    def register_topic(self, topic: str) -> None:
        self.topics[topic] = self.topics.get(topic, 0) + 1


def iter_opinion_files(root: Path) -> Iterable[Path]:
    """Yield JSON files that look like opinion metadata."""
    for path in root.rglob("*.json"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def normalise_topic(root: Path, opinion_path: Path) -> str:
    try:
        relative = opinion_path.parent.relative_to(root)
    except ValueError:
        return "unknown_topic"
    if not relative.parts:
        return "root"
    return "/".join(relative.parts)


def build_case_identifier(metadata: Dict[str, object], opinion_path: Path) -> str:
    cluster_id = metadata.get("cluster_id")
    if cluster_id:
        return f"cluster_{cluster_id}"
    docket_id = metadata.get("docket_id")
    if docket_id:
        return f"docket_{docket_id}"
    return opinion_path.stem


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def harvest_case(
    *,
    opinion_path: Path,
    topic_root: Path,
    output_root: Path,
    harvester: CourtListenerCaseHarvester,
    download_pdfs: bool,
    overwrite: bool,
) -> Optional[Dict[str, object]]:
    """Fetch docket metadata + RECAP artefacts for a single opinion JSON file."""
    try:
        metadata = json.loads(opinion_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to read %s: %s", opinion_path, exc)
        return None

    docket_id = metadata.get("docket_id")
    if not docket_id:
        return {"status": "skipped_no_docket"}

    topic = normalise_topic(topic_root, opinion_path)
    case_identifier = build_case_identifier(metadata, opinion_path)
    case_dir = output_root / topic / case_identifier

    # Always keep a pointer back to the original data.
    opinion_manifest = {
        "source_path": str(opinion_path),
        "metadata": metadata,
    }
    save_json(opinion_manifest, case_dir / "opinion_metadata.json")

    result: Dict[str, object] = {
        "topic": topic,
        "case_identifier": case_identifier,
        "docket_id": docket_id,
        "downloads": [],
    }

    try:
        docket_metadata = harvester.fetch_docket_metadata(int(docket_id))
        save_json(docket_metadata, case_dir / "docket_metadata.json")
        result["docket_metadata_saved"] = True
    except Exception as exc:
        logger.warning(
            "Could not fetch docket metadata for docket_id=%s (%s)", docket_id, exc
        )
        result["docket_metadata_saved"] = False

    try:
        recap_docs = harvester.fetch_recap_documents(int(docket_id))
    except Exception as exc:
        logger.warning("Could not fetch RECAP documents for docket_id=%s (%s)", docket_id, exc)
        recap_docs = []

    serialisable_docs = [doc.metadata for doc in recap_docs]
    save_json(serialisable_docs, case_dir / "recap_documents.json")
    result["recap_documents_count"] = len(serialisable_docs)

    if download_pdfs and recap_docs:
        docs_dir = case_dir / "documents"
        downloaded_files: List[str] = []
        for doc in recap_docs:
            downloaded = harvester.download_recap_document(
                doc,
                destination_dir=docs_dir,
                overwrite=overwrite,
            )
            if downloaded:
                downloaded_files.append(downloaded.name)
        result["downloaded_files"] = downloaded_files
        result["downloaded_count"] = len(downloaded_files)

    return result


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich downloaded CourtListener opinions with docket metadata and RECAP artefacts."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Directory containing opinion JSON files (can be specified multiple times).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where enriched case bundles will be written.",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download public RECAP PDFs when available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDF files even if already present.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Override CourtListener rate-limit delay in seconds.",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    input_roots = [Path(path).resolve() for path in args.inputs]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    harvester = CourtListenerCaseHarvester(rate_limit_delay=args.delay)
    stats = HarvestStats()

    for root in input_roots:
        if not root.exists():
            logger.warning("Input path %s does not exist; skipping.", root)
            continue

        for opinion_path in iter_opinion_files(root):
            stats.processed += 1

            harvest_result = harvest_case(
                opinion_path=opinion_path,
                topic_root=root,
                output_root=output_root,
                harvester=harvester,
                download_pdfs=args.download_pdfs,
                overwrite=args.overwrite,
            )

            if not harvest_result:
                stats.skipped_errors += 1
                continue

            if harvest_result.get("status") == "skipped_no_docket":
                stats.skipped_no_docket += 1
                continue

            topic = harvest_result.get("topic", "unknown_topic")
            stats.register_topic(topic)

            if harvest_result.get("docket_metadata_saved"):
                stats.docket_metadata_saved += 1
            if harvest_result.get("recap_documents_count"):
                stats.recap_documents_saved += 1
            if harvest_result.get("downloaded_count"):
                stats.recap_documents_downloaded += harvest_result["downloaded_count"]  # type: ignore[misc]

    # Persist summary for reproducibility.
    summary = {
        "inputs": [str(path) for path in input_roots],
        "output_root": str(output_root),
        "download_pdfs": args.download_pdfs,
        "stats": {
            "processed": stats.processed,
            "skipped_no_docket": stats.skipped_no_docket,
            "skipped_errors": stats.skipped_errors,
            "docket_metadata_saved": stats.docket_metadata_saved,
            "recap_documents_saved": stats.recap_documents_saved,
            "recap_documents_downloaded": stats.recap_documents_downloaded,
            "topics": stats.topics,
        },
    }
    save_json(summary, output_root / "harvest_summary.json")

    logger.info("Processed %s opinion files", stats.processed)
    logger.info("Dockets enriched: %s", stats.docket_metadata_saved)
    logger.info("RECAP docs captured: %s", stats.recap_documents_saved)
    if args.download_pdfs:
        logger.info("PDFs downloaded: %s", stats.recap_documents_downloaded)
    if stats.skipped_no_docket:
        logger.info("Skipped (no docket id): %s", stats.skipped_no_docket)
    if stats.skipped_errors:
        logger.info("Skipped due to errors: %s", stats.skipped_errors)


if __name__ == "__main__":
    main()
