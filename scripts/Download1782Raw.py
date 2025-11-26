#!/usr/bin/env python3
"""
Wide-net downloader for the full CourtListener section 1782 universe.

This script relaxes the § 1782 filter and pours every search hit into a separate
`1782_discovery_raw` corpus so we can analyze the noisy pool without polluting
the curated dataset.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import importlib.util

# Add document_ingestion to the import path.
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py",
)
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)
CourtListenerClient = download_module.CourtListenerClient

logger = logging.getLogger(__name__)

# Very broad keyword sets to surface near the entire § 1782 universe.
RAW_KEYWORD_SETS: List[List[str]] = [
    ['"1782"'],
    ['"section 1782"'],
    ['"28 u.s.c. 1782"'],
    ['"judicial assistance"', '"foreign"'],
    ['"letters rogatory"'],
    ['"foreign tribunal"'],
    ['"aid of foreign"', '"discovery"'],
]


def hydrate_raw_corpus(
    client: CourtListenerClient,
    topic: str,
    keyword_sets: Sequence[Sequence[str]],
    target_total: int,
    max_per_query: int,
    date_after: str,
) -> int:
    """
    Download the raw § 1782 search universe into a separate topic directory.

    Args:
        client: CourtListener client instance.
        topic: Topic name to use for checkpointing and storage.
        keyword_sets: Search keyword combinations to iterate through.
        target_total: Desired total count (0 means no explicit target).
        max_per_query: Upper bound per bulk_download call.
        date_after: Optional date limit (YYYY-MM-DD).

    Returns:
        Number of newly downloaded opinions.
    """
    checkpoint_state = client._load_checkpoint(topic)
    existing_hashes = set(checkpoint_state.get("case_hashes", set()))
    existing_total = len(existing_hashes) if existing_hashes else len(checkpoint_state.get("downloaded_ids", []))

    logger.info("Existing raw corpus size for '%s': %d", topic, existing_total)

    total_downloaded = 0

    for index, keywords in enumerate(keyword_sets, start=1):
        if target_total and existing_total + total_downloaded >= target_total:
            break

        if target_total:
            remaining = target_total - (existing_total + total_downloaded)
            if remaining <= 0:
                break
            max_results = min(max_per_query, remaining)
        else:
            max_results = max_per_query

        logger.info("-" * 80)
        logger.info("RAW QUERY %d", index)
        logger.info("Keywords: %s", keywords)
        logger.info("max_results requested: %d", max_results)

        opinions = client.bulk_download(
            topic=topic,
            courts=None,
            keywords=list(keywords),
            date_after=date_after,
            max_results=max_results,
            resume=True,
            apply_filters=False,
        )

        new_count = len(opinions)
        total_downloaded += new_count
        logger.info("New opinions saved this query: %d", new_count)

        if new_count == 0 and client._load_checkpoint(topic).get("last_cursor") is None:
            logger.info("No new results returned and no cursor to resume; moving on.")

    logger.info("=" * 80)
    logger.info("RAW DOWNLOAD COMPLETE")
    logger.info("New opinions downloaded this run: %d", total_downloaded)

    final_state = client._load_checkpoint(topic)
    final_hashes = set(final_state.get("case_hashes", set()))
    final_total = len(final_hashes) if final_hashes else len(final_state.get("downloaded_ids", []))
    logger.info("Total opinions now stored for '%s': %d", topic, final_total)

    return total_downloaded


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the raw CourtListener § 1782 search universe into a separate corpus."
    )
    parser.add_argument("--topic", default="1782_discovery_raw", help="Target topic directory for raw downloads.")
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Desired total opinion count (0 means run each keyword set once without a hard cap).",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=8000,
        help="Maximum opinions to request per keyword query.",
    )
    parser.add_argument(
        "--date-after",
        default="1950-01-01",
        help="Only download opinions filed after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Optional custom keyword set to override the defaults.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    client = CourtListenerClient()

    if args.keywords:
        keyword_sets: Sequence[Sequence[str]] = [args.keywords]
    else:
        keyword_sets = RAW_KEYWORD_SETS

    hydrate_raw_corpus(
        client=client,
        topic=args.topic,
        keyword_sets=keyword_sets,
        target_total=args.target,
        max_per_query=args.max_per_query,
        date_after=args.date_after,
    )


if __name__ == "__main__":
    main()
