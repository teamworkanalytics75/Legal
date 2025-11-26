#!/usr/bin/env python3
"""
Download all available RECAP PDFs for 28 U.S.C. § 1782 matters using the CourtListener REST API.

The script queries the search endpoint with `type=r` (RECAP documents), walks the pagination
cursor, and downloads every available `recap_documents` attachment that matches the query.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

CONFIG_PATH = Path("document_ingestion/courtlistener_config.json")
RECAP_BASE_URL = "https://www.courtlistener.com/"


def load_config(path: Path) -> Dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file {path} not found.") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Configuration file {path} is not valid JSON: {exc}") from exc


def sanitize(text: str, fallback: str) -> str:
    if not text:
        return fallback
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_", " "))
    cleaned = "_".join(cleaned.strip().split())
    return cleaned or fallback


class CourtListenerRecapDownloader:
    def __init__(
        self,
        api_token: str,
        user_agent: str,
        output_dir: Path,
        delay: float,
        logger: logging.Logger,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {api_token}",
                "User-Agent": user_agent,
            }
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = max(delay, 0.0)
        self.logger = logger

        self._downloaded_ids: Set[int] = set()
        self._downloaded_paths: Set[str] = set()

    def _rate_limit(self) -> None:
        if self.delay:
            time.sleep(self.delay)

    def fetch_recap_results(
        self,
        query: str,
        page_size: int,
        max_results: Optional[int],
        max_pages: Optional[int],
    ) -> Iterable[Dict[str, object]]:
        params = {
            "q": query,
            "type": "r",
            "order_by": "dateFiled desc",
            "page_size": page_size,
        }
        url = urljoin(RECAP_BASE_URL, "api/rest/v4/search/")
        pages_seen = 0
        results_yielded = 0

        while url:
            if max_pages is not None and pages_seen >= max_pages:
                break

            self._rate_limit()
            response = self.session.get(url, params=params, timeout=60)
            params = None  # Only apply params to first request
            if response.status_code != 200:
                raise SystemExit(
                    f"Search request failed ({response.status_code}): {response.text[:200]}"
                )

            payload = response.json()
            results: List[Dict[str, object]] = payload.get("results", [])
            for result in results:
                yield result
                results_yielded += 1
                if max_results is not None and results_yielded >= max_results:
                    return

            url = payload.get("next")
            pages_seen += 1

    def iter_recap_documents(
        self,
        results: Iterable[Dict[str, object]],
    ) -> Iterable[Tuple[Dict[str, object], Dict[str, object]]]:
        for result in results:
            recap_docs: List[Dict[str, object]] = result.get("recap_documents") or []
            if not recap_docs:
                continue
            for doc in recap_docs:
                yield result, doc

    def download_document(
        self,
        docket_meta: Dict[str, object],
        doc_meta: Dict[str, object],
        overwrite: bool,
    ) -> Optional[Path]:
        doc_id = doc_meta.get("id")
        if doc_id is None or doc_id in self._downloaded_ids:
            return None

        if not doc_meta.get("is_available", False):
            self.logger.debug("Skipping unavailable document %s", doc_id)
            return None

        relative_path = doc_meta.get("filepath_local")
        if not relative_path:
            self.logger.debug("Skipping document %s with missing filepath", doc_id)
            return None

        pdf_url = urljoin(RECAP_BASE_URL, relative_path)

        docket_number = sanitize(str(docket_meta.get("docketNumber", "unknown")), "unknown")
        doc_number = str(doc_meta.get("document_number") or doc_meta.get("entry_number") or "na")
        short_desc = sanitize(doc_meta.get("short_description", "") or "", "document")
        attachment_suffix = ""
        attachment_number = doc_meta.get("attachment_number")
        if attachment_number not in (None, "", 0):
            attachment_suffix = f"_att{attachment_number}"

        filename = f"{docket_number}_doc{doc_number}{attachment_suffix}_{short_desc}.pdf"
        destination = self.output_dir / filename

        if destination.exists() and not overwrite:
            self.logger.debug("Skipping existing file %s", destination.name)
            self._downloaded_ids.add(doc_id)
            self._downloaded_paths.add(str(destination))
            return destination

        self._rate_limit()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Referer": urljoin(RECAP_BASE_URL, docket_meta.get("absolute_url", "/")),
        }
        response = self.session.get(pdf_url, headers=headers, timeout=120)
        if response.status_code != 200 or "pdf" not in response.headers.get("Content-Type", "").lower():
            self.logger.warning(
                "Failed to download %s (status %s)", pdf_url, response.status_code
            )
            return None

        destination.write_bytes(response.content)
        self._downloaded_ids.add(doc_id)
        self._downloaded_paths.add(str(destination))
        self.logger.info("Downloaded %s", destination.name)
        return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all 28 U.S.C. § 1782 RECAP PDFs from CourtListener."
    )
    parser.add_argument(
        "--query",
        default='"28 usc 1782"',
        help="Search query to submit to CourtListener (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/case_law/1782_recap_api_pdfs",
        help="Directory where PDFs will be stored (default: %(default)s).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of search results per API page (default: %(default)s).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Optional cap on total search results to process.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on the number of API pages to read.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay (seconds) between HTTP requests (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite PDFs that already exist locally.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("courtlistener_recap")
    logger.setLevel(getattr(logging, level))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    api_config: Dict[str, object] = config.get("api") or {}
    api_token = api_config.get("api_token")
    user_agent = api_config.get("user_agent", "CourtListenerRecapDownloader/1.0")
    if not api_token:
        raise SystemExit("API token missing from configuration file.")

    logger = setup_logger(args.log_level)
    output_dir = Path(args.output_dir)

    downloader = CourtListenerRecapDownloader(
        api_token=api_token,
        user_agent=user_agent,
        output_dir=output_dir,
        delay=args.delay,
        logger=logger,
    )

    logger.info("Starting RECAP download with query %s", args.query)
    results_iter = downloader.fetch_recap_results(
        query=args.query,
        page_size=args.page_size,
        max_results=args.max_results,
        max_pages=args.max_pages,
    )
    total_seen = 0
    total_saved = 0

    for docket_meta, doc_meta in downloader.iter_recap_documents(results_iter):
        total_seen += 1
        path = downloader.download_document(
            docket_meta=docket_meta,
            doc_meta=doc_meta,
            overwrite=args.overwrite,
        )
        if path is not None:
            total_saved += 1

    logger.info("Processed %s documents; downloaded %s new files.", total_seen, total_saved)


if __name__ == "__main__":
    main()
