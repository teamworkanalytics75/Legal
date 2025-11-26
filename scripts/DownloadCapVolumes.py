#!/usr/bin/env python3
"""
Download Harvard CAP bulk ZIP volumes for a given reporter series.

Example:
    py scripts\\download_cap_volumes.py --reporter f-supp-3d --start 1 --end 20

The ZIPs will be saved under data/case_law/cap_bulk/<reporter>/<volume>.zip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

BASE_URL = "https://static.case.law"
DEFAULT_OUTDIR = Path("data/case_law/cap_bulk")


def download_volume(reporter: str, volume: int, outdir: Path, force: bool = False) -> bool:
    """Download a single CAP volume ZIP."""

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{volume}.zip"

    if outfile.exists() and not force:
        logging.info("Skipping %s (already exists)", outfile)
        return False

    url = f"{BASE_URL}/{reporter}/{volume}.zip"
    logging.info("Downloading %s -> %s", url, outfile)

    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        logging.warning("Failed to download %s (status %s)", url, response.status_code)
        return False

    with outfile.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1_048_576):
            if chunk:
                fh.write(chunk)

    logging.info("Saved %s (%s bytes)", outfile, outfile.stat().st_size)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CAP bulk ZIP volumes.")
    parser.add_argument(
        "--reporter",
        required=True,
        help="Reporter slug (e.g., f-supp, f-supp-2d, f-supp-3d).",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="First volume number to download.",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Last volume number to download (inclusive).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Base directory for downloads (default: data/case_law/cap_bulk).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the ZIP already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    reporter_dir = args.outdir / args.reporter

    downloaded = 0
    for volume in range(args.start, args.end + 1):
        if download_volume(args.reporter, volume, reporter_dir, force=args.force):
            downloaded += 1

    logging.info("Finished. Volumes attempted: %d; newly downloaded: %d", args.end - args.start + 1, downloaded)


if __name__ == "__main__":
    main()
