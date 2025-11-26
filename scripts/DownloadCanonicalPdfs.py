#!/usr/bin/env python3
"""Download PDFs for all canonical §1782 cases with CourtListener fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

CANONICAL_DIR = Path("data/case_law/The Art of War - Database")
OUTPUT_DIR = Path("data/case_law/pdfs/canonical_all")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CL = "https://www.courtlistener.com"
STORAGE_BASE = "https://storage.courtlistener.com/"
HEADERS = {"User-Agent": "TheMatrixLegalAI/2.0"}

SKIP_FILES = {"drive_verification_results.json"}

def pick_pdf_url(data: dict) -> Optional[str]:
    for opinion in data.get("opinions") or []:
        url = opinion.get("download_url")
        if url:
            return url
    top_level = data.get("download_url")
    if top_level:
        return top_level
    for opinion in data.get("opinions") or []:
        local_path = opinion.get("local_path")
        if local_path:
            return urljoin(STORAGE_BASE, local_path)
    local_path = data.get("local_path")
    if local_path:
        return urljoin(STORAGE_BASE, local_path)
    absolute_url = data.get("absolute_url")
    if absolute_url:
        return urljoin(BASE_CL, absolute_url)
    return None


def download_pdf(name: str, url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=90)
        resp.raise_for_status()
        if resp.headers.get("Content-Type", "").startswith("text/html") and not url.lower().endswith(".pdf"):
            raise ValueError("Received HTML instead of PDF")
        dest.write_bytes(resp.content)
        print(f"[ok]   {dest.name}")
        return True
    except Exception as exc:
        print(f"[err]  {dest.name}: {exc}")
        return False


def main() -> None:
    json_files = sorted(CANONICAL_DIR.glob("*.json"))
    successes = 0
    failures = []

    for path in json_files:
        if path.name in SKIP_FILES:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        pdf_url = pick_pdf_url(data)
        if not pdf_url:
            print(f"[warn] {path.name} has no usable PDF URL")
            failures.append(path.name)
            continue

        pdf_path = OUTPUT_DIR / (path.stem + ".pdf")
        if download_pdf(path.name, pdf_url, pdf_path):
            successes += 1
        else:
            failures.append(path.name)

    print("=" * 60)
    print(f"PDF download complete -> saved: {successes}, failed: {len(failures)}")
    if failures:
        print("Failures:")
        for name in failures:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
