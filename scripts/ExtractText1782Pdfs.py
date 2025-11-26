#!/usr/bin/env python3
"""
Lightweight text extraction pipeline for the 1782 RECAP PDF corpus.

This script walks the `data/case_law/1782_recap_api_pdfs` directory,
attempts direct text extraction, and saves plain-text outputs in
`data/case_law/1782_text`. Files that still lack usable text are logged
for follow-up OCR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import PyPDF2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PDF_DIR = Path("data/case_law/1782_recap_api_pdfs")
TEXT_DIR = Path("data/case_law/1782_text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def extract_with_pypdf2(pdf_path: Path) -> str:
    """Return concatenated text using PyPDF2 (direct text extraction)."""
    text_parts: List[str] = []
    try:
        with pdf_path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        logger.debug("PyPDF2 failed for %s: %s", pdf_path.name, exc)
    return "\n".join(text_parts).strip()


def extract_with_pymupdf(pdf_path: Path) -> str:
    """Return concatenated text using PyMuPDF's native extractor."""
    text_parts: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception as exc:
        logger.debug("PyMuPDF failed for %s: %s", pdf_path.name, exc)
    return "\n".join(text_parts).strip()


def extract_text(pdf_path: Path) -> Dict[str, Optional[str]]:
    """
    Attempt direct extraction via PyPDF2, fall back to PyMuPDF.
    Return dictionary containing text and which extractor succeeded.
    """
    text = extract_with_pypdf2(pdf_path)
    method = "pypdf2"

    if len(text) < 200:
        alt_text = extract_with_pymupdf(pdf_path)
        if len(alt_text) > len(text):
            text = alt_text
            method = "pymupdf"

    return {"text": text if text else None, "method": method}


def main() -> None:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in %s", PDF_DIR)
        return

    summary = {
        "total_pdfs": len(pdf_files),
        "text_saved": 0,
        "needs_ocr": 0,
        "failed": 0,
        "details": [],
    }

    for pdf_path in pdf_files:
        result = extract_text(pdf_path)
        text = result["text"]

        if text and len(text.strip()) >= 200:
            txt_path = TEXT_DIR / f"{pdf_path.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            summary["text_saved"] += 1
            summary["details"].append(
                {
                    "pdf": pdf_path.name,
                    "text_file": txt_path.name,
                    "chars": len(text),
                    "method": result["method"],
                }
            )
        elif text:
            # Short text available; store it but mark for review.
            txt_path = TEXT_DIR / f"{pdf_path.stem}.txt"
            txt_path.write_text(text, encoding="utf-8")
            summary["needs_ocr"] += 1
            summary["details"].append(
                {
                    "pdf": pdf_path.name,
                    "text_file": txt_path.name,
                    "chars": len(text),
                    "method": result["method"],
                    "note": "short_text_possible_scan",
                }
            )
        else:
            summary["needs_ocr"] += 1
            summary["details"].append(
                {
                    "pdf": pdf_path.name,
                    "text_file": None,
                    "chars": 0,
                    "method": result["method"],
                    "note": "no_text_extracted",
                }
            )

    summary_path = TEXT_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Extraction complete: %s saved (%s require OCR). Summary at %s",
        summary["text_saved"],
        summary["needs_ocr"],
        summary_path,
    )


if __name__ == "__main__":
    main()
