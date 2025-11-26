#!/usr/bin/env python3
"""
Publish a DOCX as a native Google Doc and log the result.

Usage:
  python scripts/PublishMotionToDocs.py \
      --path reports/motion_to_seal/motion_to_seal_v2_clean.docx \
      --name "Motion to Seal — Master" \
      --folder "Motion_to_Seal_Master"

Writes publish log to reports/motion_to_seal/publish_log.json and prints the Doc URL.
Credentials: place OAuth JSON at document_ingestion/credentials.json (not committed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import sys

from document_ingestion.GoogleDriveBackup import GoogleDriveBackup


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def publish_docx(
    docx_path: Path,
    doc_name: str,
    folder: Optional[str] = None,
    credentials_path: Path = Path("document_ingestion/credentials.json"),
    log_path: Path = Path("reports/motion_to_seal/publish_log.json"),
) -> int:
    gdrive = GoogleDriveBackup(credentials_path=str(credentials_path))

    # Choose or create destination folder
    if folder:
        folder_id = gdrive.create_backup_folder(folder)
    else:
        folder_id = gdrive.create_backup_folder()

    file_id = gdrive.upload_docx_as_google_doc(docx_path, doc_name, parent_id=folder_id)
    if not file_id:
        print("ERROR: Upload/convert failed", file=sys.stderr)
        return 2

    # Retrieve a shareable link (webViewLink)
    url: Optional[str] = None
    try:
        meta = gdrive.service.files().get(fileId=file_id, fields='webViewLink').execute()
        url = meta.get('webViewLink')
    except Exception:
        url = None

    # Write publish log
    ensure_dir(log_path)
    payload = {
        "fileId": file_id,
        "url": url,
        "doc_name": doc_name,
        "folder": folder or "The Matrix_CaseLaw_Backup",
        "source": str(docx_path),
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Print URL for convenience
    if url:
        print(url)
    else:
        print(file_id)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish DOCX as native Google Doc")
    parser.add_argument("--path", required=True, help="Path to .docx file")
    parser.add_argument("--name", required=True, help="Google Doc name")
    parser.add_argument("--folder", help="Target Drive folder name (optional)")
    parser.add_argument(
        "--log",
        help="Output publish log path (default: reports/motion_to_seal/publish_log.json)",
    )
    parser.add_argument(
        "--creds",
        help="Path to OAuth credentials.json (default: document_ingestion/credentials.json)",
    )
    args = parser.parse_args()

    docx_path = Path(args.path)
    if not docx_path.exists():
        print(f"ERROR: File not found: {docx_path}", file=sys.stderr)
        return 2

    log_path = Path(args.log) if args.log else Path("reports/motion_to_seal/publish_log.json")
    creds_path = Path(args.creds) if args.creds else Path("document_ingestion/credentials.json")

    return publish_docx(docx_path, args.name, args.folder, creds_path, log_path)


if __name__ == "__main__":
    raise SystemExit(main())

