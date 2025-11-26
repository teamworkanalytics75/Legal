#!/usr/bin/env python3
"""
Upload a local directory of files to a specific Google Drive folder.

Usage:
    python upload_directory_to_drive.py <local_dir> <drive_folder_id>

The script uploads JSON/TXT files, replacing existing Drive files
that share the same name (to avoid duplicates).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "google_drive_backup",
    Path(__file__).parent.parent / "document_ingestion" / "google_drive_backup.py"
)
gdrive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdrive_module)
GoogleDriveBackup = gdrive_module.GoogleDriveBackup

from googleapiclient.http import MediaFileUpload


def ensure_args() -> tuple[Path, str]:
    if len(sys.argv) != 3:
        print("Usage: python upload_directory_to_drive.py <local_dir> <drive_folder_id>")
        sys.exit(1)

    local_dir = Path(sys.argv[1]).resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        print(f"Local directory not found: {local_dir}")
        sys.exit(1)

    folder_id = sys.argv[2]
    return local_dir, folder_id


def main() -> None:
    local_dir, folder_id = ensure_args()
    gdrive = GoogleDriveBackup()

    files = [p for p in local_dir.glob("*") if p.is_file() and p.suffix.lower() in {".json", ".txt"}]
    if not files:
        print(f"No JSON/TXT files found in {local_dir}")
        return

    print(f"Uploading {len(files)} files from {local_dir} to Drive folder {folder_id} …")

    for path in files:
        drive_name = path.name

        # Delete existing file with same name (if any) to prevent duplicates.
        search = gdrive.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false and name='{drive_name}'",
            spaces='drive',
            fields='files(id)'
        ).execute()

        for existing in search.get('files', []):
            gdrive.service.files().delete(fileId=existing['id']).execute()

        media = MediaFileUpload(str(path), resumable=False)
        gdrive.service.files().create(
            body={'name': drive_name, 'parents': [folder_id]},
            media_body=media,
            fields='id'
        ).execute()

        print(f"  ✓ {drive_name}")
        time.sleep(0.2)  # gentle rate limiting

    print("Upload complete.")


if __name__ == "__main__":
    main()
