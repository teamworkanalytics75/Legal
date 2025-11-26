#!/usr/bin/env python3
"""
Remove files from a Google Drive folder if their names match the canonical 1782 set.

Usage:
    python remove_drive_files_matching_canonical.py <drive_folder_id>
"""

from __future__ import annotations

import sys
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


def get_canonical_names() -> set[str]:
    canonical_dir = Path("data/case_law/The Art of War - Database")
    return {p.name for p in canonical_dir.glob("*.json")}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python remove_drive_files_matching_canonical.py <drive_folder_id>")
        sys.exit(1)

    folder_id = sys.argv[1]
    canonical_names = get_canonical_names()

    backup = GoogleDriveBackup()
    service = backup.service

    removed = 0
    for name in sorted(canonical_names):
        query = (
            f"'{folder_id}' in parents and trashed=false and name='{name}'"
        )
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        for file in response.get("files", []):
            service.files().delete(fileId=file["id"]).execute()
            removed += 1
            print(f"Deleted {file['name']} ({file['id']})")

    print(f"Done. Removed {removed} files.")


if __name__ == "__main__":
    main()
