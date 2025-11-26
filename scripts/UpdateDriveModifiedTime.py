#!/usr/bin/env python3
"""
Refresh Google Drive file timestamps for the §1782 corpus.

This script iterates over the canonical local directory and re-uploads the
matching JSON files to their corresponding Google Drive objects using
`files().update`. The content remains unchanged, but Drive records a fresh
`modifiedTime`, so the folder appears recently updated without creating new
duplicates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure we can import the shared Google Drive helper.
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


def refresh_modified_times(
    local_dir: Path,
    folder_id: str,
    sleep_seconds: float = 0.3
) -> None:
    """Re-upload Drive files so their modified time reflects the current run."""

    if not local_dir.exists():
        print(f"⚠️  Local directory not found: {local_dir}")
        return

    print("=" * 80)
    print("🔄 REFRESHING GOOGLE DRIVE FILE TIMESTAMPS")
    print("=" * 80)

    gdrive = GoogleDriveBackup()

    print("🌐 Fetching Drive file list…")
    response = gdrive.service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
        pageSize=500
    ).execute()

    drive_files = {file["name"]: file["id"] for file in response.get("files", [])}
    print(f"📁 Drive files discovered: {len(drive_files)}")

    local_files = sorted(local_dir.glob("*.json"))
    print(f"💾 Local JSON files: {len(local_files)}\n")

    matched = 0
    missing = []

    for local_path in local_files:
        name = local_path.name
        file_id = drive_files.get(name)

        if not file_id:
            missing.append(name)
            continue

        print(f"   🔁 Updating {name} …")
        media = MediaFileUpload(
            filename=str(local_path),
            mimetype="application/json",
            resumable=False
        )

        gdrive.service.files().update(
            fileId=file_id,
            media_body=media,
            body={"name": name}
        ).execute()

        matched += 1
        time.sleep(sleep_seconds)

    print("\n🧾 SUMMARY")
    print("-" * 40)
    print(f"Files updated: {matched}")
    print(f"Local files missing on Drive: {len(missing)}")

    if missing:
        print("\n⚠️  The following files were not found on Drive:")
        for name in missing:
            print(f"   • {name}")

    print("\n✅ Timestamp refresh complete.")
    print("=" * 80)


if __name__ == "__main__":
    canonical_dir = Path("data/case_law/The Art of War - Database")
    drive_folder_id = "1uQ9-7Y-iTkO7KYZ7yCkLp_9IzZljGbIb"

    refresh_modified_times(canonical_dir, drive_folder_id)
