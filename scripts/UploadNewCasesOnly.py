from __future__ import annotations

import sys
import time
import json
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


def load_canonical_names() -> set[str]:
    canonical_dir = Path("data/case_law/The Art of War - Database")
    return {p.name for p in canonical_dir.glob("*.json")}


def load_baseline_list(baseline_path: Path | None) -> set[str]:
    names: set[str] = set()
    if baseline_path and baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            names.update(data)
    return names


def upload_new_cases(folder_id: str, baseline_path: Path | None = None) -> None:
    canonical_names = load_canonical_names()
    baseline_names = load_baseline_list(baseline_path)
    skip_names = canonical_names | baseline_names

    source_dir = Path("data/case_law/1782_discovery")
    files_to_upload = [
        p for p in source_dir.glob("*.json") if p.name not in skip_names
    ]

    if not files_to_upload:
        print("No new cases found to upload.")
        return

    print(f"Uploading {len(files_to_upload)} new JSON files to folder {folder_id} …")
    gdrive = GoogleDriveBackup()

    for json_path in sorted(files_to_upload):
        name = json_path.name

        search = gdrive.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false and name='{name}'",
            spaces='drive',
            fields='files(id)'
        ).execute()
        for existing in search.get("files", []):
            gdrive.service.files().delete(fileId=existing["id"]).execute()

        media = MediaFileUpload(str(json_path), resumable=False)
        gdrive.service.files().create(
            body={'name': name, 'parents': [folder_id]},
            media_body=media,
            fields='id'
        ).execute()

        txt_path = json_path.with_suffix('.txt')
        if txt_path.exists():
            txt_name = txt_path.name
            search_txt = gdrive.service.files().list(
                q=f"'{folder_id}' in parents and trashed=false and name='{txt_name}'",
                spaces='drive',
                fields='files(id)'
            ).execute()
            for existing in search_txt.get("files", []):
                gdrive.service.files().delete(fileId=existing["id"]).execute()

            txt_media = MediaFileUpload(str(txt_path), resumable=False)
            gdrive.service.files().create(
                body={'name': txt_name, 'parents': [folder_id]},
                media_body=txt_media,
                fields='id'
            ).execute()

        print(f"  - {name}")
        time.sleep(0.2)

    print("Upload complete.")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("Usage: python upload_new_cases_only.py <drive_folder_id> [baseline_json]")
        sys.exit(1)

    folder_id = sys.argv[1]
    baseline_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    upload_new_cases(folder_id, baseline_path)


if __name__ == "__main__":
    main()
