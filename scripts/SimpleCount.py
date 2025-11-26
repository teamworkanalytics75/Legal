#!/usr/bin/env python3
"""
Simple Google Drive File Count

This script counts files in the Google Drive folder.
"""

import sys
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

def count_files():
    """Count files in Google Drive."""

    backup = GoogleDriveBackup()
    folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

    # Get subfolder
    query = f"'{folder_id}' in parents"
    files = backup.service.files().list(q=query, fields='files(id, name, mimeType)').execute()

    subfolder_id = None
    for file in files.get('files', []):
        if file.get('mimeType') == 'application/vnd.google-apps.folder':
            subfolder_id = file['id']
            break

    if subfolder_id:
        # Get files
        query2 = f"'{subfolder_id}' in parents"
        files = backup.service.files().list(q=query2, fields='files(id, name)').execute()
        file_count = len(files.get('files', []))
        print(f"Files in subfolder: {file_count}")

        # Show first 5 files
        for i, file in enumerate(files.get('files', [])[:5]):
            print(f"{i+1}. {file['name']}")
    else:
        print("No subfolder found")

if __name__ == "__main__":
    count_files()


