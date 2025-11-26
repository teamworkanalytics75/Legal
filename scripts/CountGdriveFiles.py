#!/usr/bin/env python3
"""
Count Google Drive Files

This script counts all files in the Google Drive folder to get the exact total.
"""

import sys
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

def count_gdrive_files():
    """Count all files in the Google Drive folder."""

    # Initialize Google Drive backup
    backup = GoogleDriveBackup()

    # Get the folder ID for 1782_discovery
    folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

    print(f"Counting files in Google Drive folder: {folder_id}")

    try:
        # First, find the subfolder
        files = backup.service.files().list(
            q=f"'{folder_id}' in parents",
            fields='files(id, name, size, mimeType)'
        ).execute()

        file_list = files.get('files', [])

        # Find the subfolder
        subfolder_id = None
        for file in file_list:
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                subfolder_id = file['id']
                print(f"Found subfolder: {file['name']} (ID: {subfolder_id})")
                break

        if not subfolder_id:
            print("No subfolder found")
            return 0

        # Count all files in the subfolder
        all_files = []
        page_token = None

        while True:
            query = f"'{subfolder_id}' in parents"
            if page_token:
                query += f" and pageToken='{page_token}'"

            files = backup.service.files().list(
                q=query,
                fields='files(id, name, size), nextPageToken'
            ).execute()

            all_files.extend(files.get('files', []))
            page_token = files.get('nextPageToken')

            if not page_token:
                break

        print(f"Total files found: {len(all_files)}")

        # Show first 10 files
        print("\nFirst 10 files:")
        for i, file in enumerate(all_files[:10]):
            print(f"{i+1}. {file['name']} ({file.get('size', 'unknown')} bytes)")

        if len(all_files) > 10:
            print(f"... and {len(all_files) - 10} more files")

        return len(all_files)

    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        return 0

if __name__ == "__main__":
    count = count_gdrive_files()
    print(f"\nFinal count: {count} files")


