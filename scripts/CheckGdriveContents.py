#!/usr/bin/env python3
"""
Check Google Drive Folder Contents

This script checks what's actually in the Google Drive folder and lists all files.
"""

import sys
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

def check_gdrive_contents():
    """Check what's actually in the Google Drive folder."""

    # Initialize Google Drive backup
    backup = GoogleDriveBackup()

    # Get the folder ID for 1782_discovery
    folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

    print(f"Checking Google Drive folder: {folder_id}")

    try:
        # List all files in the folder
        files = backup.service.files().list(
            q=f"'{folder_id}' in parents",
            fields='files(id, name, size, mimeType)'
        ).execute()

        file_list = files.get('files', [])
        print(f"Found {len(file_list)} items in Google Drive folder")

        for i, file in enumerate(file_list):
            print(f"{i+1}. {file['name']} ({file.get('size', 'unknown')} bytes, {file.get('mimeType', 'unknown')})")

        # If there are subfolders, check them too
        for file in file_list:
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                print(f"\nChecking subfolder: {file['name']}")
                subfolder_files = backup.service.files().list(
                    q=f"'{file['id']}' in parents",
                    fields='files(id, name, size, mimeType)'
                ).execute()

                subfolder_list = subfolder_files.get('files', [])
                print(f"Found {len(subfolder_list)} items in subfolder")

                for j, subfile in enumerate(subfolder_list[:10]):  # Show first 10
                    print(f"  {j+1}. {subfile['name']} ({subfile.get('size', 'unknown')} bytes)")

                if len(subfolder_list) > 10:
                    print(f"  ... and {len(subfolder_list) - 10} more files")

    except Exception as e:
        print(f"Error accessing Google Drive: {e}")

if __name__ == "__main__":
    check_gdrive_contents()


