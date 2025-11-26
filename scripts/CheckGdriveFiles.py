#!/usr/bin/env python3
"""
Check Google Drive Files for 1782 Cases

This script checks the files in Google Drive and verifies they're actually about 1782 discovery.
"""

import sys
import json
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

def check_gdrive_files():
    """Check files in Google Drive and verify they're about 1782 discovery."""

    # Initialize Google Drive backup
    backup = GoogleDriveBackup()

    # Get the folder ID for 1782_discovery
    folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

    print('=== Checking Google Drive Files ===')
    print('Folder ID:', folder_id)

    try:
        # List files in the folder
        query = f"'{folder_id}' in parents"
        files = backup.service.files().list(
            q=query,
            fields='files(id, name, size)'
        ).execute()

        file_list = files.get('files', [])
        print('Total files found:', len(file_list))

        # Show first 20 files
        print('\nFirst 20 files:')
        for i, file in enumerate(file_list[:20]):
            print(f'{i+1}. {file["name"]} ({file.get("size", "unknown")} bytes)')

        if len(file_list) > 20:
            print(f'... and {len(file_list) - 20} more files')

        # Check a few files for content
        print('\n=== Checking File Contents ===')
        sample_files = file_list[:5]  # Check first 5 files

        for i, file in enumerate(sample_files):
            print(f'\n--- File {i+1}: {file["name"]} ---')

            try:
                # Download file content
                content = backup.service.files().get_media(fileId=file['id']).execute()

                # Try to parse as JSON
                try:
                    case_data = json.loads(content.decode('utf-8'))
                    case_name = case_data.get('caseName', 'Unknown')
                    court = case_data.get('court', 'Unknown')

                    print(f'Case Name: {case_name}')
                    print(f'Court: {court}')

                    # Check if it's actually about 1782
                    case_text = case_data.get('plain_text', '') or case_data.get('html_with_citations', '')
                    if '1782' in case_text.lower() or '28 u.s.c. 1782' in case_text.lower():
                        print('✓ Contains 1782 references')
                    else:
                        print('✗ No 1782 references found')

                except json.JSONDecodeError:
                    print('Not a JSON file')

            except Exception as e:
                print(f'Error reading file: {e}')

    except Exception as e:
        print(f'Error accessing Google Drive: {e}')

if __name__ == "__main__":
    check_gdrive_files()


