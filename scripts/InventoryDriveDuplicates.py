#!/usr/bin/env python3
"""
Inventory Google Drive Duplicates

This script lists all files in the Drive folder and identifies duplicates
by grouping files with the same name.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Google Drive backup module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "google_drive_backup",
    Path(__file__).parent.parent / "document_ingestion" / "google_drive_backup.py"
)
gdrive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdrive_module)
GoogleDriveBackup = gdrive_module.GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def inventory_drive_duplicates():
    """Inventory all files in Drive folder and identify duplicates."""

    print("="*80)
    print("🔍 PHASE 1: INVENTORYING GOOGLE DRIVE DUPLICATES")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Initialize Google Drive backup
        print("🔐 Authenticating with Google Drive...")
        backup = GoogleDriveBackup()

        # Target folder ID
        folder_id = "1uQ9-7Y-iTkO7KYZ7yCkLp_9IzZljGbIb"
        print(f"📁 Scanning folder: {folder_id}")
        print()

        # List all files in folder
        print("📋 Fetching file list from Drive...")
        results = backup.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType, size, modifiedTime)",
            orderBy="name"
        ).execute()

        files = results.get('files', [])
        print(f"✅ Found {len(files)} total files")
        print()

        # Group files by name
        print("🔍 Grouping files by name...")
        by_name = defaultdict(list)

        for file_info in files:
            name = file_info['name']
            by_name[name].append({
                'id': file_info['id'],
                'name': name,
                'size': file_info.get('size', 'Unknown'),
                'modified': file_info.get('modifiedTime', 'Unknown')
            })

        # Analyze duplicates
        print("📊 Analyzing duplicates...")
        total_files = len(files)
        unique_names = len(by_name)
        duplicate_groups = {name: files for name, files in by_name.items() if len(files) > 1}

        print()
        print("="*60)
        print("📈 INVENTORY RESULTS")
        print("="*60)
        print(f"Total files in Drive: {total_files}")
        print(f"Unique filenames: {unique_names}")
        print(f"Duplicate groups: {len(duplicate_groups)}")
        print(f"Files to delete: {total_files - unique_names}")
        print()

        # Show duplicate groups
        if duplicate_groups:
            print("🔍 DUPLICATE GROUPS FOUND:")
            print("-" * 60)
            for name, file_list in duplicate_groups.items():
                print(f"📄 {name} ({len(file_list)} copies):")
                for i, file_info in enumerate(file_list, 1):
                    status = "🆕 KEEP (newest)" if i == len(file_list) else "🗑️ DELETE"
                    print(f"   {i}. {file_info['id']} - {file_info['modified']} {status}")
                print()

        # Summary table
        print("="*60)
        print("📋 SUMMARY TABLE")
        print("="*60)
        print(f"{'Metric':<25} {'Count':<10} {'Status'}")
        print("-" * 50)
        print(f"{'Total files':<25} {total_files:<10} {'Current'}")
        print(f"{'Unique names':<25} {unique_names:<10} {'Target'}")
        print(f"{'Duplicates':<25} {total_files - unique_names:<10} {'To delete'}")
        print(f"{'Final count':<25} {unique_names:<10} {'After cleanup'}")
        print()

        # Save inventory to file
        inventory_data = {
            'timestamp': datetime.now().isoformat(),
            'folder_id': folder_id,
            'total_files': total_files,
            'unique_names': unique_names,
            'duplicate_groups': len(duplicate_groups),
            'files_to_delete': total_files - unique_names,
            'duplicates': dict(duplicate_groups)
        }

        inventory_file = Path("data/case_law/drive_inventory.json")
        inventory_file.parent.mkdir(parents=True, exist_ok=True)

        with open(inventory_file, 'w', encoding='utf-8') as f:
            json.dump(inventory_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Inventory saved to: {inventory_file}")
        print()
        print("✅ PHASE 1 COMPLETE: Drive inventory finished")
        print("="*80)

        return inventory_data

    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.error(f"Error during inventory: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    inventory_drive_duplicates()
