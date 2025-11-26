#!/usr/bin/env python3
"""
Delete Google Drive Duplicates

This script deletes duplicate files from Drive, keeping only the newest version
of each file based on modifiedTime.
"""

import sys
import json
import logging
import time
from pathlib import Path
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


def delete_drive_duplicates():
    """Delete duplicate files from Drive, keeping newest version."""

    print("="*80)
    print("🗑️ PHASE 1B: DELETING GOOGLE DRIVE DUPLICATES")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Load inventory data
        inventory_file = Path("data/case_law/drive_inventory.json")
        if not inventory_file.exists():
            print("❌ ERROR: No inventory file found. Run inventory_drive_duplicates.py first.")
            return None

        print("📋 Loading inventory data...")
        with open(inventory_file, 'r', encoding='utf-8') as f:
            inventory_data = json.load(f)

        duplicates = inventory_data.get('duplicates', {})
        folder_id = inventory_data.get('folder_id')

        print(f"📁 Target folder: {folder_id}")
        print(f"🔍 Found {len(duplicates)} duplicate groups")
        print()

        # Initialize Google Drive backup
        print("🔐 Authenticating with Google Drive...")
        backup = GoogleDriveBackup()

        # Track deletion progress
        total_deleted = 0
        total_kept = 0
        errors = 0

        print()
        print("🗑️ DELETING DUPLICATES...")
        print("-" * 60)

        for filename, file_list in duplicates.items():
            print(f"📄 Processing: {filename}")

            # Sort by modified time (newest last)
            sorted_files = sorted(file_list, key=lambda x: x.get('modified', ''))

            # Keep the newest (last in sorted list)
            keep_file = sorted_files[-1]
            delete_files = sorted_files[:-1]

            print(f"   🆕 KEEPING: {keep_file['id']} ({keep_file['modified']})")
            total_kept += 1

            # Delete older copies
            for file_info in delete_files:
                try:
                    print(f"   🗑️ DELETING: {file_info['id']} ({file_info['modified']})")
                    backup.service.files().delete(fileId=file_info['id']).execute()
                    total_deleted += 1

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    print(f"   ❌ ERROR deleting {file_info['id']}: {e}")
                    errors += 1

            print()

        # Summary
        print("="*60)
        print("📊 DELETION SUMMARY")
        print("="*60)
        print(f"Files kept: {total_kept}")
        print(f"Files deleted: {total_deleted}")
        print(f"Errors: {errors}")
        print()

        # Verify final state
        print("🔍 Verifying final state...")
        results = backup.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name)"
        ).execute()

        final_files = results.get('files', [])
        final_count = len(final_files)

        print(f"✅ Final file count: {final_count}")

        # Update inventory
        inventory_data['cleanup_completed'] = datetime.now().isoformat()
        inventory_data['files_deleted'] = total_deleted
        inventory_data['files_kept'] = total_kept
        inventory_data['final_count'] = final_count
        inventory_data['errors'] = errors

        with open(inventory_file, 'w', encoding='utf-8') as f:
            json.dump(inventory_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Updated inventory saved to: {inventory_file}")
        print()

        if errors == 0:
            print("✅ PHASE 1B COMPLETE: Drive duplicates cleaned successfully!")
        else:
            print(f"⚠️ PHASE 1B COMPLETE: Drive cleaned with {errors} errors")

        print("="*80)

        return {
            'files_deleted': total_deleted,
            'files_kept': total_kept,
            'final_count': final_count,
            'errors': errors
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.error(f"Error during deletion: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    delete_drive_duplicates()
