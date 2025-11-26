#!/usr/bin/env python3
"""
Move Canonical Copies to Clean Database

This script moves exactly one vetted copy per unique hash from the cluttered
1782_discovery folder to the clean "The Art of War - Database" folder.
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

def move_canonical_copies():
    """Move one canonical copy per hash to clean database."""

    print("="*80)
    print("MOVING CANONICAL COPIES TO CLEAN DATABASE")
    print("="*80)

    source = Path("data/case_law/1782_discovery")
    target = Path("data/case_law/The Art of War - Database")

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target}")

    # Group files by hash
    buckets = defaultdict(list)
    files_without_hash = []

    for path in source.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            case_hash = data.get("_matrix_case_hash")
            if case_hash:
                buckets[case_hash].append(path)
            else:
                files_without_hash.append(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    print(f"Found {len(buckets)} unique hash groups")
    print(f"Found {len(files_without_hash)} files without hash")

    # Move one canonical copy per hash group
    moved_count = 0
    keep_prefixes = ("case_",)  # Prefer case_XX_*.json files

    for case_hash, files in buckets.items():
        print(f"\nProcessing hash: {case_hash[:12]}... ({len(files)} copies)")

        # Find the best canonical copy
        chosen = None

        # First, try to find a case_XX_*.json file
        for path in files:
            if path.name.startswith(keep_prefixes):
                chosen = path
                print(f"  Selected: {path.name} (preferred naming)")
                break

        # If no case_XX_ file found, skip search_*, unknown_*, and subdirectory files
        if not chosen:
            for path in files:
                if (not path.name.startswith(('search_', 'unknown_')) and
                    path.parent.name not in ('cap_test', 'landmark_cases', 'phase1_binding', 'phase2_post_zf') and
                    path.name not in ('1782_case_count_research_report.json', 'drive_verification_results.json')):
                    chosen = path
                    print(f"  Selected: {path.name} (clean copy)")
                    break

        # If still no good choice, use the first file
        if not chosen:
            chosen = files[0]
            print(f"  Selected: {chosen.name} (fallback)")

        # Move the chosen file
        target_file = target / chosen.name
        shutil.move(str(chosen), str(target_file))
        moved_count += 1

        # Also move matching TXT file if it exists
        txt_file = chosen.with_suffix('.txt')
        if txt_file.exists():
            target_txt = target / txt_file.name
            shutil.move(str(txt_file), str(target_txt))
            print(f"  Also moved: {txt_file.name}")

    print(f"\n" + "="*80)
    print("CANONICAL COPY MOVEMENT COMPLETE")
    print("="*80)
    print(f"Files moved to canonical database: {moved_count}")
    print(f"Canonical directory: {target}")

    return moved_count

if __name__ == "__main__":
    move_canonical_copies()
