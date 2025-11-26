#!/usr/bin/env python3
"""Debug script to see what's actually in the lawsuit documents."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.query_lawsuit_db import search_documents

def debug_documents():
    """Show actual content from lawsuit documents."""

    # Search for McGrath specifically
    results = search_documents("McGrath", limit=3)

    print(f"Found {len(results)} McGrath documents:")
    for i, (rowid, preview, doc_length) in enumerate(results):
        print(f"\n--- Document {i+1} (ID: {rowid}, Length: {doc_length}) ---")
        print(preview)
        print("---")

if __name__ == "__main__":
    debug_documents()
