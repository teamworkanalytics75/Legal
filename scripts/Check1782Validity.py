#!/usr/bin/env python3
"""Quick validation: flag JSON files in 1782_discovery that lack clear §1782 references."""

from __future__ import annotations

import json
from pathlib import Path

STRINGS = [
    "28 u.s.c. 1782",
    "28 usc 1782",
    "section 1782",
    "§ 1782",
    "foreign tribunal",
    "letters rogatory",
]

SOURCE_DIR = Path('data/case_law/1782_discovery')
OUTPUT_PATH = Path('data/case_law/flagged_missing_1782.json')

def main() -> None:
    flagged = []
    for path in sorted(SOURCE_DIR.glob('*.json')):
        text = path.read_text(encoding='utf-8').lower()
        if not any(s in text for s in STRINGS):
            data = json.loads(path.read_text(encoding='utf-8'))
            flagged.append({'filename': path.name, 'case_name': data.get('caseName', '')})

    print(f"Files checked: {len(list(SOURCE_DIR.glob('*.json')))}")
    print(f"Flagged (missing §1782 references): {len(flagged)}")
    for entry in flagged[:20]:
        print(f" - {entry['filename']} | {entry['case_name']}")
    OUTPUT_PATH.write_text(json.dumps(flagged, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Full list saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
