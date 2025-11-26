"""
Utility script to remove or replace non-ASCII glyphs from text files.

The The Matrix repository accumulated a wide mix of emoji, smart quotes, and
control characters that render poorly on some developer environments
(`dY...` artefacts). This script normalises affected files to plain ASCII so
documentation and Python sources stay portable.

Usage (from repository root):

    py -3 sanitize_unicode.py

The script focuses on Markdown, Python, configuration, and text assets while
skipping virtual environments and cache directories.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Dict, Iterable


# Character replacements for the most common glyphs we want to preserve in ASCII.
CHAR_REPLACEMENTS: Dict[str, str] = {
    "\u00a0": " ", # non-breaking space
    "\u00a7": "Section",
    "\u00ad": "-", # soft hyphen
    "\u2010": "-", # hyphen
    "\u2011": "-", # non-breaking hyphen
    "\u2012": "-", # figure dash
    "\u2013": "-", # en dash
    "\u2014": "-", # em dash
    "\u2018": "'", # left single quotation mark
    "\u2019": "'", # right single quotation mark
    "\u201a": ",",
    "\u201c": '"', # left double quotation mark
    "\u201d": '"', # right double quotation mark
    "\u201e": '"',
    "\u2022": "- ", # bullet
    "\u2024": ".",
    "\u2026": "...", # ellipsis
    "\u202f": " ", # narrow no-break space
    "\u2122": "TM",
    "\u2190": "<-",
    "\u2191": "^",
    "\u2192": "->",
    "\u2193": "v",
    "\u2194": "<->",
    "\u21a9": "<-",
    "\u21aa": "->",
    "\u2212": "-", # minus sign
    "\u221e": "infinity",
    "\u2260": "!=",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2605": "*",
    "\u2606": "*",
    "\u2611": "[x]",
    "\u2610": "[ ]",
    "\u2615": "coffee",
    "\u2620": "skull",
    "\u2622": "hazard",
    "\u262a": "star",
    "\u263a": ":)",
    "\u2640": "female",
    "\u2642": "male",
    "\u26a0": "WARNING",
    "\u26a1": "lightning",
    "\u2705": "[ok]",
    "\u270f": "pencil",
    "\u2713": "[ok]",
    "\u2714": "[ok]",
    "\u2716": "x",
    "\u2728": "*",
    "\u274c": "x",
    "\u2753": "?",
    "\u3001": ",",
    "\u3002": ".",
    "\u300c": '"',
    "\u300d": '"',
    "\uff08": "(",
    "\uff09": ")",
    "\uff1a": ":",
}

# Prefixes to skip when traversing the repository.
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}

# File extensions we consider "text" for sanitisation.
TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".py",
}


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield candidate text files beneath *root*."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {"Makefile", "Dockerfile"}:
            yield path


def sanitize_text(content: str) -> str:
    """Return a sanitised ASCII representation of *content*."""
    # Apply replacements first.
    for src, dst in CHAR_REPLACEMENTS.items():
        content = content.replace(src, dst)

    # Normalise accents (e.g. e -> e) where possible.
    content = unicodedata.normalize("NFKD", content)

    result_chars: list[str] = []
    at_line_start = True
    skip_next_space = False

    for ch in content:
        code_point = ord(ch)

        if ch in ("\ufeff", "\u200d"): # BOM & zero-width joiner
            continue

        if unicodedata.combining(ch):
            continue

        if ch == "\n":
            result_chars.append(ch)
            at_line_start = True
            skip_next_space = False
            continue

        if ch == "\r":
            result_chars.append(ch)
            at_line_start = True
            skip_next_space = False
            continue

        if ch == "\t":
            result_chars.append(ch)
            skip_next_space = False
            continue

        if ch == " ":
            if skip_next_space and not at_line_start:
                skip_next_space = False
                continue
            skip_next_space = False
            if at_line_start:
                result_chars.append(ch)
            elif result_chars and result_chars[-1] == " ":
                continue
            else:
                result_chars.append(ch)
            continue

        if code_point < 32:
            # Drop remaining control characters.
            continue

        if code_point < 128:
            result_chars.append(ch)
            at_line_start = False
            skip_next_space = False
            continue

        if unicodedata.category(ch).startswith("Z"):
            if at_line_start:
                result_chars.append(" ")
            elif result_chars and result_chars[-1] == " ":
                continue
            else:
                result_chars.append(" ")
            skip_next_space = False
            continue

        # Drop any other high code-point glyphs (emoji, box drawing, etc.).
        skip_next_space = True
        continue

    sanitised = "".join(result_chars)
    return sanitised


def process_file(path: Path, dry_run: bool = False) -> bool:
    """Sanitise *path*. Returns True when modifications were made."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip files we cannot decode safely.
        return False

    cleaned = sanitize_text(original)

    if cleaned == original:
        return False

    if not dry_run:
        path.write_text(cleaned, encoding="utf-8", newline="")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip problematic non-ASCII characters from text files.")
    parser.add_argument("--dry-run", action="store_true", help="Report files that would change without modifying them.")
    args = parser.parse_args()

    root = Path(".").resolve()
    modified = []

    for file_path in iter_text_files(root):
        if process_file(file_path, dry_run=args.dry_run):
            modified.append(file_path)

    action = "Would modify" if args.dry_run else "Modified"
    def safe_print(message: str) -> None:
        try:
            print(message)
        except UnicodeEncodeError:
            print(message.encode("ascii", "ignore").decode("ascii"))

    for path in modified:
        safe_print(f"{action}: {path}")

    safe_print(f"{action} {len(modified)} files.")


if __name__ == "__main__":
    main()
