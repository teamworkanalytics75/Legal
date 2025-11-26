#!/usr/bin/env python3
"""
Legacy AutoGen writer workflow (deprecated).

This script is preserved only for reference and debugging. The production
writer pipeline now always runs the LangChain-enabled atomic workflow,
and manual SQLite logic survives solely as an automatic fallback when
LangChain encounters an error.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Notify callers that the AutoGen workflow is retired."""
    print("⚠️  The AutoGen writer workflow is deprecated and no longer executes.")
    print("    Production runs use the LangChain-enabled atomic pipeline by default.")
    print("    Manual SQL remains available only as an automatic fallback if LangChain fails.")
    print("    Retrieve an earlier revision if you need the full legacy behaviour.")
    sys.exit(0)


if __name__ == "__main__":
    main()
