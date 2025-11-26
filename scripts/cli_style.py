#!/usr/bin/env python3
"""
CLI Help Style Utilities (ADHD/Autism‑Friendly)
================================================

Use these helpers to make `--help` output compact, scannable, and consistent.

Quick Start
-----------
import argparse
from scripts.cli_style import CompactHelpFormatter, action_box_epilog

parser = argparse.ArgumentParser(
    prog="example_tool",
    description="Do one thing well — compact help by default",
    formatter_class=CompactHelpFormatter,
    epilog=action_box_epilog([
        "example_tool --input data.json --output out.json",
        "example_tool --help",
    ]),
)

parser.add_argument("--input", required=True, help="Input file path")
parser.add_argument("--output", required=True, help="Output file path")

args = parser.parse_args()
"""

from __future__ import annotations

import argparse
from textwrap import dedent


class CompactHelpFormatter(argparse.HelpFormatter):
    """Compact, aligned argparse formatter with wider text width.

    - Short option/argument columns
    - Wrapped descriptions
    - Keeps help succinct and scannable
    """

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=28, width=92)


def action_box_epilog(commands: list[str]) -> str:
    """Return an epilog string with a visually distinct Action Box.

    Example:
        epilog=action_box_epilog([
            "tool --foo bar",
            "tool --help",
        ])
    """
    lines = ["\nDo This Next:", "```"]
    lines.extend(commands)
    lines.append("```")
    return "\n".join(lines)


def compact_parser(
    prog: str,
    description: str,
    commands: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Create a preconfigured ArgumentParser with compact style and action box.

    Args:
        prog: Program name
        description: Short, action-oriented description
        commands: Optional list of example commands for the Action Box
    """
    epilog = action_box_epilog(commands or [f"{prog} --help"])  # default example
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=CompactHelpFormatter,
        epilog=epilog,
    )


__all__ = [
    "CompactHelpFormatter",
    "action_box_epilog",
    "compact_parser",
]
