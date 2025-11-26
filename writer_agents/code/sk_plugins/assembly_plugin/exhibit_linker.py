"""
Simple exhibit link generator.
"""

from __future__ import annotations

from typing import Dict, List

from ..base_plugin import kernel_function


@kernel_function(
    name="LinkExhibits",
    description="Create an exhibit appendix mapping references to file names.",
)
def link_exhibits(context: Dict[str, List[Dict[str, str]]]) -> str:
    exhibits = context.get("exhibits") or []
    if not exhibits:
        return "No exhibits attached."
    lines = ["# Exhibit Appendix", ""]
    for exhibit in exhibits:
        label = exhibit.get("label", "Exhibit")
        description = exhibit.get("description", "")
        filename = exhibit.get("filename", "")
        entry = f"- **{label}**: {description}"
        if filename:
            entry += f" (file: {filename})"
        lines.append(entry)
    return "\n".join(lines)

