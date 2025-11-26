"""
Assemble individual sections into a cohesive document.
"""

from __future__ import annotations

from typing import Dict, List

from ..base_plugin import kernel_function


@kernel_function(
    name="AssembleDocument",
    description="Combine ordered sections into a single motion-ready document.",
)
def assemble_document(context: Dict[str, List[Dict[str, str]]]) -> str:
    sections = context.get("sections") or []
    buffer: List[str] = []
    for section in sections:
        title = section.get("title")
        content = section.get("content", "")
        if title:
            buffer.append(f"# {title}")
        buffer.append(content.strip())
        buffer.append("")  # spacing
    return "\n".join(buffer).strip()

