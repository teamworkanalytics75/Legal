"""
Native function to create an ordered factual timeline.
"""

from __future__ import annotations

from typing import Dict, List

from ..base_plugin import kernel_function


@kernel_function(
    name="FactualTimelineNative",
    description="Compile a chronological timeline from structured events.",
)
def factual_timeline_native_function(context: Dict[str, List[Dict]]) -> str:
    events = context.get("events") or []
    if not events:
        return "No timeline events supplied."

    sorted_events = sorted(events, key=lambda e: e.get("date", ""))
    lines = ["# Case Timeline", ""]
    for event in sorted_events:
        date = event.get("date", "Unknown date")
        title = event.get("title", "Event")
        detail = event.get("detail", "")
        lines.append(f"- **{date}** – {title}")
        if detail:
            lines.append(f"  - {detail}")
    return "\n".join(lines)

