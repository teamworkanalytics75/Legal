"""Shared helpers for Semantic Kernel plugin compatibility."""

from typing import Any


def extract_prompt_text(value: Any) -> str:
    """
    Normalize Semantic Kernel prompt outputs into plain text.

    Handles ChatMessageContent lists, plain strings, and arbitrary objects.
    """
    candidate = value

    if isinstance(candidate, list) and candidate:
        first = candidate[0]
        items = getattr(first, "items", None)
        if items:
            item = items[0]
            text = getattr(item, "text", None)
            if text is not None:
                return text
            return str(item)
        text = getattr(first, "text", None)
        if text is not None:
            return text
        return str(first)

    text_attr = getattr(candidate, "text", None)
    if text_attr is not None:
        return text_attr

    if isinstance(candidate, bytes):
        return candidate.decode("utf-8", errors="ignore")

    if not isinstance(candidate, str):
        candidate = str(candidate)

    return candidate


__all__ = ["extract_prompt_text"]
