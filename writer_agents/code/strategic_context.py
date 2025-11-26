"""Strategic Context Loader

Loads and provides strategic context to all modules.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default strategic context (if file doesn't exist)
DEFAULT_CONTEXT = {
    "ultimate_goal": "Settlement with President and Fellows of Harvard College",
    "hk_case": "Defamation case against Harvard Club of Hong Kong",
    "us_filings": {
        "first": "Motion for seal and pseudonym (D. Mass)",
        "second": "Section 1782 discovery motion (D. Mass)"
    },
    "primary_keywords": [
        "harvard", "china", "torture", "political retaliation",
        "defamation", "harassment", "academic institution",
        "foreign government", "section 1782", "28 U.S.C. 1782",
        "discovery for use in foreign proceeding", "intel factors",
        "foreign tribunal", "judicial assistance", "xi jinping",
        "xi mingze", "communist party", "chinese communist party", "ccp",
        "esuwiki", "spoliation", "litigation hold"
    ],
    "corpus_priority": "section_1782_discovery",
    "target_defendant": "President and Fellows of Harvard College"
}


def load_strategic_context(context_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load strategic context from file or return defaults.

    Args:
        context_path: Path to STRATEGIC_CONTEXT.md or JSON file.
                     If None, searches in project root.

    Returns:
        Dictionary with strategic context
    """
    if context_path is None:
        # Try to find in project root (go up from writer_agents/code/)
        project_root = Path(__file__).parents[2]  # writer_agents/code -> writer_agents -> root
        context_path = project_root / "STRATEGIC_CONTEXT.md"

    # For now, return default context
    # TODO: Parse markdown file if it exists
    if context_path.exists():
        logger.info(f"Found strategic context file: {context_path}")
        # Could parse markdown here if needed
        return DEFAULT_CONTEXT.copy()

    logger.debug("Using default strategic context")
    return DEFAULT_CONTEXT.copy()


def get_priority_keywords() -> list:
    """Get priority keywords from strategic context."""
    context = load_strategic_context()
    return context.get("primary_keywords", [])


def should_prioritize_1782() -> bool:
    """Check if Section 1782 cases should be prioritized."""
    context = load_strategic_context()
    return context.get("corpus_priority") == "section_1782_discovery"


def get_ultimate_goal() -> str:
    """Get the ultimate strategic goal."""
    context = load_strategic_context()
    return context.get("ultimate_goal", "")


def get_target_defendant() -> str:
    """Get the target defendant for settlement."""
    context = load_strategic_context()
    return context.get("target_defendant", "")

