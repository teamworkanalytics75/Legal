"""Compatibility shim for lowercase import of LangchainIntegration.

This module provides a lowercase alias for LangchainIntegration.py to support
both naming conventions. The original file uses capital letters for better
readability (especially for ADHD/autistic users), while this shim allows
backwards-compatible lowercase imports.

Import everything from the capitalized module.
"""

from __future__ import annotations

# Import everything from the capitalized module
# Try relative import first, then absolute
try:
    from .LangchainIntegration import (
        LangChainSQLAgent,
        EvidenceRetrievalAgent,
        create_evidence_retrieval_agent,
    )
except ImportError:
    # Fallback to absolute import
    from writer_agents.code.LangchainIntegration import (
        LangChainSQLAgent,
        EvidenceRetrievalAgent,
        create_evidence_retrieval_agent,
    )

# Re-export everything so it's available via lowercase import
__all__ = [
    "LangChainSQLAgent",
    "EvidenceRetrievalAgent", 
    "create_evidence_retrieval_agent",
]

