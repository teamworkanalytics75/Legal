"""Atomic single-duty micro-agents for distributed cognition.

This package contains atomic agents organized by function:
- citations: Citation finding, normalization, verification, location, insertion
- research: Fact extraction, precedent finding/ranking, statute location, exhibit fetching
- drafting: Outline building, section writing, paragraph writing, transitions
- review: Grammar, style, logic, consistency, redaction, compliance, expert QA
- output: Markdown/DOCX export, metadata tagging
"""

# Import all atomic agent types for convenient access
from .citations import (
    CitationFinderAgent,
    CitationNormalizerAgent,
    CitationVerifierAgent,
    CitationLocatorAgent,
    CitationInserterAgent,
)
from .research import (
    FactExtractorAgent,
    PrecedentFinderAgent,
    PrecedentRankerAgent,
    PrecedentSummarizerAgent,
    StatuteLocatorAgent,
    ExhibitFetcherAgent,
)
from .enhanced_research import (
    EnhancedPrecedentFinderAgent,
    EnhancedFactExtractorAgent,
)
from .timeline_analyzer import (
    TimelineAnalyzerAgent,
)
from .evidence_correlator import (
    EvidenceCorrelatorAgent,
)
from .drafting import (
    OutlineBuilderAgent,
    SectionWriterAgent,
    ParagraphWriterAgent,
    TransitionAgent,
)
from .review import (
    GrammarFixerAgent,
    StyleCheckerAgent,
    LogicCheckerAgent,
    ConsistencyCheckerAgent,
    RedactionAgent,
    ComplianceAgent,
    ExpertQAAgent,
)
from .output import (
    MarkdownExporterAgent,
    DocxExporterAgent,
    MetadataTaggerAgent,
)

__all__ = [
    # Citations (5 agents - all deterministic)
    'CitationFinderAgent',
    'CitationNormalizerAgent',
    'CitationVerifierAgent',
    'CitationLocatorAgent',
    'CitationInserterAgent',

    # Research (6 agents - mix)
    'FactExtractorAgent',
    'PrecedentFinderAgent',
    'PrecedentRankerAgent',
    'PrecedentSummarizerAgent',
    'StatuteLocatorAgent',
    'ExhibitFetcherAgent',

    # Enhanced Research (2 agents - LangChain enabled)
    'EnhancedPrecedentFinderAgent',
    'EnhancedFactExtractorAgent',

    # Timeline Analysis (1 agent - LangChain + LLM)
    'TimelineAnalyzerAgent',

    # Evidence Correlation (1 agent - LangChain + LLM)
    'EvidenceCorrelatorAgent',

    # Drafting (4 agents - all LLM)
    'OutlineBuilderAgent',
    'SectionWriterAgent',
    'ParagraphWriterAgent',
    'TransitionAgent',

    # Review/QA (7 agents - mix)
    'GrammarFixerAgent',
    'StyleCheckerAgent',
    'LogicCheckerAgent',
    'ConsistencyCheckerAgent',
    'RedactionAgent',
    'ComplianceAgent',
    'ExpertQAAgent',

    # Output (3 agents - all deterministic)
    'MarkdownExporterAgent',
    'DocxExporterAgent',
    'MetadataTaggerAgent',
]

# Total: 25 atomic agents
# - Deterministic (zero cost): 14 agents (56%)
# - LLM-based (cost): 11 agents (44%)
# - gpt-4o-mini: 10 agents
# - gpt-4o (premium): 1 agent (ExpertQAAgent, conditional)
