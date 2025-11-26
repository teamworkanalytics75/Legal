"""Agent tier and meta-category configuration."""

COMPLETENESS_AGENTS = [
    # Research agents - need comprehensive analysis
    "FactExtractorAgent",
    "PrecedentFinderAgent",
    "EnhancedPrecedentFinderAgent",
    "EnhancedFactExtractorAgent",
    "StatuteLocatorAgent",
    "ExhibitFetcherAgent",
    # Drafting agents - need comprehensive content generation
    "OutlineBuilderAgent",
    "SectionWriterAgent",
    "ParagraphWriterAgent",
    "TransitionAgent",
]

PRECISION_AGENTS = [
    # Research precision agents
    "PrecedentRankerAgent",
    "PrecedentSummarizerAgent",
    # Review agents - need high precision
    "LogicCheckerAgent",
    "ConsistencyCheckerAgent",
    "ExpertQAAgent",
    "GrammarFixerAgent",
    "StyleCheckerAgent",
]

STANDARD_AGENTS = [
    # Citation agents - mostly deterministic
    "CitationFinderAgent",
    "CitationNormalizerAgent",
    "CitationVerifierAgent",
    "CitationLocatorAgent",
    "CitationInserterAgent",
    # Output agents - deterministic formatting
    "MarkdownExporterAgent",
    "DocxExporterAgent",
    "MetadataTaggerAgent",
    # Compliance agents - rule-based
    "RedactionAgent",
    "ComplianceAgent",
]

META_CATEGORY_CONFIG = {
    "completeness": {
        "model_tier": "premium",
        "temperature": 0.3,
        "max_tokens": 8000,
        "instruction_suffix": "Think deeply and comprehensively. Explore all relevant angles and produce thorough analysis."
    },
    "precision": {
        "model_tier": "premium",
        "temperature": 0.1,
        "max_tokens": 6000,
        "instruction_suffix": "Analyze critically and precisely. Select the best, most accurate information."
    },
    "standard": {
        "model_tier": "mini",
        "temperature": 0.0,
        "max_tokens": 2000,
        "instruction_suffix": ""
    }
}

def get_agent_meta_category(agent_name: str) -> str:
    """Get meta-category for an agent."""
    if agent_name in COMPLETENESS_AGENTS:
        return "completeness"
    elif agent_name in PRECISION_AGENTS:
        return "precision"
    else:
        return "standard"

def get_agent_config(agent_name: str) -> dict:
    """Get configuration for an agent based on its meta-category."""
    meta_category = get_agent_meta_category(agent_name)
    return META_CATEGORY_CONFIG[meta_category]
