"""Task decomposition and spawn policy engine.

Analyzes case characteristics and determines optimal atomic agent allocation.
Prioritizes cost efficiency through deterministic heuristics and minimal LLM usage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .insights import CaseInsights
except ImportError:
    from insights import CaseInsights


@dataclass
class TaskProfile:
    """Analysis profile for determining agent spawn policies."""

    evidence_count: int
    citation_density: float # citations per 1000 words
    unique_jurisdictions: List[str]
    report_length_estimate: str # short, medium, long
    complexity_score: float # 0.0-1.0
    requires_redaction: bool
    highest_priority_facts: List[str]
    spawn_policies: Dict[str, int] = field(default_factory=dict) # agent_type -> count
    estimated_sections: int = 0
    estimated_words: int = 0


class TaskDecomposer:
    """Decomposes legal writing tasks into atomic agent assignments."""

    # Deterministic thresholds for spawn decisions
    CITATION_DENSITY_THRESHOLD = 8.0 # per 1k words
    HIGH_EVIDENCE_THRESHOLD = 250
    LONG_REPORT_WORDS = 5000
    MEDIUM_REPORT_WORDS = 2000
    MAX_SECTION_WRITERS = 6

    def __init__(self) -> None:
        """Initialize task decomposer."""
        pass

    async def compute(
        self,
        insights: CaseInsights,
        summary: str,
        session: Optional[Any] = None,
    ) -> TaskProfile:
        """Compute task profile from case insights.

        Args:
            insights: Bayesian network case insights
            summary: Case summary text
            session: Optional SQLite session for persistence

        Returns:
            TaskProfile with spawn policies
        """
        # Extract basic metrics deterministically
        evidence_count = self._count_evidence(insights)
        estimated_words = self._estimate_word_count(summary, insights)
        estimated_sections = self._estimate_sections(insights)
        citation_density = self._estimate_citation_density(summary)

        # Determine report length
        if estimated_words > self.LONG_REPORT_WORDS:
            report_length = "long"
        elif estimated_words > self.MEDIUM_REPORT_WORDS:
            report_length = "medium"
        else:
            report_length = "short"

        # Detect jurisdictions from summary (deterministic regex)
        jurisdictions = self._extract_jurisdictions(summary)

        # Check for redaction needs
        requires_redaction = self._check_redaction_needs(summary)

        # Compute complexity score (deterministic factors)
        complexity = self._compute_complexity(
            evidence_count, len(jurisdictions), estimated_words
        )

        # Extract priority facts from insights
        priority_facts = self._extract_priority_facts(insights)

        # Build spawn policies based on deterministic rules
        spawn_policies = self._build_spawn_policies(
            evidence_count=evidence_count,
            citation_density=citation_density,
            estimated_sections=estimated_sections,
            requires_redaction=requires_redaction,
            complexity=complexity,
        )

        return TaskProfile(
            evidence_count=evidence_count,
            citation_density=citation_density,
            unique_jurisdictions=jurisdictions,
            report_length_estimate=report_length,
            complexity_score=complexity,
            requires_redaction=requires_redaction,
            highest_priority_facts=priority_facts,
            spawn_policies=spawn_policies,
            estimated_sections=estimated_sections,
            estimated_words=estimated_words,
        )

    def _count_evidence(self, insights: CaseInsights) -> int:
        """Count evidence items from insights."""
        if not insights:
            return 0

        # Count from posteriors if available
        count = len(insights.posteriors) if hasattr(insights, 'posteriors') else 0

        # Add evidence nodes if tracked separately
        if hasattr(insights, 'evidence'):
            count += len(insights.evidence)

        return count

    def _estimate_word_count(self, summary: str, insights: CaseInsights) -> int:
        """Estimate final report word count.

        Uses heuristics based on:
        - Summary length
        - Number of evidence items
        - Complexity indicators
        """
        # Base estimate from summary
        summary_words = len(summary.split())

        # Evidence contributes to length
        evidence_count = self._count_evidence(insights)
        evidence_factor = evidence_count * 50 # ~50 words per evidence item

        # Total estimate
        estimated = summary_words * 3 + evidence_factor # 3x expansion typical

        return max(1000, min(10000, estimated)) # Clamp to reasonable range

    def _estimate_sections(self, insights: CaseInsights) -> int:
        """Estimate number of major sections needed."""
        # Base sections: Introduction, Analysis, Conclusion
        base = 3

        # Add sections for major topics
        evidence_count = self._count_evidence(insights)

        if evidence_count > 10:
            additional = min(evidence_count // 5, 6) # 1 section per 5 evidence items, cap at 6
        else:
            additional = 1

        return base + additional

    def _estimate_citation_density(self, summary: str) -> float:
        """Estimate citations per 1000 words.

        Uses regex patterns to detect potential citation markers.
        """
        # Patterns that suggest citations
        patterns = [
            r'\d+ [A-Z][a-z]+\.?\s+\d+', # "123 U.S. 456"
            r'v\.', # versus
            r'\d{4}\)', # year in parens
            r'[A-Z][a-z]+ v\. [A-Z][a-z]+', # Case names
        ]

        citation_count = 0
        for pattern in patterns:
            citation_count += len(re.findall(pattern, summary))

        word_count = len(summary.split())
        if word_count == 0:
            return 0.0

        # Citations per 1000 words
        return (citation_count / word_count) * 1000

    def _extract_jurisdictions(self, summary: str) -> List[str]:
        """Extract jurisdiction references from text."""
        jurisdictions = []

        # Common jurisdiction patterns
        patterns = {
            'federal': r'\b(?:federal|Federal|U\.?S\.?|United States)\b',
            'massachusetts': r'\b(?:Massachusetts|Mass\.|MA)\b',
            'california': r'\b(?:California|Cal\.|CA)\b',
            'new_york': r'\b(?:New York|N\.?Y\.?)\b',
            # Add more as needed
        }

        for jurisdiction, pattern in patterns.items():
            if re.search(pattern, summary):
                jurisdictions.append(jurisdiction)

        return jurisdictions

    def _check_redaction_needs(self, summary: str) -> bool:
        """Check if document requires redaction."""
        # Look for redaction markers or PII indicators
        redaction_markers = [
            r'\[REDACTED\]',
            r'\[SEALED\]',
            r'\bDoe\b', # Jane/John Doe
            r'\bconfidential\b',
            r'\bsealed\b',
        ]

        for marker in redaction_markers:
            if re.search(marker, summary, re.IGNORECASE):
                return True

        return False

    def _compute_complexity(
        self,
        evidence_count: int,
        jurisdiction_count: int,
        word_count: int,
    ) -> float:
        """Compute complexity score (0.0-1.0).

        Higher scores indicate more complex analysis requiring more agents.
        """
        # Normalize factors
        evidence_score = min(evidence_count / 100, 1.0) # Max at 100 items
        jurisdiction_score = min(jurisdiction_count / 3, 1.0) # Max at 3 jurisdictions
        length_score = min(word_count / 10000, 1.0) # Max at 10k words

        # Weighted average
        complexity = (
            evidence_score * 0.4 +
            jurisdiction_score * 0.3 +
            length_score * 0.3
        )

        return round(complexity, 2)

    def _extract_priority_facts(self, insights: CaseInsights) -> List[str]:
        """Extract highest priority facts from insights."""
        priority_facts = []

        # Get facts from posteriors if available
        if hasattr(insights, 'posteriors') and insights.posteriors:
            # Sort by probability/confidence
            sorted_posteriors = sorted(
                insights.posteriors.items(),
                key=lambda x: max(x[1].values()) if isinstance(x[1], dict) else 0,
                reverse=True
            )

            # Take top 5
            for node_id, distribution in sorted_posteriors[:5]:
                if isinstance(distribution, dict):
                    top_state = max(distribution.items(), key=lambda x: x[1])
                    priority_facts.append(f"{node_id}: {top_state[0]} ({top_state[1]:.2f})")

        return priority_facts

    def _build_spawn_policies(
        self,
        evidence_count: int,
        citation_density: float,
        estimated_sections: int,
        requires_redaction: bool,
        complexity: float,
    ) -> Dict[str, int]:
        """Build spawn policies based on task characteristics.

        Returns:
            Dictionary mapping agent types to spawn counts
        """
        policies: Dict[str, int] = {}

        # Research agents (scale with evidence)
        if evidence_count > 0:
            policies['FactExtractorAgent'] = 1
            policies['PrecedentFinderAgent'] = 1

            if evidence_count > 100:
                policies['PrecedentRankerAgent'] = 1

            if evidence_count > 200:
                policies['StatuteLocatorAgent'] = 1

        # Citation agents (scale with density)
        if citation_density > self.CITATION_DENSITY_THRESHOLD:
            # High citation density requires more verifiers
            policies['CitationFinderAgent'] = 1
            policies['CitationNormalizerAgent'] = 1
            policies['CitationVerifierAgent'] = min(3, int(citation_density / 10))
        else:
            # Standard citation processing
            policies['CitationFinderAgent'] = 1
            policies['CitationNormalizerAgent'] = 1
            policies['CitationVerifierAgent'] = 1

        # Always need locator and inserter
        policies['CitationLocatorAgent'] = 1
        policies['CitationInserterAgent'] = 1

        # Drafting agents (scale with sections)
        policies['OutlineBuilderAgent'] = 1
        section_writers = min(estimated_sections, self.MAX_SECTION_WRITERS)
        policies['SectionWriterAgent'] = section_writers

        if complexity > 0.5:
            policies['TransitionAgent'] = 1

        # QA agents (always needed)
        policies['GrammarFixerAgent'] = 1
        policies['StyleCheckerAgent'] = 1
        policies['LogicCheckerAgent'] = 1
        policies['ConsistencyCheckerAgent'] = 1
        policies['ComplianceAgent'] = 1

        # Conditional QA agents
        if requires_redaction:
            policies['RedactionAgent'] = 1

        if complexity > 0.7:
            policies['ExpertQAAgent'] = 1 # Only for complex cases

        # Output agents (always needed)
        policies['MarkdownExporterAgent'] = 1
        policies['DocxExporterAgent'] = 1
        policies['MetadataTaggerAgent'] = 1

        return policies

    def estimate_cost(self, profile: TaskProfile, premium_mode: bool = False) -> Dict[str, float]:
        """Estimate cost for executing task profile.

        Args:
            profile: Task profile with spawn policies
            premium_mode: Whether premium models are enabled

        Returns:
            Dictionary with cost estimates
        """
        # Cost per agent type (estimated)
        # Deterministic agents are free (0 cost)
        # Standard LLM agents use gpt-4o-mini (~$0.0015 per call)
        # Premium LLM agents use gpt-4o (~$0.05 per call, ~33x more expensive)

        # Import agent tiers for meta-category lookup
        try:
            from .agent_tiers import get_agent_meta_category
        except ImportError:
            # Fallback if agent_tiers not available
            def get_agent_meta_category(agent_name: str) -> str:
                return "standard"

        # Base costs for standard mode (gpt-4o-mini)
        base_agent_costs = {
            'CitationFinderAgent': 0.0, # Deterministic
            'CitationNormalizerAgent': 0.0, # Deterministic
            'CitationVerifierAgent': 0.0, # Deterministic
            'CitationLocatorAgent': 0.0, # Deterministic
            'CitationInserterAgent': 0.0, # Deterministic
            'FactExtractorAgent': 0.002, # LLM
            'PrecedentFinderAgent': 0.001, # Mostly DB
            'PrecedentRankerAgent': 0.001, # LLM
            'StatuteLocatorAgent': 0.0, # DB
            'OutlineBuilderAgent': 0.002, # LLM
            'SectionWriterAgent': 0.005, # LLM (larger)
            'TransitionAgent': 0.001, # LLM
            'GrammarFixerAgent': 0.001, # LLM or deterministic
            'StyleCheckerAgent': 0.001, # Mixed
            'LogicCheckerAgent': 0.002, # LLM
            'ConsistencyCheckerAgent': 0.0, # Deterministic
            'RedactionAgent': 0.0, # Deterministic
            'ComplianceAgent': 0.0, # Deterministic
            'ExpertQAAgent': 0.01, # LLM (gpt-4o if needed)
            'MarkdownExporterAgent': 0.0, # Deterministic
            'DocxExporterAgent': 0.0, # Deterministic
            'MetadataTaggerAgent': 0.0, # Deterministic
        }

        # Calculate costs based on premium mode
        total_cost = 0.0
        cost_breakdown = {}
        premium_cost = 0.0
        standard_cost = 0.0
        premium_agents_count = 0
        standard_agents_count = 0

        for agent_type, count in profile.spawn_policies.items():
            base_cost = base_agent_costs.get(agent_type, 0.001)

            # Determine if this agent gets premium pricing
            meta_category = get_agent_meta_category(agent_type)
            if premium_mode and meta_category in ["completeness", "precision"]:
                # Premium agents use GPT-4o (~33x more expensive)
                agent_cost = base_cost * 33 * count
                premium_cost += agent_cost
                premium_agents_count += count
            else:
                # Standard agents use GPT-4o-mini
                agent_cost = base_cost * count
                standard_cost += agent_cost
                standard_agents_count += count

            total_cost += agent_cost
            cost_breakdown[agent_type] = agent_cost

        return {
            'total_estimated_cost': round(total_cost, 4),
            'premium_cost': round(premium_cost, 4),
            'standard_cost': round(standard_cost, 4),
            'by_agent': cost_breakdown,
            'deterministic_agents': sum(
                1 for agent in profile.spawn_policies
                if base_agent_costs.get(agent, 0) == 0
            ),
            'llm_agents': sum(
                1 for agent in profile.spawn_policies
                if base_agent_costs.get(agent, 0) > 0
            ),
            'premium_agents': premium_agents_count,
            'standard_agents': standard_agents_count,
            'premium_mode': premium_mode,
        }
