"""Factual Timeline drafting function for Semantic Kernel."""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction, KernelFunctionFromPrompt, KernelArguments

from ..base_plugin import kernel_function
from ..base_plugin import BaseSKPlugin, PluginMetadata, DraftingFunction, FunctionResult
from ..utils import extract_prompt_text

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """Individual timeline event."""
    date: str
    event: str
    significance: str
    evidence_refs: List[str]


@dataclass
class TimelineSection:
    """Structured output for timeline section."""
    title: str
    introduction: str
    events: List[TimelineEvent]
    analysis: str
    conclusion: str
    citations: List[str]
    word_count: int


class FactualTimelineNativeFunction(DraftingFunction):
    """Native function for factual timeline section drafting."""

    def __init__(self):
        super().__init__(
            name="FactualTimelineNative",
            description="Generate factual timeline section using deterministic templates",
            section_type="factual_timeline"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute factual timeline drafting."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: evidence, posteriors, case_summary"
            )

        try:
            evidence = kwargs["evidence"]
            posteriors = kwargs["posteriors"]
            case_summary = kwargs["case_summary"]
            jurisdiction = kwargs.get("jurisdiction", "US")

            # Extract timeline-relevant evidence
            timeline_events = self._extract_timeline_events(evidence, posteriors)

            # Format citations
            citations = self._format_citations(evidence)

            # Generate structured section
            section = self._generate_timeline_section(
                timeline_events, case_summary, jurisdiction, citations
            )

            return FunctionResult(
                success=True,
                value=section,
                metadata={"section_type": "factual_timeline", "events_count": len(timeline_events)}
            )

        except Exception as e:
            logger.error(f"Error in FactualTimelineNative: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _extract_timeline_events(self, evidence: Dict[str, Any], posteriors: Dict[str, Any]) -> List[TimelineEvent]:
        """Extract timeline events from evidence and posteriors."""

        events = []

        # Look for date-based evidence
        for node, state in evidence.items():
            if any(date_indicator in str(node).lower() for date_indicator in ["apr", "may", "jun", "2019", "2020", "date"]):
                event = TimelineEvent(
                    date=self._extract_date_from_node(node),
                    event=f"{node}: {state}",
                    significance=self._assess_event_significance(node, state, posteriors),
                    evidence_refs=[f"[{node}:{state}]"]
                )
                events.append(event)

        # Add high-confidence posterior events
        for node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.8:
                if any(keyword in str(node).lower() for keyword in ["event", "occur", "happen", "action"]):
                    event = TimelineEvent(
                        date="Timeline Event",
                        event=f"{node} (Confidence: {prob:.1%})",
                        significance="High-confidence event based on analysis",
                        evidence_refs=[f"[{node}:{prob:.1%}]"]
                    )
                    events.append(event)

        # Sort events by date if possible
        events.sort(key=lambda e: e.date)

        return events

    def _extract_date_from_node(self, node: str) -> str:
        """Extract date from node name."""
        import re

        # Look for date patterns
        date_patterns = [
            r'(\d{4})',  # Year
            r'(Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # Month
            r'(\d{1,2})',  # Day
        ]

        for pattern in date_patterns:
            match = re.search(pattern, node)
            if match:
                return match.group(1)

        return "Timeline Event"

    def _assess_event_significance(self, node: str, state: str, posteriors: Dict[str, Any]) -> str:
        """Assess the significance of a timeline event."""

        # Check if this event affects high-confidence posteriors
        for posterior_node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.7:
                if any(keyword in str(node).lower() for keyword in ["email", "awareness", "violation"]):
                    return f"Significant event affecting {posterior_node} ({prob:.1%} confidence)"

        return "Timeline event requiring analysis"

    def _generate_timeline_section(
        self,
        events: List[TimelineEvent],
        case_summary: str,
        jurisdiction: str,
        citations: str
    ) -> TimelineSection:
        """Generate timeline section using templates."""

        # Title
        title = "Factual Timeline"

        # Introduction
        introduction = f"""
This section presents a chronological timeline of key events relevant to this case.
The timeline is based on the evidence presented {citations} and provides
the factual foundation for the legal analysis that follows.
        """.strip()

        # Analysis
        analysis = self._analyze_timeline_significance(events)

        # Conclusion
        conclusion = f"""
The timeline demonstrates a clear sequence of events {citations} that
establishes the factual basis for the legal claims asserted in this case.
The chronological progression supports the causation analysis and
provides context for the alleged harm.
        """.strip()

        # Calculate word count
        full_text = f"{introduction} {analysis} {conclusion}"
        word_count = len(full_text.split())

        return TimelineSection(
            title=title,
            introduction=introduction,
            events=events,
            analysis=analysis,
            conclusion=conclusion,
            citations=citations.split() if citations else [],
            word_count=word_count
        )

    def _analyze_timeline_significance(self, events: List[TimelineEvent]) -> str:
        """Analyze the significance of timeline events."""

        if not events:
            return """
The timeline analysis indicates that additional evidence may be required to
establish a complete chronological sequence of events relevant to this case.
            """.strip()

        significant_events = [e for e in events if "Significant" in e.significance]

        if significant_events:
            return f"""
The timeline analysis reveals {len(significant_events)} significant events that
directly impact the legal analysis. These events demonstrate a clear progression
that supports the causation chain and establishes the factual basis for the claims.

Key events include:
{chr(10).join(f"• {event.date}: {event.event}" for event in significant_events)}
            """.strip()
        else:
            return f"""
The timeline analysis identifies {len(events)} relevant events that provide
context for the legal analysis. While additional evidence may strengthen
the timeline, the identified events support the factual foundation of the case.
            """.strip()


class FactualTimelineSemanticFunction(DraftingFunction):
    """Semantic function for factual timeline section drafting using LLM."""

    def __init__(self):
        super().__init__(
            name="FactualTimelineSemantic",
            description="Generate factual timeline section using LLM with structured prompts",
            section_type="factual_timeline"
        )
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """Get the prompt template for timeline drafting."""
        return """
You are a legal expert specializing in factual analysis. Draft a comprehensive factual timeline section for a legal memorandum.

CONTEXT:
Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence: {{$evidence}}
Bayesian Network Posteriors: {{$posteriors}}

{{$research_findings}}

{{$quality_constraints}}

REQUIREMENTS:
1. Create a chronological timeline of key events
2. Structure events with dates, descriptions, and significance
3. Cite evidence using [Node:State] format
4. Analyze the timeline's legal significance
5. Maintain objective, factual tone
6. Target 600-1000 words
7. Incorporate relevant case law from research findings if provided
8. Follow quality constraints and feature targets if provided

OUTPUT FORMAT:
Title: Factual Timeline

I. Introduction
[Brief overview of timeline purpose]

II. Chronological Events
[Date: Event Description - Significance]
[Evidence citations for each event]

III. Timeline Analysis
[Analysis of event significance and patterns]

IV. Conclusion
[Summary of timeline's legal relevance]

Ensure all events are supported by evidence citations in [Node:State] format.
        """

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic timeline drafting."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: evidence, posteriors, case_summary"
            )

        try:
            prompt_func = KernelFunctionFromPrompt(
                function_name=self.name,
                plugin_name="FactualTimelinePlugin",
                description=self.description,
                prompt=self.prompt_template,
            )

            # Prepare variables, including optional research findings and quality constraints
            variables = {
                "case_summary": kwargs["case_summary"],
                "jurisdiction": kwargs.get("jurisdiction", "US"),
                "evidence": json.dumps(kwargs["evidence"], indent=2),
                "posteriors": json.dumps(kwargs["posteriors"], indent=2)
            }
            
            # Add research findings if provided (always include, even if empty)
            if "research_findings" in kwargs and kwargs["research_findings"]:
                try:
                    if isinstance(kwargs["research_findings"], str):
                        research_data = json.loads(kwargs["research_findings"])
                    else:
                        research_data = kwargs["research_findings"]
                    research_text = self._format_research_findings(research_data)
                    variables["research_findings"] = research_text if research_text else ""
                except Exception as e:
                    logger.warning(f"Error formatting research findings: {e}")
                    variables["research_findings"] = ""
            else:
                variables["research_findings"] = ""
            
            # Add quality constraints if provided (always include, even if empty)
            if "quality_constraints" in kwargs and kwargs["quality_constraints"]:
                variables["quality_constraints"] = kwargs["quality_constraints"]
            else:
                variables["quality_constraints"] = ""
            
            arguments = KernelArguments(**variables)
            result = await prompt_func.invoke(kernel, arguments)
            result_text = extract_prompt_text(result.value).strip()

            return FunctionResult(
                success=True,
                value=result_text,
                metadata={
                    "section_type": "factual_timeline",
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as e:
            logger.error(f"Error in FactualTimelineSemantic: {e}")
            return FunctionResult(success=False, value=None, error=str(e))
    
    def _format_research_findings(self, research_data: Dict[str, Any]) -> str:
        """Format research findings for inclusion in prompt."""
        if not research_data:
            return ""
        
        formatted = []
        cases = research_data.get("cases", [])[:5]  # Top 5 cases
        explanations = research_data.get("explanations", {})
        overall = explanations.get("overall", {})
        
        if overall.get("summary"):
            formatted.append(f"Research Summary: {overall.get('summary')}")
        
        if cases:
            formatted.append(f"\nTop Relevant Cases ({len(cases)} cases):")
            for i, case in enumerate(cases, 1):
                case_name = case.get("case_name", "Unknown")
                court = case.get("court", "Unknown")
                similarity = case.get("similarity_score", 0.0)
                formatted.append(f"{i}. {case_name} ({court}) - Relevance: {similarity:.2f}")
        
        by_theme = explanations.get("by_theme", {})
        if by_theme:
            formatted.append("\nFindings by Theme:")
            for theme, theme_data in list(by_theme.items())[:3]:
                formatted.append(f"- {theme}: {theme_data.get('count', 0)} cases found")
        
        return "\n".join(formatted)


class FactualTimelinePlugin(BaseSKPlugin):
    """Plugin for factual timeline section drafting."""

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.native_function = FactualTimelineNativeFunction()
        self.semantic_function = FactualTimelineSemanticFunction()

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="FactualTimelinePlugin",
            description="Plugin for drafting factual timeline sections",
            version="1.0.0",
            functions=["FactualTimelineNative", "FactualTimelineSemantic"]
        )

    async def _register_functions(self) -> None:
        """Register timeline functions with the kernel."""

        # Register native function
        @kernel_function(
            name="FactualTimelineNative",
            description="Generate factual timeline section using deterministic templates"
        )
        async def factual_timeline_native(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US"
        ) -> str:
            """Native function for timeline drafting."""
            result = await self.native_function.execute(
                evidence=json.loads(evidence),
                posteriors=json.loads(posteriors),
                case_summary=case_summary,
                jurisdiction=jurisdiction
            )

            if result.success:
                section = result.value
                events_text = "\n".join(f"• {event.date}: {event.event}" for event in section.events)
                return f"""
# {section.title}

## Introduction
{section.introduction}

## Chronological Events
{events_text}

## Analysis
{section.analysis}

## Conclusion
{section.conclusion}

---
*Word count: {section.word_count} | Events: {len(section.events)}*
                """.strip()
            else:
                raise RuntimeError(f"Timeline drafting failed: {result.error}")

        # Register semantic function
        @kernel_function(
            name="FactualTimelineSemantic",
            description="Generate factual timeline section using LLM with structured prompts"
        )
        async def factual_timeline_semantic(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US"
        ) -> str:
            """Semantic function for timeline drafting."""
            result = await self.semantic_function.execute(
                kernel=self.kernel,
                evidence=json.loads(evidence),
                posteriors=json.loads(posteriors),
                case_summary=case_summary,
                jurisdiction=jurisdiction
            )

            if result.success:
                return result.value
            else:
                raise RuntimeError(f"Timeline semantic drafting failed: {result.error}")

        # Store function references
        self._functions["FactualTimelineNative"] = factual_timeline_native
        self._functions["FactualTimelineSemantic"] = factual_timeline_semantic


# Export classes
__all__ = [
    "FactualTimelinePlugin",
    "FactualTimelineNativeFunction",
    "FactualTimelineSemanticFunction",
    "TimelineSection",
    "TimelineEvent"
]
