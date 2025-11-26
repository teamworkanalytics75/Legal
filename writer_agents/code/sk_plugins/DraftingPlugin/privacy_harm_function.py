"""Privacy Harm drafting function for Semantic Kernel."""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction, KernelFunctionFromPrompt, KernelArguments

try:
    from ..base_plugin import kernel_function
    from ..base_plugin import BaseSKPlugin, PluginMetadata, DraftingFunction, FunctionResult
    from ..utils import extract_prompt_text
except ImportError:  # pragma: no cover - fallback for script execution
    from base_plugin import kernel_function  # type: ignore
    from base_plugin import BaseSKPlugin, PluginMetadata, DraftingFunction, FunctionResult  # type: ignore
    from utils import extract_prompt_text  # type: ignore

logger = logging.getLogger(__name__)

FACT_GUARD_BLOCK = """
KEY FACTS SUMMARY (use names/dates verbatim; never invent facts):
{{$key_facts_summary}}

MISSING FACTS TODO (blank if validation already confirmed full coverage):
{{$fact_retry_todo}}

FACT CHECKLIST (blank if coverage already met):
{{$fact_checklist_block}}

CATBOOST / SHAP GUIDANCE (blank if score already ≥ target):
{{$feature_guidance}}

STRICT FACTUAL INSTRUCTIONS:
- Use ONLY the facts listed above and in the Structured Facts section; do not add new parties, dates, or evidence.
- Mention every named person, date, and number explicitly (you may rephrase but never omit them).
- Cite supporting evidence with [Node:State] for each factual assertion.
- If a detail is missing from Key Facts or Structured Facts, leave it out instead of guessing.
- If the Missing Facts TODO block lists items, explicitly incorporate each into this section.
"""


@dataclass
class PrivacyHarmSection:
    """Structured output for privacy harm section."""
    title: str
    introduction: str
    harm_analysis: str
    legal_framework: str
    evidence_summary: str
    conclusion: str
    citations: List[str]
    word_count: int


class PrivacyHarmNativeFunction(DraftingFunction):
    """Native function for privacy harm section drafting."""

    def __init__(self):
        super().__init__(
            name="PrivacyHarmNative",
            description="Generate privacy harm section using deterministic templates",
            section_type="privacy_harm"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute privacy harm drafting."""
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

            # Extract high-confidence posteriors
            key_posteriors = self._extract_key_posteriors(posteriors, threshold=0.6)

            # Format citations
            citations = self._format_citations(evidence)

            # Generate structured section
            section = self._generate_privacy_harm_section(
                evidence, key_posteriors, case_summary, jurisdiction, citations
            )

            return FunctionResult(
                success=True,
                value=section,
                metadata={"section_type": "privacy_harm", "citations_count": len(citations)}
            )

        except Exception as e:
            logger.error(f"Error in PrivacyHarmNative: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _generate_privacy_harm_section(
        self,
        evidence: Dict[str, Any],
        posteriors: Dict[str, Any],
        case_summary: str,
        jurisdiction: str,
        citations: str
    ) -> PrivacyHarmSection:
        """Generate privacy harm section using templates."""

        # Title
        title = "Privacy Harm Analysis"

        # Introduction
        introduction = f"""
This section analyzes the privacy harm resulting from the alleged conduct described in this case.
Based on the evidence presented {citations}, the privacy violations constitute significant
harm warranting legal relief under {jurisdiction} privacy law.
        """.strip()

        # Harm analysis based on posteriors
        harm_analysis = self._analyze_privacy_harm(evidence, posteriors)

        # Legal framework
        legal_framework = self._get_legal_framework(jurisdiction)

        # Evidence summary
        evidence_summary = self._summarize_evidence(evidence, posteriors)

        # Conclusion
        conclusion = f"""
The privacy harm analysis demonstrates that the alleged conduct resulted in significant
privacy violations {citations}. The evidence supports a finding of actionable privacy harm
under applicable {jurisdiction} law, warranting the requested relief.
        """.strip()

        # Calculate word count
        full_text = f"{introduction} {harm_analysis} {legal_framework} {evidence_summary} {conclusion}"
        word_count = len(full_text.split())

        return PrivacyHarmSection(
            title=title,
            introduction=introduction,
            harm_analysis=harm_analysis,
            legal_framework=legal_framework,
            evidence_summary=evidence_summary,
            conclusion=conclusion,
            citations=citations.split() if citations else [],
            word_count=word_count
        )

    def _analyze_privacy_harm(self, evidence: Dict[str, Any], posteriors: Dict[str, Any]) -> str:
        """Analyze privacy harm based on evidence and posteriors."""

        harm_factors = []

        # Check for direct privacy violations
        if any("privacy" in str(node).lower() for node in evidence.keys()):
            harm_factors.append("Direct privacy violations evidenced by the conduct")

        # Check for high-confidence harm indicators
        for node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.7:
                if "harm" in str(node).lower() or "damage" in str(node).lower():
                    harm_factors.append(f"High probability of harm indicated by {node} ({prob:.1%})")

        # Check for data exposure
        if any("data" in str(node).lower() or "information" in str(node).lower() for node in evidence.keys()):
            harm_factors.append("Personal data exposure and misuse")

        if harm_factors:
            return f"""
The privacy harm analysis reveals multiple factors supporting a finding of actionable harm:

{chr(10).join(f"• {factor}" for factor in harm_factors)}

These factors collectively demonstrate that the alleged conduct resulted in significant
privacy violations that exceed mere inconvenience and constitute actionable harm under
applicable privacy law.
            """.strip()
        else:
            return """
The privacy harm analysis indicates potential privacy concerns based on the evidence
presented. While the specific harm factors require further development, the alleged
conduct raises privacy issues that warrant legal consideration.
            """.strip()

    def _get_legal_framework(self, jurisdiction: str) -> str:
        """Get legal framework based on jurisdiction."""

        frameworks = {
            "US": """
Under United States privacy law, privacy harm analysis focuses on:
• Intrusion upon seclusion
• Public disclosure of private facts
• False light publicity
• Appropriation of name or likeness

The harm must be substantial and offensive to a reasonable person, and the conduct
must exceed bounds of decency tolerated by society.
            """,
            "EU": """
Under European Union privacy law (GDPR), privacy harm analysis considers:
• Breach of personal data protection principles
• Unlawful processing of personal data
• Failure to implement appropriate safeguards
• Non-consensual data processing

The harm must be material or non-material damage resulting from unlawful processing.
            """,
            "CA": """
Under California privacy law (CCPA/CPRA), privacy harm analysis examines:
• Unauthorized collection of personal information
• Failure to provide required disclosures
• Unlawful sale or sharing of personal information
• Failure to implement reasonable security measures

The harm must be actual and material, not merely hypothetical or speculative.
            """
        }

        return frameworks.get(jurisdiction, frameworks["US"]).strip()

    def _summarize_evidence(self, evidence: Dict[str, Any], posteriors: Dict[str, Any]) -> str:
        """Summarize evidence supporting privacy harm."""

        evidence_items = []

        # Direct evidence
        for node, state in evidence.items():
            if "privacy" in str(node).lower() or "data" in str(node).lower():
                evidence_items.append(f"• {node}: {state}")

        # High-confidence posteriors
        for node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.8:
                evidence_items.append(f"• {node}: {prob:.1%} confidence")

        if evidence_items:
            return f"""
The evidence supporting privacy harm includes:

{chr(10).join(evidence_items)}

This evidence collectively supports a finding of actionable privacy harm warranting
legal relief.
            """.strip()
        else:
            return """
The evidence presented supports privacy harm analysis based on the alleged conduct
and circumstances of the case. Further development of specific harm factors may be
warranted based on additional evidence.
            """.strip()


class PrivacyHarmSemanticFunction(DraftingFunction):
    """Semantic function for privacy harm section drafting using LLM."""

    def __init__(self):
        super().__init__(
            name="PrivacyHarmSemantic",
            description="Generate privacy harm section using LLM with structured prompts",
            section_type="privacy_harm"
        )
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """Get the prompt template for privacy harm drafting."""
        return """
You are a legal expert specializing in privacy law. Draft a comprehensive privacy harm analysis section for a legal memorandum.

CONTEXT:
Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence: {{$evidence}}
Bayesian Network Posteriors: {{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

{{$quality_constraints}}

REQUIREMENTS:
1. Write a formal legal analysis of privacy harm
2. Structure the section with clear headings
3. Cite evidence using [Node:State] format
4. Apply appropriate legal framework for the jurisdiction
5. Conclude with actionable harm finding
6. Maintain professional legal tone
7. Target 800-1200 words
8. Incorporate relevant case law from research findings if provided
9. Follow quality constraints and feature targets if provided

OUTPUT FORMAT:
Title: Privacy Harm Analysis

I. Introduction
[Brief overview of privacy harm analysis]

II. Legal Framework
[Applicable privacy law principles]

III. Harm Analysis
[Detailed analysis of privacy violations]

IV. Evidence Summary
[Supporting evidence with citations]

V. Conclusion
[Actionable harm finding]

Ensure all claims are supported by evidence citations in [Node:State] format.
        """

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic privacy harm drafting."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: evidence, posteriors, case_summary"
            )

        try:
            prompt_func = KernelFunctionFromPrompt(
                function_name=self.name,
                plugin_name="PrivacyHarmPlugin",
                description=self.description,
                prompt=self.prompt_template,
            )

            # Prepare variables, including optional research findings and quality constraints
            variables = {
                "case_summary": kwargs["case_summary"],
                "jurisdiction": kwargs.get("jurisdiction", "US"),
                "evidence": json.dumps(kwargs["evidence"], indent=2),
                "posteriors": json.dumps(kwargs["posteriors"], indent=2),
                "structured_facts": kwargs.get("structured_facts", "") or "",
                "autogen_notes": kwargs.get("autogen_notes", "") or "",
                "key_facts_summary": kwargs.get("key_facts_summary", "") or "",
                "fact_retry_todo": kwargs.get("fact_retry_todo", "") or "",
                "fact_checklist_block": kwargs.get("fact_checklist_block", "") or "",
                "feature_guidance": kwargs.get("feature_guidance", "") or "",
            }
            
            # Add research findings if provided (always include, even if empty)
            if "research_findings" in kwargs and kwargs["research_findings"]:
                try:
                    # Parse if it's a JSON string, otherwise use as-is
                    if isinstance(kwargs["research_findings"], str):
                        research_data = json.loads(kwargs["research_findings"])
                    else:
                        research_data = kwargs["research_findings"]
                    
                    # Format research findings for prompt
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
            
            result = await prompt_func.invoke(kernel, KernelArguments(**variables))
            result_text = extract_prompt_text(result.value).strip()
            return FunctionResult(
                success=True,
                value=result_text,
                metadata={
                    "section_type": "privacy_harm",
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as e:
            logger.error(f"Error in PrivacyHarmSemantic: {e}")
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


class PrivacyHarmPlugin(BaseSKPlugin):
    """Plugin for privacy harm section drafting."""

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.native_function = PrivacyHarmNativeFunction()
        self.semantic_function = PrivacyHarmSemanticFunction()

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="PrivacyHarmPlugin",
            description="Plugin for drafting privacy harm analysis sections",
            version="1.0.0",
            functions=["PrivacyHarmNative", "PrivacyHarmSemantic"]
        )

    async def _register_functions(self) -> None:
        """Register privacy harm functions with the kernel."""

        # Register native function
        @kernel_function(
            name="PrivacyHarmNative",
            description="Generate privacy harm section using deterministic templates"
        )
        async def privacy_harm_native(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US"
        ) -> str:
            """Native function for privacy harm drafting."""
            result = await self.native_function.execute(
                evidence=json.loads(evidence),
                posteriors=json.loads(posteriors),
                case_summary=case_summary,
                jurisdiction=jurisdiction
            )

            if result.success:
                section = result.value
                return f"""
# {section.title}

## Introduction
{section.introduction}

## Harm Analysis
{section.harm_analysis}

## Legal Framework
{section.legal_framework}

## Evidence Summary
{section.evidence_summary}

## Conclusion
{section.conclusion}

---
*Word count: {section.word_count} | Citations: {len(section.citations)}*
                """.strip()
            else:
                raise RuntimeError(f"Privacy harm drafting failed: {result.error}")

        # Register semantic function
        @kernel_function(
            name="PrivacyHarmSemantic",
            description="Generate privacy harm section using LLM with structured prompts"
        )
        async def privacy_harm_semantic(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US",
            research_findings: str = "",
            quality_constraints: str = "",
            structured_facts: str = "",
            autogen_notes: str = "",
            fact_key_summary: str = "",
            filtered_evidence: str = "",
            fact_filter_stats: str = "",
            key_facts_summary: str = "",
            fact_retry_todo: str = "",
            fact_checklist_block: str = "",
            feature_guidance: str = "",
        ) -> str:
            """Semantic function for privacy harm drafting."""
            try:
                research_data = json.loads(research_findings) if research_findings else {}
            except Exception:
                research_data = {}

            try:
                fact_keys = json.loads(fact_key_summary) if fact_key_summary else []
                if not isinstance(fact_keys, list):
                    fact_keys = [str(fact_keys)]
            except Exception:
                fact_keys = [fact_key_summary] if fact_key_summary else []

            try:
                filtered_data = json.loads(filtered_evidence) if filtered_evidence else []
            except Exception:
                filtered_data = []

            try:
                filter_stats_data = json.loads(fact_filter_stats) if fact_filter_stats else {}
            except Exception:
                filter_stats_data = {}

            logger.info(
                "[FACTS][PrivacyHarm] structured_len=%d fact_keys=%s filtered_evidence=%d summary_len=%d todo_len=%d dropped=%s",
                len(structured_facts or ""),
                ", ".join(fact_keys[:5]) or "n/a",
                len(filtered_data) if isinstance(filtered_data, list) else 0,
                len(key_facts_summary or ""),
                len(fact_retry_todo or ""),
                filter_stats_data.get("dropped_count"),
            )

            result = await self.semantic_function.execute(
                kernel=self.kernel,
                evidence=json.loads(evidence),
                posteriors=json.loads(posteriors),
                case_summary=case_summary,
                jurisdiction=jurisdiction,
                research_findings=research_data,
                quality_constraints=quality_constraints or "",
                structured_facts=structured_facts or "",
                autogen_notes=autogen_notes or "",
                key_facts_summary=key_facts_summary or "",
                fact_retry_todo=fact_retry_todo or "",
                fact_checklist_block=fact_checklist_block or "",
                feature_guidance=feature_guidance or "",
            )

            if result.success:
                return result.value
            else:
                raise RuntimeError(f"Privacy harm semantic drafting failed: {result.error}")

        # Store function references
        self._functions["PrivacyHarmNative"] = privacy_harm_native
        self._functions["PrivacyHarmSemantic"] = privacy_harm_semantic


# Export classes
__all__ = [
    "PrivacyHarmPlugin",
    "PrivacyHarmNativeFunction",
    "PrivacyHarmSemanticFunction",
    "PrivacyHarmSection"
]
