"""Causation Analysis drafting function for Semantic Kernel."""

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
class CausalLink:
    """Individual causal link in the chain."""
    cause: str
    effect: str
    strength: float
    evidence_refs: List[str]
    legal_standard: str


@dataclass
class CausationSection:
    """Structured output for causation analysis section."""
    title: str
    introduction: str
    causal_chain: List[CausalLink]
    legal_framework: str
    analysis: str
    conclusion: str
    citations: List[str]
    word_count: int


class CausationAnalysisNativeFunction(DraftingFunction):
    """Native function for causation analysis section drafting."""

    def __init__(self):
        super().__init__(
            name="CausationAnalysisNative",
            description="Generate causation analysis section using deterministic templates",
            section_type="causation_analysis"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute causation analysis drafting."""
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

            # Build causal chain from evidence and posteriors
            causal_chain = self._build_causal_chain(evidence, posteriors)

            # Format citations
            citations = self._format_citations(evidence)

            # Generate structured section
            section = self._generate_causation_section(
                causal_chain, case_summary, jurisdiction, citations
            )

            return FunctionResult(
                success=True,
                value=section,
                metadata={"section_type": "causation_analysis", "links_count": len(causal_chain)}
            )

        except Exception as e:
            logger.error(f"Error in CausationAnalysisNative: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _build_causal_chain(self, evidence: Dict[str, Any], posteriors: Dict[str, Any]) -> List[CausalLink]:
        """Build causal chain from evidence and posteriors."""

        links = []

        # Look for causation-related evidence
        causation_evidence = {}
        for node, state in evidence.items():
            if any(keyword in str(node).lower() for keyword in ["cause", "effect", "lead", "result", "causation"]):
                causation_evidence[node] = state

        # Build links based on high-confidence posteriors
        for node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.7:
                # Find potential causes
                potential_causes = self._find_potential_causes(node, evidence, posteriors)

                for cause in potential_causes:
                    link = CausalLink(
                        cause=cause,
                        effect=node,
                        strength=prob,
                        evidence_refs=[f"[{cause}:{evidence.get(cause, 'Unknown')}]", f"[{node}:{prob:.1%}]"],
                        legal_standard=self._get_legal_standard(cause, node, jurisdiction="US")
                    )
                    links.append(link)

        # Add evidence-based links
        for cause_node, cause_state in evidence.items():
            for effect_node, effect_prob in posteriors.items():
                if isinstance(effect_prob, (int, float)) and effect_prob > 0.6:
                    if self._is_causally_related(cause_node, effect_node):
                        link = CausalLink(
                            cause=f"{cause_node}: {cause_state}",
                            effect=f"{effect_node} (Confidence: {effect_prob:.1%})",
                            strength=effect_prob,
                            evidence_refs=[f"[{cause_node}:{cause_state}]", f"[{effect_node}:{effect_prob:.1%}]"],
                            legal_standard=self._get_legal_standard(cause_node, effect_node, jurisdiction="US")
                        )
                        links.append(link)

        return links

    def _find_potential_causes(self, effect: str, evidence: Dict[str, Any], posteriors: Dict[str, Any]) -> List[str]:
        """Find potential causes for an effect."""

        causes = []

        # Look for evidence that could cause this effect
        for node, state in evidence.items():
            if self._is_causally_related(node, effect):
                causes.append(f"{node}: {state}")

        # Look for high-confidence posteriors that could cause this effect
        for node, prob in posteriors.items():
            if isinstance(prob, (int, float)) and prob > 0.8:
                if self._is_causally_related(node, effect):
                    causes.append(f"{node} (Confidence: {prob:.1%})")

        return causes

    def _is_causally_related(self, cause: str, effect: str) -> bool:
        """Determine if two nodes are causally related."""

        cause_lower = str(cause).lower()
        effect_lower = str(effect).lower()

        # Simple heuristic for causal relationships
        causal_patterns = [
            ("email", "awareness"),
            ("awareness", "action"),
            ("action", "harm"),
            ("violation", "damage"),
            ("conduct", "result"),
            ("breach", "liability")
        ]

        for cause_pattern, effect_pattern in causal_patterns:
            if cause_pattern in cause_lower and effect_pattern in effect_lower:
                return True

        return False

    def _get_legal_standard(self, cause: str, effect: str, jurisdiction: str) -> str:
        """Get applicable legal standard for causation."""

        if jurisdiction == "US":
            return "But-for causation and proximate cause"
        elif jurisdiction == "EU":
            return "Direct causation and foreseeability"
        elif jurisdiction == "CA":
            return "Substantial factor causation"
        else:
            return "Applicable causation standard"

    def _generate_causation_section(
        self,
        causal_chain: List[CausalLink],
        case_summary: str,
        jurisdiction: str,
        citations: str
    ) -> CausationSection:
        """Generate causation section using templates."""

        # Title
        title = "Causation Analysis"

        # Introduction
        introduction = f"""
This section analyzes the causal relationship between the alleged conduct and
the resulting harm. The analysis examines both factual causation (but-for cause)
and legal causation (proximate cause) under applicable {jurisdiction} law.
        """.strip()

        # Legal framework
        legal_framework = self._get_legal_framework(jurisdiction)

        # Analysis
        analysis = self._analyze_causal_chain(causal_chain)

        # Conclusion
        conclusion = f"""
The causation analysis demonstrates a clear causal chain {citations} that
establishes both factual and legal causation. The evidence supports a finding
that the alleged conduct was both a but-for cause and proximate cause of the harm.
        """.strip()

        # Calculate word count
        full_text = f"{introduction} {legal_framework} {analysis} {conclusion}"
        word_count = len(full_text.split())

        return CausationSection(
            title=title,
            introduction=introduction,
            causal_chain=causal_chain,
            legal_framework=legal_framework,
            analysis=analysis,
            conclusion=conclusion,
            citations=citations.split() if citations else [],
            word_count=word_count
        )

    def _get_legal_framework(self, jurisdiction: str) -> str:
        """Get legal framework for causation analysis."""

        frameworks = {
            "US": """
Under United States law, causation requires both factual causation (but-for cause)
and legal causation (proximate cause). Factual causation asks whether the harm
would have occurred but for the defendant's conduct. Legal causation asks whether
the harm was a foreseeable result of the conduct.
            """,
            "EU": """
Under European Union law, causation analysis focuses on direct causation and
foreseeability. The harm must be a direct result of the conduct, and the
harm must have been foreseeable at the time of the conduct.
            """,
            "CA": """
Under California law, causation may be established through substantial factor
analysis. The conduct need not be the sole cause, but must be a substantial
factor in bringing about the harm.
            """
        }

        return frameworks.get(jurisdiction, frameworks["US"]).strip()

    def _analyze_causal_chain(self, causal_chain: List[CausalLink]) -> str:
        """Analyze the causal chain."""

        if not causal_chain:
            return """
The causation analysis indicates that additional evidence may be required to
establish a clear causal relationship between the alleged conduct and the harm.
            """.strip()

        strong_links = [link for link in causal_chain if link.strength > 0.8]
        moderate_links = [link for link in causal_chain if 0.6 <= link.strength <= 0.8]

        analysis_parts = []

        if strong_links:
            analysis_parts.append(f"""
The analysis reveals {len(strong_links)} strong causal links with high confidence:

{chr(10).join(f"• {link.cause} → {link.effect} (Strength: {link.strength:.1%})" for link in strong_links)}
            """.strip())

        if moderate_links:
            analysis_parts.append(f"""
Additionally, {len(moderate_links)} moderate causal links support the causation analysis:

{chr(10).join(f"• {link.cause} → {link.effect} (Strength: {link.strength:.1%})" for link in moderate_links)}
            """.strip())

        analysis_parts.append("""
The causal chain demonstrates a logical progression from the alleged conduct
to the resulting harm, satisfying both factual and legal causation requirements.
        """.strip())

        return "\n\n".join(analysis_parts)


class CausationAnalysisSemanticFunction(DraftingFunction):
    """Semantic function for causation analysis section drafting using LLM."""

    def __init__(self):
        super().__init__(
            name="CausationAnalysisSemantic",
            description="Generate causation analysis section using LLM with structured prompts",
            section_type="causation_analysis"
        )
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """Get the prompt template for causation analysis drafting."""
        return """
You are a legal expert specializing in causation analysis. Draft a comprehensive causation analysis section for a legal memorandum.

CONTEXT:
Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence: {{$evidence}}
Bayesian Network Posteriors: {{$posteriors}}

{{$research_findings}}

{{$quality_constraints}}

REQUIREMENTS:
1. Analyze both factual causation (but-for cause) and legal causation (proximate cause)
2. Identify causal links in the evidence chain
3. Apply appropriate legal standards for the jurisdiction
4. Cite evidence using [Node:State] format
5. Maintain analytical, objective tone
6. Target 800-1200 words
7. Incorporate relevant case law from research findings if provided
8. Follow quality constraints and feature targets if provided

OUTPUT FORMAT:
Title: Causation Analysis

I. Introduction
[Overview of causation requirements]

II. Legal Framework
[Applicable causation standards]

III. Factual Causation Analysis
[But-for cause analysis with evidence]

IV. Legal Causation Analysis
[Proximate cause analysis]

V. Conclusion
[Causation finding]

Ensure all causal links are supported by evidence citations in [Node:State] format.
        """

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic causation analysis drafting."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: evidence, posteriors, case_summary"
            )

        try:
            prompt_func = KernelFunctionFromPrompt(
                function_name=self.name,
                plugin_name="CausationAnalysisPlugin",
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
                    "section_type": "causation_analysis",
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as e:
            logger.error(f"Error in CausationAnalysisSemantic: {e}")
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


class CausationAnalysisPlugin(BaseSKPlugin):
    """Plugin for causation analysis section drafting."""

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.native_function = CausationAnalysisNativeFunction()
        self.semantic_function = CausationAnalysisSemanticFunction()

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="CausationAnalysisPlugin",
            description="Plugin for drafting causation analysis sections",
            version="1.0.0",
            functions=["CausationAnalysisNative", "CausationAnalysisSemantic"]
        )

    async def _register_functions(self) -> None:
        """Register causation analysis functions with the kernel."""

        # Register native function
        @kernel_function(
            name="CausationAnalysisNative",
            description="Generate causation analysis section using deterministic templates"
        )
        async def causation_analysis_native(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US"
        ) -> str:
            """Native function for causation analysis drafting."""
            result = await self.native_function.execute(
                evidence=json.loads(evidence),
                posteriors=json.loads(posteriors),
                case_summary=case_summary,
                jurisdiction=jurisdiction
            )

            if result.success:
                section = result.value
                links_text = "\n".join(f"• {link.cause} → {link.effect}" for link in section.causal_chain)
                return f"""
# {section.title}

## Introduction
{section.introduction}

## Legal Framework
{section.legal_framework}

## Causal Chain Analysis
{links_text}

## Analysis
{section.analysis}

## Conclusion
{section.conclusion}

---
*Word count: {section.word_count} | Causal links: {len(section.causal_chain)}*
                """.strip()
            else:
                raise RuntimeError(f"Causation analysis drafting failed: {result.error}")

        # Register semantic function
        @kernel_function(
            name="CausationAnalysisSemantic",
            description="Generate causation analysis section using LLM with structured prompts"
        )
        async def causation_analysis_semantic(
            evidence: str,
            posteriors: str,
            case_summary: str,
            jurisdiction: str = "US"
        ) -> str:
            """Semantic function for causation analysis drafting."""
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
                raise RuntimeError(f"Causation analysis semantic drafting failed: {result.error}")

        # Store function references
        self._functions["CausationAnalysisNative"] = causation_analysis_native
        self._functions["CausationAnalysisSemantic"] = causation_analysis_semantic


# Export classes
__all__ = [
    "CausationAnalysisPlugin",
    "CausationAnalysisNativeFunction",
    "CausationAnalysisSemanticFunction",
    "CausationSection",
    "CausalLink"
]
