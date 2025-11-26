"""Motion section drafting plugin for Semantic Kernel."""

import json
import logging
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction, KernelArguments
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from ..base_plugin import kernel_function
from ..utils import extract_prompt_text

from ..base_plugin import BaseSKPlugin, PluginMetadata, DraftingFunction, FunctionResult

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
- Use ONLY the facts listed above and in Structured Facts below; do not add new parties, dates, or evidence.
- Mention every named person, date, and number explicitly in the section (rephrase is fine, omission is not).
- Cite supporting evidence with [Node:State] for each factual assertion.
- If information is not provided above, leave it out instead of guessing.
- If the Missing Facts TODO block contains items, explicitly incorporate each item into this section.
"""


SECTION_PROMPTS: Dict[str, str] = {
    "introduction": """
You are a senior legal writer. Draft a persuasive introduction for a motion to seal and proceed under pseudonym.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Frame the urgent privacy/safety concerns and requested relief in 2–3 paragraphs.
2. Reference the most salient evidence and case-law themes without revealing sensitive names.
3. Highlight why immediate sealing/pseudonym relief is necessary and narrowly tailored.
4. Keep tone formal, confident, and action-oriented (target 200–300 words).
5. Use [Node:State] citations where applicable.
""",
    "legal_standard": """
You are a legal analyst. Draft the legal standard section for a motion to seal and proceed pseudonymously.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Identify the governing sealing standard and pseudonym factors in this jurisdiction.
2. Summarize the multi-factor tests (privacy interest, harm, balancing, alternatives) with citations.
3. Integrate relevant precedents from the research findings.
4. Keep tone authoritative and structured with clear subheadings.
5. Target 350–450 words, using [Node:State] citations where applicable.
""",
    "danger_safety": """
Draft the danger/safety section analyzing concrete harms if the motion is denied.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Provide a fact-based narrative of harassment, retaliation, or doxxing risks.
2. Tie each risk to evidence/posteriors and supporting cases.
3. Explain why sealing/pseudonym relief is the least restrictive means to avoid irreparable harm.
4. Target 250–400 words, formal tone, [Node:State] citations.
""",
    "public_interest": """
Draft the public-interest section balancing transparency against privacy/safety needs.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Explain why sealing limited materials does not undermine public oversight.
2. Reference precedents showing courts routinely protect similar sensitive data.
3. Offer meaningful alternatives (redactions, limited access) already considered.
4. Target 250–350 words, cite research cases where possible.
""",
    "balancing_test": """
Draft the balancing-test section weighing privacy and public access factors.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Explicitly walk through each balancing factor used by the court.
2. Tie every factor to concrete facts, evidence citations, and precedent.
3. Emphasize why privacy harms outweigh generalized public curiosity.
4. Target 300–400 words, structured with subheadings or numbered factors.
""",
    "protective_measures": """
Draft the protective-measures section summarizing proposed sealing scope and safeguards.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Specify exactly which documents/identifiers will be sealed or pseudonymized.
2. Describe redaction protocols, limited-access procedures, and monitoring.
3. Show the request is narrowly tailored and reviewable.
4. Target 200–300 words, formal tone, bullet lists encouraged.
""",
    "conclusion": """
Draft the conclusion and requested-relief section for the motion.

Case Summary: {{$case_summary}}
Jurisdiction: {{$jurisdiction}}
Evidence Highlights:
{{$evidence}}
Bayesian Posteriors:
{{$posteriors}}

""" + FACT_GUARD_BLOCK + """

Structured Facts:
{{$structured_facts}}

AutoGen Notes:
{{$autogen_notes}}

{{$research_findings}}

QUALITY CONSTRAINTS:
{{$quality_constraints}}

Requirements:
1. Summarize key reasons sealing/pseudonym relief is warranted.
2. Itemize the requested orders (sealing scope, pseudonym, additional relief).
3. Include a respectful closing paragraph and signature block placeholder.
4. Target 200–250 words, concise and action oriented.
"""
}


class MotionSectionSemanticFunction(DraftingFunction):
    """Semantic drafting function for a specific motion section."""

    def __init__(self, section_key: str, prompt_template: str):
        super().__init__(
            name=f"Draft{section_key.title().replace('_', '')}Semantic",
            description=f"Generate the {section_key.replace('_', ' ')} section using LLM prompts",
            section_type=section_key
        )
        self.section_key = section_key
        self.prompt_template = prompt_template

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic drafting for the section."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: evidence, posteriors, case_summary"
            )

        try:
            semantic_func = KernelFunctionFromPrompt(
                function_name=self.name,
                plugin_name="MotionSectionsPlugin",
                description=self.description,
                prompt=self.prompt_template,
            )

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

            research_text = self._format_research_findings(kwargs.get("research_findings"))
            variables["research_findings"] = research_text
            variables["quality_constraints"] = kwargs.get("quality_constraints", "")

            arguments = KernelArguments(**variables)
            if hasattr(semantic_func, "invoke"):
                result = await semantic_func.invoke(kernel, arguments=arguments)
            else:  # Fallback for legacy KernelFunction objects
                result = await kernel.invoke_function(semantic_func, variables=variables)
            result_text = extract_prompt_text(result.value).strip()
            return FunctionResult(
                success=True,
                value=result_text,
                metadata={
                    "section_type": self.section_key,
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as exc:
            logger.error(f"Error drafting {self.section_key} section: {exc}")
            return FunctionResult(success=False, value=None, error=str(exc))

    def _format_research_findings(self, research_data: Optional[Any]) -> str:
        """Format research findings for prompt inclusion."""
        if not research_data:
            return ""

        try:
            if isinstance(research_data, str):
                research = json.loads(research_data) if research_data.strip() else {}
            else:
                research = research_data
        except Exception:
            return ""

        formatted = []
        explanations = research.get("explanations", {})
        overall = explanations.get("overall", {})
        if overall.get("summary"):
            formatted.append(f"Research Summary: {overall['summary']}")

        cases = research.get("cases", [])[:5]
        if cases:
            formatted.append("\nTop Relevant Cases:")
            for idx, case in enumerate(cases, 1):
                case_name = case.get("case_name", "Unknown")
                court = case.get("court", "Unknown")
                similarity = case.get("similarity_score", 0.0)
                formatted.append(f"{idx}. {case_name} ({court}) - Relevance {similarity:.2f}")

        themes = explanations.get("by_theme", {})
        if themes:
            formatted.append("\nThemes:")
            for theme, data in list(themes.items())[:3]:
                formatted.append(f"- {theme}: {data.get('count', 0)} cases")

        return "\n".join(formatted)


class MotionSectionsPlugin(BaseSKPlugin):
    """Plugin registering semantic drafting functions for multiple motion sections."""

    def __init__(self, kernel: Kernel):
        self.section_functions: Dict[str, MotionSectionSemanticFunction] = {
            section_key: MotionSectionSemanticFunction(section_key, prompt)
            for section_key, prompt in SECTION_PROMPTS.items()
        }
        super().__init__(kernel)

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MotionSectionsPlugin",
            description="Semantic drafting functions for motion sections",
            version="1.0.0",
            functions=[func.name for func in self.section_functions.values()]
        )

    async def _register_functions(self) -> None:
        """Register semantic drafting functions for each section."""

        def _register(section_key: str, section_function: MotionSectionSemanticFunction) -> KernelFunction:
            @kernel_function(
                name=section_function.name,
                description=section_function.description
            )
            async def section_semantic(
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
                """Semantic drafting entry point for a motion section."""
                try:
                    evidence_data = json.loads(evidence) if evidence else []
                except Exception:
                    evidence_data = []

                try:
                    posteriors_data = json.loads(posteriors) if posteriors else {}
                except Exception:
                    posteriors_data = {}

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
                    "[FACTS][MotionSections:%s] structured_len=%d fact_keys=%s filtered_evidence=%d summary_len=%d todo_len=%d dropped=%s",
                    section_key,
                    len(structured_facts or ""),
                    ", ".join(fact_keys[:5]) or "n/a",
                    len(filtered_data) if isinstance(filtered_data, list) else 0,
                    len(key_facts_summary or ""),
                    len(fact_retry_todo or ""),
                    filter_stats_data.get("dropped_count"),
                )

                result = await section_function.execute(
                    kernel=self.kernel,
                    evidence=evidence_data,
                    posteriors=posteriors_data,
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
                raise RuntimeError(f"{section_key} drafting failed: {result.error}")

            return section_semantic

        for key, section_function in self.section_functions.items():
            registered = _register(key, section_function)
            self._functions[section_function.name] = registered


__all__ = [
    "MotionSectionsPlugin",
    "MotionSectionSemanticFunction"
]
