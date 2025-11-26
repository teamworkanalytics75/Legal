"""Orchestrator that coordinates planner, writer, double checker, and editor agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .agents import (
    AgentFactory,
    DoubleCheckerAgent,
    EditorAgent,
    ModelConfig,
    PlannerAgent,
    StylistAgent,
    WriterAgent,
)
from .idioms import IdiomSelector
from .insights import CaseInsights
from .tasks import DraftSection, PlanDirective, ReviewFindings, SectionPlan, WriterDeliverable
from .validators import CitationCheckConfig, CitationValidator, StructureValidator


@dataclass(slots=True)
class WriterOrchestratorConfig:
    """Configures how the writer orchestration should behave."""

    model_config: ModelConfig = field(default_factory=ModelConfig)
    include_stylist: bool = True
    max_writer_rounds: int = 1
    required_nodes: Sequence[str] = ()

    # LangChain configuration
    enable_langchain: bool = False
    langchain_db_path: Optional[Path] = None
    langchain_fallback: bool = True


class PlannerParseError(RuntimeError):
    """Raised when the planner output cannot be parsed."""




def _clean_planner_output(output: str) -> str:
    """Normalize planner output to improve JSON parsing resilience."""

    cleaned = output.lstrip("\ufeff\n \t")
    if cleaned.startswith("```"):
        cleaned = cleaned.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Find the first complete JSON object
    start = cleaned.find("{")
    if start == -1:
        return cleaned

    # Count braces to find the end of the JSON object
    brace_count = 0
    end = start
    for i, char in enumerate(cleaned[start:], start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end > start:
        cleaned = cleaned[start : end + 1]

    return cleaned

def _build_planner_prompt(insights: CaseInsights) -> str:
    """Create the planning prompt using case insights."""
    return (
        "You are the Planner agent for a legal memorandum drafting team. "
        "Produce a JSON object with keys 'directive' and 'sections'.\n\n"
        "The 'directive' must include: objective, deliverable_format, tone, style_constraints, citation_expectations.\n"
        "The 'sections' must be a list where each entry includes: section_id, title, objective, key_points, idiom_tags, tone, evidence_refs.\n"
        "Use the Bayesian analysis to drive the outline.\n\n"
        "Bayesian Findings:\n"
        f"{insights.to_prompt_block()}\n"
    )


def _parse_plan(output: str) -> tuple[PlanDirective, List[SectionPlan]]:
    """Parse the planner response into structured plan objects."""
    try:
        print(f"DEBUG: Original output length: {len(output)}")
        print(f"DEBUG: Original output preview: {output[:200]}...")

        cleaned_output = _clean_planner_output(output)
        print(f"DEBUG: Cleaned output length: {len(cleaned_output)}")
        print(f"DEBUG: Cleaned output preview: {cleaned_output[:200]}...")

        data = json.loads(cleaned_output)
    except json.JSONDecodeError as exc:
        raise PlannerParseError(f"Planner did not return valid JSON: {exc}") from exc

    # Debug logging
    print(f"DEBUG: Parsed JSON keys: {list(data.keys())}")
    print(f"DEBUG: Directive value: {data.get('directive')}")
    print(f"DEBUG: Sections value: {data.get('sections')}")

    directive_payload = data.get("directive")
    sections_payload = data.get("sections", [])

    if not isinstance(directive_payload, dict):
        raise PlannerParseError(f"Planner directive missing or not an object. Got: {type(directive_payload)}, value: {directive_payload}")

    directive = PlanDirective(
        objective=str(directive_payload.get("objective", "Draft a memorandum summarizing BN insights.")),
        deliverable_format=str(directive_payload.get("deliverable_format", "Legal memorandum")),
        tone=str(directive_payload.get("tone", "Formal and analytical")),
        style_constraints=[str(item) for item in directive_payload.get("style_constraints", [])],
        citation_expectations=str(
            directive_payload.get(
                "citation_expectations",
                "Embed evidence references as [Node:Outcome] once per paragraph.",
            )
        ),
    )

    sections: List[SectionPlan] = []
    if not isinstance(sections_payload, list) or not sections_payload:
        raise PlannerParseError("Planner did not provide any sections.")

    for entry in sections_payload:
        if not isinstance(entry, dict):
            continue
        section = SectionPlan(
            section_id=str(entry.get("section_id", "section")),
            title=str(entry.get("title", "Unnamed Section")),
            objective=str(entry.get("objective", "")),
            key_points=[str(item) for item in entry.get("key_points", [])],
            idiom_tags=[str(tag) for tag in entry.get("idiom_tags", [])],
            tone=str(entry.get("tone", "")) if entry.get("tone") else None,
            evidence_refs=[str(ref) for ref in entry.get("evidence_refs", [])],
        )
        sections.append(section)

    if not sections:
        raise PlannerParseError("Planner sections list was empty after parsing.")

    return directive, sections


class WriterOrchestrator:
    """Coordinates the multi-agent legal writing workflow."""

    def __init__(
        self,
        config: Optional[WriterOrchestratorConfig] = None,
        idiom_selector: Optional[IdiomSelector] = None,
    ) -> None:
        self._config = config or WriterOrchestratorConfig()
        self._factory = AgentFactory(self._config.model_config)
        self._idioms = idiom_selector
        self._planner = PlannerAgent(
            self._factory,
            name="Planner",
            system_message="Design structured outlines grounded in Bayesian insights. Respond in JSON only.",
        )
        self._writer = WriterAgent(
            self._factory,
            name="Writer",
            system_message="Draft legal sections with precise citations and idiomatic phrasing where appropriate.",
        )
        self._checker = DoubleCheckerAgent(
            self._factory,
            name="DoubleChecker",
            system_message="Verify factual consistency, cite mismatches, and respond in concise bullet points.",
        )
        self._editor = EditorAgent(
            self._factory,
            name="Editor",
            system_message="Synthesize drafts into a polished memorandum while respecting structure and citations.",
        )
        self._stylist = (
            StylistAgent(
                self._factory,
                name="Stylist",
                system_message="Suggest global improvements to tone, coherence, and rhetorical force in bullet form.",
            )
            if self._config.include_stylist
            else None
        )

    async def close(self) -> None:
        """Dispose the shared model client."""
        await self._factory.close()

    async def run(self, insights: CaseInsights) -> WriterDeliverable:
        """Execute the complete writing workflow."""
        directive, sections = await self._plan(insights)
        drafts = await self._draft_sections(directive, sections, insights)
        reviews = await self._double_check(drafts, sections, directive)
        edited_document = await self._edit_document(directive, drafts, reviews, insights)
        metadata = {}
        if self._stylist is not None:
            metadata["stylist_feedback"] = await self._stylist_feedback(drafts, directive)
        return WriterDeliverable(
            plan=directive,
            sections=drafts,
            edited_document=edited_document,
            reviews=reviews,
            metadata=metadata,
        )

    async def _plan(self, insights: CaseInsights) -> tuple[PlanDirective, List[SectionPlan]]:
        prompt = _build_planner_prompt(insights)
        response = await self._planner.run(task=prompt)
        return _parse_plan(response)

    async def _draft_sections(
        self,
        directive: PlanDirective,
        sections: Sequence[SectionPlan],
        insights: CaseInsights,
    ) -> List[DraftSection]:
        drafts: List[DraftSection] = []
        context_block = insights.to_prompt_block()
        instruction_header = (
            f"Global Objective: {directive.objective}\n"
            f"Tone: {directive.tone}\n"
            f"Citation Rules: {directive.citation_expectations}\n"
        )
        for section in sections:
            idioms: List[str] = []
            if self._idioms is not None and section.idiom_tags:
                idioms = self._idioms.render_many(section.idiom_tags, count=2)
            idiom_block = "\n".join(f"Suggested Idiom: {phrase}" for phrase in idioms) if idioms else ""
            writer_prompt = (
                f"{instruction_header}\n"
                f"{section.to_prompt()}\n\n"
                f"Bayesian Context:\n{context_block}\n"
            )
            if idiom_block:
                writer_prompt += f"\n{idiom_block}\n"
            body = await self._writer.run(task=writer_prompt)
            drafts.append(DraftSection(section_id=section.section_id, title=section.title, body=body))
        return drafts

    async def _double_check(
        self,
        drafts: Sequence[DraftSection],
        expected_sections: Sequence[SectionPlan],
        directive: PlanDirective,
    ) -> List[ReviewFindings]:

        findings: List[ReviewFindings] = []
        if self._config.required_nodes:
            validator = CitationValidator(CitationCheckConfig(required_nodes=self._config.required_nodes))
        else:
            validator = None
        structure_validator = StructureValidator(section.section_id for section in expected_sections)
        findings.extend(structure_validator.run(list(drafts)))
        for draft in drafts:
            if validator is not None:
                findings.extend(validator.run(draft))
            checker_prompt = (
                f"Evaluate the following section for factual consistency with the BN plan.\n"
                f"Directive: {directive.to_prompt()}\n"
                f"Section Title: {draft.title}\n"
                "Respond with bullet findings and cite problematic sentences explicitly."
                f"\n\nSection Body:\n{draft.body}\n"
            )
            feedback = await self._checker.run(task=checker_prompt)
            findings.append(
                ReviewFindings(
                    section_id=draft.section_id,
                    severity="info",
                    message="Double-check feedback",
                    suggestions=feedback.strip(),
                )
            )
        return findings

    async def _edit_document(
        self,
        directive: PlanDirective,
        drafts: Sequence[DraftSection],
        reviews: Sequence[ReviewFindings],
        insights: CaseInsights,
    ) -> str:
        combined_sections = "\n\n".join(f"# {draft.title}\n\n{draft.body}" for draft in drafts)
        review_block = "\n".join(
            f"- [{finding.severity.upper()}] {finding.section_id or 'general'}: {finding.message}"
            for finding in reviews
            if finding.message
        )
        editor_prompt = (
            f"You are the final editor. Apply directive rules and produce cohesive prose.\n"
            f"Directive Summary:\n{directive.to_prompt()}\n\n"
            f"Bayesian Findings:\n{insights.to_prompt_block()}\n\n"
            f"Draft Sections:\n{combined_sections}\n\n"
            f"Quality Review Notes:\n{review_block or 'No outstanding issues reported.'}\n"
            "Return the polished memorandum only."
        )
        return await self._editor.run(task=editor_prompt)

    async def _stylist_feedback(self, drafts: Sequence[DraftSection], directive: PlanDirective) -> str:
        if self._stylist is None:
            return ""
        body = "\n\n".join(f"{draft.title}:\n{draft.body}" for draft in drafts)
        stylist_prompt = (
            f"Directive:\n{directive.to_prompt()}\n\nDraft Content:\n{body}\n\n"
            "Provide high-level style recommendations in bullet form."
        )
        return await self._stylist.run(task=stylist_prompt)


__all__ = ["WriterOrchestrator", "WriterOrchestratorConfig", "PlannerParseError"]





