#!/usr/bin/env python3
"""
Citation Retrieval Plugin - Atomic SK plugin for citation requirements.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

if TYPE_CHECKING:
    from .document_structure import DocumentStructure

logger = logging.getLogger(__name__)


class CitationRetrievalPlugin(BaseFeaturePlugin):
    """Atomic plugin for citation requirements and retrieval."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "citation_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        self.top_citations = self._load_top_citations()
        self.winning_citations = self._load_winning_citations()
        logger.info(f"CitationRetrievalPlugin initialized with {len(self.top_citations)} top citations")

    def _load_top_citations(self) -> List[Dict[str, Any]]:
        """Load top 20 motion citations from config file."""
        try:
            config_path = Path(__file__).parents[3] / "config" / "top_motion_citations.json"
            if not config_path.exists():
                # Try alternative path
                config_path = Path(__file__).parents[2] / "config" / "top_motion_citations.json"

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    citations = config.get("citations", [])
                logger.info(f"Loaded {len(citations)} top citations from config")
                return citations
            else:
                logger.warning(f"Top citations config not found at {config_path}")
                return []
        except Exception as e:
            logger.error(f"Failed to load top citations: {e}")
            return []

    def _load_winning_citations(self) -> List[Dict[str, Any]]:
        """Load winning citations database (fallback/legacy)."""
        try:
            citations_file = Path(__file__).parents[3] / "case_law_data" / "analysis" / "winning_citations_federal.json"
            if citations_file.exists():
                with open(citations_file, 'r') as f:
                    citations = json.load(f)
                logger.info(f"Loaded {len(citations)} winning citations (legacy)")
                return citations
            else:
                # Use top citations as winning citations if legacy file not found
                return self.top_citations
        except Exception as e:
            logger.error(f"Failed to load winning citations: {e}")
            return self.top_citations

    async def analyze_citation_strength(self, draft_text: str) -> FunctionResult:
        """Analyze citation strength and coverage in draft."""
        try:
            import re

            # Extract citations from text
            citation_pattern = r'\b\d+\s+(?:F\.|Mass\.|U\.S\.)\s+(?:Supp\.|App\.|2d|3d)?\s*\d+'
            citations_found = re.findall(citation_pattern, draft_text)

            # Check against top citations (prioritized) and winning citations
            top_cite_list = [cite["citation"] for cite in self.top_citations]
            winning_cite_list = [cite["citation"] for cite in self.winning_citations]
            all_preferred_citations = list(set(top_cite_list + winning_cite_list))

            # Separate top citations vs other winning citations
            top_cites_used = [cite for cite in citations_found if cite in top_cite_list]
            winning_cites_used = [cite for cite in citations_found if cite in all_preferred_citations]

            # Calculate citation density by section
            sections = self._extract_sections(draft_text)
            section_density = {}

            for section_name, section_text in sections.items():
                section_citations = re.findall(citation_pattern, section_text)
                word_count = len(section_text.split())
                density = len(section_citations) / max(word_count / 100, 1)
                section_density[section_name] = {
                    "citations": len(section_citations),
                    "density_per_100_words": density,
                    "citation_list": section_citations
                }

            # Get requirements from rules
            section_requirements = self.rules.get("sections", {})

            # Evaluate against requirements
            evaluation = {}
            for section_name, requirements in section_requirements.items():
                if section_name in section_density:
                    current = section_density[section_name]
                    min_required = requirements.get("min_citations", 1)
                    target_density = requirements.get("density_per_100_words", 1.0)

                    evaluation[section_name] = {
                        "current_citations": current["citations"],
                        "required_citations": min_required,
                        "current_density": current["density_per_100_words"],
                        "target_density": target_density,
                        "meets_requirements": current["citations"] >= min_required,
                        "density_adequate": current["density_per_100_words"] >= target_density
                    }

            # Overall score
            total_citations = len(citations_found)
            top_citations_count = len(top_cites_used)
            winning_citations_count = len(winning_cites_used)
            min_total = self.rules.get("minimum_threshold", 3)

            strength_score = min(1.0, total_citations / max(min_total, 1))
            top_score = min(1.0, top_citations_count / max(total_citations, 1)) if total_citations > 0 else 0.0
            winning_score = min(1.0, winning_citations_count / max(total_citations, 1)) if total_citations > 0 else 0.0

            return FunctionResult(
                success=True,
                value={
                    "total_citations": total_citations,
                    "top_citations_used": top_citations_count,
                    "top_citations_list": top_cites_used,
                    "winning_citations_used": winning_citations_count,
                    "winning_citations_list": winning_cites_used,
                    "section_analysis": section_density,
                    "requirements_evaluation": evaluation,
                    "strength_score": strength_score,
                    "top_score": top_score,
                    "winning_score": winning_score,
                    "meets_minimum": total_citations >= min_total,
                    "recommendations": self._get_citation_recommendations(evaluation, winning_cites_used, top_cites_used)
                },
                metadata={"analysis_type": "citation_strength"}
            )

        except Exception as e:
            logger.error(f"Citation analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from motion text."""
        import re

        sections = {}

        # Look for section headers
        section_patterns = [
            (r'(?:^|\n)\s*(?:I\.|1\.|INTRODUCTION)', 'introduction'),
            (r'(?:^|\n)\s*(?:II\.|2\.|PRIVACY HARM|HARM ANALYSIS)', 'harm_analysis'),
            (r'(?:^|\n)\s*(?:III\.|3\.|LEGAL STANDARD)', 'legal_standard'),
            (r'(?:^|\n)\s*(?:IV\.|4\.|ARGUMENT)', 'argument'),
            (r'(?:^|\n)\s*(?:V\.|5\.|CONCLUSION)', 'conclusion')
        ]

        for pattern, section_name in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                start = matches[0].start()
                # Find next section or end of text
                next_section_start = len(text)
                for other_pattern, _ in section_patterns:
                    if other_pattern != pattern:
                        other_matches = list(re.finditer(other_pattern, text[start+1:], re.IGNORECASE | re.MULTILINE))
                        if other_matches:
                            next_section_start = min(next_section_start, start + 1 + other_matches[0].start())

                sections[section_name] = text[start:next_section_start]

        return sections

    def _get_citation_recommendations(self, evaluation: Dict, winning_cites_used: List[str], top_cites_used: List[str]) -> List[Dict]:
        """Get recommendations for improving citations."""
        recommendations = []

        # Check section requirements
        for section_name, eval_data in evaluation.items():
            if not eval_data["meets_requirements"]:
                recommendations.append({
                    "type": "insufficient_citations",
                    "section": section_name,
                    "priority": "high",
                    "message": f"Add {eval_data['required_citations'] - eval_data['current_citations']} more citations to {section_name}",
                    "current": eval_data["current_citations"],
                    "required": eval_data["required_citations"]
                })

            if not eval_data["density_adequate"]:
                recommendations.append({
                    "type": "low_density",
                    "section": section_name,
                    "priority": "medium",
                    "message": f"Increase citation density in {section_name}",
                    "current_density": eval_data["current_density"],
                    "target_density": eval_data["target_density"]
                })

        # Check for missing top citations (highest priority)
        top_cite_list = [cite["citation"] for cite in self.top_citations[:10]]
        missing_top = [cite for cite in top_cite_list if cite not in top_cites_used]

        if missing_top:
            recommendations.append({
                "type": "missing_top_citations",
                "priority": "high",
                "message": f"Add these top-performing citations: {', '.join(missing_top[:3])}",
                "missing_citations": missing_top[:5],
                "source": "top_motion_citations"
            })

        # Check for missing winning citations (fallback)
        top_winning_cites = [cite["citation"] for cite in self.winning_citations[:10]]
        missing_winning = [cite for cite in top_winning_cites if cite not in winning_cites_used and cite not in top_cite_list]

        if missing_winning:
            recommendations.append({
                "type": "missing_winning_citations",
                "priority": "medium",
                "message": f"Consider adding these high-success citations: {', '.join(missing_winning[:3])}",
                "missing_citations": missing_winning[:5]
            })

        return recommendations

    async def suggest_citation_improvements(self, draft_text: str) -> FunctionResult:
        """Suggest specific citation improvements."""
        try:
            analysis_result = await self.analyze_citation_strength(draft_text)
            if not analysis_result.success:
                return analysis_result

            analysis_data = analysis_result.value
            recommendations = analysis_data["recommendations"]

            # Generate specific suggestions
            suggestions = []

            for rec in recommendations:
                if rec["type"] == "insufficient_citations":
                    suggestions.append({
                        "action": "add_citations",
                        "section": rec["section"],
                        "count": rec["required"] - rec["current"],
                        "suggested_citations": self._get_suggested_citations_for_section(rec["section"]),
                        "priority": rec["priority"]
                    })

                elif rec["type"] == "missing_winning_citations":
                    suggestions.append({
                        "action": "add_winning_citations",
                        "citations": rec["missing_citations"],
                        "context": "Add these high-success citations to strengthen your argument",
                        "priority": rec["priority"]
                    })

            return FunctionResult(
                success=True,
                value={
                    "suggestions": suggestions,
                    "total_suggestions": len(suggestions),
                    "high_priority": len([s for s in suggestions if s["priority"] == "high"]),
                    "analysis_summary": {
                        "total_citations": analysis_data["total_citations"],
                        "winning_citations": analysis_data["winning_citations_used"],
                        "strength_score": analysis_data["strength_score"]
                    }
                },
                metadata={"analysis_type": "citation_improvements"}
            )

        except Exception as e:
            logger.error(f"Citation improvement suggestions failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_suggested_citations_for_section(self, section_name: str) -> List[str]:
        """Get suggested citations for a specific section, prioritizing top citations."""
        # Use top citations first, then fallback to section-specific defaults
        if self.top_citations:
            # Return top 5 citations as suggestions
            return [cite["citation"] for cite in self.top_citations[:5]]

        # Fallback to section-specific defaults if no top citations available
        section_citations = {
            "introduction": ["469 U.S. 310", "353 Mass. 614"],
            "harm_analysis": ["605 F. Supp. 3", "740 F. Supp. 2"],
            "legal_standard": ["457 U.S. 307", "462 U.S. 213"],
            "argument": ["715 F.2d 1", "895 F.2d 38"],
            "conclusion": ["637 F.3d 53", "490 F. Supp. 3"]
        }

        return section_citations.get(section_name, ["469 U.S. 310", "353 Mass. 614"])

    def _normalize_section_key(self, section_name: Optional[str]) -> str:
        """Normalize section identifiers to predictable keys."""
        if not section_name:
            return "argument"
        normalized = section_name.strip().lower().replace(" ", "_")
        return normalized or "argument"

    def _section_keywords(self, section_name: str) -> List[str]:
        """Return keywords used to match document paragraphs for a section."""
        normalized = self._normalize_section_key(section_name)
        keyword_map = {
            "introduction": ["introduction", "summary", "overview"],
            "harm_analysis": ["harm", "privacy", "safety", "risk"],
            "privacy_harm": ["privacy", "harm", "safety"],
            "legal_standard": ["legal standard", "standard", "rule", "precedent"],
            "argument": ["argument", "analysis", "court", "authority"],
            "conclusion": ["conclusion", "relief", "request", "respectfully"],
        }
        keywords = keyword_map.get(normalized, [])
        display = normalized.replace("_", " ")
        if display not in keywords:
            keywords.append(display)
        return keywords

    def _find_section_anchor_index(self, structure: Optional["DocumentStructure"], section_name: str) -> Optional[int]:
        """Return the paragraph index that best anchors the requested section header."""
        if not structure or not getattr(structure, "paragraphs", None):
            return None

        normalized_target = section_name.replace("_", " ")
        for paragraph in structure.paragraphs:
            try:
                normalized_title = structure._normalize_section_title(paragraph.text)  # type: ignore[attr-defined]
                is_header = structure._looks_like_section_header(paragraph.text)  # type: ignore[attr-defined]
            except AttributeError:
                normalized_title = paragraph.text.strip()
                is_header = False

            if not normalized_title:
                continue

            if is_header and normalized_target in normalized_title.lower():
                anchor = paragraph.paragraph_index + 1
                if anchor >= len(structure.paragraphs):
                    return paragraph.paragraph_index
                return anchor
        return None

    def _score_paragraphs_for_section(self, structure: Optional["DocumentStructure"], section_name: str) -> Optional[int]:
        """Fallback scoring to choose a paragraph when no explicit section header found."""
        if not structure or not getattr(structure, "paragraphs", None):
            return None

        keywords = self._section_keywords(section_name)
        best_index: Optional[int] = None
        best_score = -1

        for para in structure.paragraphs:
            text_lower = para.text.lower()
            score = 0
            for keyword in keywords:
                if keyword and keyword.lower() in text_lower:
                    score += 5
            if "citation" in text_lower or "authority" in text_lower:
                score += 2
            if len(para.sentences) >= 2:
                score += 1

            if score > best_score:
                best_score = score
                best_index = para.paragraph_index

        return best_index

    def _find_location_for_section(self, structure: Optional["DocumentStructure"], section_name: str) -> DocumentLocation:
        """Create a DocumentLocation near the requested section."""
        normalized_section = self._normalize_section_key(section_name)

        anchor_idx = self._find_section_anchor_index(structure, normalized_section)
        if anchor_idx is None:
            anchor_idx = self._score_paragraphs_for_section(structure, normalized_section)

        if structure is None or not getattr(structure, "paragraphs", None):
            char_offset = len(getattr(structure, "original_text", "") or "")
            return DocumentLocation(
                character_offset=char_offset,
                position_type="after",
                section_name=normalized_section
            )

        if anchor_idx is None:
            anchor_idx = max(len(structure.paragraphs) - 1, 0)

        paragraph = structure.paragraphs[anchor_idx]
        sentence_index = paragraph.sentences[-1].sentence_index if paragraph.sentences else None
        return DocumentLocation(
            paragraph_index=anchor_idx,
            sentence_index=sentence_index,
            position_type="after",
            section_name=normalized_section
        )

    def _format_citation_sentence(self, citations: List[str], section_name: str, needed: int) -> str:
        """Create the text that will be inserted for the citation improvement."""
        normalized = self._normalize_section_key(section_name).replace("_", " ")
        trimmed = [cite for cite in citations if cite]
        if not trimmed:
            trimmed = self.get_top_citations(max(1, needed))
        trimmed = trimmed[:max(1, needed)]
        joined = "; ".join(trimmed)
        return f"See {joined} (supporting the {normalized} section)."

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests inserting citations into weak sections.

        Args:
            text: Draft text that needs improvements.
            structure: Parsed document structure.
            context: Optional extra context (unused).
        """
        try:
            analysis = await self.analyze_citation_strength(text)
            if not analysis.success or not analysis.value:
                logger.debug("Citation analysis failed or returned empty data; skipping edit generation")
                return []

            recommendations = analysis.value.get("recommendations", [])
            if not recommendations:
                return []

            edit_requests: List[EditRequest] = []

            for rec in recommendations:
                rec_type = rec.get("type")
                if rec_type not in {"insufficient_citations", "low_density"}:
                    continue

                section_name = rec.get("section") or "argument"
                needed = 1
                if rec_type == "insufficient_citations":
                    current = rec.get("current", 0)
                    required = rec.get("required", 1)
                    needed = max(1, required - current)

                suggested = rec.get("suggested_citations") or self._get_suggested_citations_for_section(section_name)
                location = self._find_location_for_section(structure, section_name)
                content = self._format_citation_sentence(suggested, section_name, needed)

                priority_map = {"high": 85, "medium": 70}
                priority = priority_map.get(rec.get("priority", "").lower(), 60)
                reason = rec.get("message") or f"Add citation support to the {section_name} section."

                edit_requests.append(
                    EditRequest(
                        plugin_name="citation_count",
                        location=location,
                        edit_type="insert",
                        content=content,
                        priority=priority,
                        affected_plugins=["citation_format", "petition_quality", "word_count", "sentence_count"],
                        metadata={
                            "section": section_name,
                            "recommendation_type": rec_type,
                            "suggested_citations": suggested[:max(1, needed)],
                            "needed_citations": needed,
                            "priority": rec.get("priority"),
                        },
                        reason=reason,
                    )
                )

                if len(edit_requests) >= 5:
                    break

            # Fallback: if no section-specific edits were created, use missing top citations
            if not edit_requests:
                fallback_rec = next(
                    (r for r in recommendations if r.get("type") in {"missing_top_citations", "missing_winning_citations"}),
                    None,
                )
                if fallback_rec:
                    section_name = "argument"
                    suggested = fallback_rec.get("missing_citations") or self.get_top_citations(2)
                    location = self._find_location_for_section(structure, section_name)
                    content = self._format_citation_sentence(suggested, section_name, len(suggested))
                    edit_requests.append(
                        EditRequest(
                            plugin_name="citation_count",
                            location=location,
                            edit_type="insert",
                            content=content,
                            priority=65,
                            affected_plugins=["citation_format", "petition_quality"],
                            metadata={
                                "section": section_name,
                                "recommendation_type": fallback_rec.get("type"),
                                "suggested_citations": suggested,
                                "needed_citations": len(suggested),
                                "priority": fallback_rec.get("priority"),
                            },
                            reason=fallback_rec.get("message", "Add top-performing citations to strengthen argument."),
                        )
                    )

            if edit_requests:
                logger.info(f"CitationRetrievalPlugin generated {len(edit_requests)} citation edit request(s)")

            return edit_requests

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Citation edit request generation failed: {exc}")
            return []

    def get_top_citations(self, limit: int = 10) -> List[str]:
        """Get top N citations for prioritization."""
        if self.top_citations:
            return [cite["citation"] for cite in self.top_citations[:limit]]
        return [cite["citation"] for cite in self.winning_citations[:limit]] if self.winning_citations else []
