#!/usr/bin/env python3
"""
Required Case Citation Plugin - Enforces citation of specific required cases.

This plugin ensures that drafts cite all required case law cases identified through research.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

# Forward reference for DocumentStructure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class RequiredCaseCitationPlugin(BaseFeaturePlugin):
    """Plugin that enforces citation of specific required cases from research."""

    def __init__(
        self,
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        required_cases: Optional[List[Dict[str, Any]]] = None,
        memory_store=None,  # EpisodicMemoryBank for learning
        **kwargs
    ):
        """
        Initialize required case citation plugin.

        Args:
            kernel: SK kernel
            chroma_store: Chroma store (optional)
            rules_dir: Rules directory
            required_cases: List of required cases to enforce. If None, loads from research results.
            memory_store: EpisodicMemoryBank for learning
            **kwargs: Additional parameters (db_paths, enable_langchain, etc.)
        """
        super().__init__(kernel, "required_case_citations", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        self.required_cases = required_cases or self._load_required_cases_from_research()
        logger.info(f"RequiredCaseCitationPlugin initialized with {len(self.required_cases)} required cases")

        # Log required cases
        high_priority = [c for c in self.required_cases if c.get('priority') == 'high']
        if high_priority:
            logger.info(f"High-priority required cases ({len(high_priority)}):")
            for case in high_priority[:5]:
                logger.info(f"  - {case.get('case_name', 'Unknown')} ({case.get('reason', '')[:60]})")

    def _load_required_cases_from_research(self) -> List[Dict[str, Any]]:
        """Load required cases from latest research results or master database."""
        try:
            # First try to load from master citations database
            master_citations_paths = [
                Path(__file__).parents[3] / "config" / "master_case_citations.json",
                Path(__file__).parents[4] / "config" / "master_case_citations.json",
            ]

            for master_path in master_citations_paths:
                if master_path.exists():
                    with open(master_path, 'r', encoding='utf-8') as f:
                        master_data = json.load(f)

                    # Extract all cases marked for enforcement
                    required = []
                    categories = master_data.get('categories', {})
                    for category_name, category_data in categories.items():
                        cases = category_data.get('cases', [])
                        for case in cases:
                            if case.get('enforce', True):
                                required.append(case)

                    if required:
                        logger.info(f"Loaded {len(required)} required cases from master citations database")
                        return required

            # Fallback: Try to load from test results
            research_paths = [
                Path(__file__).parents[3] / "test_results" / "research_national_security" / "test_results.json",
                Path(__file__).parents[4] / "test_results" / "research_national_security" / "test_results.json",
            ]

            for research_path in research_paths:
                if research_path.exists():
                    with open(research_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract winning cases and top cases from research
                    baseline = data.get('baseline', {})
                    research = baseline.get('metrics', {}).get('research_findings', {})

                    required = []

                    # Add winning cases (high priority)
                    winning_cases = research.get('winning_cases', [])
                    for case in winning_cases[:10]:  # Top 10 winning cases
                        required.append({
                            "case_name": case.get('case_name', ''),
                            "citation": case.get('citation', ''),
                            "court": case.get('court', ''),
                            "priority": "high",
                            "reason": "Winning case matching research themes",
                            "keywords": case.get('keywords_found', [])
                        })

                    # Add top semantic similarity cases (medium priority)
                    top_cases = research.get('cases', [])[:10]  # Top 10 by similarity
                    for case in top_cases:
                        case_name = case.get('case_name', '')
                        # Skip if already added as winning case
                        if not any(c['case_name'] == case_name for c in required):
                            required.append({
                                "case_name": case_name,
                                "citation": case.get('citation', ''),
                                "court": case.get('court', ''),
                                "similarity_score": case.get('similarity_score', 0.0),
                                "priority": "medium",
                                "reason": f"High semantic similarity ({case.get('similarity_score', 0.0):.3f})",
                            })

                    if required:
                        logger.info(f"Loaded {len(required)} required cases from research results")
                        return required
        except Exception as e:
            logger.warning(f"Could not load required cases from research: {e}")

        # Fallback: use default required cases (key cases from research)
        return self._get_default_required_cases()

    def update_required_cases_from_research(self, research_results: Dict[str, Any]) -> None:
        """
        Update required cases from research results in workflow state.

        Args:
            research_results: Research results dictionary from WorkflowState.research_results
        """
        try:
            if not research_results or not research_results.get('success'):
                logger.debug("Research results not available or unsuccessful, keeping existing required cases")
                return

            # Extract cases from research results
            required = []

            # Add winning cases (high priority)
            winning_cases = research_results.get('winning_cases', [])
            for case in winning_cases[:10]:  # Top 10 winning cases
                required.append({
                    "case_name": case.get('case_name', ''),
                    "citation": case.get('citation', ''),
                    "court": case.get('court', ''),
                    "priority": "high",
                    "reason": "Winning case matching research themes",
                    "keywords": case.get('keywords_found', []),
                    "enforce": True
                })

            # Add top semantic similarity cases (medium priority)
            top_cases = research_results.get('cases', [])[:10]  # Top 10 by similarity
            for case in top_cases:
                case_name = case.get('case_name', '')
                # Skip if already added as winning case
                if not any(c['case_name'] == case_name for c in required):
                    required.append({
                        "case_name": case_name,
                        "citation": case.get('citation', ''),
                        "court": case.get('court', ''),
                        "similarity_score": case.get('similarity_score', 0.0),
                        "priority": "medium",
                        "reason": f"High semantic similarity ({case.get('similarity_score', 0.0):.3f})",
                        "enforce": True
                    })

            if required:
                self.required_cases = required
                high_priority = [c for c in required if c.get('priority') == 'high']
                logger.info(f"Updated required cases from research: {len(required)} cases ({len(high_priority)} high-priority)")
            else:
                logger.warning("No cases extracted from research results, keeping existing required cases")

        except Exception as e:
            logger.warning(f"Failed to update required cases from research: {e}")

    def _get_default_required_cases(self) -> List[Dict[str, Any]]:
        """Get default required cases (key cases from national security research)."""
        return [
            {
                "case_name": "United States v. President and Fellows of Harvard College",
                "citation": "United States v. President and Fellows of Harvard College, District Court, D. Massachusetts (2004)",
                "court": "District Court, D. Massachusetts",
                "priority": "high",
                "reason": "Directly relevant: Harvard + foreign government connections, same jurisdiction",
                "keywords": ["harvard", "foreign government"],
                "enforce": True
            },
            {
                "case_name": "United States v. LaRouche Campaign",
                "citation": "United States v. LaRouche Campaign, District Court, D. Massachusetts (1988)",
                "court": "District Court, D. Massachusetts",
                "priority": "high",
                "reason": "Highest similarity (0.789), same jurisdiction, national security frameworks",
                "keywords": ["national security"],
                "enforce": True
            },
            {
                "case_name": "Morris v. People's Republic of China",
                "citation": "Morris v. People's Republic of China, District Court, S.D. New York (2007)",
                "court": "District Court, S.D. New York",
                "priority": "high",
                "reason": "PRC as foreign government defendant, winning case",
                "keywords": ["foreign government", "communist party"],
                "enforce": True
            },
            {
                "case_name": "Jackson v. People's Republic of China",
                "citation": "Jackson v. People's Republic of China, District Court, N.D. Alabama (1982)",
                "court": "District Court, N.D. Alabama",
                "priority": "high",
                "reason": "Early precedent: PRC as foreign government defendant, 550 F.Supp. 869",
                "keywords": ["foreign government", "communist party"],
                "enforce": True
            },
            {
                "case_name": "Doe v. Amherst College",
                "citation": "Doe v. Amherst College, District Court, D. Massachusetts",
                "court": "District Court, D. Massachusetts",
                "priority": "medium",
                "reason": "Academic institution case, same jurisdiction",
                "keywords": ["academic institution"],
                "enforce": True
            },
            {
                "case_name": "Gailius v. Immigration & Naturalization Service",
                "citation": "Gailius v. Immigration & Naturalization Service, Court of Appeals for the First Circuit (1998)",
                "court": "Court of Appeals for the First Circuit",
                "priority": "medium",
                "reason": "Harvard + communist party connections, First Circuit (controls D. Mass)",
                "keywords": ["harvard", "communist party"],
                "enforce": True
            },
            {
                "case_name": "Doe I v. Bush",
                "citation": "Doe I v. Bush, District Court, D. Massachusetts (2002)",
                "court": "District Court, D. Massachusetts",
                "priority": "medium",
                "reason": "Same jurisdiction, national security + privacy balance",
                "keywords": ["national security", "privacy"],
                "enforce": True
            },
        ]

    async def enforce_required_citations(self, draft_text: str) -> FunctionResult:
        """
        Enforce citation of all required cases with memory integration and database search.

        Phase 7: Now uses database search to verify cases exist and find related precedents.

        Queries past research before enforcement and stores results for learning.
        Uses database search to verify required cases exist and find supporting cases.

        Args:
            draft_text: Draft text to check

        Returns:
            FunctionResult with citation enforcement analysis
        """
        try:
            # Query past research before enforcement
            if self.memory_store and self.required_cases:
                # Build query from draft text and case names
                case_names = [c.get('case_name', '') for c in self.required_cases[:5] if c.get('case_name')]
                query_parts = [draft_text[:200]]
                if case_names:
                    query_parts.append(" ".join(case_names))
                query = " ".join(query_parts).strip()

                if query:
                    past_research = self._query_memory(
                        query=query,
                        agent_type="CaseLawResearcher",
                        k=5,
                        memory_types=["query"]
                    )
                    if past_research:
                        logger.info(f"Found {len(past_research)} past research memories to inform enforcement")
                        # Optionally use past research to update required cases priority
                        # (Could be enhanced to learn from past successes)

            # Phase 7: Verify required cases exist in database and find related cases
            verified_cases = []
            unverified_cases = []

            if self.sqlite_searcher and self.required_cases:
                for case in self.required_cases:
                    case_name = case.get('case_name', '')
                    if not case_name:
                        continue

                    # Search database to verify case exists
                    try:
                        db_results = await self.search_case_law(
                            query=case_name,
                            top_k=5,
                            min_similarity=0.7  # High threshold for verification
                        )

                        # Check if we found a match
                        found_match = False
                        for result in db_results:
                            if case_name.lower() in result.get('case_name', '').lower():
                                found_match = True
                                # Update case with database metadata
                                case['verified_in_db'] = True
                                case['db_citation'] = result.get('citation', case.get('citation', ''))
                                case['db_similarity'] = result.get('similarity_score', 0.0)
                                break

                        if found_match:
                            verified_cases.append(case)
                        else:
                            case['verified_in_db'] = False
                            unverified_cases.append(case)
                            logger.warning(f"Required case not found in database: {case_name}")
                    except Exception as e:
                        logger.debug(f"Database verification failed for {case_name}: {e}")
                        case['verified_in_db'] = None  # Unknown
                        verified_cases.append(case)  # Include anyway
            else:
                # No database access, use all cases as-is
                verified_cases = self.required_cases

            text_lower = draft_text.lower()

            # Check each required case (use verified cases if available)
            citation_status = []
            missing_cases = []
            cited_cases = []

            cases_to_check = verified_cases if verified_cases else self.required_cases
            for case in cases_to_check:
                case_name = case.get('case_name', '')
                citation = case.get('citation', '')
                priority = case.get('priority', 'medium')
                reason = case.get('reason', '')

                # Check if case is cited (check for case name or citation)
                is_cited = False
                citation_found = None

                # Check for case name variations
                name_variations = [
                    case_name.lower(),
                    case_name.split(' v. ')[0].lower(),  # First party
                    case_name.split(' v. ')[-1].lower() if ' v. ' in case_name else '',  # Second party
                ]

                # Check for citation format (e.g., "550 F.Supp. 869")
                citation_patterns = [
                    r'\d+\s+F\.\s+Supp\.?\s*(?:2d|3d)?\s*\d+',  # Standard citation
                    r'\d+\s+F\.\s+(?:2d|3d)?\s*\d+',  # Reporter citation
                    r'\d+\s+Mass\.\s+(?:App\.)?\s*\d+',  # Massachusetts citation
                ]

                # Check if case name appears in text
                for variation in name_variations:
                    if variation and variation in text_lower:
                        is_cited = True
                        citation_found = f"Case name: {variation}"
                        break

                # Check if citation appears in text
                if not is_cited:
                    for pattern in citation_patterns:
                        matches = re.findall(pattern, draft_text, re.IGNORECASE)
                        if matches and citation:
                            # Check if citation contains matching reporter cite
                            if any(match.lower() in citation.lower() for match in matches):
                                is_cited = True
                                citation_found = f"Citation pattern: {matches[0]}"
                                break

                # Check for full citation text
                if not is_cited and citation:
                    # Extract key parts of citation for matching
                    citation_key = citation.split('(')[0].strip().lower()
                    if citation_key and citation_key[:100] in text_lower:
                        is_cited = True
                        citation_found = "Full citation text"

                status = {
                    "case_name": case_name,
                    "citation": citation,
                    "priority": priority,
                    "is_cited": is_cited,
                    "citation_found": citation_found,
                    "reason": reason,
                    "keywords": case.get('keywords', [])
                }

                citation_status.append(status)

                if is_cited:
                    cited_cases.append(status)
                else:
                    missing_cases.append(status)

            # Calculate compliance score
            total_cases = len(self.required_cases)
            high_priority_cases = [c for c in self.required_cases if c.get('priority') == 'high']
            high_priority_cited = [c for c in citation_status if c['priority'] == 'high' and c['is_cited']]

            overall_compliance = len(cited_cases) / max(total_cases, 1)
            high_priority_compliance = len(high_priority_cited) / max(len(high_priority_cases), 1)

            # Generate recommendations
            recommendations = self._generate_recommendations(missing_cases, cited_cases)

            # Store enforcement result in memory for learning
            if self.memory_store:
                try:
                    self._store_memory(
                        summary=f"Citation enforcement: {len(cited_cases)}/{total_cases} cases cited (compliance: {overall_compliance:.1%})",
                        context={
                            "total_required_cases": total_cases,
                            "cases_cited": len(cited_cases),
                            "cases_missing": len(missing_cases),
                            "overall_compliance_score": overall_compliance,
                            "high_priority_compliance": high_priority_compliance,
                            "missing_case_names": [c.get('case_name', 'Unknown') for c in missing_cases[:10]],
                            "cited_case_names": [c.get('case_name', 'Unknown') for c in cited_cases[:10]],
                            "meets_requirements": high_priority_compliance >= 0.8
                        },
                        memory_type="execution",
                        source="RequiredCaseCitationPlugin"
                    )
                except Exception as e:
                    logger.debug(f"Failed to store enforcement memory: {e}")

            result_value = {
                "total_required_cases": total_cases,
                "cases_cited": len(cited_cases),
                "cases_missing": len(missing_cases),
                "overall_compliance_score": overall_compliance,
                "high_priority_compliance_score": high_priority_compliance,
                "citation_status": citation_status,
                "missing_cases": missing_cases,
                "cited_cases": cited_cases,
                "recommendations": recommendations,
                "meets_requirements": high_priority_compliance >= 0.8  # At least 80% of high-priority cases
            }

            # Add database verification info if available
            if verified_cases or unverified_cases:
                result_value["verified_cases_count"] = len(verified_cases) if verified_cases else 0
                result_value["unverified_cases_count"] = len(unverified_cases) if unverified_cases else 0

            return FunctionResult(
                success=True,
                value=result_value,
                metadata={"analysis_type": "required_case_citation_enforcement"}
            )

        except Exception as e:
            logger.error(f"Required citation enforcement failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _generate_recommendations(self, missing_cases: List[Dict], cited_cases: List[Dict]) -> List[str]:
        """Generate recommendations for missing citations."""
        recommendations = []

        if not missing_cases:
            recommendations.append("✓ All required cases are cited. Excellent citation coverage.")
            return recommendations

        # Group by priority
        high_priority_missing = [c for c in missing_cases if c.get('priority') == 'high']
        medium_priority_missing = [c for c in missing_cases if c.get('priority') == 'medium']

        if high_priority_missing:
            recommendations.append(f"⚠️ CRITICAL: {len(high_priority_missing)} high-priority cases missing citations:")
            for case in high_priority_missing:
                recommendations.append(
                    f"   - {case['case_name']}: {case['reason']}. "
                    f"Citation: {case.get('citation', 'N/A')[:80]}"
                )

        if medium_priority_missing:
            recommendations.append(f"ℹ️ {len(medium_priority_missing)} medium-priority cases missing citations:")
            for case in medium_priority_missing[:5]:  # Top 5
                recommendations.append(
                    f"   - {case['case_name']}: {case['reason']}"
                )

        # Positive feedback for cited cases
        if cited_cases:
            high_priority_cited = [c for c in cited_cases if c.get('priority') == 'high']
            if high_priority_cited:
                recommendations.append(
                    f"✓ {len(high_priority_cited)} high-priority cases are properly cited."
                )

        return recommendations

    async def get_citation_suggestions(self, draft_text: str, section: Optional[str] = None) -> FunctionResult:
        """
        Get suggestions for where to add missing citations.

        Args:
            draft_text: Draft text
            section: Optional section name to focus on

        Returns:
            FunctionResult with citation suggestions
        """
        try:
            # Query past successful citation placements if memory available
            past_placements = []
            if self.memory_store and section:
                past_placements = self._query_memory(
                    query=f"{section} citation placement",
                    agent_type=self.__class__.__name__,
                    k=3,
                    memory_types=["execution"]
                )

            # First check what's missing
            enforcement_result = await self.enforce_required_citations(draft_text)
            if not enforcement_result.success:
                return enforcement_result

            missing_cases = enforcement_result.value.get('missing_cases', [])
            if not missing_cases:
                return FunctionResult(
                    success=True,
                    value={
                        "suggestions": ["All required cases are cited. No suggestions needed."]
                    }
                )

            # Extract sections
            sections = self._extract_sections(draft_text)

            suggestions = []
            for case in missing_cases[:10]:  # Top 10 missing cases
                case_name = case.get('case_name', 'Unknown')
                keywords = case.get('keywords', [])
                reason = case.get('reason', '')

                # Suggest sections where this case might fit
                suggested_sections = []
                for section_name, section_text in sections.items():
                    section_lower = section_text.lower()
                    # Check if section contains relevant keywords
                    if keywords and any(keyword.lower() in section_lower for keyword in keywords if keyword):
                        suggested_sections.append(section_name)

                if suggested_sections:
                    # Enhance suggestion with past placement data if available
                    placement_hint = ""
                    if past_placements:
                        # Check if similar cases were cited in similar sections
                        for mem in past_placements:
                            mem_context = mem.context if hasattr(mem, 'context') else {}
                            if mem_context.get('cases_cited'):
                                placement_hint = " (Past successful placements suggest similar sections)"
                                break

                    suggestions.append({
                        "case": case_name,
                        "citation": case.get('citation', ''),
                        "reason": reason,
                        "suggested_sections": suggested_sections,
                        "suggestion": f"Add citation to {case_name} in sections: {', '.join(suggested_sections)}. "
                                     f"Reason: {reason}{placement_hint}",
                        "learned_from_memory": len(past_placements) > 0
                    })
                else:
                    suggestions.append({
                        "case": case_name,
                        "citation": case.get('citation', ''),
                        "reason": reason,
                        "suggestion": f"Add citation to {case_name} in relevant section. Reason: {reason}",
                        "learned_from_memory": False
                    })

            return FunctionResult(
                success=True,
                value={
                    "suggestions": suggestions,
                    "missing_count": len(missing_cases)
                }
            )

        except Exception as e:
            logger.error(f"Citation suggestions failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract document sections by headers."""
        sections = {}
        # Look for section headers (##, ###, I., II., etc.)
        section_pattern = r'^(?:#{1,3}|\d+\.|[IVX]+\.)\s+(.+?)$'
        lines = text.split('\n')
        current_section = "Introduction"
        current_section_text = []

        for line in lines:
            # Check if line is a section header
            match = re.match(section_pattern, line.strip(), re.IGNORECASE)
            if match:
                # Save previous section
                if current_section_text:
                    sections[current_section] = '\n'.join(current_section_text)
                # Start new section
                current_section = match.group(1).strip()
                current_section_text = [line]
            else:
                current_section_text.append(line)

        # Save last section
        if current_section_text:
            sections[current_section] = '\n'.join(current_section_text)

        return sections

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure'
    ) -> List[EditRequest]:
        """
        Generate edit requests to insert missing case citations at appropriate locations.

        Args:
            text: Full document text
            structure: Parsed document structure

        Returns:
            List of EditRequest objects for inserting citations
        """
        try:
            # Check which cases are missing
            enforcement_result = await self.enforce_required_citations(text)
            if not enforcement_result.success:
                logger.warning("Citation enforcement check failed, cannot generate edit requests")
                return []

            missing_cases = enforcement_result.value.get('missing_cases', [])
            if not missing_cases:
                return []  # All citations present

            logger.info(f"Found {len(missing_cases)} missing citations, generating edit requests")

            # Extract sections for better placement
            sections = self._extract_sections(text)

            requests = []

            # For each missing case, find appropriate location
            for case in missing_cases[:10]:  # Limit to top 10 to avoid too many edits
                case_name = case.get('case_name', '')
                citation = case.get('citation', '')
                priority_level = case.get('priority', 'medium')
                keywords = case.get('keywords', [])
                reason = case.get('reason', '')

                # Calculate request priority based on case priority
                request_priority = 90 if priority_level == 'critical' else 75 if priority_level == 'high' else 60

                # Format citation text
                citation_text = f" {case_name}, {citation}" if citation else f" {case_name}"

                # Find best paragraph to insert citation
                best_location = None
                best_score = 0

                for para_idx, paragraph in enumerate(structure.paragraphs):
                    para_text_lower = paragraph.text.lower()

                    # Score paragraph based on keyword matches
                    score = 0
                    if keywords:
                        for keyword in keywords:
                            if keyword.lower() in para_text_lower:
                                score += 10

                    # Boost score if paragraph mentions related legal concepts
                    legal_terms = ["case", "holding", "precedent", "court", "decision", "ruling", "authority"]
                    for term in legal_terms:
                        if term in para_text_lower:
                            score += 2

                    # Check if paragraph is in a relevant section
                    for section_name, section_text in sections.items():
                        if paragraph.text in section_text:
                            section_lower = section_name.lower()
                            if any(keyword.lower() in section_lower for keyword in keywords if keywords):
                                score += 5
                            # Boost score for legal argument sections
                            if any(term in section_lower for term in ["argument", "legal", "precedent", "authority", "support"]):
                                score += 3

                    if score > best_score:
                        best_score = score
                        # Insert after first sentence that seems relevant, or after first sentence
                        if paragraph.sentences:
                            # Use first sentence as insertion point
                            sent_idx = 0
                            best_location = DocumentLocation(
                                paragraph_index=para_idx,
                                sentence_index=sent_idx,
                                position_type="after"
                            )

                # If no good location found, use first paragraph
                if not best_location:
                    if structure.paragraphs:
                        first_para = structure.paragraphs[0]
                        if first_para.sentences:
                            best_location = DocumentLocation(
                                paragraph_index=0,
                                sentence_index=0,
                                position_type="after"
                            )

                if best_location:
                    # Create citation text with context
                    citation_content = f"See{citation_text}."

                    # Add explanatory context if available
                    if reason:
                        citation_content += f" ({reason[:100]})"

                    request = EditRequest(
                        plugin_name="required_case_citations",
                        location=best_location,
                        edit_type="insert",
                        content=citation_content,
                        priority=request_priority,
                        affected_plugins=["word_count", "sentence_count", "character_count", "citation_format", "paragraph_structure"],
                        metadata={
                            "case_name": case_name,
                            "citation": citation,
                            "priority": priority_level,
                            "keywords": keywords,
                            "placement_score": best_score
                        },
                        reason=f"Add required citation: {case_name} ({reason})"
                    )
                    requests.append(request)

            logger.info(f"Generated {len(requests)} citation insertion edit requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to generate citation edit requests: {e}")
            return []

