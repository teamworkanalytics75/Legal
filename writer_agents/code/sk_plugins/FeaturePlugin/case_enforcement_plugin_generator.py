#!/usr/bin/env python3
"""
Case Enforcement Plugin Generator - Creates individual plugins for each verified case.

This module generates SK plugins for each case in the master citations database,
ensuring every case has its own enforcement plugin.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class IndividualCaseEnforcementPlugin(BaseFeaturePlugin):
    """
    Plugin for enforcing citation of a single specific case.

    Generated automatically for each case in master_case_citations.json
    """

    def __init__(
        self,
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        case_info: Dict[str, Any],
        memory_store=None,  # EpisodicMemoryBank for learning
        db_paths=None,  # List[Path] - case law database paths (Phase 1)
        enable_langchain: bool = True,  # Enable LangChain SQL agents (Phase 2)
        enable_courtlistener: bool = False,  # Enable CourtListener API (Phase 4, optional)
        enable_storm: bool = False  # Enable STORM research (Phase 5, optional)
    ):
        """
        Initialize plugin for a specific case.

        Args:
            kernel: SK kernel
            chroma_store: Chroma store
            rules_dir: Rules directory
            case_info: Case information from master_case_citations.json
            memory_store: EpisodicMemoryBank for learning
            db_paths: Case law database paths (Phase 1)
            enable_langchain: Enable LangChain SQL agents (Phase 2)
            enable_courtlistener: Enable CourtListener API (Phase 4, optional)
            enable_storm: Enable STORM research (Phase 5, optional)
        """
        case_name = case_info.get('case_name', 'Unknown Case')
        plugin_name = f"case_enforcement_{self._sanitize_case_name(case_name)}"
        super().__init__(
            kernel, plugin_name, chroma_store, rules_dir,
            memory_store=memory_store,
            db_paths=db_paths,
            enable_langchain=enable_langchain,
            enable_courtlistener=enable_courtlistener,
            enable_storm=enable_storm
        )

        self.case_info = case_info
        self.case_name = case_name
        self.citation = case_info.get('citation', '')
        self.full_citation = case_info.get('full_citation', '')
        self.priority = case_info.get('priority', 'medium')
        self.category = case_info.get('category', 'general')

        logger.info(f"IndividualCaseEnforcementPlugin initialized for: {case_name} ({self.priority} priority)")

    def _sanitize_case_name(self, case_name: str) -> str:
        """Convert case name to valid plugin name."""
        # Remove special characters, replace spaces with underscores
        sanitized = case_name.lower()
        sanitized = sanitized.replace(' ', '_')
        sanitized = sanitized.replace('.', '')
        sanitized = sanitized.replace("'", '')
        sanitized = sanitized.replace(',', '')
        sanitized = sanitized.replace('&', 'and')
        sanitized = sanitized.replace('(', '').replace(')', '')
        # Limit length
        return sanitized[:50]

    async def enforce_case_citation(self, draft_text: str) -> FunctionResult:
        """
        Enforce citation of this specific case.

        Args:
            draft_text: Draft text to check

        Returns:
            FunctionResult with citation status
        """
        try:
            import re
            text_lower = draft_text.lower()
            case_name_lower = self.case_name.lower()

            # Check multiple citation methods
            is_cited = False
            citation_found = None
            citation_method = None

            # Method 1: Check for case name variations
            name_variations = [
                case_name_lower,
                case_name_lower.split(' v. ')[0] if ' v. ' in case_name_lower else '',  # First party
                case_name_lower.split(' v. ')[-1] if ' v. ' in case_name_lower else '',  # Second party
                self.case_name.split(' v. ')[0].lower() if ' v. ' in self.case_name else '',  # Short first party
            ]

            for variation in name_variations:
                if variation and len(variation) > 5 and variation in text_lower:
                    is_cited = True
                    citation_found = variation
                    citation_method = "case_name"
                    break

            # Method 2: Check for citation pattern
            if not is_cited and self.citation:
                # Extract reporter citation (e.g., "542 U.S. 241", "550 F.Supp. 869")
                citation_patterns = [
                    r'\d+\s+U\.S\.\s+\d+',  # Supreme Court
                    r'\d+\s+F\.\s+(?:2d|3d)?\s*\d+',  # Federal Reporter
                    r'\d+\s+F\.\s+Supp\.?\s*(?:2d|3d)?\s*\d+',  # Federal Supplement
                    r'\d+\s+Mass\.\s+(?:App\.)?\s*\d+',  # Massachusetts
                ]

                # Extract numbers from citation
                citation_numbers = re.findall(r'\d+', self.citation)

                for pattern in citation_patterns:
                    matches = re.findall(pattern, draft_text, re.IGNORECASE)
                    if matches:
                        # Check if match contains citation numbers
                        for match in matches:
                            match_numbers = re.findall(r'\d+', match)
                            if any(num in citation_numbers for num in match_numbers):
                                is_cited = True
                                citation_found = match
                                citation_method = "citation_pattern"
                                break
                        if is_cited:
                            break

            # Method 3: Check for full citation text
            if not is_cited and self.full_citation:
                citation_key = self.full_citation.split('(')[0].strip().lower()
                if citation_key and len(citation_key) > 20 and citation_key[:100] in text_lower:
                    is_cited = True
                    citation_found = "full_citation"
                    citation_method = "full_citation_text"

            # Generate recommendation if not cited
            recommendation = None
            if not is_cited:
                recommendation = self._generate_recommendation()

            result_value = {
                "case_name": self.case_name,
                "citation": self.citation,
                "full_citation": self.full_citation,
                "is_cited": is_cited,
                "citation_found": citation_found,
                "citation_method": citation_method,
                "priority": self.priority,
                "category": self.category,
                "recommendation": recommendation,
                "relevance": self.case_info.get('relevance', ''),
                "keywords": self.case_info.get('keywords', [])
            }

            # Store enforcement result in memory for learning
            if self.memory_store:
                try:
                    self._store_memory(
                        summary=f"Case citation check: {self.case_name} - {'CITED' if is_cited else 'MISSING'}",
                        context={
                            "case_name": self.case_name,
                            "is_cited": is_cited,
                            "citation_method": citation_method,
                            "priority": self.priority,
                            "citation_found": citation_found
                        },
                        memory_type="execution",
                        source=f"CaseEnforcement_{self._sanitize_case_name(self.case_name)}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to store case enforcement memory: {e}")

            return FunctionResult(
                success=True,
                value=result_value,
                metadata={"analysis_type": "individual_case_enforcement"}
            )

        except Exception as e:
            logger.error(f"Case citation enforcement failed for {self.case_name}: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _generate_recommendation(self) -> str:
        """Generate recommendation for citing this case."""
        relevance = self.case_info.get('relevance', '')
        keywords = self.case_info.get('keywords', [])

        recommendation = f"Add citation to {self.case_name}"
        if self.full_citation:
            recommendation += f": {self.full_citation}"

        if relevance:
            recommendation += f". Relevance: {relevance}"

        if keywords:
            recommendation += f". Keywords: {', '.join(keywords[:3])}"

        return recommendation


class CaseEnforcementPluginFactory:
    """
    Factory for creating individual case enforcement plugins from master citations database.
    """

    @staticmethod
    def load_master_citations(master_citations_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load master citations database."""
        if master_citations_path is None:
            master_citations_path = Path(__file__).parents[3] / "config" / "master_case_citations.json"
            if not master_citations_path.exists():
                master_citations_path = Path(__file__).parents[2] / "config" / "master_case_citations.json"

        if master_citations_path.exists():
            with open(master_citations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Master citations file not found: {master_citations_path}")
            return {}

    @staticmethod
    def create_all_case_plugins(
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        master_citations_path: Optional[Path] = None,
        memory_store=None,  # EpisodicMemoryBank for learning
        db_paths=None,  # List[Path] - case law database paths (Phase 1)
        enable_langchain: bool = True,  # Enable LangChain SQL agents (Phase 2)
        enable_courtlistener: bool = False,  # Enable CourtListener API (Phase 4, optional)
        enable_storm: bool = False  # Enable STORM research (Phase 5, optional)
    ) -> Dict[str, IndividualCaseEnforcementPlugin]:
        """
        Create individual plugins for all cases in master citations database.

        Returns:
            Dictionary mapping case names to plugins
        """
        master_data = CaseEnforcementPluginFactory.load_master_citations(master_citations_path)

        plugins = {}

        # Extract all cases from categories
        all_cases = []
        categories = master_data.get('categories', {})
        for category_name, category_data in categories.items():
            cases = category_data.get('cases', [])
            for case in cases:
                if case.get('enforce', True):  # Only create plugins for cases marked for enforcement
                    case['category'] = category_name
                    all_cases.append(case)

        # Create plugin for each case
        for case_info in all_cases:
            try:
                case_name = case_info.get('case_name', '')
                if not case_name:
                    continue

                plugin = IndividualCaseEnforcementPlugin(
                    kernel=kernel,
                    chroma_store=chroma_store,
                    rules_dir=rules_dir,
                    case_info=case_info,
                    memory_store=memory_store,
                    db_paths=db_paths,  # Phase 1: Database access
                    enable_langchain=enable_langchain,  # Phase 2: LangChain
                    enable_courtlistener=enable_courtlistener,  # Phase 4: CourtListener
                    enable_storm=enable_storm  # Phase 5: STORM
                )

                # Use sanitized case name as key
                plugin_key = plugin._sanitize_case_name(case_name)
                plugins[plugin_key] = plugin

            except Exception as e:
                logger.error(f"Failed to create plugin for case {case_info.get('case_name', 'Unknown')}: {e}")
                continue

        logger.info(f"Created {len(plugins)} individual case enforcement plugins")
        return plugins

    @staticmethod
    def create_plugin_for_case(
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        case_info: Dict[str, Any],
        memory_store=None  # EpisodicMemoryBank for learning
    ) -> IndividualCaseEnforcementPlugin:
        """Create a single plugin for a specific case."""
        return IndividualCaseEnforcementPlugin(
            kernel=kernel,
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            case_info=case_info,
            memory_store=memory_store
        )

