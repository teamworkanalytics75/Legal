"""
Outline Manager - Organizes Plugins Based on Perfect Outline Structure

Uses the reverse-engineered perfect outline from CatBoost analysis to:
1. Organize plugins by section
2. Calibrate plugin targets based on outline requirements
3. Ensure proper section order and transitions
4. Enforce enumeration requirements
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """Types of sections in the perfect outline."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    TRANSITION = "transition"


@dataclass
class SubOutlineSection:
    """Represents a sub-section within a section (nested enumeration level)."""
    name: str
    level: int  # Enumeration level (1, 2, 3, etc.)
    optimal_word_count: Optional[Tuple[int, int]] = None  # (min, max)
    optimal_enumeration_depth: int = 0
    required_plugins: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlineSection:
    """Represents a section in the perfect outline."""
    name: str
    display_name: str
    position: int
    section_type: SectionType
    importance: float = 0.0
    required_plugins: List[str] = field(default_factory=list)
    optional_plugins: List[str] = field(default_factory=list)
    enumeration_required: bool = False
    enumeration_min_count: int = 0
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None
    # Sub-outline structures
    sub_outline: List[SubOutlineSection] = field(default_factory=list)
    optimal_word_count: Optional[Tuple[int, int]] = None  # (min, max) from analysis
    optimal_enumeration_depth: int = 0  # Optimal max depth for this section
    optimal_paragraph_count: Optional[Tuple[int, int]] = None  # (min, max) from analysis
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlineTransition:
    """Represents a critical transition between sections."""
    from_section: str
    to_section: str
    importance: float
    must_be_consecutive: bool = True
    description: str = ""


class OutlineManager:
    """
    Manages the perfect outline structure and plugin organization.

    Based on CatBoost analysis of 628 motions, this ensures:
    - Proper section order (Legal Standard → Factual Background is critical)
    - Extensive enumeration (11+ instances, bullet points throughout)
    - Strategic section placement (Danger/Safety early, Balancing Test prominent)
    - Plugin calibration based on outline requirements
    """

    # Perfect outline structure (from TOP_FEATURES_PERFECT_OUTLINE.md)
    PERFECT_OUTLINE = [
        {
            "name": "introduction",
            "display_name": "Introduction",
            "position": 1,
            "importance": 0.0,
            "enumeration_required": False,
        },
        {
            "name": "legal_standard",
            "display_name": "Legal Standard",
            "position": 2,
            "importance": 64.80,  # CRITICAL: #1 most important
            "enumeration_required": True,
            "required_plugins": ["citation_retrieval", "required_case_citation"],
            "metadata": {
                "description": "Establish legal framework before facts",
                "must_cite": ["Doe v. Public Citizen"],
            }
        },
        {
            "name": "factual_background",
            "display_name": "Factual Background",
            "position": 3,
            "importance": 64.80,  # CRITICAL: Part of #1 transition
            "enumeration_required": False,
            "required_plugins": ["factual_timeline"],
            "metadata": {
                "description": "Must immediately follow Legal Standard",
                "transition_from": "legal_standard",
            }
        },
        {
            "name": "privacy_harm",
            "display_name": "Privacy Harm / Good Cause",
            "position": 4,
            "importance": 3.36,
            "enumeration_required": True,
            "enumeration_min_count": 3,
            "required_plugins": ["mentions_privacy", "privacy_harm_count"],
            "optional_plugins": ["mentions_harassment", "mentions_safety", "mentions_retaliation"],
            "metadata": {
                "description": "Use enumeration extensively",
                "enumeration_style": "bullet_points",
            }
        },
        {
            "name": "danger_safety",
            "display_name": "Danger / Safety Arguments",
            "position": 5,
            "importance": 3.36,
            "enumeration_required": True,
            "enumeration_min_count": 2,
            "required_plugins": ["mentions_safety"],
            "metadata": {
                "description": "Place early - position matters",
                "enumeration_style": "numbered_list",
            }
        },
        {
            "name": "public_interest",
            "display_name": "Public Interest Analysis",
            "position": 6,
            "importance": 0.23,
            "enumeration_required": False,
            "optional_plugins": ["public_interest"],
            "metadata": {
                "description": "Address counterarguments",
            }
        },
        {
            "name": "balancing_test",
            "display_name": "Balancing Test",
            "position": 7,
            "importance": 0.23,
            "enumeration_required": True,
            "required_plugins": ["balancing_test_position", "balancing_outweigh"],
            "metadata": {
                "description": "Place prominently - not buried",
                "enumeration_style": "bullet_points",
            }
        },
        {
            "name": "protective_measures",
            "display_name": "Proposed Protective Measures",
            "position": 8,
            "importance": 0.05,
            "enumeration_required": True,
            "enumeration_min_count": 2,
            "required_plugins": ["protective_measures"],
            "metadata": {
                "description": "Use enumeration for measures",
                "enumeration_style": "bullet_points",
            }
        },
        {
            "name": "conclusion",
            "display_name": "Conclusion",
            "position": 9,
            "importance": 0.0,
            "enumeration_required": False,
        },
    ]

    # Critical transitions (from analysis)
    CRITICAL_TRANSITIONS = [
        {
            "from_section": "legal_standard",
            "to_section": "factual_background",
            "importance": 64.80,  # #1 MOST IMPORTANT
            "must_be_consecutive": True,
            "description": "Legal Standard → Factual Background is THE MOST IMPORTANT feature"
        }
    ]

    # Enumeration requirements (from analysis)
    ENUMERATION_REQUIREMENTS = {
        "overall_min_count": 11,  # Successful motions have 11.75 vs 6.18
        "enumeration_density": 1.68,  # Per 1000 words
        "bullet_points_required": True,  # 31.56 importance
        "sections_with_enumeration": [
            "legal_standard",
            "privacy_harm",
            "danger_safety",
            "balancing_test",
            "protective_measures",
        ]
    }

    def __init__(self, outline_source: Optional[Path] = None):
        """
        Initialize Outline Manager.

        Args:
            outline_source: Optional path to JSON file with outline data
        """
        self.sections: Dict[str, OutlineSection] = {}
        self.transitions: List[OutlineTransition] = []
        self._load_perfect_outline()
        self._load_transitions()

        if outline_source and outline_source.exists():
            self._load_from_file(outline_source)

    def _load_perfect_outline(self):
        """Load the perfect outline structure."""
        for section_data in self.PERFECT_OUTLINE:
            section = OutlineSection(
                name=section_data["name"],
                display_name=section_data["display_name"],
                position=section_data["position"],
                section_type=SectionType.REQUIRED,
                importance=section_data.get("importance", 0.0),
                required_plugins=section_data.get("required_plugins", []),
                optional_plugins=section_data.get("optional_plugins", []),
                enumeration_required=section_data.get("enumeration_required", False),
                enumeration_min_count=section_data.get("enumeration_min_count", 0),
                metadata=section_data.get("metadata", {})
            )
            self.sections[section.name] = section

        logger.info(f"Loaded {len(self.sections)} outline sections")

    def _load_transitions(self):
        """Load critical transitions."""
        for trans_data in self.CRITICAL_TRANSITIONS:
            transition = OutlineTransition(
                from_section=trans_data["from_section"],
                to_section=trans_data["to_section"],
                importance=trans_data["importance"],
                must_be_consecutive=trans_data.get("must_be_consecutive", True),
                description=trans_data.get("description", "")
            )
            self.transitions.append(transition)

        logger.info(f"Loaded {len(self.transitions)} critical transitions")

    def _load_from_file(self, file_path: Path):
        """Load outline data from JSON file (optional enhancement)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update sections with data from file if available
            if "sections" in data:
                for section_name, section_data in data["sections"].items():
                    if section_name in self.sections:
                        # Update section with file data
                        for key, value in section_data.items():
                            if hasattr(self.sections[section_name], key):
                                setattr(self.sections[section_name], key, value)

            logger.info(f"Enhanced outline with data from {file_path}")
        except Exception as e:
            logger.warning(f"Could not load outline from file: {e}")

    def load_sub_outline_data(self, analysis_json_path: Path) -> None:
        """
        Load sub-outline data from section analysis results.
        
        Args:
            analysis_json_path: Path to section_sub_outline_analysis.json
        """
        try:
            if not analysis_json_path.exists():
                logger.warning(f"Sub-outline analysis file not found: {analysis_json_path}")
                return
            
            with open(analysis_json_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            updated_count = 0
            for section_name, section_results in analysis_data.items():
                if section_name not in self.sections:
                    continue
                
                section = self.sections[section_name]
                
                # Load optimal word count range
                word_count_analysis = section_results.get('word_count_analysis')
                if word_count_analysis and 'optimal_range' in word_count_analysis:
                    section.optimal_word_count = tuple(word_count_analysis['optimal_range'])
                    updated_count += 1
                
                # Load optimal enumeration depth
                enum_depth_analysis = section_results.get('enumeration_depth_analysis')
                if enum_depth_analysis and 'optimal_threshold' in enum_depth_analysis:
                    section.optimal_enumeration_depth = int(enum_depth_analysis['optimal_threshold'])
                    updated_count += 1
                
                # Load optimal paragraph count
                para_analysis = section_results.get('paragraph_analysis', {})
                para_count_analysis = para_analysis.get('paragraph_count')
                if para_count_analysis and 'optimal_range' in para_count_analysis:
                    section.optimal_paragraph_count = tuple(para_count_analysis['optimal_range'])
                    updated_count += 1
                
                # Load sub-outline structures (if available in future analysis)
                # For now, sub_outline will be populated manually or from detailed analysis
            
            logger.info(f"Loaded sub-outline data for {updated_count} sections from {analysis_json_path}")
            
        except Exception as e:
            logger.warning(f"Could not load sub-outline data: {e}")

    def get_section_order(self) -> List[str]:
        """Get the recommended section order."""
        return [section.name for section in sorted(
            self.sections.values(),
            key=lambda s: s.position
        )]

    def get_section(self, section_name: str) -> Optional[OutlineSection]:
        """Get a section by name."""
        return self.sections.get(section_name)

    def get_plugins_for_section(self, section_name: str) -> Tuple[List[str], List[str]]:
        """
        Get required and optional plugins for a section.

        Returns:
            Tuple of (required_plugins, optional_plugins)
        """
        section = self.get_section(section_name)
        if not section:
            return [], []

        return section.required_plugins, section.optional_plugins

    def get_all_plugins_by_section(self) -> Dict[str, Dict[str, List[str]]]:
        """Get all plugins organized by section."""
        result = {}
        for section_name, section in self.sections.items():
            result[section_name] = {
                "required": section.required_plugins,
                "optional": section.optional_plugins,
            }
        return result

    def check_critical_transition(self, section1: str, section2: str) -> Optional[OutlineTransition]:
        """
        Check if two sections form a critical transition.

        Returns:
            OutlineTransition if critical, None otherwise
        """
        for transition in self.transitions:
            if (transition.from_section == section1 and
                transition.to_section == section2):
                return transition
        return None

    def validate_section_order(self, current_order: List[str]) -> Dict[str, Any]:
        """
        Validate that sections are in the correct order.

        Returns:
            Validation result with issues and recommendations
        """
        issues = []
        warnings = []
        recommendations = []

        perfect_order = self.get_section_order()

        # Check critical transitions
        for transition in self.transitions:
            if transition.must_be_consecutive:
                try:
                    idx1 = current_order.index(transition.from_section)
                    idx2 = current_order.index(transition.to_section)

                    if idx2 != idx1 + 1:
                        issues.append({
                            "type": "critical_transition",
                            "severity": "error",
                            "message": f"CRITICAL: {transition.from_section} must immediately precede {transition.to_section}",
                            "transition": transition,
                            "current_positions": (idx1, idx2),
                            "expected_positions": (idx1, idx1 + 1)
                        })
                        recommendations.append(
                            f"Move '{transition.to_section}' to immediately after '{transition.from_section}'"
                        )
                except ValueError:
                    warnings.append({
                        "type": "missing_section",
                        "severity": "warning",
                        "message": f"Missing section in order: {transition.from_section} or {transition.to_section}"
                    })

        # Check section positions (warnings for optional sections)
        section_positions = {name: idx for idx, name in enumerate(current_order)}
        perfect_positions = {name: idx for idx, name in enumerate(perfect_order)}

        for section_name, section in self.sections.items():
            if section_name in section_positions:
                current_pos = section_positions[section_name]
                perfect_pos = perfect_positions.get(section_name, current_pos)

                if section.importance > 10.0 and abs(current_pos - perfect_pos) > 2:
                    warnings.append({
                        "type": "position_deviation",
                        "severity": "warning",
                        "message": f"High-importance section '{section_name}' is {abs(current_pos - perfect_pos)} positions away from recommended",
                        "section": section_name,
                        "current_position": current_pos,
                        "recommended_position": perfect_pos
                    })

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "perfect_order": perfect_order,
            "current_order": current_order
        }

    def get_enumeration_requirements(self) -> Dict[str, Any]:
        """Get enumeration requirements for the outline."""
        return {
            **self.ENUMERATION_REQUIREMENTS,
            "section_requirements": {
                section_name: {
                    "required": section.enumeration_required,
                    "min_count": section.enumeration_min_count,
                    "style": section.metadata.get("enumeration_style", "bullet_points")
                }
                for section_name, section in self.sections.items()
                if section.enumeration_required
            }
        }

    def calibrate_plugin_targets(self, section_name: str, plugin_name: str) -> Dict[str, Any]:
        """
        Calibrate plugin targets based on outline section requirements.

        Returns:
            Calibration data for the plugin
        """
        section = self.get_section(section_name)
        if not section:
            return {}

        calibration = {
            "section": section_name,
            "section_importance": section.importance,
            "enumeration_required": section.enumeration_required,
            "enumeration_min_count": section.enumeration_min_count,
        }

        # Section-specific calibrations
        if section_name == "legal_standard":
            calibration.update({
                "citation_required": True,
                "min_citations": 3,
                "must_include_cases": section.metadata.get("must_cite", []),
            })
        elif section_name == "privacy_harm":
            calibration.update({
                "enumeration_style": "bullet_points",
                "min_privacy_mentions": 5,  # From CatBoost analysis
                "min_harm_types": 3,
            })
        elif section_name == "danger_safety":
            calibration.update({
                "enumeration_style": "numbered_list",
                "min_safety_mentions": 2,
                "position_priority": "high",  # Early placement matters
            })
        elif section_name == "balancing_test":
            calibration.update({
                "enumeration_style": "bullet_points",
                "position_priority": "high",  # Prominent placement
            })

        return calibration

    def get_outline_summary(self) -> Dict[str, Any]:
        """Get summary of the perfect outline structure."""
        return {
            "total_sections": len(self.sections),
            "required_sections": [s.name for s in self.sections.values() if s.section_type == SectionType.REQUIRED],
            "critical_transitions": [
                {
                    "from": t.from_section,
                    "to": t.to_section,
                    "importance": t.importance
                }
                for t in self.transitions
            ],
            "enumeration_requirements": self.get_enumeration_requirements(),
            "section_order": self.get_section_order(),
            "top_features": [
                {
                    "name": s.name,
                    "importance": s.importance,
                    "enumeration_required": s.enumeration_required
                }
                for s in sorted(
                    self.sections.values(),
                    key=lambda x: x.importance,
                    reverse=True
                )[:5]
            ]
        }


def load_outline_manager(outline_source: Optional[Path] = None, base_dir: Optional[Path] = None) -> OutlineManager:
    """
    Load OutlineManager with perfect outline structure.

    Args:
        outline_source: Optional path to JSON file with outline data
        base_dir: Optional base directory for relative path resolution

    Returns:
        Configured OutlineManager instance
    """
    # Try to load from analysis results if available
    if outline_source is None:
        # Try multiple possible locations for flexibility
        possible_paths = [
            Path("case_law_data/results/catboost_structure_analysis.json"),
            Path(__file__).parent.parent.parent / "case_law_data/results/catboost_structure_analysis.json",
        ]

        if base_dir:
            possible_paths.insert(0, base_dir / "case_law_data/results/catboost_structure_analysis.json")

        for analysis_path in possible_paths:
            if analysis_path.exists():
                outline_source = analysis_path
                logger.info(f"Found outline data at: {outline_source}")
                break

    return OutlineManager(outline_source=outline_source)

