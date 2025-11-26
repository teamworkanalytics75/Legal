#!/usr/bin/env python3
"""
Question Classifier - Analyzes user questions to determine type, components, and complexity.

This component classifies natural language questions to determine:
- Question type (probability, research, analysis, writing)
- Required system components (Research, ML, BN, Writing)
- Complexity level (quick answer vs full workflow)
"""

import logging
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions the system can handle."""
    PROBABILITY = "probability"      # "what's the probability", "percent chance"
    RESEARCH = "research"            # "find cases", "what does case law say"
    ANALYSIS = "analysis"            # "analyze my case", "evaluate my motion"
    WRITING = "writing"              # "write a motion", "draft a section"
    HYBRID = "hybrid"                # Complex questions requiring multiple components
    UNKNOWN = "unknown"              # Could not classify


class RequiredComponent(Enum):
    """System components that may be needed."""
    RESEARCH = "research"            # CaseLawResearcher
    ML = "ml"                        # CatBoost/RefinementLoop
    BN = "bn"                        # Bayesian Network inference
    WRITING = "writing"              # AutoGen/SK writing
    NONE = "none"                    # Quick answer only


class ComplexityLevel(Enum):
    """Question complexity determines answer strategy."""
    QUICK = "quick"                  # Direct query, <10 seconds
    MODERATE = "moderate"            # Some processing, 10-30 seconds
    COMPLEX = "complex"              # Full workflow, 30+ seconds


@dataclass
class QuestionClassification:
    """Result of question classification."""
    question_type: QuestionType
    required_components: List[RequiredComponent]
    complexity: ComplexityLevel
    confidence: float  # 0.0-1.0
    keywords: List[str]
    reasoning: str


class QuestionClassifier:
    """Classifies natural language questions for routing to appropriate system components."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize question classifier.

        Args:
            config_path: Optional path to classification rules JSON file
        """
        self.config_path = config_path or self._find_default_config()
        self.rules = self._load_rules()
        logger.info(f"QuestionClassifier initialized with {len(self.rules.get('patterns', {}))} classification patterns")

    def _find_default_config(self) -> Path:
        """Find default configuration file path."""
        possible_paths = [
            Path(__file__).parent.parent / "config" / "question_classification_rules.json",
            Path(__file__).parent.parent.parent / "writer_agents" / "config" / "question_classification_rules.json",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        # Return default path (will create if doesn't exist)
        return possible_paths[0]

    def _load_rules(self) -> Dict[str, Any]:
        """Load classification rules from config file."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load classification rules: {e}")
        return self._get_default_rules()

    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default classification rules."""
        return {
            "patterns": {
                "probability": {
                    "keywords": [
                        "probability", "percent", "percentage", "chance", "likelihood",
                        "odds", "what's the", "what is the", "how likely", "how probable"
                    ],
                    "required_components": ["bn", "research"],
                    "complexity": "quick"
                },
                "research": {
                    "keywords": [
                        "find cases", "case law", "precedents", "what does", "what did",
                        "search for", "look up", "find similar", "related cases"
                    ],
                    "required_components": ["research"],
                    "complexity": "quick"
                },
                "analysis": {
                    "keywords": [
                        "analyze", "evaluate", "assess", "review", "examine",
                        "what about", "tell me about", "explain"
                    ],
                    "required_components": ["research", "ml"],
                    "complexity": "moderate"
                },
                "writing": {
                    "keywords": [
                        "write", "draft", "create", "generate", "compose",
                        "make a", "prepare a", "build a"
                    ],
                    "required_components": ["writing", "research", "ml"],
                    "complexity": "complex"
                }
            },
            "complexity_indicators": {
                "quick": ["simple", "quick", "fast", "direct"],
                "moderate": ["analyze", "compare", "evaluate"],
                "complex": ["comprehensive", "full", "complete", "detailed", "thorough"]
            }
        }

    def classify(self, question: str, context: Optional[str] = None) -> QuestionClassification:
        """
        Classify a user question.

        Args:
            question: User's question text
            context: Optional conversation context

        Returns:
            QuestionClassification with type, components, complexity, etc.
        """
        question_lower = question.lower()

        # Extract keywords
        keywords = self._extract_keywords(question_lower)

        # Determine question type
        question_type, type_confidence = self._classify_type(question_lower, keywords)

        # Determine required components
        required_components = self._determine_components(question_lower, question_type, keywords)

        # Determine complexity
        complexity = self._determine_complexity(question_lower, question_type, keywords)

        # Generate reasoning
        reasoning = self._generate_reasoning(question_type, required_components, complexity, keywords)

        return QuestionClassification(
            question_type=question_type,
            required_components=required_components,
            complexity=complexity,
            confidence=type_confidence,
            keywords=keywords,
            reasoning=reasoning
        )

    def _extract_keywords(self, question_lower: str) -> List[str]:
        """Extract relevant keywords from question."""
        keywords = []
        all_patterns = self.rules.get("patterns", {})

        for pattern_type, pattern_data in all_patterns.items():
            pattern_keywords = pattern_data.get("keywords", [])
            for keyword in pattern_keywords:
                if keyword.lower() in question_lower:
                    keywords.append(keyword)

        return list(set(keywords))  # Deduplicate

    def _classify_type(self, question_lower: str, keywords: List[str]) -> Tuple[QuestionType, float]:
        """Classify question type based on patterns."""
        patterns = self.rules.get("patterns", {})
        scores = {}

        # Score each pattern type
        for pattern_type, pattern_data in patterns.items():
            score = 0.0
            pattern_keywords = pattern_data.get("keywords", [])

            # Count keyword matches
            matches = sum(1 for kw in pattern_keywords if kw.lower() in question_lower)
            if matches > 0:
                score = min(matches / len(pattern_keywords), 1.0) * 0.8  # Base score

            # Boost for exact phrase matches
            for keyword in pattern_keywords:
                if keyword in question_lower:
                    score += 0.1

            scores[pattern_type] = score

        # Check for hybrid (multiple strong signals)
        strong_signals = [pt for pt, score in scores.items() if score > 0.5]
        if len(strong_signals) >= 2:
            return QuestionType.HYBRID, 0.9

        # Find best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0.3:
                try:
                    return QuestionType(best_type[0]), best_type[1]
                except ValueError:
                    pass

        return QuestionType.UNKNOWN, 0.0

    def _determine_components(self, question_lower: str, question_type: QuestionType, keywords: List[str]) -> List[RequiredComponent]:
        """Determine which system components are required."""
        components = set()

        # Get components from pattern rules
        patterns = self.rules.get("patterns", {})
        if question_type != QuestionType.UNKNOWN:
            try:
                pattern_data = patterns.get(question_type.value, {})
                required = pattern_data.get("required_components", [])
                for comp in required:
                    try:
                        components.add(RequiredComponent(comp))
                    except ValueError:
                        pass
            except:
                pass

        # Add components based on keywords
        if any(kw in question_lower for kw in ["probability", "percent", "chance", "likelihood"]):
            components.add(RequiredComponent.BN)

        if any(kw in question_lower for kw in ["case", "precedent", "law", "court", "judge"]):
            components.add(RequiredComponent.RESEARCH)

        if any(kw in question_lower for kw in ["analyze", "evaluate", "assess", "features"]):
            components.add(RequiredComponent.ML)

        if any(kw in question_lower for kw in ["write", "draft", "create", "generate"]):
            components.add(RequiredComponent.WRITING)

        # Remove NONE if we have actual components
        if components and RequiredComponent.NONE in components:
            components.remove(RequiredComponent.NONE)

        if not components:
            components.add(RequiredComponent.NONE)

        return list(components)

    def _determine_complexity(self, question_lower: str, question_type: QuestionType, keywords: List[str]) -> ComplexityLevel:
        """Determine question complexity."""
        complexity_indicators = self.rules.get("complexity_indicators", {})

        # Check for explicit complexity indicators
        for level_str, indicators in complexity_indicators.items():
            if any(ind in question_lower for ind in indicators):
                try:
                    return ComplexityLevel(level_str)
                except ValueError:
                    pass

        # Default complexity based on question type
        if question_type == QuestionType.WRITING:
            return ComplexityLevel.COMPLEX
        elif question_type == QuestionType.ANALYSIS:
            return ComplexityLevel.MODERATE
        elif question_type == QuestionType.HYBRID:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.QUICK

    def _generate_reasoning(self, question_type: QuestionType, components: List[RequiredComponent],
                           complexity: ComplexityLevel, keywords: List[str]) -> str:
        """Generate human-readable reasoning for classification."""
        reasoning_parts = [
            f"Question type: {question_type.value}",
            f"Requires: {', '.join(c.value for c in components)}",
            f"Complexity: {complexity.value}",
            f"Keywords: {', '.join(keywords[:5])}"  # Limit to first 5
        ]
        return "; ".join(reasoning_parts)

