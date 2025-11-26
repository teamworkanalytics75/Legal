#!/usr/bin/env python3
"""
Paragraph Monitor Plugin - Atomic SK plugin for monitoring and aggregating paragraph-level health.

Monitors:
- Overall paragraph health distribution
- Paragraph-level issue tracking
- Cross-paragraph consistency
- Progression and flow
- Position-specific requirements (intro, body, conclusion)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class ParagraphMonitorPlugin(BaseFeaturePlugin):
    """Atomic plugin for monitoring paragraph health and consistency."""

    def __init__(
        self,
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        per_paragraph_plugin=None,
        memory_store=None,
        db_paths=None,
        enable_langchain: bool = True,
        enable_courtlistener: bool = False,
        enable_storm: bool = False
    ):
        super().__init__(
            kernel, "paragraph_monitor", chroma_store, rules_dir,
            memory_store=memory_store,
            db_paths=db_paths,
            enable_langchain=enable_langchain,
            enable_courtlistener=enable_courtlistener,
            enable_storm=enable_storm
        )
        self.per_paragraph_plugin = per_paragraph_plugin
        logger.info("ParagraphMonitorPlugin initialized")

    async def monitor_all_paragraphs(self, draft_text: str) -> FunctionResult:
        """Monitor all paragraphs and provide aggregated health metrics."""
        try:
            # Use per-paragraph plugin if available, otherwise analyze directly
            if self.per_paragraph_plugin:
                para_analysis = await self.per_paragraph_plugin.analyze_all_paragraphs(draft_text)
                if not para_analysis.success:
                    return para_analysis
                para_data = para_analysis.value
            else:
                # Fallback: basic analysis
                return FunctionResult(
                    success=False,
                    value=None,
                    error="PerParagraphPlugin not available"
                )

            paragraph_analyses = para_data.get("paragraph_analyses", [])
            total_paragraphs = len(paragraph_analyses)

            if total_paragraphs == 0:
                return FunctionResult(
                    success=False,
                    value=None,
                    error="No paragraphs to monitor"
                )

            # Aggregate metrics
            monitoring_results = {
                "total_paragraphs": total_paragraphs,
                "healthy_count": len(para_data.get("healthy_paragraphs", [])),
                "problematic_count": len(para_data.get("problematic_paragraphs", [])),
                "avg_health_score": para_data.get("avg_health_score", 0.0),
                "health_distribution": para_data.get("health_distribution", {}),

                # Position-specific analysis
                "intro_paragraph": self._analyze_position(paragraph_analyses, "intro"),
                "conclusion_paragraph": self._analyze_position(paragraph_analyses, "conclusion"),
                "body_paragraphs": self._analyze_position(paragraph_analyses, "body"),

                # Issue tracking
                "common_issues": self._identify_common_issues(paragraph_analyses),
                "critical_paragraphs": self._identify_critical_paragraphs(paragraph_analyses),

                # Consistency checks
                "length_consistency": self._check_length_consistency(paragraph_analyses),
                "sentence_consistency": self._check_sentence_consistency(paragraph_analyses),

                # Recommendations
                "recommendations": []
            }

            # Generate monitoring recommendations
            recommendations = []

            # Health threshold recommendations
            if monitoring_results["avg_health_score"] < 0.7:
                recommendations.append(
                    f"Overall paragraph health is below target (score: {monitoring_results['avg_health_score']:.2f}). "
                    f"{monitoring_results['problematic_count']} paragraphs need attention."
                )

            # Position-specific recommendations
            intro_analysis = monitoring_results["intro_paragraph"]
            if intro_analysis and intro_analysis.get("issues"):
                recommendations.append(
                    f"Introduction paragraph has issues: {', '.join(intro_analysis['issues'])}"
                )

            concl_analysis = monitoring_results["conclusion_paragraph"]
            if concl_analysis and concl_analysis.get("issues"):
                recommendations.append(
                    f"Conclusion paragraph has issues: {', '.join(concl_analysis['issues'])}"
                )

            # Consistency recommendations
            if not monitoring_results["length_consistency"]["consistent"]:
                recommendations.append(
                    f"Paragraph length inconsistency detected: {monitoring_results['length_consistency']['message']}"
                )

            # Critical paragraph recommendations
            if monitoring_results["critical_paragraphs"]:
                recommendations.append(
                    f"{len(monitoring_results['critical_paragraphs'])} critical paragraphs need immediate attention "
                    f"(health score < 0.5)"
                )

            monitoring_results["recommendations"] = recommendations

            success = len(monitoring_results["problematic_paragraphs"]) == 0 and monitoring_results["avg_health_score"] >= 0.7

            return FunctionResult(
                success=success,
                value=monitoring_results,
                error=None if success else f"{monitoring_results['problematic_count']} paragraphs need attention"
            )

        except Exception as e:
            logger.error(f"Paragraph monitoring failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _analyze_position(self, paragraph_analyses: List[Dict], position: str) -> Optional[Dict]:
        """Analyze paragraphs at a specific position."""
        position_paras = [p for p in paragraph_analyses if p.get("position") == position]

        if not position_paras:
            return None

        if position == "intro":
            para = position_paras[0]  # Should only be one intro
            target_min = self.rules.get("targets", {}).get("intro_min_words", 50)
            target_max = self.rules.get("targets", {}).get("intro_max_words", 200)
        elif position == "conclusion":
            para = position_paras[0]  # Should only be one conclusion
            target_min = self.rules.get("targets", {}).get("conclusion_min_words", 40)
            target_max = self.rules.get("targets", {}).get("conclusion_max_words", 150)
        else:  # body
            # Aggregate body paragraphs
            avg_words = sum(p["word_count"] for p in position_paras) / len(position_paras)
            avg_sentences = sum(p["sentence_count"] for p in position_paras) / len(position_paras)
            return {
                "count": len(position_paras),
                "avg_word_count": avg_words,
                "avg_sentence_count": avg_sentences,
                "health_score": sum(p["health_score"] for p in position_paras) / len(position_paras)
            }

        issues = []
        if para["word_count"] < target_min:
            issues.append(f"Too short for {position}: {para['word_count']} words (target: {target_min}+)")
        if para["word_count"] > target_max:
            issues.append(f"Too long for {position}: {para['word_count']} words (target: <{target_max})")

        return {
            "paragraph_index": para["paragraph_index"],
            "word_count": para["word_count"],
            "sentence_count": para["sentence_count"],
            "health_score": para["health_score"],
            "issues": issues
        }

    def _identify_common_issues(self, paragraph_analyses: List[Dict]) -> List[Dict]:
        """Identify issues that appear across multiple paragraphs."""
        issue_counts = {}

        for para in paragraph_analyses:
            for issue in para.get("issues", []):
                # Extract issue type (e.g., "Too short", "Too long")
                issue_type = issue.split(":")[0] if ":" in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Return issues that appear in 2+ paragraphs
        common = [
            {"issue": issue_type, "count": count, "percentage": count / len(paragraph_analyses) * 100}
            for issue_type, count in issue_counts.items()
            if count >= 2
        ]

        return sorted(common, key=lambda x: x["count"], reverse=True)

    def _identify_critical_paragraphs(self, paragraph_analyses: List[Dict]) -> List[Dict]:
        """Identify paragraphs with critical issues (health score < 0.5)."""
        critical = [
            {
                "paragraph_index": p["paragraph_index"],
                "position": p["position"],
                "health_score": p["health_score"],
                "issues": p["issues"],
                "word_count": p["word_count"]
            }
            for p in paragraph_analyses
            if p["health_score"] < 0.5
        ]

        return sorted(critical, key=lambda x: x["health_score"])

    def _check_length_consistency(self, paragraph_analyses: List[Dict]) -> Dict[str, Any]:
        """Check if paragraph lengths are reasonably consistent."""
        word_counts = [p["word_count"] for p in paragraph_analyses]

        if len(word_counts) < 2:
            return {"consistent": True, "message": "Insufficient paragraphs for consistency check"}

        avg_length = sum(word_counts) / len(word_counts)
        max_deviation = max(abs(wc - avg_length) for wc in word_counts)
        deviation_pct = (max_deviation / avg_length * 100) if avg_length > 0 else 0

        # Consider consistent if deviation is within 50% of average
        consistent = deviation_pct < 50

        return {
            "consistent": consistent,
            "avg_length": avg_length,
            "max_deviation": max_deviation,
            "deviation_percentage": deviation_pct,
            "message": f"Max deviation: {max_deviation:.0f} words ({deviation_pct:.1f}%)" if not consistent else "Lengths are reasonably consistent"
        }

    def _check_sentence_consistency(self, paragraph_analyses: List[Dict]) -> Dict[str, Any]:
        """Check if sentence counts per paragraph are reasonably consistent."""
        sentence_counts = [p["sentence_count"] for p in paragraph_analyses]

        if len(sentence_counts) < 2:
            return {"consistent": True, "message": "Insufficient paragraphs for consistency check"}

        avg_sentences = sum(sentence_counts) / len(sentence_counts)
        max_deviation = max(abs(sc - avg_sentences) for sc in sentence_counts)
        deviation_pct = (max_deviation / avg_sentences * 100) if avg_sentences > 0 else 0

        # Consider consistent if deviation is within 40% of average
        consistent = deviation_pct < 40

        return {
            "consistent": consistent,
            "avg_sentences": avg_sentences,
            "max_deviation": max_deviation,
            "deviation_percentage": deviation_pct,
            "message": f"Max deviation: {max_deviation:.1f} sentences ({deviation_pct:.1f}%)" if not consistent else "Sentence counts are reasonably consistent"
        }

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft by monitoring all paragraphs."""
        return await self.monitor_all_paragraphs(draft_text)

