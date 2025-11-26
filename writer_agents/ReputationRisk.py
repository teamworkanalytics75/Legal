"""Reputation and PR risk analysis for legal cases.

This module analyzes the reputational impact of different case outcomes,
with specific models for institutional defendants like Harvard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from writer_agents.code.insights import CaseInsights

logger = logging.getLogger(__name__)


@dataclass
class ReputationConfig:
    """Configuration for reputation analysis."""

    institution_name: str = "Harvard"

    # Reputation factors with weights
    factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "academic_prestige": {
            "weight": 0.25,
            "baseline_score": 98.0, # Out of 100
        },
        "federal_funding": {
            "weight": 0.20,
            "baseline_dollars": 600_000_000.0, # Annual federal funding
        },
        "donor_relations": {
            "weight": 0.20,
            "baseline_score": 90.0,
        },
        "student_enrollment": {
            "weight": 0.15,
            "baseline_applications": 50_000.0,
        },
        "media_perception": {
            "weight": 0.10,
            "baseline_sentiment": 0.7, # 0-1 scale
        },
        "alumni_trust": {
            "weight": 0.10,
            "baseline_score": 85.0,
        }
    })


@dataclass
class ReputationImpact:
    """Impact analysis for a specific outcome scenario."""
    scenario: str # "settle_high", "settle_low", "trial_win", "trial_lose"
    overall_score: float # Negative score indicating damage
    factor_impacts: Dict[str, float] # Impact per factor

    def to_report(self) -> str:
        """Generate markdown report."""
        impact_level = self._interpret_impact(self.overall_score)

        report = [f"### {self.scenario.replace('_', ' ').title()} Scenario"]
        report.append(f"\n**Overall Reputation Impact:** {self.overall_score:.1f} ({impact_level})")
        report.append("\n**Factor Breakdown:**")

        for factor, impact in self.factor_impacts.items():
            report.append(f"- {factor.replace('_', ' ').title()}: {impact:.1f}")

        return "\n".join(report)

    def _interpret_impact(self, score: float) -> str:
        """Interpret impact score."""
        if score > -2:
            return "Minimal Impact"
        elif score > -5:
            return "Minor Impact"
        elif score > -10:
            return "Moderate Impact"
        elif score > -15:
            return "Significant Impact"
        else:
            return "Severe Impact"


@dataclass
class MediaImpactScore:
    """Media coverage impact assessment."""
    interest_score: float # 0-100 scale
    expected_article_count: int
    major_outlets_likely: bool
    expected_negative_ratio: float # 0-1 scale

    def to_report(self) -> str:
        """Generate markdown report."""
        intensity = "High" if self.interest_score > 70 else "Moderate" if self.interest_score > 40 else "Low"

        return f"""
**Media Interest Level:** {intensity} ({self.interest_score:.0f}/100)

**Expected Articles:** ~{self.expected_article_count}

**Major Outlet Coverage:** {"Yes" if self.major_outlets_likely else "Unlikely"}

**Expected Negative Sentiment:** {self.expected_negative_ratio:.0%}
"""


class ReputationFactorAnalyzer:
    """Analyze factors affecting institutional reputation."""

    def __init__(self, config: Optional[ReputationConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or ReputationConfig()

    def analyze_reputation_impact(
        self,
        case_insights: CaseInsights,
        case_outcome: str # "settle_high", "settle_low", "trial_win", "trial_lose"
    ) -> ReputationImpact:
        """
        Analyze reputation impact for different outcomes.

        Args:
            case_insights: CaseInsights from BN analysis
            case_outcome: Outcome scenario to analyze

        Returns:
            ReputationImpact with detailed analysis
        """
        logger.info(f"Analyzing reputation impact for outcome: {case_outcome}")

        impacts = {}

        # Academic prestige impact
        impacts["academic_prestige"] = self._calculate_prestige_impact(case_outcome)

        # Federal funding risk
        involves_national_security = self._check_national_security(case_insights)
        impacts["federal_funding"] = self._calculate_funding_impact(
            case_outcome,
            involves_national_security
        )

        # Donor relations impact
        impacts["donor_relations"] = self._calculate_donor_impact(case_outcome)

        # Student enrollment impact
        impacts["student_enrollment"] = self._calculate_enrollment_impact(case_outcome)

        # Media perception impact
        impacts["media_perception"] = self._calculate_media_impact(case_outcome)

        # Alumni trust impact
        impacts["alumni_trust"] = self._calculate_alumni_impact(case_outcome)

        # Calculate overall weighted impact
        total_impact = sum(
            impacts.get(factor, 0) * self.config.factors[factor]["weight"]
            for factor in self.config.factors
        )

        logger.info(f"Overall reputation impact: {total_impact:.1f}")

        return ReputationImpact(
            scenario=case_outcome,
            overall_score=total_impact,
            factor_impacts=impacts,
        )

    def _calculate_prestige_impact(self, outcome: str) -> float:
        """Calculate academic prestige impact."""
        impact_map = {
            "trial_lose": -15.0, # Significant hit
            "settle_high": -8.0, # Moderate hit (looks guilty)
            "settle_low": -3.0, # Minor hit
            "trial_win": -2.0, # Some cost even if win
        }
        return impact_map.get(outcome, -5.0)

    def _calculate_funding_impact(self, outcome: str, national_security: bool) -> float:
        """Calculate federal funding impact (as percentage of baseline)."""
        if not national_security:
            return 0.0

        impact_map = {
            "trial_lose": -0.20, # 20% at risk
            "settle_high": -0.10, # 10% at risk
            "settle_low": -0.05, # 5% at risk
            "trial_win": -0.02, # 2% at risk
        }

        percentage_impact = impact_map.get(outcome, -0.08)
        # Convert to dollar impact
        baseline_funding = self.config.factors["federal_funding"]["baseline_dollars"]
        return percentage_impact * baseline_funding / 1_000_000 # In millions for readability

    def _calculate_donor_impact(self, outcome: str) -> float:
        """Calculate donor relations impact."""
        impact_map = {
            "trial_lose": -12.0,
            "settle_high": -7.0,
            "settle_low": -3.0,
            "trial_win": -1.0,
        }
        return impact_map.get(outcome, -5.0)

    def _calculate_enrollment_impact(self, outcome: str) -> float:
        """Calculate student enrollment impact (as % change)."""
        impact_map = {
            "trial_lose": -5.0, # 5% fewer applications
            "settle_high": -2.0,
            "settle_low": -0.5,
            "trial_win": 0.0,
        }
        return impact_map.get(outcome, -2.0)

    def _calculate_media_impact(self, outcome: str) -> float:
        """Calculate media perception impact."""
        impact_map = {
            "trial_lose": -25.0, # Major negative coverage
            "settle_high": -15.0,
            "settle_low": -8.0,
            "trial_win": -3.0,
        }
        return impact_map.get(outcome, -10.0)

    def _calculate_alumni_impact(self, outcome: str) -> float:
        """Calculate alumni trust impact."""
        impact_map = {
            "trial_lose": -18.0,
            "settle_high": -10.0,
            "settle_low": -4.0,
            "trial_win": -2.0,
        }
        return impact_map.get(outcome, -8.0)

    def _check_national_security(self, insights: CaseInsights) -> bool:
        """Check if case involves national security issues."""
        # Look for national security indicators in evidence
        for evidence_item in insights.evidence:
            if any(keyword in evidence_item.node_id.lower()
                   for keyword in ["prc", "china", "national", "security", "foreign"]):
                return True

        # Check summary
        if any(keyword in insights.summary.lower()
               for keyword in ["national security", "china", "prc", "foreign"]):
            return True

        return False


class MediaImpactModeler:
    """Model media coverage and sentiment."""

    def model_media_coverage(
        self,
        case_severity: str, # "low", "moderate", "high", "extreme"
        involves_china: bool = False,
        involves_student_safety: bool = False,
    ) -> MediaImpactScore:
        """
        Model expected media coverage.

        Args:
            case_severity: Severity level of the case
            involves_china: Whether case involves China-related issues
            involves_student_safety: Whether case involves student safety

        Returns:
            MediaImpactScore with coverage estimates
        """
        logger.info(f"Modeling media coverage for {case_severity} severity case")

        # Base media interest score
        interest_scores = {
            "low": 20.0,
            "moderate": 50.0,
            "high": 75.0,
            "extreme": 95.0,
        }
        base_interest = interest_scores.get(case_severity, 50.0)

        # Amplification factors
        if involves_china:
            base_interest *= 1.5 # China stories get huge coverage
            logger.info("Amplifying for China-related content")

        if involves_student_safety:
            base_interest *= 1.3 # Student safety is compelling
            logger.info("Amplifying for student safety concerns")

        # Cap at 100
        total_interest = min(base_interest, 100.0)

        # Estimate coverage metrics based on interest level
        if total_interest > 80:
            expected_articles = 500 # Major national story
            major_outlet_coverage = True
            expected_negative_sentiment = 0.75
        elif total_interest > 60:
            expected_articles = 150
            major_outlet_coverage = True
            expected_negative_sentiment = 0.65
        elif total_interest > 40:
            expected_articles = 50
            major_outlet_coverage = False
            expected_negative_sentiment = 0.55
        else:
            expected_articles = 15
            major_outlet_coverage = False
            expected_negative_sentiment = 0.50

        return MediaImpactScore(
            interest_score=total_interest,
            expected_article_count=expected_articles,
            major_outlets_likely=major_outlet_coverage,
            expected_negative_ratio=expected_negative_sentiment,
        )


class ReputationRiskScorer:
    """Calculate overall reputation risk scores."""

    def __init__(self, config: Optional[ReputationConfig] = None):
        """
        Initialize scorer.

        Args:
            config: Optional configuration
        """
        self.config = config or ReputationConfig()
        self.factor_analyzer = ReputationFactorAnalyzer(config)
        self.media_modeler = MediaImpactModeler()

    def score_reputation_risk(
        self,
        case_insights: CaseInsights,
        case_outcomes: Optional[List[str]] = None
    ) -> Dict[str, ReputationImpact]:
        """
        Score reputation risk across multiple outcome scenarios.

        Args:
            case_insights: CaseInsights from BN analysis
            case_outcomes: Optional list of outcomes (uses default if not provided)

        Returns:
            Dictionary mapping outcome to ReputationImpact
        """
        if case_outcomes is None:
            case_outcomes = ["settle_high", "settle_low", "trial_win", "trial_lose"]

        logger.info(f"Scoring reputation risk for {len(case_outcomes)} outcomes")

        reputation_impacts = {}
        for outcome in case_outcomes:
            impact = self.factor_analyzer.analyze_reputation_impact(case_insights, outcome)
            reputation_impacts[outcome] = impact

        return reputation_impacts

    def generate_summary_report(
        self,
        reputation_impacts: Dict[str, ReputationImpact],
        media_score: Optional[MediaImpactScore] = None
    ) -> str:
        """
        Generate comprehensive reputation risk summary.

        Args:
            reputation_impacts: Reputation impacts by outcome
            media_score: Optional media impact score

        Returns:
            Markdown formatted report
        """
        report = ["# Reputation Risk Analysis\n"]

        if media_score:
            report.append("## Media Impact Assessment\n")
            report.append(media_score.to_report())
            report.append("\n---\n")

        report.append("## Outcome Scenarios\n")

        # Sort by impact severity
        sorted_impacts = sorted(
            reputation_impacts.items(),
            key=lambda x: x[1].overall_score
        )

        for outcome, impact in sorted_impacts:
            report.append(impact.to_report())
            report.append("\n")

        # Summary recommendations
        report.append("## Strategic Recommendations\n")

        worst_impact = sorted_impacts[0][1]
        best_impact = sorted_impacts[-1][1]

        report.append(f"**Worst Case:** {worst_impact.scenario} (Impact: {worst_impact.overall_score:.1f})")
        report.append(f"**Best Case:** {best_impact.scenario} (Impact: {best_impact.overall_score:.1f})")
        report.append("\n")

        if worst_impact.overall_score < -10:
            report.append(
                "WARNING **High Reputational Risk:** The worst-case scenario carries significant "
                "reputational damage. Consider prioritizing reputation protection in strategy."
            )

        return "\n".join(report)


__all__ = [
    "ReputationConfig",
    "ReputationImpact",
    "MediaImpactScore",
    "ReputationFactorAnalyzer",
    "MediaImpactModeler",
    "ReputationRiskScorer",
]

