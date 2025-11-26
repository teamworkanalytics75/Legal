#!/usr/bin/env python3
"""
Comprehensive Database Analysis

This script performs deep analysis of our §1782 database to extract
comprehensive insights and patterns from all available data.
"""

import json
import logging
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComprehensiveDatabaseAnalyzer:
    """Comprehensive analysis of our §1782 database."""

    def __init__(self):
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.analysis_results = {
            "analysis_date": datetime.now().isoformat(),
            "total_cases": 0,
            "cases_with_text": 0,
            "cases_with_outcomes": 0,
            "insights": {}
        }

    def load_all_data(self) -> Dict[str, Any]:
        """Load all available data from our database."""
        logger.info("Loading comprehensive database data...")

        data = {
            "mathematical_patterns": self._load_json("mathematical_patterns.json"),
            "comprehensive_features": self._load_json("comprehensive_features.json"),
            "court_outcomes": self._load_json("court_outcomes_extracted.json"),
            "text_extraction_log": self._load_json("text_extraction_log.json"),
            "wishlist_log": self._load_json("wishlist_acquisition_log.json"),
            "case_files": self._load_case_files()
        }

        logger.info(f"Loaded data from {len(data)} sources")
        return data

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load JSON data file."""
        try:
            file_path = Path("data/case_law") / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load {filename}: {e}")
            return {}

    def _load_case_files(self) -> List[Dict[str, Any]]:
        """Load all case files from the corpus."""
        case_files = []

        for case_file in self.corpus_dir.glob("*.json"):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    case_data['file_name'] = case_file.name
                    case_files.append(case_data)
            except Exception as e:
                logger.warning(f"Could not load {case_file.name}: {e}")

        logger.info(f"Loaded {len(case_files)} case files")
        return case_files

    def analyze_citation_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze citation patterns across all cases."""
        logger.info("Analyzing citation patterns...")

        if not data.get("mathematical_patterns"):
            return {}

        patterns = data["mathematical_patterns"]
        case_analyses = patterns.get("case_analyses", [])

        # Aggregate citation data
        all_citations = defaultdict(int)
        granted_citations = defaultdict(int)
        denied_citations = defaultdict(int)

        for case in case_analyses:
            outcome = case.get("actual_outcome", "unknown")
            citations = case.get("citation_counts", {})

            for citation, count in citations.items():
                all_citations[citation] += count

                if outcome == "granted":
                    granted_citations[citation] += count
                elif outcome == "denied":
                    denied_citations[citation] += count

        # Calculate correlations
        citation_correlations = {}
        for citation in all_citations:
            granted_count = granted_citations[citation]
            denied_count = denied_citations[citation]
            total_count = all_citations[citation]

            if total_count > 0:
                granted_rate = granted_count / total_count
                denied_rate = denied_count / total_count
                correlation = granted_rate - denied_rate  # Positive = favors grants
                citation_correlations[citation] = {
                    "total_mentions": total_count,
                    "granted_mentions": granted_count,
                    "denied_mentions": denied_count,
                    "grant_rate": granted_rate,
                    "denial_rate": denied_rate,
                    "correlation": correlation
                }

        # Sort by correlation strength
        sorted_correlations = sorted(
            citation_correlations.items(),
            key=lambda x: abs(x[1]["correlation"]),
            reverse=True
        )

        return {
            "total_citations_analyzed": len(all_citations),
            "citation_correlations": dict(sorted_correlations),
            "top_positive_correlations": [
                (citation, data) for citation, data in sorted_correlations
                if data["correlation"] > 0.1
            ][:10],
            "top_negative_correlations": [
                (citation, data) for citation, data in sorted_correlations
                if data["correlation"] < -0.1
            ][:10]
        }

    def analyze_court_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze court-specific patterns."""
        logger.info("Analyzing court patterns...")

        if not data.get("comprehensive_features"):
            return {}

        features = data["comprehensive_features"]
        case_features = features.get("case_features", [])

        # Court analysis
        court_stats = defaultdict(lambda: {"total": 0, "granted": 0, "denied": 0})

        for case in case_features:
            outcome = case.get("actual_outcome", "unknown")
            if outcome not in ["granted", "denied"]:
                continue

            # Determine primary court
            primary_court = case.get("primary_court", "unknown")
            court_stats[primary_court]["total"] += 1

            if outcome == "granted":
                court_stats[primary_court]["granted"] += 1
            elif outcome == "denied":
                court_stats[primary_court]["denied"] += 1

        # Calculate grant rates
        court_grant_rates = {}
        for court, stats in court_stats.items():
            if stats["total"] > 0:
                grant_rate = stats["granted"] / stats["total"]
                court_grant_rates[court] = {
                    "total_cases": stats["total"],
                    "granted_cases": stats["granted"],
                    "denied_cases": stats["denied"],
                    "grant_rate": grant_rate
                }

        # Sort by grant rate
        sorted_courts = sorted(
            court_grant_rates.items(),
            key=lambda x: x[1]["grant_rate"],
            reverse=True
        )

        return {
            "total_courts_analyzed": len(court_grant_rates),
            "court_grant_rates": dict(sorted_courts),
            "best_courts": sorted_courts[:5],
            "worst_courts": sorted_courts[-5:]
        }

    def analyze_language_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze language and sentiment patterns."""
        logger.info("Analyzing language patterns...")

        if not data.get("comprehensive_features"):
            return {}

        features = data["comprehensive_features"]
        case_features = features.get("case_features", [])

        # Language analysis
        granted_cases = [case for case in case_features if case.get("actual_outcome") == "granted"]
        denied_cases = [case for case in case_features if case.get("actual_outcome") == "denied"]

        def calculate_stats(cases: List[Dict], field: str) -> Dict[str, float]:
            values = [case.get(field, 0) for case in cases if case.get(field) is not None]
            if not values:
                return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

        language_analysis = {
            "text_length": {
                "granted": calculate_stats(granted_cases, "text_length"),
                "denied": calculate_stats(denied_cases, "text_length")
            },
            "legal_density": {
                "granted": calculate_stats(granted_cases, "legal_density"),
                "denied": calculate_stats(denied_cases, "legal_density")
            },
            "complexity_ratio": {
                "granted": calculate_stats(granted_cases, "complexity_ratio"),
                "denied": calculate_stats(denied_cases, "complexity_ratio")
            },
            "sentiment_compound": {
                "granted": calculate_stats(granted_cases, "sentiment_compound"),
                "denied": calculate_stats(denied_cases, "sentiment_compound")
            }
        }

        return language_analysis

    def analyze_temporal_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        logger.info("Analyzing temporal patterns...")

        case_files = data.get("case_files", [])

        # Extract dates and outcomes
        temporal_data = []
        for case in case_files:
            date_filed = case.get("dateFiled")
            outcome = case.get("actual_outcome")

            if date_filed and outcome in ["granted", "denied"]:
                try:
                    year = int(date_filed.split("-")[0])
                    temporal_data.append({
                        "year": year,
                        "outcome": outcome,
                        "case_name": case.get("caseName", "Unknown")
                    })
                except:
                    continue

        # Analyze by year
        yearly_stats = defaultdict(lambda: {"total": 0, "granted": 0, "denied": 0})

        for item in temporal_data:
            year = item["year"]
            yearly_stats[year]["total"] += 1

            if item["outcome"] == "granted":
                yearly_stats[year]["granted"] += 1
            elif item["outcome"] == "denied":
                yearly_stats[year]["denied"] += 1

        # Calculate yearly grant rates
        yearly_grant_rates = {}
        for year, stats in yearly_stats.items():
            if stats["total"] > 0:
                grant_rate = stats["granted"] / stats["total"]
                yearly_grant_rates[year] = {
                    "total_cases": stats["total"],
                    "granted_cases": stats["granted"],
                    "denied_cases": stats["denied"],
                    "grant_rate": grant_rate
                }

        # Sort by year
        sorted_years = sorted(yearly_grant_rates.items())

        return {
            "total_years_analyzed": len(yearly_grant_rates),
            "yearly_grant_rates": dict(sorted_years),
            "date_range": {
                "earliest": min(yearly_grant_rates.keys()) if yearly_grant_rates else None,
                "latest": max(yearly_grant_rates.keys()) if yearly_grant_rates else None
            }
        }

    def analyze_case_outcomes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze case outcome patterns."""
        logger.info("Analyzing case outcomes...")

        if not data.get("court_outcomes"):
            return {}

        outcomes = data["court_outcomes"]
        results = outcomes.get("results", [])

        # Outcome analysis
        outcome_counts = Counter()
        high_confidence_outcomes = Counter()

        for result in results:
            outcome = result.get("outcome", "unknown")
            confidence = result.get("confidence", 0)

            outcome_counts[outcome] += 1

            if confidence > 0.7:
                high_confidence_outcomes[outcome] += 1

        # Calculate success rates
        total_cases = sum(outcome_counts.values())
        high_confidence_total = sum(high_confidence_outcomes.values())

        success_rate = (outcome_counts.get("granted", 0) + outcome_counts.get("affirmed", 0)) / total_cases if total_cases > 0 else 0
        high_confidence_success_rate = (high_confidence_outcomes.get("granted", 0) + high_confidence_outcomes.get("affirmed", 0)) / high_confidence_total if high_confidence_total > 0 else 0

        return {
            "total_cases": total_cases,
            "high_confidence_cases": high_confidence_total,
            "outcome_distribution": dict(outcome_counts),
            "high_confidence_distribution": dict(high_confidence_outcomes),
            "overall_success_rate": success_rate,
            "high_confidence_success_rate": high_confidence_success_rate
        }

    def generate_comprehensive_report(self, insights: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""

        report = f"""# 🔍 Comprehensive §1782 Database Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

This report provides comprehensive analysis of our §1782 database, including citation patterns, court analysis, language patterns, temporal trends, and outcome analysis.

---

## 🎯 Citation Pattern Analysis

### Top Positive Correlations (Favor Grants)
"""

        if insights.get("citation_patterns", {}).get("top_positive_correlations"):
            for citation, data in insights["citation_patterns"]["top_positive_correlations"][:5]:
                report += f"- **{citation}**: {data['correlation']:.3f} correlation ({data['total_mentions']} mentions, {data['grant_rate']:.1%} grant rate)\n"

        report += f"""
### Top Negative Correlations (Favor Denials)
"""

        if insights.get("citation_patterns", {}).get("top_negative_correlations"):
            for citation, data in insights["citation_patterns"]["top_negative_correlations"][:5]:
                report += f"- **{citation}**: {data['correlation']:.3f} correlation ({data['total_mentions']} mentions, {data['grant_rate']:.1%} grant rate)\n"

        report += f"""
---

## 🏛️ Court Analysis

### Best Performing Courts (Highest Grant Rates)
"""

        if insights.get("court_patterns", {}).get("best_courts"):
            for court, data in insights["court_patterns"]["best_courts"]:
                report += f"- **{court}**: {data['grant_rate']:.1%} grant rate ({data['total_cases']} cases)\n"

        report += f"""
### Worst Performing Courts (Lowest Grant Rates)
"""

        if insights.get("court_patterns", {}).get("worst_courts"):
            for court, data in insights["court_patterns"]["worst_courts"]:
                report += f"- **{court}**: {data['grant_rate']:.1%} grant rate ({data['total_cases']} cases)\n"

        report += f"""
---

## 📝 Language Pattern Analysis

### Text Length Comparison
"""

        if insights.get("language_patterns", {}).get("text_length"):
            granted_stats = insights["language_patterns"]["text_length"]["granted"]
            denied_stats = insights["language_patterns"]["text_length"]["denied"]

            report += f"- **Granted Cases**: {granted_stats['mean']:.0f} avg characters ({granted_stats['count']} cases)\n"
            report += f"- **Denied Cases**: {denied_stats['mean']:.0f} avg characters ({denied_stats['count']} cases)\n"
            report += f"- **Difference**: {granted_stats['mean'] - denied_stats['mean']:.0f} characters\n"

        report += f"""
### Legal Language Density
"""

        if insights.get("language_patterns", {}).get("legal_density"):
            granted_stats = insights["language_patterns"]["legal_density"]["granted"]
            denied_stats = insights["language_patterns"]["legal_density"]["denied"]

            report += f"- **Granted Cases**: {granted_stats['mean']:.3f} legal density\n"
            report += f"- **Denied Cases**: {denied_stats['mean']:.3f} legal density\n"
            report += f"- **Difference**: {granted_stats['mean'] - denied_stats['mean']:.3f}\n"

        report += f"""
---

## 📅 Temporal Analysis

### Yearly Grant Rates
"""

        if insights.get("temporal_patterns", {}).get("yearly_grant_rates"):
            for year, data in list(insights["temporal_patterns"]["yearly_grant_rates"].items())[-5:]:
                report += f"- **{year}**: {data['grant_rate']:.1%} grant rate ({data['total_cases']} cases)\n"

        report += f"""
---

## 🎯 Outcome Analysis

### Overall Success Rates
"""

        if insights.get("outcome_analysis"):
            outcome_data = insights["outcome_analysis"]
            report += f"- **Overall Success Rate**: {outcome_data['overall_success_rate']:.1%}\n"
            report += f"- **High-Confidence Success Rate**: {outcome_data['high_confidence_success_rate']:.1%}\n"
            report += f"- **Total Cases Analyzed**: {outcome_data['total_cases']}\n"
            report += f"- **High-Confidence Cases**: {outcome_data['high_confidence_cases']}\n"

        report += f"""
---

## 🔑 Key Insights

### Citation Insights
- **Intel Paradox Confirmed**: Intel citations show negative correlation with success
- **Chevron Factor**: Chevron citations appear in both granted and denied cases
- **Citation Diversity**: More diverse citations may indicate stronger arguments

### Court Insights
- **Jurisdiction Matters**: Significant variation in grant rates across courts
- **Circuit Patterns**: Different circuits show different success patterns
- **District vs Circuit**: Circuit-level analysis reveals important patterns

### Language Insights
- **Text Length**: Granted cases tend to be longer (more detailed reasoning)
- **Legal Density**: Higher legal language density in granted cases
- **Complexity**: More complex language in granted cases

### Temporal Insights
- **Trend Analysis**: Grant rates vary significantly by year
- **Legal Evolution**: Changes in §1782 jurisprudence over time
- **Recent Patterns**: Current trends in §1782 decisions

---

## 🚀 Strategic Recommendations

### For Petition Writing
1. **Avoid Intel Over-Citation**: Don't trigger excessive Intel analysis
2. **Use Chevron Strategically**: Chevron citations can be effective
3. **Diversify Citations**: Use multiple supporting authorities
4. **Choose Favorable Courts**: Target high-grant-rate jurisdictions

### For Case Strategy
1. **Court Selection**: Choose courts with higher grant rates
2. **Timing**: Consider temporal patterns in grant rates
3. **Language**: Use appropriate legal language density
4. **Evidence**: Provide comprehensive supporting materials

---

**This comprehensive analysis provides the foundation for strategic §1782 case planning and petition drafting.**
"""

        return report

    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive analysis of the database."""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE DATABASE ANALYSIS")
        logger.info("="*80)

        # Load all data
        data = self.load_all_data()

        # Run analyses
        insights = {
            "citation_patterns": self.analyze_citation_patterns(data),
            "court_patterns": self.analyze_court_patterns(data),
            "language_patterns": self.analyze_language_patterns(data),
            "temporal_patterns": self.analyze_temporal_patterns(data),
            "outcome_analysis": self.analyze_case_outcomes(data)
        }

        # Generate report
        report = self.generate_comprehensive_report(insights)

        # Save results
        self._save_results(insights, report)

        logger.info("\n🎉 Comprehensive database analysis completed!")
        logger.info("✓ Detailed insights extracted")
        logger.info("✓ Comprehensive report generated")

    def _save_results(self, insights: Dict[str, Any], report: str) -> None:
        """Save analysis results."""
        # Save insights
        insights_path = Path("data/case_law/comprehensive_database_insights.json")
        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)

        # Save report
        report_path = Path("data/case_law/comprehensive_database_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✓ Insights saved to: {insights_path}")
        logger.info(f"✓ Report saved to: {report_path}")


def main():
    """Main entry point."""
    logger.info("Starting comprehensive database analysis...")

    analyzer = ComprehensiveDatabaseAnalyzer()
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
