#!/usr/bin/env python3
"""
Generate §1782 Summary Report - Simplified Version

Creates a comprehensive markdown report with frequency analysis,
correlations, and early insights from the corpus analysis results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/summary_generation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_corpus_analysis(results_file: Path) -> Dict:
    """Load the combined corpus analysis results."""
    logger.info(f"Loading corpus analysis from: {results_file}")

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return {}

    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_summary_report(corpus_data: Dict, output_file: Path) -> bool:
    """Generate the comprehensive summary report."""
    logger.info("Generating summary report...")

    analyses = corpus_data.get('analyses', [])
    if not analyses:
        logger.error("No analyses found in corpus data")
        return False

    # Analyze outcomes
    outcomes = Counter()
    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        outcomes[outcome] += 1

    # Analyze Intel factors
    intel_stats = defaultdict(lambda: {'detected': 0, 'explicit': 0, 'semantic': 0})
    for analysis in analyses:
        intel_factors = analysis.get('intel_factors', {})
        for factor_name, factor_data in intel_factors.items():
            if factor_data.get('detected'):
                intel_stats[factor_name]['detected'] += 1
                if factor_data.get('detection_method') == 'explicit':
                    intel_stats[factor_name]['explicit'] += 1
                elif factor_data.get('detection_method') == 'semantic':
                    intel_stats[factor_name]['semantic'] += 1

    # Analyze citations
    citation_counts = Counter()
    for analysis in analyses:
        citations = analysis.get('citations', [])
        for citation in citations:
            case_name = citation.get('case_name', '')
            if case_name:
                citation_counts[case_name] += 1

    # Analyze structural features
    page_counts = {'grants': [], 'denials': [], 'partial': []}
    proposed_order_counts = {'grants': 0, 'denials': 0, 'partial': 0}
    total_by_outcome = {'grants': 0, 'denials': 0, 'partial': 0}

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        structural = analysis.get('structural', {})

        if outcome in page_counts:
            page_counts[outcome].append(structural.get('page_count', 0))
            total_by_outcome[outcome] += 1

            if structural.get('has_proposed_order', False):
                proposed_order_counts[outcome] += 1

    # Generate markdown report
    report_content = f"""# §1782 Corpus Analysis - Round 1 ({len(analyses)} Cases)

## Executive Summary
- **Total cases analyzed:** {len(analyses)}
- **Successful extractions:** {corpus_data.get('successful_analyses', 0)}
- **Failed extractions:** {corpus_data.get('failed_analyses', 0)}
- **Success rate:** {(corpus_data.get('successful_analyses', 0) / corpus_data.get('total_cases', 1)) * 100:.1f}%

### Outcome Distribution
"""

    # Add outcome distribution
    for outcome, count in outcomes.items():
        percentage = (count / len(analyses)) * 100
        report_content += f"- **{outcome.title()}:** {count} cases ({percentage:.1f}%)\n"

    # Add average page counts
    for outcome in ['grants', 'denials', 'partial']:
        if page_counts[outcome]:
            avg_pages = sum(page_counts[outcome]) / len(page_counts[outcome])
            report_content += f"- **Average pages ({outcome}):** {avg_pages:.1f}\n"

    report_content += f"""

## Intel Factor Analysis

| Factor | Explicit Mentions | Semantic Detections | Total Detected | Detection Rate |
|--------|-------------------|---------------------|----------------|----------------|
| Factor 1 (Foreign Participant) | {intel_stats['factor_1']['explicit']} | {intel_stats['factor_1']['semantic']} | {intel_stats['factor_1']['detected']} | {intel_stats['factor_1']['detected']/len(analyses)*100:.1f}% |
| Factor 2 (Receptivity) | {intel_stats['factor_2']['explicit']} | {intel_stats['factor_2']['semantic']} | {intel_stats['factor_2']['detected']} | {intel_stats['factor_2']['detected']/len(analyses)*100:.1f}% |
| Factor 3 (Circumvention) | {intel_stats['factor_3']['explicit']} | {intel_stats['factor_3']['semantic']} | {intel_stats['factor_3']['detected']} | {intel_stats['factor_3']['detected']/len(analyses)*100:.1f}% |
| Factor 4 (Unduly Intrusive) | {intel_stats['factor_4']['explicit']} | {intel_stats['factor_4']['semantic']} | {intel_stats['factor_4']['detected']} | {intel_stats['factor_4']['detected']/len(analyses)*100:.1f}% |

## Citation Patterns

| Precedent | Total Mentions | Mention Rate |
|-----------|----------------|--------------|
"""

    # Add citation table (sorted by mention count)
    sorted_citations = citation_counts.most_common()
    for case_name, count in sorted_citations:
        mention_rate = (count / len(analyses)) * 100
        report_content += f"| {case_name} | {count} | {mention_rate:.1f}% |\n"

    report_content += f"""

## Structural Correlations

### Page Count Analysis
"""

    # Add structural analysis
    for outcome in ['grants', 'denials', 'partial']:
        if page_counts[outcome]:
            avg_pages = sum(page_counts[outcome]) / len(page_counts[outcome])
            report_content += f"- **Average pages ({outcome}):** {avg_pages:.1f}\n"

    report_content += f"""

### Document Features by Outcome
"""

    for outcome in ['grants', 'denials', 'partial']:
        if total_by_outcome[outcome] > 0:
            proposed_order_rate = (proposed_order_counts[outcome] / total_by_outcome[outcome]) * 100
            report_content += f"- **Proposed Order present ({outcome}):** {proposed_order_rate:.1f}%\n"

    report_content += f"""

## Early Insights

### Key Findings
"""

    # Generate insights based on the data
    insights = []

    # Intel factor insights
    factor_4_detected = intel_stats['factor_4']['detected']
    if factor_4_detected > len(analyses) * 0.7:
        insights.append(f"**Factor 4 (Unduly Intrusive) is most frequently discussed** → appears in {factor_4_detected}/{len(analyses)} cases ({factor_4_detected/len(analyses)*100:.1f}%)")

    # Intel citation insight
    intel_mentions = citation_counts.get('Intel Corp', 0)
    if intel_mentions > len(analyses) * 0.7:
        insights.append(f"**Intel citation is near-universal** → {intel_mentions}/{len(analyses)} cases mention Intel ({intel_mentions/len(analyses)*100:.1f}%)")

    # Page count insight
    if page_counts['grants'] and page_counts['denials']:
        avg_grants = sum(page_counts['grants']) / len(page_counts['grants'])
        avg_denials = sum(page_counts['denials']) / len(page_counts['denials'])
        if avg_denials > avg_grants * 1.1:
            insights.append(f"**Longer briefs correlate with denials** → average {avg_denials:.1f} pages for denials vs {avg_grants:.1f} for grants")

    # Proposed order insight
    total_grants = total_by_outcome['grants']
    if total_grants > 0:
        proposed_order_grants = proposed_order_counts['grants']
        proposed_order_rate = (proposed_order_grants / total_grants) * 100
        if proposed_order_rate > 50:
            insights.append(f"**Proposed Order correlates with success** → {proposed_order_rate:.1f}% of grants include proposed order")

    # Add insights to report
    for i, insight in enumerate(insights, 1):
        report_content += f"{i}. {insight}\n"

    report_content += f"""

## Data Quality Assessment

- **Text Extraction:** {len(analyses)}/{corpus_data.get('total_cases', 0)} PDFs successfully processed
- **Outcome Detection:** {sum(1 for a in analyses if a.get('outcome', {}).get('confidence', 0) > 0.7)}/{len(analyses)} cases with high-confidence outcomes
- **Intel Factor Detection:** {sum(intel_stats[f]['detected'] for f in ['factor_1', 'factor_2', 'factor_3', 'factor_4'])} total factor detections across all cases
- **Citation Extraction:** {sum(citation_counts.values())} total citation instances

## Methodology Notes

- **Hybrid Detection:** Combined explicit factor mentions with semantic pattern matching
- **Confidence Scoring:** Outcome detection uses regex patterns with context windows
- **Citation Analysis:** Favorability inferred from surrounding language patterns
- **Structural Metrics:** Automated extraction from PDF text and layout analysis

---
*Report generated from corpus analysis of §1782 discovery cases*
*Analysis timestamp: {corpus_data.get('processing_timestamp', 'Unknown')}*
"""

    # Write report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"Summary report saved to: {output_file}")
    return True


def main():
    """Main execution function."""
    print("Generating §1782 Summary Report...")

    # Define paths
    results_file = Path("data/case_law/corpus_analysis_results/combined_corpus_analysis.json")
    output_file = Path("data/case_law/summary_report_round1.md")

    # Load corpus data
    corpus_data = load_corpus_analysis(results_file)
    if not corpus_data:
        logger.error("Failed to load corpus analysis data")
        sys.exit(1)

    # Generate report
    if generate_summary_report(corpus_data, output_file):
        logger.info("Summary report generation completed successfully")
        print(f"\n[SUCCESS] Summary report generated: {output_file}")

        # Print quick stats
        analyses = corpus_data.get('analyses', [])
        print(f"\nQuick Stats:")
        print(f"- Cases analyzed: {len(analyses)}")

        # Count outcomes
        outcomes = {}
        for analysis in analyses:
            outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        print(f"- Outcomes: {outcomes}")

        # Count Intel factors
        intel_count = 0
        for analysis in analyses:
            intel_factors = analysis.get('intel_factors', {})
            intel_count += sum(1 for f in intel_factors.values() if f.get('detected'))
        print(f"- Intel factors detected: {intel_count}")

        # Count citations
        citation_count = sum(len(a.get('citations', [])) for a in analyses)
        print(f"- Citations found: {citation_count}")
    else:
        logger.error("Summary report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
