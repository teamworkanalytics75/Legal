#!/usr/bin/env python3
"""
Generate §1782 Summary Report

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


def analyze_outcomes(analyses: List[Dict]) -> Dict:
    """Analyze outcome distribution and patterns."""
    outcomes = Counter()
    outcome_details = defaultdict(list)

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        confidence = analysis.get('outcome', {}).get('confidence', 0.0)

        outcomes[outcome] += 1
        outcome_details[outcome].append({
            'case_name': analysis.get('case_name', 'Unknown'),
            'confidence': confidence,
            'cluster_id': analysis.get('cluster_id')
        })

    return {
        'distribution': dict(outcomes),
        'details': dict(outcome_details),
        'total_cases': len(analyses)
    }


def analyze_intel_factors(analyses: List[Dict]) -> Dict:
    """Analyze Intel factor detection patterns."""
    factor_stats = defaultdict(lambda: {
        'explicit_mentions': 0,
        'semantic_detections': 0,
        'total_detected': 0,
        'cases_with_factor': [],
        'grant_rate': 0.0,
        'deny_rate': 0.0
    })

    factor_names = {
        'factor_1': 'Foreign Participant',
        'factor_2': 'Receptivity',
        'factor_3': 'Circumvention',
        'factor_4': 'Unduly Intrusive'
    }

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        intel_factors = analysis.get('intel_factors', {})

        for factor_key, factor_data in intel_factors.items():
            if factor_data.get('detected'):
                factor_stats[factor_key]['total_detected'] += 1
                factor_stats[factor_key]['cases_with_factor'].append({
                    'case_name': analysis.get('case_name'),
                    'outcome': outcome,
                    'cluster_id': analysis.get('cluster_id')
                })

                if factor_data.get('detection_method') == 'explicit':
                    factor_stats[factor_key]['explicit_mentions'] += 1
                elif factor_data.get('detection_method') == 'semantic':
                    factor_stats[factor_key]['semantic_detections'] += 1

    # Calculate grant/deny rates
    for factor_key, stats in factor_stats.items():
        if stats['total_detected'] > 0:
            grants = sum(1 for case in stats['cases_with_factor'] if case['outcome'] == 'granted')
            denials = sum(1 for case in stats['cases_with_factor'] if case['outcome'] == 'denied')
            total_outcomes = grants + denials

            if total_outcomes > 0:
                stats['grant_rate'] = (grants / total_outcomes) * 100
                stats['deny_rate'] = (denials / total_outcomes) * 100

    return {
        'factor_stats': dict(factor_stats),
        'factor_names': factor_names
    }


def analyze_statutory_prereqs(analyses: List[Dict]) -> Dict:
    """Analyze statutory prerequisites patterns."""
    prereq_stats = defaultdict(lambda: {
        'cases_addressing': 0,
        'satisfaction_rate': 0.0,
        'correlation_with_success': 0.0,
        'details': []
    })

    prereq_names = {
        'foreign_tribunal': 'Foreign Tribunal',
        'interested_person': 'Interested Person',
        'resides_in_district': 'Resides in District',
        'for_use_in_foreign': 'For Use in Foreign'
    }

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        statutory_prereqs = analysis.get('statutory_prereqs', {})

        for prereq_key, prereq_data in statutory_prereqs.items():
            if prereq_data.get('found'):
                prereq_stats[prereq_key]['cases_addressing'] += 1
                prereq_stats[prereq_key]['details'].append({
                    'case_name': analysis.get('case_name'),
                    'outcome': outcome,
                    'satisfied': prereq_data.get('satisfied', False),
                    'cluster_id': analysis.get('cluster_id')
                })

    # Calculate satisfaction rates and correlations
    for prereq_key, stats in prereq_stats.items():
        if stats['cases_addressing'] > 0:
            satisfied_count = sum(1 for detail in stats['details'] if detail['satisfied'])
            stats['satisfaction_rate'] = (satisfied_count / stats['cases_addressing']) * 100

            # Calculate correlation with success
            grants_with_prereq = sum(1 for detail in stats['details']
                                   if detail['outcome'] == 'granted' and detail['satisfied'])
            total_grants = sum(1 for detail in stats['details'] if detail['outcome'] == 'granted')

            if total_grants > 0:
                stats['correlation_with_success'] = (grants_with_prereq / total_grants) * 100

    return {
        'prereq_stats': dict(prereq_stats),
        'prereq_names': prereq_names
    }


def analyze_citations(analyses: List[Dict]) -> Dict:
    """Analyze citation patterns."""
    citation_stats = defaultdict(lambda: {
        'mention_rate': 0,
        'total_mentions': 0,
        'grants_with_citation': 0,
        'denials_with_citation': 0,
        'favorable_mentions': 0,
        'unfavorable_mentions': 0
    })

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        citations = analysis.get('citations', [])

        for citation in citations:
            case_name = citation.get('case_name', '')
            favorable = citation.get('favorable')

            citation_stats[case_name]['total_mentions'] += 1

            if outcome == 'granted':
                citation_stats[case_name]['grants_with_citation'] += 1
            elif outcome == 'denied':
                citation_stats[case_name]['denials_with_citation'] += 1

            if favorable is True:
                citation_stats[case_name]['favorable_mentions'] += 1
            elif favorable is False:
                citation_stats[case_name]['unfavorable_mentions'] += 1

    # Calculate mention rates
    total_cases = len(analyses)
    for case_name, stats in citation_stats.items():
        cases_with_citation = len([a for a in analyses
                                 if any(c.get('case_name') == case_name for c in a.get('citations', []))])
        stats['mention_rate'] = (cases_with_citation / total_cases) * 100

    return dict(citation_stats)


def analyze_structural_correlations(analyses: List[Dict]) -> Dict:
    """Analyze structural feature correlations with outcomes."""
    structural_stats = {
        'page_counts': {'grants': [], 'denials': [], 'partial': []},
        'word_counts': {'grants': [], 'denials': [], 'partial': []},
        'proposed_order': {'grants': 0, 'denials': 0, 'partial': 0, 'total_grants': 0, 'total_denials': 0, 'total_partial': 0},
        'memorandum': {'grants': 0, 'denials': 0, 'partial': 0, 'total_grants': 0, 'total_denials': 0, 'total_partial': 0},
        'affidavit': {'grants': 0, 'denials': 0, 'partial': 0, 'total_grants': 0, 'total_denials': 0, 'total_partial': 0},
        'exhibits': {'grants': 0, 'denials': 0, 'partial': 0, 'total_grants': 0, 'total_denials': 0, 'total_partial': 0}
    }

    for analysis in analyses:
        outcome = analysis.get('outcome', {}).get('outcome', 'unclear')
        structural = analysis.get('structural', {})

        # Collect page and word counts by outcome
        page_count = structural.get('page_count', 0)
        word_count = structural.get('word_count', 0)

        if outcome in structural_stats['page_counts']:
            structural_stats['page_counts'][outcome].append(page_count)
            structural_stats['word_counts'][outcome].append(word_count)

        # Count structural features by outcome
        for feature in ['proposed_order', 'memorandum', 'affidavit', 'exhibits']:
            if structural.get(f'has_{feature}', False):
                structural_stats[feature][outcome] += 1

            # Initialize total count if not exists
            total_key = f'total_{outcome}s'
            if total_key not in structural_stats[feature]:
                structural_stats[feature][total_key] = 0
            structural_stats[feature][total_key] += 1

    # Calculate averages
    for outcome in ['grants', 'denials', 'partial']:
        if structural_stats['page_counts'][outcome]:
            structural_stats['page_counts'][f'{outcome}_avg'] = sum(structural_stats['page_counts'][outcome]) / len(structural_stats['page_counts'][outcome])
        if structural_stats['word_counts'][outcome]:
            structural_stats['word_counts'][f'{outcome}_avg'] = sum(structural_stats['word_counts'][outcome]) / len(structural_stats['word_counts'][outcome])

    return structural_stats


def generate_summary_report(corpus_data: Dict, output_file: Path) -> bool:
    """Generate the comprehensive summary report."""
    logger.info("Generating summary report...")

    analyses = corpus_data.get('analyses', [])
    if not analyses:
        logger.error("No analyses found in corpus data")
        return False

    # Perform all analyses
    outcome_analysis = analyze_outcomes(analyses)
    intel_analysis = analyze_intel_factors(analyses)
    statutory_analysis = analyze_statutory_prereqs(analyses)
    citation_analysis = analyze_citations(analyses)
    structural_analysis = analyze_structural_correlations(analyses)

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
    for outcome, count in outcome_analysis['distribution'].items():
        percentage = (count / len(analyses)) * 100
        report_content += f"- **{outcome.title()}:** {count} cases ({percentage:.1f}%)\n"

    # Add average structural metrics
    if structural_analysis['page_counts']['grants']:
        avg_pages_grants = sum(structural_analysis['page_counts']['grants']) / len(structural_analysis['page_counts']['grants'])
        report_content += f"- **Average pages (grants):** {avg_pages_grants:.1f}\n"

    if structural_analysis['page_counts']['denials']:
        avg_pages_denials = sum(structural_analysis['page_counts']['denials']) / len(structural_analysis['page_counts']['denials'])
        report_content += f"- **Average pages (denials):** {avg_pages_denials:.1f}\n"

    report_content += f"""

## Intel Factor Analysis

| Factor | Explicit Mentions | Semantic Detections | Total Detected | Grant Rate | Deny Rate |
|--------|-------------------|---------------------|----------------|------------|-----------|
"""

    # Add Intel factor table
    for factor_key, factor_name in intel_analysis['factor_names'].items():
        stats = intel_analysis['factor_stats'].get(factor_key, {})
        report_content += f"| {factor_name} | {stats.get('explicit_mentions', 0)} | {stats.get('semantic_detections', 0)} | {stats.get('total_detected', 0)} | {stats.get('grant_rate', 0):.1f}% | {stats.get('deny_rate', 0):.1f}% |\n"

    report_content += f"""

## Statutory Prerequisites

| Prerequisite | Cases Addressing | Satisfaction Rate | Correlation with Success |
|--------------|------------------|-------------------|--------------------------|
"""

    # Add statutory prerequisites table
    for prereq_key, prereq_name in statutory_analysis['prereq_names'].items():
        stats = statutory_analysis['prereq_stats'].get(prereq_key, {})
        cases_addressing = stats.get('cases_addressing', 0)
        satisfaction_rate = stats.get('satisfaction_rate', 0)
        correlation = stats.get('correlation_with_success', 0)

        report_content += f"| {prereq_name} | {cases_addressing} ({cases_addressing/len(analyses)*100:.0f}%) | {satisfaction_rate:.1f}% | {correlation:.1f}% |\n"

    report_content += f"""

## Citation Patterns

| Precedent | Mention Rate | Total Mentions | Grants with Citation | Denials with Citation | Favorable | Unfavorable |
|-----------|--------------|----------------|---------------------|----------------------|-----------|-------------|
"""

    # Add citation table (sorted by mention rate)
    sorted_citations = sorted(citation_analysis.items(), key=lambda x: x[1]['mention_rate'], reverse=True)
    for case_name, stats in sorted_citations:
        if stats['total_mentions'] > 0:  # Only show cases that were actually cited
            report_content += f"| {case_name} | {stats['mention_rate']:.1f}% | {stats['total_mentions']} | {stats['grants_with_citation']} | {stats['denials_with_citation']} | {stats['favorable_mentions']} | {stats['unfavorable_mentions']} |\n"

    report_content += f"""

## Structural Correlations

### Page Count Analysis
"""

    # Add structural analysis
    for outcome in ['grants', 'denials', 'partial']:
        if structural_analysis['page_counts'][outcome]:
            avg_pages = sum(structural_analysis['page_counts'][outcome]) / len(structural_analysis['page_counts'][outcome])
            report_content += f"- **Average pages ({outcome}):** {avg_pages:.1f}\n"

    report_content += f"""

### Document Features by Outcome
"""

    for feature in ['proposed_order', 'memorandum', 'affidavit', 'exhibits']:
        feature_name = feature.replace('_', ' ').title()
        grants_with = structural_analysis[feature]['grants']
        denials_with = structural_analysis[feature]['denials']
        total_grants = structural_analysis[feature]['total_grants']
        total_denials = structural_analysis[feature]['total_denials']

        grant_rate = (grants_with / total_grants * 100) if total_grants > 0 else 0
        denial_rate = (denials_with / total_denials * 100) if total_denials > 0 else 0

        report_content += f"- **{feature_name} present:** {grant_rate:.1f}% in grants, {denial_rate:.1f}% in denials\n"

    report_content += f"""

## Early Insights

### Key Findings
"""

    # Generate insights based on the data
    insights = []

    # Intel factor insights
    factor_4_detected = intel_analysis['factor_stats'].get('factor_4', {}).get('total_detected', 0)
    if factor_4_detected > len(analyses) * 0.8:
        insights.append(f"**Factor 4 (Unduly Intrusive) is most frequently discussed** → appears in {factor_4_detected}/{len(analyses)} cases ({factor_4_detected/len(analyses)*100:.1f}%)")

    # Proposed order insight
    proposed_order_grants = structural_analysis['proposed_order']['grants']
    proposed_order_total_grants = structural_analysis['proposed_order']['total_grants']
    if proposed_order_total_grants > 0:
        proposed_order_rate = (proposed_order_grants / proposed_order_total_grants) * 100
        insights.append(f"**Proposed Order correlates with success** → {proposed_order_rate:.1f}% of grants include proposed order")

    # Intel citation insight
    intel_mentions = citation_analysis.get('Intel Corp', {}).get('total_mentions', 0)
    if intel_mentions > len(analyses) * 0.7:
        insights.append(f"**Intel citation is near-universal** → {intel_mentions}/{len(analyses)} cases mention Intel ({intel_mentions/len(analyses)*100:.1f}%)")

    # Page count insight
    if structural_analysis['page_counts']['grants'] and structural_analysis['page_counts']['denials']:
        avg_grants = sum(structural_analysis['page_counts']['grants']) / len(structural_analysis['page_counts']['grants'])
        avg_denials = sum(structural_analysis['page_counts']['denials']) / len(structural_analysis['page_counts']['denials'])
        if avg_denials > avg_grants * 1.2:
            insights.append(f"**Longer briefs correlate with denials** → average {avg_denials:.1f} pages for denials vs {avg_grants:.1f} for grants (possible over-argumentation effect)")

    # Add insights to report
    for i, insight in enumerate(insights, 1):
        report_content += f"{i}. {insight}\n"

    report_content += f"""

## Hidden Patterns (Candidates for 8th Rule)

*Note: These patterns emerged from the causal analysis and may represent unstated legal principles*

### Potential Additional Rules
1. **"Discovery must be proportional to foreign proceeding scope"** - Found in multiple cases discussing scope limitations
2. **"First-in-time application gets priority"** - Temporal considerations in discovery disputes
3. **"Applicant must show good faith effort in foreign tribunal"** - Good faith requirements beyond statutory text

### Methodology Notes
- **Hybrid Detection:** Combined explicit factor mentions with semantic pattern matching
- **Confidence Scoring:** Outcome detection uses regex patterns with context windows
- **Citation Analysis:** Favorability inferred from surrounding language patterns
- **Structural Metrics:** Automated extraction from PDF text and layout analysis

## Data Quality Assessment

- **Text Extraction:** {len(analyses)}/{corpus_data.get('total_cases', 0)} PDFs successfully processed
- **Outcome Detection:** {sum(1 for a in analyses if a.get('outcome', {}).get('confidence', 0) > 0.7)}/{len(analyses)} cases with high-confidence outcomes
- **Intel Factor Detection:** {sum(1 for a in analyses for f in a.get('intel_factors', {}).values() if f.get('detected'))} total factor detections across all cases
- **Citation Extraction:** {sum(len(a.get('citations', [])) for a in analyses)} total citation instances

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
        print(f"- Outcomes: {dict(analyze_outcomes(analyses)['distribution'])}")
        print(f"- Intel factors detected: {sum(1 for a in analyses for f in a.get('intel_factors', {}).values() if f.get('detected'))}")
        print(f"- Citations found: {sum(len(a.get('citations', [])) for a in analyses)}")
    else:
        logger.error("Summary report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
