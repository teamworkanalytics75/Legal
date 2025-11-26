#!/usr/bin/env python3
"""
Combined NLP Analysis Report

This script combines existing PDF analysis with new text extraction analysis
to provide comprehensive insights into the §1782 caselaw corpus.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_existing_pdf_results() -> Dict[str, Any]:
    """Analyze the existing PDF analysis results."""
    logger.info("Analyzing existing PDF analysis results...")

    pdf_analysis_path = Path("data/case_law/simple_pdf_analysis.json")
    if not pdf_analysis_path.exists():
        logger.warning("No existing PDF analysis found")
        return {}

    with open(pdf_analysis_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)

    analysis = {
        'total_pdfs_analyzed': len(pdf_data),
        'outcomes': Counter(),
        'intel_factors': defaultdict(int),
        'text_lengths': [],
        'section_1782_mentions': [],
        'intel_citations': [],
        'notable_patterns': {}
    }

    for case in pdf_data:
        # Outcomes
        analysis['outcomes'][case['outcome']['outcome']] += 1

        # Intel factors
        for factor, value in case['intel_factors'].items():
            if value:
                analysis['intel_factors'][factor] += 1

        # Text metrics
        analysis['text_lengths'].append(case['text_length'])
        analysis['section_1782_mentions'].append(case['section_1782_mentions'])

        # Intel citations
        for citation in case['citations']:
            if 'Intel' in citation['pattern']:
                analysis['intel_citations'].append({
                    'case': case['case_name'],
                    'count': citation['count']
                })

    # Calculate averages
    analysis['avg_text_length'] = sum(analysis['text_lengths']) / len(analysis['text_lengths'])
    analysis['avg_section_1782_mentions'] = sum(analysis['section_1782_mentions']) / len(analysis['section_1782_mentions'])

    # Identify patterns
    analysis['notable_patterns'] = {
        'most_common_outcome': analysis['outcomes'].most_common(1)[0] if analysis['outcomes'] else None,
        'most_common_intel_factor': max(analysis['intel_factors'].items(), key=lambda x: x[1]) if analysis['intel_factors'] else None,
        'longest_case': max(pdf_data, key=lambda x: x['text_length'])['case_name'],
        'most_section_1782_mentions': max(pdf_data, key=lambda x: x['section_1782_mentions'])['case_name']
    }

    return analysis


def analyze_new_text_extraction() -> Dict[str, Any]:
    """Analyze the new text extraction results."""
    logger.info("Analyzing new text extraction results...")

    extraction_log_path = Path("data/case_law/text_extraction_log.json")
    if not extraction_log_path.exists():
        logger.warning("No text extraction log found")
        return {}

    with open(extraction_log_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)

    analysis = {
        'total_cases_processed': extraction_data['total_cases'],
        'successful_extractions': extraction_data['cases_with_text'],
        'pdfs_downloaded': extraction_data['cases_with_pdfs'],
        'failed_extractions': extraction_data['cases_failed'],
        'success_rate': extraction_data['cases_with_text'] / extraction_data['total_cases'] * 100,
        'text_lengths': [],
        'case_types': Counter()
    }

    # Analyze individual cases
    for case in extraction_data['cases']:
        if case['status'] == 'success':
            analysis['text_lengths'].append(case['text_length'])

            # Categorize case types
            file_name = case['file']
            if file_name.startswith('cap_'):
                analysis['case_types']['CAP'] += 1
            elif file_name.startswith('landmark_'):
                analysis['case_types']['Landmark'] += 1
            elif file_name.startswith('search_'):
                analysis['case_types']['Search'] += 1
            elif file_name.startswith('case_'):
                analysis['case_types']['Case'] += 1
            else:
                analysis['case_types']['CourtListener'] += 1

    if analysis['text_lengths']:
        analysis['avg_text_length'] = sum(analysis['text_lengths']) / len(analysis['text_lengths'])
        analysis['total_text_extracted'] = sum(analysis['text_lengths'])
    else:
        analysis['avg_text_length'] = 0
        analysis['total_text_extracted'] = 0

    return analysis


def generate_combined_report(pdf_analysis: Dict, text_analysis: Dict) -> str:
    """Generate a comprehensive combined analysis report."""

    report_content = f"""# 🧠 Combined NLP Analysis Report for §1782 Caselaw Corpus

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary

This report combines analysis from two phases:
1. **Phase 1**: Deep analysis of 13 PDFs with detailed legal pattern extraction
2. **Phase 2**: Text extraction and analysis of 82 cases from CourtListener

## 🔍 Phase 1: PDF Analysis Results (13 Cases)

### Key Findings from PDF Analysis

| Metric | Value |
|--------|-------|
| **Total PDFs Analyzed** | {pdf_analysis.get('total_pdfs_analyzed', 0)} |
| **Average Text Length** | {pdf_analysis.get('avg_text_length', 0):,.0f} characters |
| **Average §1782 Mentions** | {pdf_analysis.get('avg_section_1782_mentions', 0):.1f} per case |

### Case Outcomes Distribution
"""

    if pdf_analysis.get('outcomes'):
        for outcome, count in pdf_analysis['outcomes'].most_common():
            percentage = count / pdf_analysis['total_pdfs_analyzed'] * 100
            report_content += f"- **{outcome.title()}**: {count} cases ({percentage:.1f}%)\n"

    report_content += f"""
### Intel Factors Analysis
"""

    if pdf_analysis.get('intel_factors'):
        for factor, count in pdf_analysis['intel_factors'].items():
            percentage = count / pdf_analysis['total_pdfs_analyzed'] * 100
            report_content += f"- **{factor.replace('_', ' ').title()}**: {count} cases ({percentage:.1f}%)\n"

    report_content += f"""
### Notable Patterns from PDF Analysis
"""

    if pdf_analysis.get('notable_patterns'):
        patterns = pdf_analysis['notable_patterns']
        if patterns.get('most_common_outcome'):
            outcome, count = patterns['most_common_outcome']
            report_content += f"- **Most Common Outcome**: {outcome.title()} ({count} cases)\n"

        if patterns.get('most_common_intel_factor'):
            factor, count = patterns['most_common_intel_factor']
            report_content += f"- **Most Common Intel Factor**: {factor.replace('_', ' ').title()} ({count} cases)\n"

        if patterns.get('longest_case'):
            report_content += f"- **Longest Case**: {patterns['longest_case']}\n"

        if patterns.get('most_section_1782_mentions'):
            report_content += f"- **Most §1782 References**: {patterns['most_section_1782_mentions']}\n"

    report_content += f"""
## 📄 Phase 2: Text Extraction Results (82 Cases)

### Text Extraction Summary

| Metric | Value |
|--------|-------|
| **Total Cases Processed** | {text_analysis.get('total_cases_processed', 0)} |
| **Successful Extractions** | {text_analysis.get('successful_extractions', 0)} |
| **PDFs Downloaded** | {text_analysis.get('pdfs_downloaded', 0)} |
| **Success Rate** | {text_analysis.get('success_rate', 0):.1f}% |
| **Total Text Extracted** | {text_analysis.get('total_text_extracted', 0):,} characters |
| **Average Text Length** | {text_analysis.get('avg_text_length', 0):,.0f} characters |

### Case Type Distribution
"""

    if text_analysis.get('case_types'):
        for case_type, count in text_analysis['case_types'].most_common():
            percentage = count / text_analysis['successful_extractions'] * 100
            report_content += f"- **{case_type}**: {count} cases ({percentage:.1f}%)\n"

    report_content += f"""
## 🔍 Combined Insights

### Text Volume Analysis
- **Phase 1 (PDFs)**: {pdf_analysis.get('total_pdfs_analyzed', 0)} cases with detailed legal analysis
- **Phase 2 (Text)**: {text_analysis.get('successful_extractions', 0)} cases with extracted text
- **Total Text Volume**: {text_analysis.get('total_text_extracted', 0) + sum(pdf_analysis.get('text_lengths', [])):,} characters

### Legal Pattern Observations
"""

    # Combine insights from both phases
    if pdf_analysis.get('outcomes') and text_analysis.get('successful_extractions'):
        total_analyzed = pdf_analysis['total_pdfs_analyzed'] + text_analysis['successful_extractions']
        report_content += f"- **Total Cases with Text Analysis**: {total_analyzed}\n"

        if pdf_analysis.get('outcomes'):
            most_common_outcome = pdf_analysis['outcomes'].most_common(1)[0]
            report_content += f"- **Most Common Outcome Pattern**: {most_common_outcome[0].title()} ({most_common_outcome[1]} cases)\n"

    report_content += f"""
## 🎯 Key Discoveries

### 1. Legal Outcome Patterns
"""

    if pdf_analysis.get('outcomes'):
        outcomes = pdf_analysis['outcomes']
        if 'denied' in outcomes and 'granted' in outcomes:
            denied_pct = outcomes['denied'] / pdf_analysis['total_pdfs_analyzed'] * 100
            granted_pct = outcomes['granted'] / pdf_analysis['total_pdfs_analyzed'] * 100
            report_content += f"- **Denial Rate**: {denied_pct:.1f}% of analyzed cases\n"
            report_content += f"- **Grant Rate**: {granted_pct:.1f}% of analyzed cases\n"

    report_content += f"""
### 2. Intel Factor Analysis
"""

    if pdf_analysis.get('intel_factors'):
        factors = pdf_analysis['intel_factors']
        total_cases = pdf_analysis['total_pdfs_analyzed']

        for factor, count in factors.items():
            factor_name = factor.replace('_', ' ').title()
            percentage = count / total_cases * 100
            report_content += f"- **{factor_name}**: Present in {percentage:.1f}% of cases\n"

    report_content += f"""
### 3. Text Extraction Success
- **High Success Rate**: {text_analysis.get('success_rate', 0):.1f}% of cases successfully processed
- **Comprehensive Coverage**: {text_analysis.get('successful_extractions', 0)} cases now have searchable text
- **PDF Archive**: {text_analysis.get('pdfs_downloaded', 0)} PDFs available for offline analysis

### 4. Corpus Composition
"""

    if text_analysis.get('case_types'):
        case_types = text_analysis['case_types']
        total_cases = sum(case_types.values())

        for case_type, count in case_types.most_common():
            percentage = count / total_cases * 100
            report_content += f"- **{case_type} Cases**: {percentage:.1f}% of corpus\n"

    report_content += f"""
## 🚀 Recommendations for Next Steps

### Immediate Actions
1. **Pattern Validation**: Cross-reference PDF analysis patterns with text extraction results
2. **Outcome Prediction**: Use Intel factor analysis to predict case outcomes
3. **Text Mining**: Apply NLP techniques to the {text_analysis.get('total_text_extracted', 0):,} characters of extracted text

### Advanced Analysis
1. **Temporal Analysis**: Compare patterns across different time periods
2. **Court-Specific Analysis**: Analyze patterns by court jurisdiction
3. **Party Analysis**: Examine patterns by type of foreign entity
4. **Citation Network**: Build citation networks between cases

### Research Applications
1. **Predictive Modeling**: Use patterns to predict §1782 case outcomes
2. **Legal Research**: Enable full-text search across the corpus
3. **Policy Analysis**: Identify trends in judicial interpretation
4. **Comparative Study**: Compare §1782 patterns with other discovery statutes

## 📁 Data Sources

- **PDF Analysis**: `data/case_law/simple_pdf_analysis.json`
- **Text Extraction Log**: `data/case_law/text_extraction_log.json`
- **NLP Analysis Results**: `data/case_law/nlp_analysis_results/`
- **Extracted PDFs**: `data/case_law/1782_discovery/pdfs/`

---

**This combined analysis provides a comprehensive foundation for advanced legal research and pattern recognition in §1782 caselaw.**
"""

    return report_content


def main():
    """Main entry point."""
    logger.info("Generating combined NLP analysis report...")

    # Analyze existing PDF results
    pdf_analysis = analyze_existing_pdf_results()

    # Analyze new text extraction results
    text_analysis = analyze_new_text_extraction()

    # Generate combined report
    report_content = generate_combined_report(pdf_analysis, text_analysis)

    # Save report
    report_path = Path("data/case_law/combined_nlp_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"✓ Combined analysis report saved to: {report_path}")

    # Print key findings
    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS SUMMARY")
    logger.info("="*80)

    if pdf_analysis:
        logger.info(f"📊 PDF Analysis: {pdf_analysis['total_pdfs_analyzed']} cases analyzed")
        if pdf_analysis.get('outcomes'):
            most_common = pdf_analysis['outcomes'].most_common(1)[0]
            logger.info(f"   Most common outcome: {most_common[0]} ({most_common[1]} cases)")

    if text_analysis:
        logger.info(f"📄 Text Extraction: {text_analysis['successful_extractions']} cases with text")
        logger.info(f"   Success rate: {text_analysis['success_rate']:.1f}%")
        logger.info(f"   Total text: {text_analysis['total_text_extracted']:,} characters")

    logger.info("\n🎉 Combined analysis completed!")


if __name__ == "__main__":
    main()
