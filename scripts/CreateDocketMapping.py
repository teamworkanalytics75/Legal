#!/usr/bin/env python3
"""
Docket Mapping Script

Creates a CSV mapping opinions to their original dockets for petition retrieval.
Extracts docket_id, docketNumber, case names, outcomes, and court information.

Usage: python scripts/create_docket_mapping.py
"""

import json
import csv
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocketMapper:
    def __init__(self):
        self.corpus_dir = "data/case_law/1782_discovery"
        self.output_file = "data/case_law/docket_mapping.csv"
        self.mapping_data = []

    def load_all_cases(self) -> List[Dict[str, Any]]:
        """Load all case files from the corpus."""
        logger.info("Loading all case files...")

        cases = []
        json_files = [f for f in os.listdir(self.corpus_dir) if f.endswith('.json')]

        logger.info(f"Found {len(json_files)} JSON files")

        for filename in json_files:
            filepath = os.path.join(self.corpus_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    case_data['filename'] = filename
                    case_data['filepath'] = filepath
                    cases.append(case_data)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")

        logger.info(f"Successfully loaded {len(cases)} cases")
        return cases

    def extract_outcome(self, case: Dict[str, Any]) -> str:
        """Extract outcome from case data."""
        # Try to get outcome from various fields
        outcome_fields = ['outcome', 'court_outcome', 'result', 'decision']

        for field in outcome_fields:
            if field in case and case[field]:
                return str(case[field]).upper()

        # Try to extract from text if available
        text_fields = ['opinion_text', 'extracted_text', 'caseNameFull_text', 'attorney_text']
        all_text = ""

        for field in text_fields:
            if field in case and case[field]:
                all_text += " " + str(case[field])

        if all_text:
            all_text = all_text.lower()

            # Look for clear outcome indicators
            if any(phrase in all_text for phrase in ['motion granted', 'application granted', 'petition granted', 'court grants']):
                return 'GRANTED'
            elif any(phrase in all_text for phrase in ['motion denied', 'application denied', 'petition denied', 'court denies']):
                return 'DENIED'
            elif any(phrase in all_text for phrase in ['granted in part', 'denied in part']):
                return 'MIXED'
            elif any(phrase in all_text for phrase in ['affirmed', 'court affirms']):
                return 'AFFIRMED'
            elif any(phrase in all_text for phrase in ['reversed', 'court reverses']):
                return 'REVERSED'

        return 'UNCLEAR'

    def extract_docket_info(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Extract docket information from case."""
        docket_info = {
            'filename': case.get('filename', ''),
            'filepath': case.get('filepath', ''),
            'cluster_id': case.get('cluster_id'),
            'docket_id': case.get('docket_id'),
            'docket_number': case.get('docketNumber', ''),
            'case_name': case.get('caseName', ''),
            'case_name_full': case.get('caseNameFull', ''),
            'court': case.get('court', ''),
            'court_id': case.get('court_id', ''),
            'date_filed': case.get('dateFiled', ''),
            'outcome': self.extract_outcome(case),
            'has_opinion_text': bool(case.get('opinion_text')),
            'has_extracted_text': bool(case.get('extracted_text')),
            'text_length': len(str(case.get('opinion_text', '')) + str(case.get('extracted_text', ''))),
            'citation_count': case.get('citeCount', 0)
        }

        return docket_info

    def create_mapping(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create docket mapping from all cases."""
        logger.info("Creating docket mapping...")

        mapping_data = []

        for case in cases:
            docket_info = self.extract_docket_info(case)
            mapping_data.append(docket_info)

        logger.info(f"Created mapping for {len(mapping_data)} cases")
        return mapping_data

    def analyze_mapping_quality(self, mapping_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality of the docket mapping."""
        logger.info("Analyzing mapping quality...")

        analysis = {
            'total_cases': len(mapping_data),
            'cases_with_docket_id': len([c for c in mapping_data if c['docket_id']]),
            'cases_with_docket_number': len([c for c in mapping_data if c['docket_number']]),
            'cases_with_both': len([c for c in mapping_data if c['docket_id'] and c['docket_number']]),
            'cases_with_neither': len([c for c in mapping_data if not c['docket_id'] and not c['docket_number']]),
            'cases_with_text': len([c for c in mapping_data if c['text_length'] > 0]),
            'outcome_distribution': {},
            'court_distribution': {}
        }

        # Count outcomes
        for case in mapping_data:
            outcome = case['outcome']
            analysis['outcome_distribution'][outcome] = analysis['outcome_distribution'].get(outcome, 0) + 1

        # Count courts
        for case in mapping_data:
            court = case['court_id'] or 'unknown'
            analysis['court_distribution'][court] = analysis['court_distribution'].get(court, 0) + 1

        return analysis

    def save_mapping(self, mapping_data: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save the docket mapping to CSV and create analysis report."""
        logger.info("Saving docket mapping...")

        # Save CSV
        df = pd.DataFrame(mapping_data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"✓ Docket mapping saved to: {self.output_file}")

        # Save analysis report
        report_file = "data/case_law/docket_mapping_analysis.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📋 Docket Mapping Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Cases**: {analysis['total_cases']}\n\n")

            f.write("## 📊 Mapping Quality\n\n")
            f.write(f"- **Cases with docket_id**: {analysis['cases_with_docket_id']}\n")
            f.write(f"- **Cases with docket_number**: {analysis['cases_with_docket_number']}\n")
            f.write(f"- **Cases with both**: {analysis['cases_with_both']}\n")
            f.write(f"- **Cases with neither**: {analysis['cases_with_neither']}\n")
            f.write(f"- **Cases with text**: {analysis['cases_with_text']}\n\n")

            f.write("## 🎯 Outcome Distribution\n\n")
            for outcome, count in analysis['outcome_distribution'].items():
                f.write(f"- **{outcome}**: {count} cases\n")

            f.write("\n## 🏛️ Court Distribution\n\n")
            for court, count in sorted(analysis['court_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{court}**: {count} cases\n")

            f.write("\n## 🔍 Petition Retrieval Readiness\n\n")

            # Calculate readiness metrics
            ready_for_retrieval = analysis['cases_with_docket_id'] + analysis['cases_with_docket_number']
            readiness_pct = (ready_for_retrieval / analysis['total_cases']) * 100

            f.write(f"- **Cases ready for petition retrieval**: {ready_for_retrieval} ({readiness_pct:.1f}%)\n")
            f.write(f"- **Cases needing manual docket lookup**: {analysis['cases_with_neither']}\n")

            if analysis['cases_with_both'] > 0:
                f.write(f"- **High-confidence matches**: {analysis['cases_with_both']} (both docket_id and docket_number)\n")

            f.write("\n## 📝 Next Steps\n\n")
            f.write("1. **High Priority**: Focus on cases with docket_id for automated retrieval\n")
            f.write("2. **Medium Priority**: Use docket_number for manual PACER lookup\n")
            f.write("3. **Low Priority**: Manual case name matching for cases without docket info\n")

        logger.info(f"✓ Analysis report saved to: {report_file}")

    def run_mapping(self):
        """Run the complete docket mapping process."""
        logger.info("🚀 Starting Docket Mapping Process")
        logger.info("="*80)

        # Load all cases
        cases = self.load_all_cases()

        # Create mapping
        mapping_data = self.create_mapping(cases)

        # Analyze quality
        analysis = self.analyze_mapping_quality(mapping_data)

        # Save results
        self.save_mapping(mapping_data, analysis)

        logger.info("🎉 Docket mapping process complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total cases: {analysis['total_cases']}")
        print(f"  Cases with docket_id: {analysis['cases_with_docket_id']}")
        print(f"  Cases with docket_number: {analysis['cases_with_docket_number']}")
        print(f"  Cases ready for retrieval: {analysis['cases_with_docket_id'] + analysis['cases_with_docket_number']}")
        print(f"  High-confidence matches: {analysis['cases_with_both']}")

        return mapping_data, analysis

def main():
    """Main function."""
    print("📋 Docket Mapping Tool")
    print("="*80)

    mapper = DocketMapper()
    mapping_data, analysis = mapper.run_mapping()

    print("\n✅ Docket mapping complete!")
    print("Check docket_mapping.csv and docket_mapping_analysis.md for results.")

if __name__ == "__main__":
    main()
