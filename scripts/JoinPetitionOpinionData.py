#!/usr/bin/env python3
"""
Unified Dataset Creator

Joins petition features with opinion data and outcomes to create a unified dataset
for training the petition-outcome predictive model.

Usage: python scripts/join_petition_opinion_data.py
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

class UnifiedDatasetCreator:
    def __init__(self):
        self.petition_features_file = "data/case_law/petition_features.json"
        self.docket_mapping_file = "data/case_law/docket_mapping.csv"
        self.comprehensive_nlp_file = "data/case_law/comprehensive_nlp_insights.json"
        self.output_file = "data/case_law/unified_dataset.csv"
        self.unified_json_file = "data/case_law/unified_dataset.json"

    def load_petition_features(self) -> List[Dict[str, Any]]:
        """Load petition features."""
        logger.info("Loading petition features...")

        if not os.path.exists(self.petition_features_file):
            logger.error(f"Petition features file not found: {self.petition_features_file}")
            return []

        with open(self.petition_features_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features = data.get('features', [])
        logger.info(f"Loaded {len(features)} petition features")
        return features

    def load_docket_mapping(self) -> List[Dict[str, Any]]:
        """Load docket mapping."""
        logger.info("Loading docket mapping...")

        if not os.path.exists(self.docket_mapping_file):
            logger.error(f"Docket mapping file not found: {self.docket_mapping_file}")
            return []

        mapping_data = []
        with open(self.docket_mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping_data.append(row)

        logger.info(f"Loaded {len(mapping_data)} docket mappings")
        return mapping_data

    def load_opinion_features(self) -> Dict[str, Any]:
        """Load opinion features from comprehensive NLP analysis."""
        logger.info("Loading opinion features...")

        if not os.path.exists(self.comprehensive_nlp_file):
            logger.warning(f"Comprehensive NLP file not found: {self.comprehensive_nlp_file}")
            return {}

        with open(self.comprehensive_nlp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info("Loaded comprehensive NLP insights")
        return data

    def match_petition_to_docket(self, petition: Dict[str, Any], docket_mapping: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match a petition to its corresponding docket information."""
        petition_docket_id = petition.get('docket_id')
        petition_case_name = petition.get('case_name', '').lower()

        # Try to match by docket_id first
        if petition_docket_id:
            for docket in docket_mapping:
                if docket.get('docket_id') == str(petition_docket_id):
                    return docket

        # Try to match by case name
        for docket in docket_mapping:
            docket_case_name = docket.get('case_name', '').lower()
            if petition_case_name and docket_case_name:
                # Simple name matching
                if petition_case_name in docket_case_name or docket_case_name in petition_case_name:
                    return docket

        return None

    def extract_outcome_from_docket(self, docket: Dict[str, Any]) -> str:
        """Extract outcome from docket information."""
        outcome = docket.get('outcome', 'UNCLEAR')

        # Map outcomes to standard format
        outcome_mapping = {
            'GRANTED': 'SUCCESS',
            'DENIED': 'FAILURE',
            'MIXED': 'MIXED',
            'AFFIRMED': 'SUCCESS',
            'REVERSED': 'FAILURE',
            'VACATED': 'UNCLEAR',
            'UNCLEAR': 'UNCLEAR'
        }

        return outcome_mapping.get(outcome, 'UNCLEAR')

    def create_unified_features(self, petition: Dict[str, Any], docket: Optional[Dict[str, Any]], opinion_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create unified features combining petition, docket, and opinion data."""

        # Start with petition features
        unified = petition.copy()

        # Add docket information
        if docket:
            unified.update({
                'docket_number': docket.get('docket_number', ''),
                'court_id': docket.get('court_id', ''),
                'date_filed': docket.get('date_filed', ''),
                'has_opinion_text': docket.get('has_opinion_text', False),
                'has_extracted_text': docket.get('has_extracted_text', False),
                'citation_count': docket.get('citation_count', 0)
            })

            # Extract outcome
            unified['outcome'] = self.extract_outcome_from_docket(docket)
        else:
            # Default values if no docket match
            unified.update({
                'docket_number': '',
                'court_id': '',
                'date_filed': '',
                'has_opinion_text': False,
                'has_extracted_text': False,
                'citation_count': 0,
                'outcome': 'UNCLEAR'
            })

        # Add opinion-level features if available
        if opinion_features:
            insights = opinion_features.get('insights', {})

            # Add citation patterns from opinion analysis
            citation_patterns = insights.get('citation_patterns', {})
            if citation_patterns:
                citation_counts = citation_patterns.get('citation_counts', {})
                for citation, count in citation_counts.items():
                    unified[f'opinion_citation_{citation}'] = count

            # Add outcome patterns
            outcome_patterns = insights.get('outcome_patterns', {})
            if outcome_patterns:
                outcome_distribution = outcome_patterns.get('outcome_distribution', {})
                for outcome_type, count in outcome_distribution.items():
                    unified[f'opinion_outcome_{outcome_type.lower()}'] = count

        # Create binary outcome for ML
        unified['is_success'] = 1 if unified['outcome'] == 'SUCCESS' else 0

        return unified

    def create_unified_dataset(self) -> List[Dict[str, Any]]:
        """Create the unified dataset."""
        logger.info("Creating unified dataset...")

        # Load all data sources
        petition_features = self.load_petition_features()
        docket_mapping = self.load_docket_mapping()
        opinion_features = self.load_opinion_features()

        if not petition_features:
            logger.error("No petition features to process")
            return []

        unified_dataset = []

        for petition in petition_features:
            # Match petition to docket
            docket = self.match_petition_to_docket(petition, docket_mapping)

            # Create unified features
            unified_features = self.create_unified_features(petition, docket, opinion_features)

            unified_dataset.append(unified_features)

        logger.info(f"Created unified dataset with {len(unified_dataset)} records")
        return unified_dataset

    def analyze_unified_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the unified dataset."""
        logger.info("Analyzing unified dataset...")

        analysis = {
            'total_records': len(dataset),
            'outcome_distribution': {},
            'court_distribution': {},
            'feature_completeness': {},
            'success_rate': 0,
            'matched_dockets': 0,
            'unmatched_dockets': 0
        }

        if not dataset:
            return analysis

        # Count outcomes
        for record in dataset:
            outcome = record.get('outcome', 'UNKNOWN')
            analysis['outcome_distribution'][outcome] = analysis['outcome_distribution'].get(outcome, 0) + 1

        # Count courts
        for record in dataset:
            court = record.get('court_id', 'UNKNOWN')
            analysis['court_distribution'][court] = analysis['court_distribution'].get(court, 0) + 1

        # Calculate success rate
        success_count = sum(1 for record in dataset if record.get('is_success') == 1)
        analysis['success_rate'] = success_count / len(dataset) if dataset else 0

        # Count matched vs unmatched dockets
        analysis['matched_dockets'] = sum(1 for record in dataset if record.get('docket_number'))
        analysis['unmatched_dockets'] = len(dataset) - analysis['matched_dockets']

        # Feature completeness
        if dataset:
            sample_record = dataset[0]
            for feature, value in sample_record.items():
                if isinstance(value, (int, float)):
                    non_zero_count = sum(1 for record in dataset if record.get(feature, 0) != 0)
                    analysis['feature_completeness'][feature] = non_zero_count / len(dataset)

        return analysis

    def save_unified_dataset(self, dataset: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save the unified dataset."""
        logger.info("Saving unified dataset...")

        # Save as CSV
        df = pd.DataFrame(dataset)
        df.to_csv(self.output_file, index=False)

        # Save as JSON with analysis
        with open(self.unified_json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'creation_date': datetime.now().isoformat(),
                'analysis': analysis,
                'dataset': dataset
            }, f, indent=2, ensure_ascii=False)

        # Save analysis report
        report_file = "data/case_law/unified_dataset_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📊 Unified Dataset Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Records**: {analysis['total_records']}\n\n")

            f.write("## 🎯 Outcome Distribution\n\n")
            for outcome, count in analysis['outcome_distribution'].items():
                f.write(f"- **{outcome}**: {count} records\n")

            f.write(f"\n## 📈 Success Rate\n\n")
            f.write(f"- **Overall Success Rate**: {analysis['success_rate']:.2%}\n")
            f.write(f"- **Matched Dockets**: {analysis['matched_dockets']}\n")
            f.write(f"- **Unmatched Dockets**: {analysis['unmatched_dockets']}\n\n")

            f.write("## 🏛️ Court Distribution\n\n")
            for court, count in sorted(analysis['court_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{court}**: {count} records\n")

            f.write("\n## 🔍 Feature Completeness\n\n")
            for feature, completeness in sorted(analysis['feature_completeness'].items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"- **{feature}**: {completeness:.2%}\n")

            f.write("\n## 🚀 Next Steps\n\n")
            f.write("1. **Train ML Model**: Use unified dataset for petition-outcome prediction\n")
            f.write("2. **Feature Engineering**: Refine features based on model performance\n")
            f.write("3. **Validation**: Test model on held-out data\n")
            f.write("4. **Deployment**: Apply model to new petition analysis\n")

        logger.info(f"✓ Unified dataset saved to: {self.output_file}")
        logger.info(f"✓ Unified dataset JSON saved to: {self.unified_json_file}")
        logger.info(f"✓ Analysis report saved to: {report_file}")

    def run_unified_creation(self):
        """Run the complete unified dataset creation process."""
        logger.info("🚀 Starting Unified Dataset Creation")
        logger.info("="*80)

        # Create unified dataset
        dataset = self.create_unified_dataset()

        if not dataset:
            logger.error("Failed to create unified dataset")
            return

        # Analyze dataset
        analysis = self.analyze_unified_dataset(dataset)

        # Save results
        self.save_unified_dataset(dataset, analysis)

        logger.info("🎉 Unified dataset creation complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total records: {analysis['total_records']}")
        print(f"  Success rate: {analysis['success_rate']:.2%}")
        print(f"  Matched dockets: {analysis['matched_dockets']}")
        print(f"  Outcome distribution: {analysis['outcome_distribution']}")

        return dataset, analysis

def main():
    """Main function."""
    print("🔗 Unified Dataset Creator")
    print("="*80)

    creator = UnifiedDatasetCreator()
    dataset, analysis = creator.run_unified_creation()

    print("\n✅ Unified dataset creation complete!")
    print("Check unified_dataset.csv and unified_dataset_report.md for results.")

if __name__ == "__main__":
    main()
