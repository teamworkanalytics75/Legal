#!/usr/bin/env python3
"""
Hybrid Petition Retrieval System

Creates a system that:
1. Attempts to retrieve petitions from available free sources
2. Prepares docket information for manual PACER retrieval
3. Creates a working prototype with available data

Usage: python scripts/build_petition_retrieval.py
"""

import json
import csv
import os
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPetitionRetriever:
    def __init__(self):
        self.docket_mapping_file = "data/case_law/docket_mapping.csv"
        self.petitions_raw_dir = "data/petitions_raw"
        self.petitions_text_dir = "data/petitions_text"
        self.retrieval_log_file = "data/case_law/petition_retrieval_log.json"

        # Create directories
        os.makedirs(self.petitions_raw_dir, exist_ok=True)
        os.makedirs(self.petitions_text_dir, exist_ok=True)

        self.retrieval_log = {
            'retrieval_date': datetime.now().isoformat(),
            'total_cases_processed': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'manual_retrieval_needed': 0,
            'retrieval_details': []
        }

    def load_docket_mapping(self) -> List[Dict[str, Any]]:
        """Load the docket mapping CSV."""
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

    def check_existing_petitions(self) -> Dict[str, Any]:
        """Check for existing petition files in the corpus."""
        logger.info("Checking for existing petition files...")

        existing_petitions = {
            'pdf_files': [],
            'text_files': [],
            'total_found': 0
        }

        # Check for PDFs in the corpus
        corpus_dir = "data/case_law/1782_discovery"
        if os.path.exists(corpus_dir):
            for filename in os.listdir(corpus_dir):
                if filename.endswith('.pdf'):
                    existing_petitions['pdf_files'].append(filename)
                elif filename.endswith('.txt'):
                    existing_petitions['text_files'].append(filename)

        existing_petitions['total_found'] = len(existing_petitions['pdf_files']) + len(existing_petitions['text_files'])

        logger.info(f"Found {existing_petitions['total_found']} existing petition files")
        return existing_petitions

    def attempt_courtlistener_retrieval(self, docket_info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to retrieve petition from CourtListener (with authentication)."""
        docket_id = docket_info['docket_id']
        case_name = docket_info['case_name']

        result = {
            'method': 'courtlistener',
            'docket_id': docket_id,
            'case_name': case_name,
            'success': False,
            'error': None,
            'files_retrieved': []
        }

        try:
            # Note: This would require API authentication
            # For now, we'll simulate the attempt and log the need for authentication

            logger.info(f"Attempting CourtListener retrieval for {case_name} (docket {docket_id})")

            # Simulate API call (would need actual authentication)
            # url = f"https://www.courtlistener.com/api/rest/v4/recap/"
            # params = {'docket_id': docket_id}
            # response = requests.get(url, params=params, headers={'Authorization': 'Token YOUR_TOKEN'})

            result['error'] = "API authentication required"
            result['success'] = False

        except Exception as e:
            result['error'] = str(e)
            result['success'] = False

        return result

    def prepare_manual_retrieval_info(self, docket_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare information for manual PACER retrieval."""
        return {
            'case_name': docket_info['case_name'],
            'case_name_full': docket_info['case_name_full'],
            'docket_id': docket_info['docket_id'],
            'docket_number': docket_info['docket_number'],
            'court': docket_info['court'],
            'court_id': docket_info['court_id'],
            'date_filed': docket_info['date_filed'],
            'outcome': docket_info['outcome'],
            'filename': docket_info['filename'],
            'retrieval_priority': 'high' if docket_info['outcome'] in ['GRANTED', 'DENIED'] else 'medium'
        }

    def create_sample_petition_dataset(self) -> Dict[str, Any]:
        """Create a sample petition dataset using available case text."""
        logger.info("Creating sample petition dataset from available case text...")

        sample_petitions = []
        corpus_dir = "data/case_law/1782_discovery"

        # Look for cases with substantial text that might contain petition-like content
        for filename in os.listdir(corpus_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(corpus_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)

                    # Check if case has substantial text
                    text_length = len(str(case_data.get('opinion_text', '')) + str(case_data.get('extracted_text', '')))

                    if text_length > 5000:  # Substantial text
                        # Extract petition-like content (applications, motions, etc.)
                        all_text = str(case_data.get('opinion_text', '')) + str(case_data.get('extracted_text', ''))

                        # Look for petition-related keywords
                        petition_keywords = ['application', 'motion', 'petition', 'request', 'order pursuant', 'discovery']
                        if any(keyword.lower() in all_text.lower() for keyword in petition_keywords):

                            sample_petition = {
                                'filename': filename,
                                'case_name': case_data.get('caseName', ''),
                                'docket_id': case_data.get('docket_id'),
                                'docket_number': case_data.get('docketNumber', ''),
                                'court': case_data.get('court', ''),
                                'text_length': text_length,
                                'text_content': all_text[:10000],  # First 10k characters
                                'outcome': self.extract_outcome_from_text(all_text),
                                'petition_type': 'sample_from_opinion'
                            }

                            sample_petitions.append(sample_petition)

                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")

        logger.info(f"Created {len(sample_petitions)} sample petitions from case text")
        return sample_petitions

    def extract_outcome_from_text(self, text: str) -> str:
        """Extract outcome from text content."""
        text_lower = text.lower()

        if any(phrase in text_lower for phrase in ['motion granted', 'application granted', 'petition granted']):
            return 'GRANTED'
        elif any(phrase in text_lower for phrase in ['motion denied', 'application denied', 'petition denied']):
            return 'DENIED'
        elif any(phrase in text_lower for phrase in ['granted in part', 'denied in part']):
            return 'MIXED'
        else:
            return 'UNCLEAR'

    def save_sample_petitions(self, sample_petitions: List[Dict[str, Any]]):
        """Save sample petitions to files."""
        logger.info("Saving sample petitions...")

        # Save as JSON
        sample_file = "data/case_law/sample_petitions.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({
                'creation_date': datetime.now().isoformat(),
                'total_petitions': len(sample_petitions),
                'petitions': sample_petitions
            }, f, indent=2, ensure_ascii=False)

        # Save individual text files
        for i, petition in enumerate(sample_petitions):
            text_file = os.path.join(self.petitions_text_dir, f"sample_petition_{i+1}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Case: {petition['case_name']}\n")
                f.write(f"Docket: {petition['docket_number']}\n")
                f.write(f"Court: {petition['court']}\n")
                f.write(f"Outcome: {petition['outcome']}\n")
                f.write("="*80 + "\n\n")
                f.write(petition['text_content'])

        logger.info(f"✓ Saved {len(sample_petitions)} sample petitions")

    def create_manual_retrieval_guide(self, manual_cases: List[Dict[str, Any]]):
        """Create a guide for manual PACER retrieval."""
        logger.info("Creating manual retrieval guide...")

        guide_file = "data/case_law/manual_pacer_retrieval_guide.md"

        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("# 📋 Manual PACER Retrieval Guide\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Cases Needing Manual Retrieval**: {len(manual_cases)}\n\n")

            f.write("## 🎯 High Priority Cases (Clear Outcomes)\n\n")
            high_priority = [c for c in manual_cases if c['retrieval_priority'] == 'high']
            f.write(f"**Count**: {len(high_priority)}\n\n")

            for case in high_priority[:10]:  # Show first 10
                f.write(f"### {case['case_name']}\n")
                f.write(f"- **Docket ID**: {case['docket_id']}\n")
                f.write(f"- **Docket Number**: {case['docket_number']}\n")
                f.write(f"- **Court**: {case['court']}\n")
                f.write(f"- **Outcome**: {case['outcome']}\n")
                f.write(f"- **Date Filed**: {case['date_filed']}\n\n")

            f.write("## 🔍 PACER Retrieval Steps\n\n")
            f.write("1. **Login to PACER** with your credentials\n")
            f.write("2. **Search by docket number** or case name\n")
            f.write("3. **Download petition/motion PDFs** from the docket\n")
            f.write("4. **Save as**: `{docket_number}_{case_name}.pdf`\n")
            f.write("5. **Place in**: `data/petitions_raw/`\n\n")

            f.write("## 📊 Case Summary\n\n")
            f.write(f"- **High Priority**: {len(high_priority)} cases\n")
            f.write(f"- **Medium Priority**: {len(manual_cases) - len(high_priority)} cases\n")
            f.write(f"- **Total**: {len(manual_cases)} cases\n\n")

        logger.info(f"✓ Manual retrieval guide saved to: {guide_file}")

    def run_retrieval(self):
        """Run the hybrid petition retrieval process."""
        logger.info("🚀 Starting Hybrid Petition Retrieval")
        logger.info("="*80)

        # Load docket mapping
        mapping_data = self.load_docket_mapping()
        if not mapping_data:
            return

        # Check for existing petitions
        existing = self.check_existing_petitions()
        logger.info(f"Found {existing['total_found']} existing petition files")

        # Create sample petition dataset from available text
        sample_petitions = self.create_sample_petition_dataset()
        self.save_sample_petitions(sample_petitions)

        # Prepare manual retrieval cases
        manual_cases = []
        for docket_info in mapping_data:
            if docket_info['docket_id'] and docket_info['docket_id'] != 'None':
                manual_case = self.prepare_manual_retrieval_info(docket_info)
                manual_cases.append(manual_case)

        self.create_manual_retrieval_guide(manual_cases)

        # Update retrieval log
        self.retrieval_log.update({
            'total_cases_processed': len(mapping_data),
            'sample_petitions_created': len(sample_petitions),
            'manual_retrieval_needed': len(manual_cases),
            'existing_petitions_found': existing['total_found']
        })

        # Save retrieval log
        with open(self.retrieval_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.retrieval_log, f, indent=2, ensure_ascii=False)

        logger.info("🎉 Hybrid petition retrieval complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total cases processed: {len(mapping_data)}")
        print(f"  Sample petitions created: {len(sample_petitions)}")
        print(f"  Manual retrieval needed: {len(manual_cases)}")
        print(f"  Existing petitions found: {existing['total_found']}")

        return {
            'sample_petitions': sample_petitions,
            'manual_cases': manual_cases,
            'existing_petitions': existing
        }

def main():
    """Main function."""
    print("🔄 Hybrid Petition Retrieval System")
    print("="*80)

    retriever = HybridPetitionRetriever()
    results = retriever.run_retrieval()

    print("\n✅ Hybrid retrieval complete!")
    print("Check sample_petitions.json and manual_pacer_retrieval_guide.md for results.")

if __name__ == "__main__":
    main()
