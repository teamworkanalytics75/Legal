#!/usr/bin/env python3
"""
Targeted RECAP Petition Retriever

Check RECAP for original petitions from our existing §1782 cases.
"""

import requests
import json
import os
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetedRECAPRetriever:
    def __init__(self):
        self.api_base_url = "https://www.courtlistener.com/api/rest/v4"
        self.recap_base_url = "https://www.courtlistener.com/api/rest/v4/recap"
        self.dockets_base_url = "https://www.courtlistener.com/api/rest/v4/dockets"

        # Use existing API token
        self.api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
        self.headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json',
            'User-Agent': 'Malcolm-Grayson-Research/1.0 (malcolmgrayson00@gmail.com)'
        }

        # Output directories
        self.petitions_raw_dir = "data/petitions_raw"
        self.petitions_text_dir = "data/petitions_text"
        self.results_file = "data/case_law/targeted_recap_results.json"

        # Create directories
        os.makedirs(self.petitions_raw_dir, exist_ok=True)
        os.makedirs(self.petitions_text_dir, exist_ok=True)

    def load_our_cases(self) -> List[Dict[str, Any]]:
        """Load our existing §1782 cases."""
        logger.info("Loading our existing §1782 cases...")

        cases = []
        case_dir = "data/case_law/1782_discovery"

        if not os.path.exists(case_dir):
            logger.error(f"Case directory not found: {case_dir}")
            return cases

        # Load all JSON files
        for filename in os.listdir(case_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(case_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                        cases.append(case_data)
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")

        logger.info(f"✓ Loaded {len(cases)} cases")
        return cases

    def get_docket_id_from_case(self, case: Dict[str, Any]) -> Optional[int]:
        """Extract docket ID from our case data."""
        # Try different fields that might contain docket ID
        docket_id = case.get('docket_id')
        if docket_id:
            return int(docket_id)

        # Check if there's a docket number we can search for
        docket_number = case.get('docket_number')
        if docket_number:
            # Try to extract numeric part
            import re
            numbers = re.findall(r'\d+', str(docket_number))
            if numbers:
                return int(numbers[0])

        return None

    def search_docket_by_case_info(self, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Search for docket using case information."""
        case_name = case.get('case_name', '')
        docket_number = case.get('docket_number', '')

        if not case_name and not docket_number:
            return None

        logger.info(f"Searching for docket: {case_name}")

        # Try searching by case name
        if case_name:
            try:
                params = {'q': f'"{case_name}"', 'format': 'json'}
                response = requests.get(self.dockets_base_url, headers=self.headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])

                    # Look for exact match
                    for result in results:
                        if result.get('case_name', '').lower() == case_name.lower():
                            logger.info(f"✓ Found exact match: {result.get('id')}")
                            return result

                    # If no exact match, return first result
                    if results:
                        logger.info(f"✓ Found partial match: {results[0].get('id')}")
                        return results[0]

            except Exception as e:
                logger.error(f"Error searching for {case_name}: {e}")

        return None

    def get_recap_documents(self, docket_id: int) -> List[Dict[str, Any]]:
        """Get RECAP documents for a docket."""
        logger.info(f"Getting RECAP documents for docket {docket_id}")

        try:
            params = {'docket_id': docket_id, 'format': 'json'}
            response = requests.get(self.recap_base_url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"✓ Found {len(results)} RECAP documents")
                return results
            else:
                logger.warning(f"RECAP API returned {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting RECAP documents: {e}")
            return []

    def download_petition_pdf(self, recap_doc: Dict[str, Any]) -> Optional[str]:
        """Download petition PDF from RECAP document."""
        doc_id = recap_doc.get('id')
        download_url = recap_doc.get('filepath_local')

        if not download_url:
            logger.warning(f"No download URL for document {doc_id}")
            return None

        try:
            # Download PDF
            response = requests.get(download_url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                # Save PDF
                filename = f"petition_{doc_id}.pdf"
                filepath = os.path.join(self.petitions_raw_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✓ Downloaded PDF: {filename}")
                return filepath
            else:
                logger.warning(f"Failed to download PDF: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case to find RECAP documents."""
        case_name = case.get('case_name', 'Unknown')
        docket_id = self.get_docket_id_from_case(case)

        result = {
            'case_name': case_name,
            'docket_id': docket_id,
            'recap_documents': [],
            'petitions_found': 0,
            'status': 'not_found'
        }

        # If we have a docket ID, try to get RECAP documents
        if docket_id:
            recap_docs = self.get_recap_documents(docket_id)
            result['recap_documents'] = recap_docs
            result['petitions_found'] = len(recap_docs)

            if recap_docs:
                result['status'] = 'found'
                logger.info(f"✓ Found {len(recap_docs)} RECAP documents for {case_name}")
            else:
                result['status'] = 'no_recap'
                logger.info(f"✗ No RECAP documents for {case_name}")
        else:
            # Try to search for docket by case info
            docket_data = self.search_docket_by_case_info(case)
            if docket_data:
                docket_id = docket_data.get('id')
                recap_docs = self.get_recap_documents(docket_id)
                result['recap_documents'] = recap_docs
                result['petitions_found'] = len(recap_docs)

                if recap_docs:
                    result['status'] = 'found'
                    logger.info(f"✓ Found {len(recap_docs)} RECAP documents for {case_name}")
                else:
                    result['status'] = 'no_recap'
                    logger.info(f"✗ No RECAP documents for {case_name}")
            else:
                result['status'] = 'docket_not_found'
                logger.info(f"✗ Could not find docket for {case_name}")

        return result

    def run(self):
        """Run the targeted RECAP retrieval."""
        logger.info("🎯 Targeted RECAP Petition Retriever")
        logger.info("=" * 80)

        # Load our cases
        cases = self.load_our_cases()
        if not cases:
            logger.error("No cases found to process")
            return

        # Process each case
        results = []
        total_petitions = 0

        for i, case in enumerate(cases):
            logger.info(f"Processing case {i+1}/{len(cases)}: {case.get('case_name', 'Unknown')}")

            result = self.process_case(case)
            results.append(result)

            if result['petitions_found'] > 0:
                total_petitions += result['petitions_found']

            # Be respectful to the API
            import time
            time.sleep(0.5)

        # Save results
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_cases_processed': len(cases),
                'total_petitions_found': total_petitions,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        # Generate summary
        found_cases = [r for r in results if r['status'] == 'found']
        no_recap_cases = [r for r in results if r['status'] == 'no_recap']
        docket_not_found_cases = [r for r in results if r['status'] == 'docket_not_found']

        logger.info(f"\n📊 Summary:")
        logger.info(f"  Total cases processed: {len(cases)}")
        logger.info(f"  Cases with RECAP documents: {len(found_cases)}")
        logger.info(f"  Cases without RECAP documents: {len(no_recap_cases)}")
        logger.info(f"  Cases with docket not found: {len(docket_not_found_cases)}")
        logger.info(f"  Total petitions found: {total_petitions}")

        logger.info(f"\n✅ Targeted RECAP retrieval complete!")
        logger.info(f"Results saved to: {self.results_file}")

if __name__ == "__main__":
    retriever = TargetedRECAPRetriever()
    retriever.run()
