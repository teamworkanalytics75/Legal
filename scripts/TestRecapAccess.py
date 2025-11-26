#!/usr/bin/env python3
"""
Simple RECAP Test (No Authentication)

Tests what we can access from CourtListener RECAP without authentication.
This will help us understand what's available before setting up API tokens.

Usage: python scripts/test_recap_access.py
"""

import requests
import json
import time
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRECAPTest:
    def __init__(self):
        self.api_base_url = "https://www.courtlistener.com/api/rest/v4"
        self.recap_base_url = "https://www.courtlistener.com/api/rest/v4/recap"
        self.dockets_base_url = "https://www.courtlistener.com/api/rest/v4/dockets"

        # Use existing API token
        self.api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
        self.headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }

    def test_public_access(self):
        """Test what we can access without authentication."""
        logger.info("Testing public RECAP access...")

        results = {
            'dockets_api_accessible': False,
            'recap_api_accessible': False,
            'sample_dockets': [],
            'sample_recap_docs': [],
            'search_results': []
        }

        # Test dockets API
        try:
            logger.info("Testing dockets API...")
            response = requests.get(self.dockets_base_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                results['dockets_api_accessible'] = True
                data = response.json()
                results['sample_dockets'] = data.get('results', [])[:5]  # First 5 results
                logger.info(f"✓ Dockets API accessible - found {len(data.get('results', []))} dockets")
            else:
                logger.warning(f"Dockets API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Dockets API test failed: {e}")

        # Test RECAP API
        try:
            logger.info("Testing RECAP API...")
            response = requests.get(self.recap_base_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                results['recap_api_accessible'] = True
                data = response.json()
                results['sample_recap_docs'] = data.get('results', [])[:5]  # First 5 results
                logger.info(f"✓ RECAP API accessible - found {len(data.get('results', []))} documents")
            else:
                logger.warning(f"RECAP API returned {response.status_code}")

        except Exception as e:
            logger.error(f"RECAP API test failed: {e}")

        # Test search functionality
        try:
            logger.info("Testing search for §1782...")
            search_url = f"{self.dockets_base_url}/"
            params = {'q': '1782', 'format': 'json'}
            response = requests.get(search_url, headers=self.headers, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                results['search_results'] = data.get('results', [])[:10]  # First 10 results
                logger.info(f"✓ Search successful - found {len(data.get('results', []))} results for '1782'")
            else:
                logger.warning(f"Search returned {response.status_code}")

        except Exception as e:
            logger.error(f"Search test failed: {e}")

        return results

    def analyze_sample_data(self, results: Dict[str, Any]):
        """Analyze the sample data we retrieved."""
        logger.info("Analyzing sample data...")

        analysis = {
            'docket_fields': set(),
            'recap_doc_fields': set(),
            'search_result_fields': set(),
            'potential_1782_cases': []
        }

        # Analyze docket fields
        for docket in results['sample_dockets']:
            analysis['docket_fields'].update(docket.keys())

        # Analyze RECAP document fields
        for doc in results['sample_recap_docs']:
            analysis['recap_doc_fields'].update(doc.keys())

        # Analyze search results
        for result in results['search_results']:
            analysis['search_result_fields'].update(result.keys())

            # Check if this looks like a §1782 case
            case_name = str(result.get('case_name', '')).lower()
            docket_text = str(result.get('docket_text', '')).lower()

            if any(indicator in case_name or indicator in docket_text for indicator in ['1782', 'foreign discovery', 'international judicial assistance']):
                analysis['potential_1782_cases'].append({
                    'id': result.get('id'),
                    'case_name': result.get('case_name'),
                    'court': result.get('court'),
                    'date_filed': result.get('date_filed')
                })

        return analysis

    def generate_report(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Generate a test report."""
        report_file = "data/case_law/recap_access_test_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🔍 RECAP Access Test Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 API Access Results\n\n")
            f.write(f"- **Dockets API Accessible**: {'✅ Yes' if results['dockets_api_accessible'] else '❌ No'}\n")
            f.write(f"- **RECAP API Accessible**: {'✅ Yes' if results['recap_api_accessible'] else '❌ No'}\n")
            f.write(f"- **Search Results Found**: {len(results['search_results'])}\n\n")

            f.write("## 🔍 Sample Data Analysis\n\n")
            f.write(f"- **Docket Fields Available**: {len(analysis['docket_fields'])}\n")
            f.write(f"- **RECAP Document Fields Available**: {len(analysis['recap_doc_fields'])}\n")
            f.write(f"- **Potential §1782 Cases Found**: {len(analysis['potential_1782_cases'])}\n\n")

            f.write("### Available Docket Fields\n\n")
            for field in sorted(analysis['docket_fields']):
                f.write(f"- `{field}`\n")

            f.write("\n### Available RECAP Document Fields\n\n")
            for field in sorted(analysis['recap_doc_fields']):
                f.write(f"- `{field}`\n")

            f.write("\n### Potential §1782 Cases\n\n")
            for case in analysis['potential_1782_cases']:
                f.write(f"- **{case['case_name']}** (ID: {case['id']}, Court: {case['court']})\n")

            f.write("\n## 🎯 Next Steps\n\n")
            if results['dockets_api_accessible'] and results['recap_api_accessible']:
                f.write("✅ **Both APIs are accessible!** We can proceed with:\n")
                f.write("1. Set up API authentication for full access\n")
                f.write("2. Search for §1782 dockets systematically\n")
                f.write("3. Download RECAP documents for petitions\n")
            else:
                f.write("⚠️ **Limited API access.** Consider:\n")
                f.write("1. Setting up CourtListener API authentication\n")
                f.write("2. Using alternative free sources\n")
                f.write("3. Manual PACER retrieval for critical cases\n")

        logger.info(f"✓ Test report saved to: {report_file}")

    def run_test(self):
        """Run the complete test."""
        logger.info("🚀 Starting RECAP Access Test")
        logger.info("="*80)

        # Test public access
        results = self.test_public_access()

        # Analyze sample data
        analysis = self.analyze_sample_data(results)

        # Generate report
        self.generate_report(results, analysis)

        logger.info("🎉 RECAP access test complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Dockets API accessible: {results['dockets_api_accessible']}")
        print(f"  RECAP API accessible: {results['recap_api_accessible']}")
        print(f"  Search results found: {len(results['search_results'])}")
        print(f"  Potential §1782 cases: {len(analysis['potential_1782_cases'])}")

        return results, analysis

def main():
    """Main function."""
    print("🔍 Simple RECAP Access Test")
    print("="*80)

    tester = SimpleRECAPTest()
    results, analysis = tester.run_test()

    print("\n✅ RECAP access test complete!")
    print("Check recap_access_test_report.md for detailed results.")

if __name__ == "__main__":
    main()
