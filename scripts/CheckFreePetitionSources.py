#!/usr/bin/env python3
"""
Free Petition Sources Checker

Checks available free sources for petition retrieval including:
- CourtListener RECAP API
- GitHub law databases
- Public PACER alternatives

Usage: python scripts/check_free_petition_sources.py
"""

import requests
import json
import csv
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeSourceChecker:
    def __init__(self):
        self.docket_mapping_file = "data/case_law/docket_mapping.csv"
        self.output_file = "data/case_law/free_petition_sources_report.md"
        self.courtlistener_base_url = "https://www.courtlistener.com/api/rest/v4"
        self.recap_base_url = "https://www.courtlistener.com/api/rest/v4/recap"

        # Sample docket IDs to test
        self.test_dockets = []

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

        # Get sample dockets for testing
        self.test_dockets = [
            row for row in mapping_data
            if row['docket_id'] and row['docket_id'] != 'None'
        ][:10]  # Test first 10 dockets

        return mapping_data

    def test_courtlistener_recap_api(self) -> Dict[str, Any]:
        """Test CourtListener RECAP API access."""
        logger.info("Testing CourtListener RECAP API...")

        results = {
            'api_accessible': False,
            'test_dockets': [],
            'total_dockets_tested': len(self.test_dockets),
            'successful_requests': 0,
            'failed_requests': 0,
            'sample_responses': []
        }

        for docket_info in self.test_dockets:
            docket_id = docket_info['docket_id']
            case_name = docket_info['case_name']

            try:
                # Test RECAP endpoint
                url = f"{self.recap_base_url}/"
                params = {
                    'docket_id': docket_id,
                    'format': 'json'
                }

                logger.info(f"Testing docket {docket_id}: {case_name}")

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    results['successful_requests'] += 1

                    # Check if we got useful data
                    if data.get('count', 0) > 0:
                        results['sample_responses'].append({
                            'docket_id': docket_id,
                            'case_name': case_name,
                            'count': data.get('count', 0),
                            'has_results': True
                        })
                    else:
                        results['sample_responses'].append({
                            'docket_id': docket_id,
                            'case_name': case_name,
                            'count': 0,
                            'has_results': False
                        })
                else:
                    results['failed_requests'] += 1
                    logger.warning(f"Failed request for docket {docket_id}: {response.status_code}")

            except Exception as e:
                results['failed_requests'] += 1
                logger.error(f"Error testing docket {docket_id}: {e}")

            # Be respectful to the API
            time.sleep(1)

        results['api_accessible'] = results['successful_requests'] > 0

        logger.info(f"RECAP API test complete: {results['successful_requests']} successful, {results['failed_requests']} failed")
        return results

    def test_courtlistener_docket_api(self) -> Dict[str, Any]:
        """Test CourtListener docket API access."""
        logger.info("Testing CourtListener docket API...")

        results = {
            'api_accessible': False,
            'test_dockets': [],
            'total_dockets_tested': len(self.test_dockets),
            'successful_requests': 0,
            'failed_requests': 0,
            'sample_responses': []
        }

        for docket_info in self.test_dockets:
            docket_id = docket_info['docket_id']
            case_name = docket_info['case_name']

            try:
                # Test docket endpoint
                url = f"{self.courtlistener_base_url}/dockets/{docket_id}/"

                logger.info(f"Testing docket API for {docket_id}: {case_name}")

                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    results['successful_requests'] += 1

                    # Check for RECAP documents
                    recap_docs = data.get('recap_documents', [])

                    results['sample_responses'].append({
                        'docket_id': docket_id,
                        'case_name': case_name,
                        'has_docket_data': True,
                        'recap_documents_count': len(recap_docs),
                        'has_recap_docs': len(recap_docs) > 0
                    })

                else:
                    results['failed_requests'] += 1
                    logger.warning(f"Failed docket API request for {docket_id}: {response.status_code}")

            except Exception as e:
                results['failed_requests'] += 1
                logger.error(f"Error testing docket API for {docket_id}: {e}")

            # Be respectful to the API
            time.sleep(1)

        results['api_accessible'] = results['successful_requests'] > 0

        logger.info(f"Docket API test complete: {results['successful_requests']} successful, {results['failed_requests']} failed")
        return results

    def check_github_law_databases(self) -> Dict[str, Any]:
        """Check for GitHub law databases that might have petition data."""
        logger.info("Checking GitHub law databases...")

        # Known law-related GitHub repositories
        github_repos = [
            "https://github.com/freelawproject/courtlistener",
            "https://github.com/freelawproject/recap",
            "https://github.com/freelawproject/juriscraper",
            "https://github.com/openlegaldata/awesome-legal-tech",
            "https://github.com/lexisnexis/legal-tech-resources",
            "https://github.com/harvard-lil/capstone",
            "https://github.com/harvard-lil/courtlistener"
        ]

        results = {
            'repos_checked': len(github_repos),
            'accessible_repos': 0,
            'repos_with_data': 0,
            'repo_details': []
        }

        for repo_url in github_repos:
            try:
                # Extract repo info
                repo_name = repo_url.split('/')[-1]
                owner = repo_url.split('/')[-2]

                # Test GitHub API access
                api_url = f"https://api.github.com/repos/{owner}/{repo_name}"

                response = requests.get(api_url, timeout=10)

                if response.status_code == 200:
                    repo_data = response.json()
                    results['accessible_repos'] += 1

                    # Check if it's a law-related repo
                    description = repo_data.get('description', '').lower()
                    topics = repo_data.get('topics', [])

                    law_keywords = ['legal', 'court', 'law', 'judicial', 'pacer', 'recap', 'federal']
                    is_law_repo = any(keyword in description for keyword in law_keywords) or \
                                any(keyword in ' '.join(topics).lower() for keyword in law_keywords)

                    if is_law_repo:
                        results['repos_with_data'] += 1

                    results['repo_details'].append({
                        'name': repo_name,
                        'owner': owner,
                        'description': description,
                        'is_law_repo': is_law_repo,
                        'stars': repo_data.get('stargazers_count', 0),
                        'last_updated': repo_data.get('updated_at', '')
                    })

                else:
                    logger.warning(f"Failed to access {repo_url}: {response.status_code}")

            except Exception as e:
                logger.error(f"Error checking {repo_url}: {e}")

            # Be respectful to GitHub API
            time.sleep(1)

        logger.info(f"GitHub check complete: {results['accessible_repos']} accessible, {results['repos_with_data']} law-related")
        return results

    def analyze_petition_availability(self, mapping_data: List[Dict[str, Any]],
                                   recap_results: Dict[str, Any],
                                   docket_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall petition availability."""
        logger.info("Analyzing petition availability...")

        analysis = {
            'total_cases': len(mapping_data),
            'cases_with_docket_id': len([c for c in mapping_data if c['docket_id'] and c['docket_id'] != 'None']),
            'cases_with_docket_number': len([c for c in mapping_data if c['docket_number']]),
            'recap_api_success_rate': 0,
            'docket_api_success_rate': 0,
            'estimated_petition_availability': 0,
            'recommendations': []
        }

        # Calculate success rates
        if recap_results['total_dockets_tested'] > 0:
            analysis['recap_api_success_rate'] = (recap_results['successful_requests'] / recap_results['total_dockets_tested']) * 100

        if docket_results['total_dockets_tested'] > 0:
            analysis['docket_api_success_rate'] = (docket_results['successful_requests'] / docket_results['total_dockets_tested']) * 100

        # Estimate petition availability
        cases_with_recap_access = analysis['cases_with_docket_id'] * (analysis['recap_api_success_rate'] / 100)
        analysis['estimated_petition_availability'] = int(cases_with_recap_access)

        # Generate recommendations
        if analysis['recap_api_success_rate'] > 50:
            analysis['recommendations'].append("✅ CourtListener RECAP API is reliable for automated retrieval")
        else:
            analysis['recommendations'].append("⚠️ CourtListener RECAP API has limited availability")

        if analysis['docket_api_success_rate'] > 50:
            analysis['recommendations'].append("✅ CourtListener docket API is reliable for metadata")
        else:
            analysis['recommendations'].append("⚠️ CourtListener docket API has limited availability")

        if analysis['estimated_petition_availability'] > 100:
            analysis['recommendations'].append(f"🎯 Estimated {analysis['estimated_petition_availability']} petitions available for free retrieval")
        else:
            analysis['recommendations'].append("💰 Consider PACER for comprehensive petition access")

        return analysis

    def generate_report(self, mapping_data: List[Dict[str, Any]],
                       recap_results: Dict[str, Any],
                       docket_results: Dict[str, Any],
                       github_results: Dict[str, Any],
                       analysis: Dict[str, Any]):
        """Generate comprehensive report."""
        logger.info("Generating free sources report...")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("# 🔍 Free Petition Sources Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Cases Analyzed**: {analysis['total_cases']}\n\n")

            f.write("## 📊 Summary\n\n")
            f.write(f"- **Cases with docket_id**: {analysis['cases_with_docket_id']}\n")
            f.write(f"- **Cases with docket_number**: {analysis['cases_with_docket_number']}\n")
            f.write(f"- **Estimated petition availability**: {analysis['estimated_petition_availability']}\n")
            f.write(f"- **RECAP API success rate**: {analysis['recap_api_success_rate']:.1f}%\n")
            f.write(f"- **Docket API success rate**: {analysis['docket_api_success_rate']:.1f}%\n\n")

            f.write("## 🔗 CourtListener RECAP API\n\n")
            f.write(f"- **API Accessible**: {'✅ Yes' if recap_results['api_accessible'] else '❌ No'}\n")
            f.write(f"- **Successful Requests**: {recap_results['successful_requests']}/{recap_results['total_dockets_tested']}\n")
            f.write(f"- **Failed Requests**: {recap_results['failed_requests']}\n\n")

            f.write("### Sample RECAP Results\n\n")
            for result in recap_results['sample_responses'][:5]:
                f.write(f"- **{result['case_name']}** (docket {result['docket_id']}): {result['count']} documents\n")

            f.write("\n## 🔗 CourtListener Docket API\n\n")
            f.write(f"- **API Accessible**: {'✅ Yes' if docket_results['api_accessible'] else '❌ No'}\n")
            f.write(f"- **Successful Requests**: {docket_results['successful_requests']}/{docket_results['total_dockets_tested']}\n")
            f.write(f"- **Failed Requests**: {docket_results['failed_requests']}\n\n")

            f.write("### Sample Docket Results\n\n")
            for result in docket_results['sample_responses'][:5]:
                f.write(f"- **{result['case_name']}** (docket {result['docket_id']}): {result['recap_documents_count']} RECAP docs\n")

            f.write("\n## 🐙 GitHub Law Databases\n\n")
            f.write(f"- **Repositories Checked**: {github_results['repos_checked']}\n")
            f.write(f"- **Accessible Repositories**: {github_results['accessible_repos']}\n")
            f.write(f"- **Law-Related Repositories**: {github_results['repos_with_data']}\n\n")

            f.write("### Law-Related Repositories\n\n")
            for repo in github_results['repo_details']:
                if repo['is_law_repo']:
                    f.write(f"- **{repo['owner']}/{repo['name']}**: {repo['description']} ({repo['stars']} stars)\n")

            f.write("\n## 🎯 Recommendations\n\n")
            for rec in analysis['recommendations']:
                f.write(f"- {rec}\n")

            f.write("\n## 📝 Next Steps\n\n")
            f.write("1. **High Priority**: Implement automated retrieval using CourtListener APIs\n")
            f.write("2. **Medium Priority**: Explore GitHub repositories for additional petition data\n")
            f.write("3. **Low Priority**: Consider PACER for cases not available through free sources\n")
            f.write("4. **Implementation**: Build petition retrieval system based on successful API tests\n")

        logger.info(f"✓ Report saved to: {self.output_file}")

    def run_check(self):
        """Run the complete free sources check."""
        logger.info("🚀 Starting Free Petition Sources Check")
        logger.info("="*80)

        # Load docket mapping
        mapping_data = self.load_docket_mapping()
        if not mapping_data:
            return

        # Test APIs
        recap_results = self.test_courtlistener_recap_api()
        docket_results = self.test_courtlistener_docket_api()
        github_results = self.check_github_law_databases()

        # Analyze results
        analysis = self.analyze_petition_availability(mapping_data, recap_results, docket_results)

        # Generate report
        self.generate_report(mapping_data, recap_results, docket_results, github_results, analysis)

        logger.info("🎉 Free sources check complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total cases: {analysis['total_cases']}")
        print(f"  Cases with docket_id: {analysis['cases_with_docket_id']}")
        print(f"  Estimated petition availability: {analysis['estimated_petition_availability']}")
        print(f"  RECAP API success rate: {analysis['recap_api_success_rate']:.1f}%")
        print(f"  Docket API success rate: {analysis['docket_api_success_rate']:.1f}%")

        return analysis

def main():
    """Main function."""
    print("🔍 Free Petition Sources Checker")
    print("="*80)

    checker = FreeSourceChecker()
    analysis = checker.run_check()

    print("\n✅ Free sources check complete!")
    print("Check free_petition_sources_report.md for detailed results.")

if __name__ == "__main__":
    main()
