#!/usr/bin/env python3
"""
Analyze Available Case Data

Instead of trying to download full opinions (which are restricted),
let's analyze what case data we actually have and extract useful information.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Google Drive backup module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "google_drive_backup",
    Path(__file__).parent.parent / "document_ingestion" / "google_drive_backup.py"
)
gdrive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdrive_module)
GoogleDriveBackup = gdrive_module.GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaseDataAnalyzer:
    """Analyze the case data we have and extract useful information."""

    def __init__(self):
        """Initialize the analyzer."""
        self.gdrive = GoogleDriveBackup()

        # Clean folder ID
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

        # Analysis results
        self.case_analysis = {
            'total_cases': 0,
            'cases_with_content': 0,
            'cases_without_content': 0,
            'court_distribution': {},
            'year_distribution': {},
            'case_types': {},
            'key_1782_cases': [],
            'sample_cases': []
        }

    def analyze_case_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Analyze a single case's content."""
        analysis = {
            'filename': filename,
            'has_content': bool(content and len(content.strip()) > 0),
            'content_length': len(content) if content else 0,
            'court': 'Unknown',
            'year': 'Unknown',
            'case_type': 'Unknown',
            'is_1782_case': False,
            'key_terms': []
        }

        if not content:
            return analysis

        content_lower = content.lower()

        # Check if it's a §1782 case
        section_1782_terms = [
            '28 u.s.c. § 1782', '28 usc 1782', 'section 1782',
            'foreign tribunal', 'international tribunal',
            'judicial assistance', 'foreign proceeding', 'international discovery'
        ]

        for term in section_1782_terms:
            if term in content_lower:
                analysis['is_1782_case'] = True
                analysis['key_terms'].append(term)
                break

        # Extract court information
        court_patterns = [
            r'(d\.d\.c\.|district of columbia)',
            r'(s\.d\.n\.y\.|southern district of new york)',
            r'(n\.d\. cal\.|northern district of california)',
            r'(d\. mass\.|district of massachusetts)',
            r'(1st cir\.|first circuit)',
            r'(2nd cir\.|second circuit)',
            r'(11th cir\.|eleventh circuit)',
            r'(10th cir\.|tenth circuit)',
            r'(scotus|supreme court)'
        ]

        import re
        for pattern in court_patterns:
            match = re.search(pattern, content_lower)
            if match:
                analysis['court'] = match.group(1).upper()
                break

        # Extract year
        year_match = re.search(r'(\d{4})', content)
        if year_match:
            analysis['year'] = year_match.group(1)

        # Determine case type
        if 'application' in content_lower:
            analysis['case_type'] = 'Application'
        elif 'in re' in content_lower:
            analysis['case_type'] = 'In Re'
        elif 'petition' in content_lower:
            analysis['case_type'] = 'Petition'
        elif 'motion' in content_lower:
            analysis['case_type'] = 'Motion'
        else:
            analysis['case_type'] = 'Opinion'

        return analysis

    def analyze_all_cases(self):
        """Analyze all cases in the Google Drive folder."""
        logger.info("Starting case data analysis...")

        # Get all files from the folder
        query = f"'{self.clean_folder_id}' in parents and trashed=false"
        results = self.gdrive.service.files().list(
            q=query,
            fields='files(id, name, size, createdTime)'
        ).execute()
        files = results.get('files', [])

        logger.info(f"Found {len(files)} files to analyze")

        # Process each file
        for i, file_metadata in enumerate(files, 1):
            filename = file_metadata['name']

            # Skip if it's already a full opinion file or report
            if filename.startswith('full_opinion_') or filename.endswith('_report.json'):
                logger.info(f"Skipping {filename} (analysis file)")
                continue

            logger.info(f"Analyzing case {i}/{len(files)}: {filename}")

            try:
                # Download file content
                content = self.gdrive.download_file_content(file_metadata['id'])

                # Analyze the content
                analysis = self.analyze_case_content(content, filename)

                # Update statistics
                self.case_analysis['total_cases'] += 1

                if analysis['has_content']:
                    self.case_analysis['cases_with_content'] += 1

                    # Update distributions
                    court = analysis['court']
                    year = analysis['year']
                    case_type = analysis['case_type']

                    self.case_analysis['court_distribution'][court] = self.case_analysis['court_distribution'].get(court, 0) + 1
                    self.case_analysis['year_distribution'][year] = self.case_analysis['year_distribution'].get(year, 0) + 1
                    self.case_analysis['case_types'][case_type] = self.case_analysis['case_types'].get(case_type, 0) + 1

                    # Track key §1782 cases
                    if analysis['is_1782_case']:
                        self.case_analysis['key_1782_cases'].append(analysis)

                    # Keep sample cases
                    if len(self.case_analysis['sample_cases']) < 10:
                        self.case_analysis['sample_cases'].append(analysis)
                else:
                    self.case_analysis['cases_without_content'] += 1

            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                self.case_analysis['cases_without_content'] += 1

        # Generate analysis report
        self.generate_analysis_report()

    def generate_analysis_report(self):
        """Generate a comprehensive analysis report."""
        logger.info(f"\n{'='*80}")
        logger.info("CASE DATA ANALYSIS REPORT")
        logger.info(f"{'='*80}")

        logger.info(f"Total cases analyzed: {self.case_analysis['total_cases']}")
        logger.info(f"Cases with content: {self.case_analysis['cases_with_content']}")
        logger.info(f"Cases without content: {self.case_analysis['cases_without_content']}")
        logger.info(f"Content coverage: {(self.case_analysis['cases_with_content']/self.case_analysis['total_cases'])*100:.1f}%")

        logger.info(f"\nCourt Distribution:")
        for court, count in sorted(self.case_analysis['court_distribution'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {court}: {count} cases")

        logger.info(f"\nYear Distribution:")
        for year, count in sorted(self.case_analysis['year_distribution'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {year}: {count} cases")

        logger.info(f"\nCase Type Distribution:")
        for case_type, count in sorted(self.case_analysis['case_types'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {case_type}: {count} cases")

        logger.info(f"\nKey §1782 Cases Found: {len(self.case_analysis['key_1782_cases'])}")
        for case in self.case_analysis['key_1782_cases'][:5]:  # Show first 5
            logger.info(f"  - {case['filename']} ({case['court']}, {case['year']})")

        logger.info(f"\nSample Cases:")
        for case in self.case_analysis['sample_cases']:
            logger.info(f"  - {case['filename']}: {case['case_type']} in {case['court']} ({case['year']})")

        # Save detailed report
        report_data = {
            'analysis_summary': {
                'total_cases': self.case_analysis['total_cases'],
                'cases_with_content': self.case_analysis['cases_with_content'],
                'cases_without_content': self.case_analysis['cases_without_content'],
                'content_coverage_percent': (self.case_analysis['cases_with_content']/self.case_analysis['total_cases'])*100,
                'key_1782_cases_count': len(self.case_analysis['key_1782_cases'])
            },
            'distributions': {
                'courts': self.case_analysis['court_distribution'],
                'years': self.case_analysis['year_distribution'],
                'case_types': self.case_analysis['case_types']
            },
            'key_1782_cases': self.case_analysis['key_1782_cases'],
            'sample_cases': self.case_analysis['sample_cases']
        }

        # Upload report to Google Drive
        try:
            self.gdrive.upload_file_content(
                content=json.dumps(report_data, indent=2),
                filename="case_data_analysis_report.json",
                folder_id=self.clean_folder_id
            )
            logger.info(f"\n📊 Detailed analysis report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        logger.info(f"\n🎯 Key Findings:")
        logger.info(f"1. We have {self.case_analysis['cases_with_content']} cases with actual content")
        logger.info(f"2. {len(self.case_analysis['key_1782_cases'])} cases are confirmed §1782 discovery cases")
        logger.info(f"3. Cases span {len(self.case_analysis['court_distribution'])} different courts")
        logger.info(f"4. Cases range from {min(self.case_analysis['year_distribution'].keys())} to {max(self.case_analysis['year_distribution'].keys())}")
        logger.info(f"5. Most common case type: {max(self.case_analysis['case_types'].items(), key=lambda x: x[1])[0]}")


def main():
    """Main entry point."""
    analyzer = CaseDataAnalyzer()
    analyzer.analyze_all_cases()


if __name__ == "__main__":
    main()
