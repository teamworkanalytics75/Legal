#!/usr/bin/env python3
"""
CourtListener API PDF Downloader
Uses the CourtListener API instead of browser automation to download PDFs
"""

import json
import time
import logging
import hashlib
import mysql.connector
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourtListenerAPIPDFDownloader:
    def __init__(self, config_path: str = "document_ingestion/courtlistener_config.json"):
        """Initialize the API-based PDF downloader."""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.base_url = self.config['api']['base_url']

        # Database connection
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'thetimeisN0w!',
            'database': 'lawsuit_docs',
            'charset': 'utf8mb4'
        }

        # PDF storage directory
        self.pdf_dir = Path("data/case_law/pdfs")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.base_delay = 1.0
        self.last_request_time = 0

        # Statistics tracking
        self.stats = {
            'total_cases_processed': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'free_pdfs_found': 0,
            'paid_pdfs_skipped': 0,
            'cases_with_pdfs': 0,
            'cases_without_pdfs': 0,
            'last_run': None
        }

        # Load existing PDF hashes
        self.downloaded_hashes = set()
        self.load_existing_pdfs()

        logger.info("CourtListener API PDF Downloader initialized")
        logger.info(f"PDF storage: {self.pdf_dir.absolute()}")
        logger.info(f"Database: {self.db_config['database']}")

    def load_existing_pdfs(self):
        """Load existing PDF files to avoid duplicates."""
        for pdf_file in self.pdf_dir.rglob("*.pdf"):
            try:
                with open(pdf_file, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    self.downloaded_hashes.add(file_hash)
            except Exception as e:
                logger.warning(f"Could not hash existing file {pdf_file}: {e}")

        logger.info(f"✓ Loaded {len(self.downloaded_hashes)} existing PDF hashes")

    def rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.base_delay:
            time.sleep(self.base_delay - elapsed)
        self.last_request_time = time.time()

    def make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a request to the CourtListener API."""
        self.rate_limit()

        url = f"{self.base_url}{endpoint}"

        # Add API token to headers
        headers = {
            'Authorization': f'Token {self.config["api"]["api_token"]}',
            'User-Agent': self.config["api"]["user_agent"]
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self.make_api_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def get_cases_from_database(self, limit: int = 100, topic: str = None) -> List[Dict[str, Any]]:
        """Get cases from the database that need PDF downloads."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)

            if topic:
                query = """
                    SELECT id, case_name, citation, court, date_filed, url, source_id, topic
                    FROM case_law
                    WHERE topic = %s AND download_url IS NULL
                    ORDER BY date_filed DESC
                    LIMIT %s
                """
                cursor.execute(query, (topic, limit))
            else:
                query = """
                    SELECT id, case_name, citation, court, date_filed, url, source_id, topic
                    FROM case_law
                    WHERE download_url IS NULL
                    ORDER BY date_filed DESC
                    LIMIT %s
                """
                cursor.execute(query, (limit,))

            cases = cursor.fetchall()
            cursor.close()
            conn.close()

            logger.info(f"✓ Retrieved {len(cases)} cases from database")
            return cases

        except Exception as e:
            logger.error(f"Error retrieving cases from database: {e}")
            return []

    def search_courtlistener_api(self, case_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Search CourtListener API for a specific case."""
        case_name = case_info['case_name']
        citation = case_info.get('citation', '')

        logger.info(f"🔍 API search for: {case_name}")

        # Try different search strategies
        search_queries = [
            case_name,
            case_name.replace(' v. ', ' vs '),
            case_name.replace(' v. ', ' v ')
        ]

        if citation:
            search_queries.append(citation)

        for query in search_queries:
            if not query.strip():
                continue

            logger.info(f"  Trying query: {query}")

            # Search opinions endpoint
            params = {
                'q': query,
                'format': 'json',
                'order_by': 'score desc',
                'stat_Precedential': 'on'  # Only precedential opinions
            }

            response = self.make_api_request('/search/', params)

            if response and 'results' in response:
                results = response['results']
                logger.info(f"  Found {len(results)} results")

                # Look for matching case
                for result in results:
                    if self.case_matches_api(case_name, citation, result):
                        logger.info(f"✓ Found matching case: {result.get('caseName', 'Unknown')}")
                        logger.info(f"Result fields: {list(result.keys())}")
                        logger.info(f"Opinion ID: {result.get('id')}")
                        logger.info(f"Resource URL: {result.get('resource')}")
                        logger.info(f"PDF URL: {result.get('pdf_url')}")
                        return result

            time.sleep(1)  # Small delay between queries

        logger.info(f"✗ No matching case found for: {case_name}")
        return None

    def case_matches_api(self, case_name: str, citation: str, result: Dict[str, Any]) -> bool:
        """Check if an API result matches our case."""
        result_name = result.get('caseName', '').lower()
        case_name_lower = case_name.lower()

        # Extract key words from case name (remove common words)
        common_words = {'v', 'vs', 'versus', 'in', 're', 'ex', 'parte', 'the', 'of', 'and', 'or', 'for', 'to', 'a', 'an'}
        case_words = set(case_name_lower.split())
        case_words = case_words - common_words

        result_words = set(result_name.split())

        # Check for significant overlap
        if len(case_words) >= 2:
            common_words = case_words.intersection(result_words)
            if len(common_words) >= 2:
                logger.info(f"✓ Match found: {len(common_words)} common words: {common_words}")
                return True

        # Check for citation match
        if citation and citation.strip():
            citation_lower = citation.lower()
            if citation_lower in result_name:
                logger.info(f"✓ Citation match found: {citation}")
                return True

        return False

    def get_pdf_urls_from_opinion(self, opinion_id: int) -> List[str]:
        """Get PDF URLs from an opinion."""
        logger.info(f"📄 Getting PDFs for opinion ID: {opinion_id}")

        # Get opinion details
        response = self.make_api_request(f'/opinions/{opinion_id}/')

        if not response:
            logger.warning(f"No response for opinion {opinion_id}")
            return []

        logger.info(f"Opinion response fields: {list(response.keys())}")

        pdf_urls = []

        # Check for PDF URL in the opinion
        pdf_url = response.get('pdf_url') or response.get('download_url')
        if pdf_url:
            pdf_urls.append(pdf_url)
            logger.info(f"✓ Found PDF URL: {pdf_url}")
        else:
            logger.info("No pdf_url or download_url field found")

        # Check for resource URL
        resource_url = response.get('resource')
        if resource_url and resource_url.endswith('.pdf'):
            pdf_urls.append(resource_url)
            logger.info(f"✓ Found resource PDF: {resource_url}")
        elif resource_url:
            logger.info(f"Resource URL found but not PDF: {resource_url}")
        else:
            logger.info("No resource field found")

        # Check for absolute URL
        absolute_url = response.get('absolute_url')
        if absolute_url:
            logger.info(f"Absolute URL: {absolute_url}")

        logger.info(f"Total PDF URLs found: {len(pdf_urls)}")
        return pdf_urls

    def download_pdf(self, pdf_url: str, filename: str) -> Optional[Path]:
        """Download a PDF file."""
        try:
            logger.info(f"📥 Downloading PDF: {filename}")

            response = requests.get(pdf_url, timeout=60)

            if response.status_code == 200:
                save_path = self.pdf_dir / filename

                # Skip if already downloaded
                if save_path.exists():
                    logger.info(f"  - PDF already exists: {filename}")
                    return save_path

                # Save the file
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                # Check for duplicates
                with open(save_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in self.downloaded_hashes:
                    logger.info(f"  ✓ Duplicate detected, removing: {filename}")
                    save_path.unlink()
                    self.stats['duplicates_skipped'] += 1
                    return None
                else:
                    self.downloaded_hashes.add(file_hash)
                    self.stats['total_pdfs_downloaded'] += 1
                    self.stats['free_pdfs_found'] += 1
                    logger.info(f"  ✓ Downloaded PDF: {filename}")
                    return save_path
            else:
                logger.warning(f"  ✗ Failed to download {filename}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"  ✗ Error downloading {filename}: {e}")
            return None

    def update_database_with_pdfs(self, case_info: Dict[str, Any], pdf_paths: List[Path]):
        """Update database with PDF information."""
        if not pdf_paths:
            return

        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()

            # Update case with PDF information
            pdf_urls = [str(pdf_path) for pdf_path in pdf_paths]
            pdf_urls_json = json.dumps(pdf_urls)

            update_query = """
                UPDATE case_law
                SET download_url = %s
                WHERE id = %s
            """

            cursor.execute(update_query, (
                pdf_urls_json,
                case_info['id']
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"✓ Updated database for case {case_info['case_name']} with {len(pdf_paths)} PDFs")

        except Exception as e:
            logger.error(f"Error updating database for case {case_info['case_name']}: {e}")

    def process_cases_batch(self, cases: List[Dict[str, Any]]):
        """Process a batch of cases for PDF downloads."""
        logger.info(f"🔄 Processing batch of {len(cases)} cases")

        for i, case_info in enumerate(cases):
            case_num = i + 1
            logger.info(f"\n📋 Processing case {case_num}/{len(cases)}: {case_info['case_name']}")

            try:
                # Search for the case using API
                opinion_data = self.search_courtlistener_api(case_info)

                if opinion_data:
                    # Check if this is a cluster with opinions
                    opinions = opinion_data.get('opinions', [])
                    logger.info(f"Found {len(opinions)} opinions in cluster")

                    if opinions:
                        downloaded_pdfs = []

                        # Process each opinion in the cluster
                        for i, opinion in enumerate(opinions):
                            opinion_id = opinion.get('id')
                            logger.info(f"Processing opinion {i+1}: ID {opinion_id}")

                            if opinion_id:
                                pdf_urls = self.get_pdf_urls_from_opinion(opinion_id)

                                for j, pdf_url in enumerate(pdf_urls):
                                    filename = f"{case_info['case_name'].replace(' ', '_')}_opinion_{i+1}_{j+1}.pdf"
                                    pdf_path = self.download_pdf(pdf_url, filename)

                                    if pdf_path:
                                        downloaded_pdfs.append(pdf_path)

                        if downloaded_pdfs:
                            # Update database
                            self.update_database_with_pdfs(case_info, downloaded_pdfs)
                            self.stats['cases_with_pdfs'] += 1
                            logger.info(f"✓ Downloaded {len(downloaded_pdfs)} PDFs for {case_info['case_name']}")
                        else:
                            self.stats['cases_without_pdfs'] += 1
                            logger.info(f"✗ No PDFs downloaded for {case_info['case_name']}")
                    else:
                        self.stats['cases_without_pdfs'] += 1
                        logger.info(f"✗ No opinions found in cluster for {case_info['case_name']}")
                else:
                    self.stats['cases_without_pdfs'] += 1
                    logger.info(f"✗ Case not found on CourtListener: {case_info['case_name']}")

                self.stats['total_cases_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing case {case_info['case_name']}: {e}")
                self.stats['total_errors'] += 1

    def run_pdf_download_session(self, topic: str = None, limit: int = 50):
        """Run a complete PDF download session."""
        logger.info("🚀 CourtListener API PDF Downloader")
        logger.info("=" * 70)
        logger.info("🎯 Using CourtListener API for PDF downloads")
        logger.info("📊 Integrating with existing case law database")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Get cases from database
        cases = self.get_cases_from_database(limit=limit, topic=topic)

        if not cases:
            logger.info("No cases found in database that need PDF downloads")
            return

        logger.info(f"Found {len(cases)} cases to process")

        # Process cases in batches
        batch_size = 5
        for i in range(0, len(cases), batch_size):
            batch_cases = cases[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"\n🔄 Processing Batch {batch_num}: {len(batch_cases)} cases")

            self.process_cases_batch(batch_cases)

            # Save progress after each batch
            self.save_progress()

            # Pause between batches
            if i + batch_size < len(cases):
                pause_time = 10
                logger.info(f"⏸️  Pause between batches: {pause_time} seconds")
                time.sleep(pause_time)

        # Generate final report
        self.generate_summary_report()

        logger.info(f"\n✅ PDF download session complete!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   Cases processed: {self.stats['total_cases_processed']}")
        logger.info(f"   Cases with PDFs: {self.stats['cases_with_pdfs']}")
        logger.info(f"   Cases without PDFs: {self.stats['cases_without_pdfs']}")
        logger.info(f"   FREE PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
        logger.info(f"   PAID PDFs skipped: {self.stats['paid_pdfs_skipped']}")
        logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"   Errors: {self.stats['total_errors']}")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/api_pdf_download_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'download_date': datetime.now().isoformat(),
                'stats': self.stats,
                'pdf_directory': str(self.pdf_dir.absolute())
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/api_pdf_download_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# CourtListener API PDF Download Report\n\n")
            f.write(f"**Date**: {self.stats['last_run']}\n")
            f.write(f"**Cases Processed**: {self.stats['total_cases_processed']}\n")
            f.write(f"**Cases with PDFs**: {self.stats['cases_with_pdfs']}\n")
            f.write(f"**Cases without PDFs**: {self.stats['cases_without_pdfs']}\n")
            f.write(f"**FREE PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\n")
            f.write(f"**PAID PDFs Skipped**: {self.stats['paid_pdfs_skipped']}\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\n\n")

            f.write("## Strategy\n")
            f.write("- **API Integration**: CourtListener REST API v3\n")
            f.write("- **Database Integration**: MySQL case_law table\n")
            f.write("- **Target**: PDFs available through API\n")
            f.write("- **Rate Limiting**: 1 second between requests\n\n")

            f.write("## PDF Storage\n")
            f.write(f"PDFs are stored in: `{self.pdf_dir.absolute()}`\n")
            f.write("Database is updated with PDF paths and metadata.\n\n")

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Download PDFs from CourtListener API")
    parser.add_argument("--topic", help="Topic to process (e.g., 1782_discovery)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum cases to process")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CourtListener API PDF Downloader")
    print("="*60 + "\n")

    downloader = CourtListenerAPIPDFDownloader()
    downloader.stats['last_run'] = datetime.now().isoformat()

    downloader.run_pdf_download_session(
        topic=args.topic,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
