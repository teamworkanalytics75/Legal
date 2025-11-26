#!/usr/bin/env python3
"""
CourtListener RECAP Petition Retriever

Accesses CourtListener RECAP to download original §1782 petitions and filings.
RECAP mirrors PACER documents and makes them freely available.

Usage: python scripts/recap_petition_retriever.py
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

class RECAPPetitionRetriever:
    def __init__(self):
        self.api_base_url = "https://www.courtlistener.com/api/rest/v4"
        self.recap_base_url = "https://www.courtlistener.com/api/rest/v4/recap"
        self.dockets_base_url = "https://www.courtlistener.com/api/rest/v4/dockets"

        # API token (use existing token from config)
        self.api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"

        # Output directories
        self.petitions_raw_dir = "data/petitions_raw"
        self.petitions_text_dir = "data/petitions_text"
        self.recap_log_file = "data/case_law/recap_retrieval_log.json"

        # Create directories
        os.makedirs(self.petitions_raw_dir, exist_ok=True)
        os.makedirs(self.petitions_text_dir, exist_ok=True)

        self.retrieval_log = {
            'retrieval_date': datetime.now().isoformat(),
            'api_token_available': bool(self.api_token),
            'total_dockets_searched': 0,
            'dockets_with_recap_docs': 0,
            'petitions_downloaded': 0,
            'failed_downloads': 0,
            'retrieval_details': []
        }

    def setup_authentication(self) -> bool:
        """Setup API authentication."""
        if not self.api_token:
            logger.warning("No API token found. Set COURTLISTENER_API_TOKEN environment variable.")
            logger.info("To get an API token:")
            logger.info("1. Go to https://www.courtlistener.com/api/")
            logger.info("2. Create an account and request an API token")
            logger.info("3. Set environment variable: export COURTLISTENER_API_TOKEN=your_token")
            return False

        self.headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }

        logger.info("✓ API authentication configured")
        return True

    def test_api_access(self) -> bool:
        """Test API access."""
        logger.info("Testing CourtListener API access...")

        try:
            # Test basic API access
            test_url = f"{self.api_base_url}/dockets/"
            response = requests.get(test_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                logger.info("✓ API access successful")
                return True
            elif response.status_code == 401:
                logger.error("❌ API authentication failed - check your token")
                return False
            else:
                logger.warning(f"⚠️ API returned status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False

    def search_1782_dockets(self) -> List[Dict[str, Any]]:
        """Search for §1782 dockets in CourtListener."""
        logger.info("Searching for §1782 dockets...")

        dockets = []

        # Search queries for §1782 cases
        search_queries = [
            "1782 application",
            "28 U.S.C. 1782",
            "section 1782",
            "foreign discovery",
            "international judicial assistance"
        ]

        for query in search_queries:
            logger.info(f"Searching for: {query}")

            try:
                params = {
                    'q': query,
                    'format': 'json',
                    'order_by': 'dateFiled desc',
                    'stat_Precedential': 'on'
                }

                response = requests.get(self.dockets_base_url, headers=self.headers, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])

                    logger.info(f"Found {len(results)} dockets for query: {query}")

                    for docket in results:
                        # Check if this is a §1782 case
                        if self.is_1782_docket(docket):
                            dockets.append(docket)
                        else:
                            # Log why it didn't qualify
                            logger.info(f"Skipped docket {docket.get('id')}: {docket.get('case_name', 'Unknown')}")

                else:
                    logger.warning(f"Search failed for '{query}': {response.status_code}")

            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")

            # Be respectful to the API
            time.sleep(2)

        # Remove duplicates
        unique_dockets = []
        seen_ids = set()

        for docket in dockets:
            docket_id = docket.get('id')
            if docket_id and docket_id not in seen_ids:
                unique_dockets.append(docket)
                seen_ids.add(docket_id)

        logger.info(f"✓ Found {len(unique_dockets)} unique §1782 dockets")
        return unique_dockets

    def is_1782_docket(self, docket: Dict[str, Any]) -> bool:
        """Check if a docket is related to §1782."""
        # Check docket text for §1782 indicators
        docket_text = str(docket.get('docket_text', '')).lower()
        case_name = str(docket.get('case_name', '')).lower()

        # Look for §1782 indicators (relaxed)
        indicators = [
            '1782', 'section 1782', '28 u.s.c. 1782',
            'foreign discovery', 'international judicial assistance',
            'application for', 'motion for discovery', 'in re'
        ]

        # More lenient check - if it contains any indicator, include it
        has_indicator = any(indicator in docket_text or indicator in case_name for indicator in indicators)

        if has_indicator:
            logger.info(f"✓ Qualifying docket: {docket.get('case_name', 'Unknown')} (ID: {docket.get('id')})")

        return has_indicator

    def get_recap_documents(self, docket_id: int) -> List[Dict[str, Any]]:
        """Get RECAP documents for a docket."""
        logger.info(f"Getting RECAP documents for docket {docket_id}")

        try:
            # Get docket details
            docket_url = f"{self.dockets_base_url}/{docket_id}/"
            response = requests.get(docket_url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                docket_data = response.json()
                recap_docs = docket_data.get('recap_documents', [])

                logger.info(f"Found {len(recap_docs)} RECAP documents for docket {docket_id}")
                return recap_docs
            else:
                logger.warning(f"Failed to get docket {docket_id}: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting RECAP documents for docket {docket_id}: {e}")
            return []

    def download_recap_document(self, doc: Dict[str, Any]) -> Optional[str]:
        """Download a RECAP document PDF."""
        doc_id = doc.get('id')
        doc_type = doc.get('document_type')
        attachment_number = doc.get('attachment_number')

        logger.info(f"Downloading RECAP document {doc_id} (type: {doc_type})")

        try:
            # Get document details
            doc_url = f"{self.recap_base_url}/{doc_id}/"
            response = requests.get(doc_url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                doc_data = response.json()

                # Check if it's a petition or motion
                if self.is_petition_document(doc_data):
                    # Download PDF
                    pdf_url = doc_data.get('filepath_local')
                    if pdf_url:
                        pdf_response = requests.get(pdf_url, timeout=60)

                        if pdf_response.status_code == 200:
                            # Save PDF
                            filename = f"recap_{doc_id}_{doc_type}.pdf"
                            filepath = os.path.join(self.petitions_raw_dir, filename)

                            with open(filepath, 'wb') as f:
                                f.write(pdf_response.content)

                            logger.info(f"✓ Downloaded PDF: {filename}")
                            return filepath
                        else:
                            logger.warning(f"Failed to download PDF: {pdf_response.status_code}")
                    else:
                        logger.warning(f"No PDF URL for document {doc_id}")
                else:
                    logger.info(f"Skipping non-petition document {doc_id}")

            else:
                logger.warning(f"Failed to get document {doc_id}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error downloading document {doc_id}: {e}")

        return None

    def is_petition_document(self, doc_data: Dict[str, Any]) -> bool:
        """Check if a document is a petition or motion."""
        doc_type = str(doc_data.get('document_type', '')).lower()
        description = str(doc_data.get('description', '')).lower()

        # Look for petition-related document types
        petition_types = [
            'motion', 'application', 'petition', 'request',
            'memorandum', 'brief', 'declaration'
        ]

        # Look for §1782 related content
        content_indicators = [
            '1782', 'foreign discovery', 'international judicial assistance',
            'application for', 'motion for discovery'
        ]

        is_petition_type = any(ptype in doc_type for ptype in petition_types)
        has_1782_content = any(indicator in description for indicator in content_indicators)

        return is_petition_type or has_1782_content

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfminer.six."""
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            return text
        except ImportError:
            logger.warning("pdfminer.six not installed. Install with: pip install pdfminer.six")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def save_petition_text(self, pdf_path: str, text: str, doc_info: Dict[str, Any]):
        """Save extracted petition text."""
        if not text.strip():
            return

        # Create text filename
        pdf_filename = os.path.basename(pdf_path)
        text_filename = pdf_filename.replace('.pdf', '.txt')
        text_path = os.path.join(self.petitions_text_dir, text_filename)

        # Save text with metadata
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"RECAP Document ID: {doc_info.get('id', 'unknown')}\n")
            f.write(f"Document Type: {doc_info.get('document_type', 'unknown')}\n")
            f.write(f"Description: {doc_info.get('description', 'unknown')}\n")
            f.write(f"Date Filed: {doc_info.get('date_filed', 'unknown')}\n")
            f.write("="*80 + "\n\n")
            f.write(text)

        logger.info(f"✓ Saved text: {text_filename}")

    def run_recap_retrieval(self):
        """Run the complete RECAP retrieval process."""
        logger.info("🚀 Starting RECAP Petition Retrieval")
        logger.info("="*80)

        # Setup authentication
        if not self.setup_authentication():
            logger.error("Cannot proceed without API authentication")
            return

        # Test API access
        if not self.test_api_access():
            logger.error("API access test failed")
            return

        # Search for §1782 dockets
        dockets = self.search_1782_dockets()
        self.retrieval_log['total_dockets_searched'] = len(dockets)

        if not dockets:
            logger.warning("No §1782 dockets found")
            return

        # Process each docket
        for docket in dockets:
            docket_id = docket.get('id')
            case_name = docket.get('case_name', 'Unknown')

            logger.info(f"Processing docket {docket_id}: {case_name}")

            # Get RECAP documents
            recap_docs = self.get_recap_documents(docket_id)

            if recap_docs:
                self.retrieval_log['dockets_with_recap_docs'] += 1

                # Download and process each document
                for doc in recap_docs:
                    pdf_path = self.download_recap_document(doc)

                    if pdf_path:
                        self.retrieval_log['petitions_downloaded'] += 1

                        # Extract text
                        text = self.extract_text_from_pdf(pdf_path)

                        if text:
                            self.save_petition_text(pdf_path, text, doc)
                        else:
                            logger.warning(f"No text extracted from {pdf_path}")
                    else:
                        self.retrieval_log['failed_downloads'] += 1

            # Be respectful to the API
            time.sleep(3)

        # Save retrieval log
        with open(self.recap_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.retrieval_log, f, indent=2, ensure_ascii=False)

        logger.info("🎉 RECAP retrieval complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Dockets searched: {self.retrieval_log['total_dockets_searched']}")
        print(f"  Dockets with RECAP docs: {self.retrieval_log['dockets_with_recap_docs']}")
        print(f"  Petitions downloaded: {self.retrieval_log['petitions_downloaded']}")
        print(f"  Failed downloads: {self.retrieval_log['failed_downloads']}")

def main():
    """Main function."""
    print("📥 CourtListener RECAP Petition Retriever")
    print("="*80)

    retriever = RECAPPetitionRetriever()
    retriever.run_recap_retrieval()

    print("\n✅ RECAP retrieval complete!")
    print("Check data/petitions_raw/ and data/petitions_text/ for results.")

if __name__ == "__main__":
    main()
