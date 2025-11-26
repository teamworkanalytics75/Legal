#!/usr/bin/env python3
"""
Enhanced Text Extraction for Wishlist Cases

This script extracts text from all wishlist cases we found but haven't fully analyzed yet,
then updates our predictive model with the additional data.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
import re

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WishlistTextExtractor:
    """Extract text from wishlist cases and update predictive model."""

    def __init__(self):
        self.client = CourtListenerClient()
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.extraction_log = {
            "extraction_date": datetime.now().isoformat(),
            "wishlist_cases_processed": 0,
            "wishlist_cases_with_text": 0,
            "wishlist_cases_failed": 0,
            "cases": []
        }

    def load_wishlist_cases(self) -> List[Dict]:
        """Load all wishlist cases that were found."""
        try:
            with open("data/case_law/wishlist_acquisition_log.json", 'r', encoding='utf-8') as f:
                log_data = json.load(f)

            found_cases = [case for case in log_data['cases'] if case['status'] == 'found']
            logger.info(f"Found {len(found_cases)} wishlist cases to process")
            return found_cases

        except Exception as e:
            logger.error(f"Error loading wishlist cases: {e}")
            return []

    def fetch_opinion_text(self, opinion_id: int) -> Optional[str]:
        """Fetch full text for an opinion by ID."""
        try:
            logger.info(f"Fetching text for opinion {opinion_id}")

            # Use CourtListener API to get opinion text
            response = self.client.session.get(
                f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/",
                timeout=30
            )

            if response.status_code == 200:
                opinion_data = response.json()

                # Try to get text from various fields
                text = opinion_data.get('text')
                if not text:
                    text = opinion_data.get('plain_text')
                if not text:
                    text = opinion_data.get('html')
                    if text:
                        # Simple HTML tag removal
                        text = re.sub(r'<[^>]+>', '', text)

                if text and len(text.strip()) > 100:
                    logger.info(f"✓ Retrieved {len(text)} characters of text")
                    return text
                else:
                    logger.warning(f"✗ No substantial text found for opinion {opinion_id}")
                    return None
            else:
                logger.warning(f"✗ Failed to fetch opinion {opinion_id}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"✗ Error fetching opinion {opinion_id}: {e}")
            return None

    def download_pdf(self, pdf_url: str, case_name: str) -> Optional[Path]:
        """Download PDF from URL."""
        try:
            logger.info(f"Downloading PDF: {pdf_url}")

            response = self.client.session.get(pdf_url, timeout=30)
            if response.status_code == 200:
                # Create safe filename
                safe_name = re.sub(r'[^\w\-_\.]', '_', case_name)[:100]
                pdf_path = Path("data/case_law/pdfs") / f"{safe_name}.pdf"
                pdf_path.parent.mkdir(exist_ok=True)

                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✓ PDF saved to: {pdf_path}")
                return pdf_path
            else:
                logger.warning(f"✗ Failed to download PDF: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"✗ Error downloading PDF: {e}")
            return None

    def extract_wishlist_case_text(self, case_info: Dict) -> Dict[str, Any]:
        """Extract text from a wishlist case."""
        result = {
            "wishlist_name": case_info['wishlist_name'],
            "citation": case_info['citation'],
            "cluster_id": case_info['cluster_id'],
            "status": "failed",
            "text_extracted": False,
            "pdf_downloaded": False,
            "text_length": 0,
            "pdf_path": None,
            "notes": ""
        }

        try:
            # Search for the case in our corpus
            case_file = None
            for file_path in self.corpus_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)

                    if case_data.get('cluster_id') == case_info['cluster_id']:
                        case_file = file_path
                        break
                except:
                    continue

            if not case_file:
                # Case not in corpus yet, fetch it
                logger.info(f"Case {case_info['wishlist_name']} not in corpus, fetching...")

                # Search for the case using CourtListener
                search_results = self.client.search_opinions(
                    courts=[],
                    keywords=[case_info['wishlist_name']],
                    date_start=None,
                    date_end=None,
                    page_size=10
                )

                if search_results and len(search_results) > 0:
                    # Find the matching case by cluster_id
                    matching_case = None
                    for case in search_results:
                        if case.get('cluster_id') == case_info['cluster_id']:
                            matching_case = case
                            break

                    if matching_case:
                        # Save the case
                        safe_name = re.sub(r'[^\w\-_\.]', '_', case_info['wishlist_name'])[:100]
                        case_file = self.corpus_dir / f"{case_info['cluster_id']}_{safe_name}.json"

                        with open(case_file, 'w', encoding='utf-8') as f:
                            json.dump(matching_case, f, indent=2, ensure_ascii=False)

                        logger.info(f"✓ Case saved to: {case_file}")
                    else:
                        result["notes"] = "Case not found in search results"
                        return result
                else:
                    result["notes"] = "No search results found"
                    return result

            # Extract text from the case
            with open(case_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)

            opinions = case_data.get('opinions', [])
            if not opinions:
                result["notes"] = "No opinions found"
                return result

            # Extract text from opinions
            extracted_text = ""
            for opinion in opinions:
                opinion_id = opinion.get('id')
                if opinion_id:
                    text = self.fetch_opinion_text(opinion_id)
                    if text:
                        extracted_text += text + "\n\n"
                        result["text_extracted"] = True

                # Download PDF if available
                pdf_url = opinion.get('download_url')
                if pdf_url:
                    pdf_path = self.download_pdf(pdf_url, case_info['wishlist_name'])
                    if pdf_path:
                        result["pdf_downloaded"] = True
                        result["pdf_path"] = str(pdf_path)

            # Update case data with extracted text
            if extracted_text:
                case_data['extracted_text'] = extracted_text
                case_data['text_extraction_date'] = datetime.now().isoformat()

                # Save updated case data
                with open(case_file, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, indent=2, ensure_ascii=False)

                result["text_length"] = len(extracted_text)
                result["status"] = "success"
                result["notes"] = f"Extracted {len(extracted_text)} characters"
            else:
                result["status"] = "no_text"
                result["notes"] = "No text content available"

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing {case_info['wishlist_name']}: {e}")
            result["notes"] = f"Error: {e}"

        return result

    def extract_all_wishlist_cases(self) -> None:
        """Extract text from all wishlist cases."""
        logger.info("="*80)
        logger.info("STARTING WISHLIST TEXT EXTRACTION")
        logger.info("="*80)

        wishlist_cases = self.load_wishlist_cases()
        if not wishlist_cases:
            logger.error("No wishlist cases found to process")
            return

        logger.info(f"Processing {len(wishlist_cases)} wishlist cases...")

        for i, case_info in enumerate(wishlist_cases, 1):
            logger.info(f"\nProgress: {i}/{len(wishlist_cases)}")
            logger.info(f"Processing: {case_info['wishlist_name']}")

            result = self.extract_wishlist_case_text(case_info)
            self.extraction_log["cases"].append(result)

            if result["status"] == "success":
                self.extraction_log["wishlist_cases_processed"] += 1
                if result["text_extracted"]:
                    self.extraction_log["wishlist_cases_with_text"] += 1
            else:
                self.extraction_log["wishlist_cases_failed"] += 1

            # Log progress every 5 cases
            if i % 5 == 0:
                logger.info(f"Processed {i}/{len(wishlist_cases)} cases")

        # Save extraction log
        log_path = Path("data/case_law/wishlist_text_extraction_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_log, f, indent=2, ensure_ascii=False)

        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("WISHLIST EXTRACTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total wishlist cases processed: {self.extraction_log['wishlist_cases_processed']}")
        logger.info(f"Wishlist cases with text extracted: {self.extraction_log['wishlist_cases_with_text']}")
        logger.info(f"Wishlist cases failed: {self.extraction_log['wishlist_cases_failed']}")
        logger.info(f"Extraction log saved to: {log_path}")

    def generate_wishlist_report(self) -> None:
        """Generate a report on wishlist text extraction."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING WISHLIST EXTRACTION REPORT")
        logger.info("="*80)

        # Analyze results
        successful_cases = [c for c in self.extraction_log["cases"] if c["status"] == "success"]
        failed_cases = [c for c in self.extraction_log["cases"] if c["status"] == "failed"]
        no_text_cases = [c for c in self.extraction_log["cases"] if c["status"] == "no_text"]

        # Calculate statistics
        total_text_length = sum(c["text_length"] for c in successful_cases)
        avg_text_length = total_text_length / len(successful_cases) if successful_cases else 0

        # Generate report
        report_content = f"""# 📄 Wishlist Text Extraction Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Extraction Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Wishlist Cases** | {len(self.extraction_log['cases'])} | 100% |
| **Successfully Processed** | {self.extraction_log['wishlist_cases_processed']} | {self.extraction_log['wishlist_cases_processed']/len(self.extraction_log['cases'])*100:.1f}% |
| **Text Extracted** | {self.extraction_log['wishlist_cases_with_text']} | {self.extraction_log['wishlist_cases_with_text']/len(self.extraction_log['cases'])*100:.1f}% |
| **Failed** | {self.extraction_log['wishlist_cases_failed']} | {self.extraction_log['wishlist_cases_failed']/len(self.extraction_log['cases'])*100:.1f}% |

## 📈 Text Statistics

- **Total Text Extracted**: {total_text_length:,} characters
- **Average Text Length**: {avg_text_length:.0f} characters per case
- **Cases with Text**: {len(successful_cases)} cases
- **Cases without Text**: {len(no_text_cases)} cases

## ✅ Successfully Processed Cases ({len(successful_cases)})

"""

        for case in successful_cases[:20]:  # Show first 20
            report_content += f"- **{case['wishlist_name']}**: {case['text_length']:,} chars"
            if case['pdf_downloaded']:
                report_content += f" + PDF"
            report_content += f"\n"

        if len(successful_cases) > 20:
            report_content += f"- ... and {len(successful_cases) - 20} more cases\n"

        report_content += f"""
## ❌ Failed Cases ({len(failed_cases)})

"""

        for case in failed_cases[:10]:  # Show first 10
            report_content += f"- **{case['wishlist_name']}**: {case['notes']}\n"

        if len(failed_cases) > 10:
            report_content += f"- ... and {len(failed_cases) - 10} more cases\n"

        report_content += f"""
## 📋 Next Steps

1. **Run Mathematical Analysis**: Analyze the new text data for patterns
2. **Update Predictive Model**: Retrain ML models with expanded dataset
3. **Extract Court Outcomes**: Determine actual outcomes for new cases
4. **Generate Updated Reports**: Create comprehensive analysis reports

---

**This report shows the results of extracting text from wishlist cases to expand our predictive model dataset.**
"""

        # Save report
        report_path = Path("data/case_law/wishlist_text_extraction_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✓ Wishlist extraction report saved to: {report_path}")


def main():
    """Main entry point."""
    logger.info("Starting wishlist text extraction...")

    extractor = WishlistTextExtractor()
    extractor.extract_all_wishlist_cases()
    extractor.generate_wishlist_report()

    logger.info("\n🎉 Wishlist text extraction completed!")
    logger.info("Next steps:")
    logger.info("1. Run mathematical analysis on new cases")
    logger.info("2. Update predictive model with expanded dataset")
    logger.info("3. Extract court outcomes for new cases")


if __name__ == "__main__":
    main()
