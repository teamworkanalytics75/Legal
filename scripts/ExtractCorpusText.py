#!/usr/bin/env python3
"""
PDF/Text Extraction System for §1782 Caselaw Corpus

This script extracts full text content and PDFs for all cases in our corpus,
providing comprehensive text analysis capabilities.
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


class CorpusTextExtractor:
    """Extract text content and PDFs for all cases in the corpus."""

    def __init__(self):
        self.client = CourtListenerClient()
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.extraction_log = {
            "extraction_date": datetime.now().isoformat(),
            "total_cases": 0,
            "cases_with_text": 0,
            "cases_with_pdfs": 0,
            "cases_processed": 0,
            "cases_failed": 0,
            "cases": []
        }

    def analyze_corpus(self) -> Dict[str, Any]:
        """Analyze the current corpus to understand text availability."""
        logger.info("="*80)
        logger.info("ANALYZING CORPUS TEXT AVAILABILITY")
        logger.info("="*80)

        case_files = list(self.corpus_dir.glob("*.json"))
        logger.info(f"Found {len(case_files)} case files")

        analysis = {
            "total_cases": len(case_files),
            "cases_with_opinions": 0,
            "cases_with_text": 0,
            "cases_with_pdf_urls": 0,
            "cases_with_cluster_ids": 0,
            "text_lengths": [],
            "missing_text_cases": []
        }

        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                # Check for cluster ID
                if case_data.get('cluster_id'):
                    analysis["cases_with_cluster_ids"] += 1

                # Check for opinions
                opinions = case_data.get('opinions', [])
                if opinions:
                    analysis["cases_with_opinions"] += 1

                    # Check for text content
                    has_text = False
                    for opinion in opinions:
                        if opinion.get('text') or opinion.get('plain_text'):
                            has_text = True
                            text_length = len(opinion.get('text', '') or opinion.get('plain_text', ''))
                            analysis["text_lengths"].append(text_length)
                            break

                    if has_text:
                        analysis["cases_with_text"] += 1
                    else:
                        analysis["missing_text_cases"].append({
                            "file": case_file.name,
                            "cluster_id": case_data.get('cluster_id'),
                            "case_name": case_data.get('caseName', 'Unknown')
                        })

                    # Check for PDF URLs
                    for opinion in opinions:
                        if opinion.get('download_url'):
                            analysis["cases_with_pdf_urls"] += 1
                            break

            except Exception as e:
                logger.error(f"Error analyzing {case_file.name}: {e}")

        # Print analysis results
        logger.info(f"Total cases: {analysis['total_cases']}")
        logger.info(f"Cases with cluster IDs: {analysis['cases_with_cluster_ids']}")
        logger.info(f"Cases with opinions: {analysis['cases_with_opinions']}")
        logger.info(f"Cases with text: {analysis['cases_with_text']}")
        logger.info(f"Cases with PDF URLs: {analysis['cases_with_pdf_urls']}")
        logger.info(f"Cases missing text: {len(analysis['missing_text_cases'])}")

        if analysis['text_lengths']:
            avg_length = sum(analysis['text_lengths']) / len(analysis['text_lengths'])
            logger.info(f"Average text length: {avg_length:.0f} characters")

        return analysis

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
                    logger.warning(f"✗ Insufficient text content for opinion {opinion_id}")
                    return None
            else:
                logger.warning(f"✗ Failed to fetch opinion {opinion_id}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"✗ Error fetching opinion {opinion_id}: {e}")
            return None

    def download_pdf(self, pdf_url: str, case_name: str) -> Optional[Path]:
        """Download PDF for a case."""
        try:
            logger.info(f"Downloading PDF: {pdf_url}")

            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                # Create safe filename
                safe_name = re.sub(r'[^\w\s-]', '', case_name)
                safe_name = re.sub(r'[-\s]+', '_', safe_name)
                pdf_filename = f"{safe_name}.pdf"
                pdf_path = self.corpus_dir / "pdfs" / pdf_filename

                # Create pdfs directory if it doesn't exist
                pdf_path.parent.mkdir(exist_ok=True)

                # Save PDF
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

    def extract_case_text(self, case_file: Path) -> Dict[str, Any]:
        """Extract text and PDF for a single case."""
        logger.info(f"Processing: {case_file.name}")

        result = {
            "file": case_file.name,
            "status": "failed",
            "text_extracted": False,
            "pdf_downloaded": False,
            "text_length": 0,
            "pdf_path": None,
            "notes": ""
        }

        try:
            with open(case_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)

            case_name = case_data.get('caseName', 'Unknown')
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
                    pdf_path = self.download_pdf(pdf_url, case_name)
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
            logger.error(f"Error processing {case_file.name}: {e}")
            result["notes"] = f"Error: {e}"

        return result

    def extract_all_cases(self) -> None:
        """Extract text and PDFs for all cases in the corpus."""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE TEXT EXTRACTION")
        logger.info("="*80)

        case_files = list(self.corpus_dir.glob("*.json"))
        self.extraction_log["total_cases"] = len(case_files)

        logger.info(f"Processing {len(case_files)} cases...")

        for i, case_file in enumerate(case_files, 1):
            logger.info(f"\nProgress: {i}/{len(case_files)}")

            result = self.extract_case_text(case_file)
            self.extraction_log["cases"].append(result)

            if result["status"] == "success":
                self.extraction_log["cases_processed"] += 1
                if result["text_extracted"]:
                    self.extraction_log["cases_with_text"] += 1
                if result["pdf_downloaded"]:
                    self.extraction_log["cases_with_pdfs"] += 1
            else:
                self.extraction_log["cases_failed"] += 1

            # Log progress every 10 cases
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(case_files)} cases")

        # Save extraction log
        log_path = Path("data/case_law/text_extraction_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_log, f, indent=2, ensure_ascii=False)

        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total cases processed: {self.extraction_log['cases_processed']}")
        logger.info(f"Cases with text extracted: {self.extraction_log['cases_with_text']}")
        logger.info(f"Cases with PDFs downloaded: {self.extraction_log['cases_with_pdfs']}")
        logger.info(f"Cases failed: {self.extraction_log['cases_failed']}")
        logger.info(f"Extraction log saved to: {log_path}")

    def generate_text_report(self) -> None:
        """Generate a comprehensive report on text extraction results."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING TEXT EXTRACTION REPORT")
        logger.info("="*80)

        # Analyze results
        successful_cases = [c for c in self.extraction_log["cases"] if c["status"] == "success"]
        failed_cases = [c for c in self.extraction_log["cases"] if c["status"] == "failed"]
        no_text_cases = [c for c in self.extraction_log["cases"] if c["status"] == "no_text"]

        # Calculate statistics
        total_text_length = sum(c["text_length"] for c in successful_cases)
        avg_text_length = total_text_length / len(successful_cases) if successful_cases else 0

        # Generate report
        report_content = f"""# 📄 Text Extraction Report for §1782 Caselaw Corpus

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Extraction Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Cases** | {self.extraction_log['total_cases']} | 100% |
| **Successfully Processed** | {self.extraction_log['cases_processed']} | {self.extraction_log['cases_processed']/self.extraction_log['total_cases']*100:.1f}% |
| **Text Extracted** | {self.extraction_log['cases_with_text']} | {self.extraction_log['cases_with_text']/self.extraction_log['total_cases']*100:.1f}% |
| **PDFs Downloaded** | {self.extraction_log['cases_with_pdfs']} | {self.extraction_log['cases_with_pdfs']/self.extraction_log['total_cases']*100:.1f}% |
| **Failed** | {self.extraction_log['cases_failed']} | {self.extraction_log['cases_failed']/self.extraction_log['total_cases']*100:.1f}% |

## 📈 Text Statistics

- **Total Text Extracted**: {total_text_length:,} characters
- **Average Text Length**: {avg_text_length:.0f} characters per case
- **Cases with Text**: {len(successful_cases)} cases
- **Cases without Text**: {len(no_text_cases)} cases

## ✅ Successfully Processed Cases ({len(successful_cases)})

"""

        for case in successful_cases[:20]:  # Show first 20
            report_content += f"- **{case['file']}**: {case['text_length']:,} chars"
            if case['pdf_downloaded']:
                report_content += f" + PDF"
            report_content += f"\n"

        if len(successful_cases) > 20:
            report_content += f"... and {len(successful_cases) - 20} more cases\n"

        report_content += f"""
## ❌ Failed Cases ({len(failed_cases)})

"""

        for case in failed_cases[:10]:  # Show first 10
            report_content += f"- **{case['file']}**: {case['notes']}\n"

        if len(failed_cases) > 10:
            report_content += f"... and {len(failed_cases) - 10} more cases\n"

        report_content += f"""
## 📋 Cases Without Text ({len(no_text_cases)})

"""

        for case in no_text_cases[:10]:  # Show first 10
            report_content += f"- **{case['file']}**: {case['notes']}\n"

        if len(no_text_cases) > 10:
            report_content += f"... and {len(no_text_cases) - 10} more cases\n"

        report_content += f"""
## 🎯 Key Achievements

- ✅ **{self.extraction_log['cases_with_text']}/{self.extraction_log['total_cases']}** cases now have full text content
- ✅ **{self.extraction_log['cases_with_pdfs']}** PDFs downloaded for offline access
- ✅ **{total_text_length:,}** total characters of legal text extracted
- ✅ Complete extraction log maintained for audit trail

## 📁 Output Files

- **Extraction Log**: [text_extraction_log.json](data/case_law/text_extraction_log.json)
- **PDFs Directory**: [data/case_law/1782_discovery/pdfs/](data/case_law/1782_discovery/pdfs/)
- **Updated Case Files**: All case files now include `extracted_text` field

## 🚀 Next Steps

1. **Text Analysis**: Run NLP analysis on extracted text
2. **Search Enhancement**: Enable full-text search across corpus
3. **Content Validation**: Verify text quality and completeness
4. **Index Creation**: Build searchable text index

---

**This report documents the comprehensive text extraction process for the §1782 caselaw corpus.**
"""

        # Save report
        report_path = Path("data/case_law/text_extraction_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✓ Text extraction report saved to: {report_path}")


def main():
    """Main entry point."""
    logger.info("Starting comprehensive text extraction for §1782 caselaw corpus...")

    extractor = CorpusTextExtractor()

    # Step 1: Analyze current corpus
    analysis = extractor.analyze_corpus()

    # Step 2: Extract text and PDFs for all cases
    extractor.extract_all_cases()

    # Step 3: Generate comprehensive report
    extractor.generate_text_report()

    logger.info("\n🎉 Text extraction completed successfully!")
    logger.info("Check data/case_law/text_extraction_report.md for detailed results")


if __name__ == "__main__":
    main()
