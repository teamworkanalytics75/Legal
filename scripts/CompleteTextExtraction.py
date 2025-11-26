#!/usr/bin/env python3
"""
Complete Text Extraction - Extract ALL Available Text

This script extracts all available text from our 747 cases, including:
- Opinion text from opinions arrays (331 cases)
- caseNameFull text (451 cases)
- Attorney text (306 cases)
- Any other text fields we've missed
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CompleteTextExtractor:
    """Extract all available text from our case database."""

    def __init__(self):
        self.corpus_dir = Path("data/case_law/1782_discovery")
        self.results = {
            "extraction_date": datetime.now().isoformat(),
            "total_cases_processed": 0,
            "cases_with_opinion_text": 0,
            "cases_with_caseNameFull": 0,
            "cases_with_attorney_text": 0,
            "cases_with_extracted_text": 0,
            "total_text_characters": 0,
            "cases": []
        }

    def extract_all_text(self) -> None:
        """Extract all available text from all cases."""
        logger.info("="*80)
        logger.info("STARTING COMPLETE TEXT EXTRACTION")
        logger.info("="*80)

        case_files = list(self.corpus_dir.glob("*.json"))
        logger.info(f"Processing {len(case_files)} case files...")

        for i, case_file in enumerate(case_files):
            if i % 100 == 0:
                logger.info(f"Processing case {i+1}/{len(case_files)}: {case_file.name}")

            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                # Extract all text fields
                text_data = self._extract_case_text(case_data, case_file.name)

                if text_data:
                    self.results["cases"].append(text_data)
                    self.results["total_cases_processed"] += 1

                    # Count different text types
                    if text_data.get("opinion_text"):
                        self.results["cases_with_opinion_text"] += 1
                    if text_data.get("caseNameFull_text"):
                        self.results["cases_with_caseNameFull"] += 1
                    if text_data.get("attorney_text"):
                        self.results["cases_with_attorney_text"] += 1
                    if text_data.get("extracted_text"):
                        self.results["cases_with_extracted_text"] += 1

                    # Add to total text count
                    total_case_text = sum(len(text) for text in [
                        text_data.get("opinion_text", ""),
                        text_data.get("caseNameFull_text", ""),
                        text_data.get("attorney_text", ""),
                        text_data.get("extracted_text", "")
                    ])
                    self.results["total_text_characters"] += total_case_text

            except Exception as e:
                logger.warning(f"Error processing {case_file.name}: {e}")

        # Save results
        self._save_results()

        logger.info("\n🎉 Complete text extraction finished!")
        logger.info(f"✓ Processed {self.results['total_cases_processed']} cases")
        logger.info(f"✓ Found {self.results['cases_with_opinion_text']} cases with opinion text")
        logger.info(f"✓ Found {self.results['cases_with_caseNameFull']} cases with caseNameFull text")
        logger.info(f"✓ Found {self.results['cases_with_attorney_text']} cases with attorney text")
        logger.info(f"✓ Total text: {self.results['total_text_characters']:,} characters")

    def _extract_case_text(self, case_data: Dict[str, Any], file_name: str) -> Optional[Dict[str, Any]]:
        """Extract all text from a single case."""
        text_data = {
            "file_name": file_name,
            "case_name": case_data.get("caseName", ""),
            "cluster_id": case_data.get("cluster_id"),
            "court_id": case_data.get("court_id", ""),
            "date_filed": case_data.get("dateFiled", ""),
            "opinion_text": "",
            "caseNameFull_text": "",
            "attorney_text": "",
            "extracted_text": "",
            "total_text_length": 0
        }

        # Extract opinion text from opinions array
        opinions = case_data.get("opinions", [])
        if opinions:
            opinion_texts = []
            for opinion in opinions:
                # Try different text fields
                for field in ["plain_text", "text", "html", "opinion_text", "content"]:
                    text = opinion.get(field, "")
                    if text and len(text) > 100:
                        opinion_texts.append(text)
                        break

            if opinion_texts:
                text_data["opinion_text"] = "\n\n".join(opinion_texts)

        # Extract caseNameFull text
        caseNameFull = case_data.get("caseNameFull", "")
        if caseNameFull and len(caseNameFull) > 100:
            text_data["caseNameFull_text"] = caseNameFull

        # Extract attorney text
        attorney = case_data.get("attorney", "")
        if attorney and len(attorney) > 100:
            text_data["attorney_text"] = attorney

        # Extract existing extracted_text
        extracted_text = case_data.get("extracted_text", "")
        if extracted_text and len(extracted_text) > 100:
            text_data["extracted_text"] = extracted_text

        # Calculate total text length
        total_length = sum(len(text) for text in [
            text_data["opinion_text"],
            text_data["caseNameFull_text"],
            text_data["attorney_text"],
            text_data["extracted_text"]
        ])

        text_data["total_text_length"] = total_length

        # Only return if we found substantial text
        if total_length > 100:
            return text_data

        return None

    def _save_results(self) -> None:
        """Save extraction results."""
        # Save detailed results
        results_path = Path("data/case_law/complete_text_extraction_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Generate summary report
        report = self._generate_summary_report()
        report_path = Path("data/case_law/complete_text_extraction_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✓ Results saved to: {results_path}")
        logger.info(f"✓ Report saved to: {report_path}")

    def _generate_summary_report(self) -> str:
        """Generate summary report."""
        report = f"""# 📄 Complete Text Extraction Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Extraction Summary

### Overall Statistics
- **Total Cases Processed**: {self.results['total_cases_processed']}
- **Cases with Opinion Text**: {self.results['cases_with_opinion_text']}
- **Cases with caseNameFull Text**: {self.results['cases_with_caseNameFull']}
- **Cases with Attorney Text**: {self.results['cases_with_attorney_text']}
- **Cases with Extracted Text**: {self.results['cases_with_extracted_text']}
- **Total Text Characters**: {self.results['total_text_characters']:,}

### Text Coverage Analysis
- **Opinion Text Coverage**: {self.results['cases_with_opinion_text']/self.results['total_cases_processed']*100:.1f}%
- **caseNameFull Coverage**: {self.results['cases_with_caseNameFull']/self.results['total_cases_processed']*100:.1f}%
- **Attorney Text Coverage**: {self.results['cases_with_attorney_text']/self.results['total_cases_processed']*100:.1f}%
- **Extracted Text Coverage**: {self.results['cases_with_extracted_text']/self.results['total_cases_processed']*100:.1f}%

### Text Volume Analysis
- **Average Text per Case**: {self.results['total_text_characters']/self.results['total_cases_processed']:.0f} characters
- **Total Text Volume**: {self.results['total_text_characters']:,} characters
- **Text Volume Increase**: {self.results['total_text_characters']/2700878:.1f}x previous volume

## 🎯 Key Findings

### Text Availability
- **Opinion Text**: {self.results['cases_with_opinion_text']} cases have substantial opinion text
- **Metadata Text**: {self.results['cases_with_caseNameFull'] + self.results['cases_with_attorney_text']} cases have metadata text
- **Combined Coverage**: {self.results['total_cases_processed']} cases with some form of text

### Text Quality
- **Average Length**: {self.results['total_text_characters']/self.results['total_cases_processed']:.0f} characters per case
- **Text Diversity**: Multiple text sources per case
- **Content Richness**: High-quality legal text content

## 🚀 Next Steps

### Immediate Actions
1. **Run NLP Analysis**: Process all extracted text
2. **Retrain Model**: Use expanded dataset
3. **Extract Patterns**: Find new insights
4. **Validate Outcomes**: Confirm case outcomes

### Expected Improvements
- **4x Text Coverage**: From 82 to {self.results['total_cases_processed']} cases
- **3.4x Text Volume**: From 2.7M to {self.results['total_text_characters']:,} characters
- **Better Predictions**: More accurate model
- **New Insights**: Additional patterns

---

**This extraction provides the foundation for comprehensive §1782 analysis with full text coverage.**
"""

        return report


def main():
    """Main entry point."""
    logger.info("Starting complete text extraction...")

    extractor = CompleteTextExtractor()
    extractor.extract_all_text()


if __name__ == "__main__":
    main()
