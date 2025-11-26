#!/usr/bin/env python3
"""
Analyze the 9 PDFs you downloaded from RECAP Archive
Extract text and create petition features for our model
"""

import os
import json
import logging
from pathlib import Path
import PyPDF2
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFAnalyzer:
    def __init__(self):
        self.pdf_dir = Path("data/recap_petitions")
        self.results = []

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def analyze_petition_text(self, text, filename):
        """Analyze petition text for key features."""
        text_lower = text.lower()

        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)

        # Legal terminology
        legal_terms = {
            'section_1782': len(re.findall(r'28\s*u\.s\.c\.?\s*(?:§)?\s*1782', text_lower)),
            'intel_corp': len(re.findall(r'intel\s+corp', text_lower)),
            'foreign_tribunal': len(re.findall(r'foreign\s+tribunal', text_lower)),
            'judicial_assistance': len(re.findall(r'judicial\s+assistance', text_lower)),
            'discovery': len(re.findall(r'discovery', text_lower)),
            'protective_order': len(re.findall(r'protective\s+order', text_lower)),
            'ex_parte': len(re.findall(r'ex\s+parte', text_lower)),
            'motion': len(re.findall(r'motion', text_lower)),
            'application': len(re.findall(r'application', text_lower)),
        }

        # Court jurisdiction (from filename)
        jurisdiction = "unknown"
        if 'cand' in filename.lower():
            jurisdiction = "california_northern"
        elif 'casd' in filename.lower():
            jurisdiction = "california_southern"
        elif 'flnd' in filename.lower():
            jurisdiction = "florida_northern"
        elif 'nysd' in filename.lower():
            jurisdiction = "new_york_southern"

        # Outcome indicators (if any)
        outcome_indicators = {
            'granted': len(re.findall(r'granted', text_lower)),
            'denied': len(re.findall(r'denied', text_lower)),
            'approved': len(re.findall(r'approved', text_lower)),
            'rejected': len(re.findall(r'rejected', text_lower)),
        }

        return {
            'filename': filename,
            'jurisdiction': jurisdiction,
            'word_count': word_count,
            'char_count': char_count,
            'legal_terms': legal_terms,
            'outcome_indicators': outcome_indicators,
            'text_preview': text[:500] + "..." if len(text) > 500 else text
        }

    def analyze_all_pdfs(self):
        """Analyze all PDFs in the directory."""
        logger.info("🔍 Analyzing RECAP Archive PDFs")
        logger.info("=" * 50)

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            logger.info(f"📄 Processing: {pdf_file.name}")

            # Extract text
            text = self.extract_text_from_pdf(pdf_file)

            if text:
                # Analyze text
                analysis = self.analyze_petition_text(text, pdf_file.name)
                analysis['full_text'] = text  # Store full text for later analysis
                self.results.append(analysis)

                logger.info(f"✓ Extracted {len(text)} characters, {analysis['word_count']} words")
                logger.info(f"  Jurisdiction: {analysis['jurisdiction']}")
                logger.info(f"  §1782 mentions: {analysis['legal_terms']['section_1782']}")
            else:
                logger.warning(f"✗ No text extracted from {pdf_file.name}")

        return self.results

    def save_results(self):
        """Save analysis results."""
        # Save detailed results
        results_file = Path("data/case_law/recap_pdf_analysis.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': '2025-10-19',
                'total_pdfs': len(self.results),
                'petitions': self.results
            }, f, indent=2, ensure_ascii=False)

        # Create summary report
        report_file = Path("data/case_law/recap_pdf_report.md")
        report_content = f"""# RECAP Archive PDF Analysis Report

## Summary
- **Total PDFs Analyzed**: {len(self.results)}
- **Analysis Date**: 2025-10-19

## Individual Petitions

"""

        for i, petition in enumerate(self.results, 1):
            report_content += f"""### Petition {i}: {petition['filename']}

- **Jurisdiction**: {petition['jurisdiction']}
- **Word Count**: {petition['word_count']:,}
- **Character Count**: {petition['char_count']:,}

#### Legal Terms Found:
"""
            for term, count in petition['legal_terms'].items():
                if count > 0:
                    report_content += f"- **{term.replace('_', ' ').title()}**: {count}\n"

            report_content += f"""
#### Text Preview:
```
{petition['text_preview']}
```

---
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📊 Results saved to: {results_file}")
        logger.info(f"📄 Report saved to: {report_file}")

def main():
    analyzer = PDFAnalyzer()
    results = analyzer.analyze_all_pdfs()
    analyzer.save_results()

    logger.info(f"\n✅ Analysis complete!")
    logger.info(f"📊 Analyzed {len(results)} petition PDFs")
    logger.info(f"📁 Check results in: data/case_law/")

if __name__ == "__main__":
    main()
