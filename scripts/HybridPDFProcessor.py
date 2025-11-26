#!/usr/bin/env python3
"""
Hybrid Manual + Automated PDF Processor
Manual discovery + automated text extraction and analysis
"""

import json
import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import PyPDF2
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPDFProcessor:
    def __init__(self):
        self.pdf_dir = Path("data/recap_petitions_manual")
        self.text_dir = Path("data/recap_petitions_text")
        self.features_dir = Path("data/petition_features_expanded")

        # Create directories
        for dir_path in [self.pdf_dir, self.text_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.processed_files = set()
        self.stats = {
            'total_pdfs_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'genuine_1782_cases': 0,
            'last_run': None
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

                if text.strip():
                    logger.info(f"✓ Extracted {len(text)} characters from {pdf_path.name}")
                    return text
                else:
                    logger.warning(f"⚠️  No text extracted from {pdf_path.name}")
                    return None

        except Exception as e:
            logger.error(f"✗ Error extracting text from {pdf_path.name}: {e}")
            return None

    def is_genuine_1782_case(self, text: str) -> bool:
        """Check if text represents a genuine §1782 case."""
        text_lower = text.lower()

        # Strong indicators
        strong_patterns = [
            '28 u.s.c. § 1782',
            '28 usc 1782',
            'section 1782',
            'discovery for use in foreign proceeding',
            'ex parte application',
            'letter rogatory',
            'foreign tribunal',
            'foreign proceeding'
        ]

        # Count matches
        matches = sum(1 for pattern in strong_patterns if pattern in text_lower)

        # Must have at least 2 strong indicators
        is_genuine = matches >= 2

        if is_genuine:
            logger.info(f"✓ Genuine §1782 case detected ({matches} indicators)")
        else:
            logger.info(f"✗ Not a genuine §1782 case ({matches} indicators)")

        return is_genuine

    def extract_petition_features(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract features from petition text."""
        features = {
            'filename': filename,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
        }

        # Count specific patterns
        text_lower = text.lower()

        # Intel factors
        features['intel_cited'] = 'intel corp' in text_lower
        features['intel_factors_mentioned'] = sum(1 for factor in [
            'non-party', 'receptivity', 'circumvention', 'burdensome'
        ] if factor in text_lower)

        # Discovery scope
        features['single_request'] = 'single' in text_lower and 'request' in text_lower
        features['narrowly_tailored'] = 'narrowly tailored' in text_lower
        features['not_burdensome'] = 'not burdensome' in text_lower

        # Legal citations
        features['local_precedent_count'] = text_lower.count('district court')
        features['other_citations_count'] = text_lower.count('v.')

        # Document structure
        features['has_table_contents'] = 'table of contents' in text_lower
        features['has_intel_headings'] = 'intel' in text_lower and 'factor' in text_lower

        # Sector identification
        if 'patent' in text_lower or 'frand' in text_lower:
            features['sector'] = 'Patent/FRAND'
        elif 'bank' in text_lower or 'financial' in text_lower:
            features['sector'] = 'Financial'
        elif 'criminal' in text_lower:
            features['sector'] = 'Criminal'
        else:
            features['sector'] = 'Commercial'

        return features

    def process_pdf_file(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file."""
        logger.info(f"📄 Processing: {pdf_path.name}")

        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            self.stats['failed_extractions'] += 1
            return {'status': 'failed', 'reason': 'no_text_extracted'}

        self.stats['successful_extractions'] += 1

        # Check if genuine §1782 case
        is_genuine = self.is_genuine_1782_case(text)
        if not is_genuine:
            return {'status': 'skipped', 'reason': 'not_genuine_1782'}

        self.stats['genuine_1782_cases'] += 1

        # Extract features
        features = self.extract_petition_features(text, pdf_path.name)

        # Save extracted text
        text_file = self.text_dir / f"{pdf_path.stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)

        # Save features
        features_file = self.features_dir / f"{pdf_path.stem}_features.json"
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Processed: {pdf_path.name}")
        return {
            'status': 'success',
            'features': features,
            'text_file': str(text_file),
            'features_file': str(features_file)
        }

    def process_all_pdfs(self):
        """Process all PDFs in the manual directory."""
        logger.info("🚀 HYBRID PDF PROCESSOR")
        logger.info("=" * 50)
        logger.info("📁 Processing PDFs from manual downloads")
        logger.info("🎯 Extracting text and features")

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"📊 Found {len(pdf_files)} PDF files")

        if not pdf_files:
            logger.warning("⚠️  No PDF files found in manual directory")
            logger.info("   Please download PDFs manually to: data/recap_petitions_manual/")
            return

        results = []

        for pdf_path in pdf_files:
            if pdf_path.name in self.processed_files:
                logger.info(f"⏭️  Skipping already processed: {pdf_path.name}")
                continue

            result = self.process_case(pdf_path)
            results.append(result)
            self.processed_files.add(pdf_path.name)

            # Brief pause between files
            time.sleep(1)

        # Generate summary
        self.generate_summary_report(results)

        logger.info(f"\\n✅ Processing complete!")
        logger.info(f"📊 Processed: {self.stats['total_pdfs_processed']} PDFs")
        logger.info(f"✓ Successful: {self.stats['successful_extractions']}")
        logger.info(f"✗ Failed: {self.stats['failed_extractions']}")
        logger.info(f"🎯 Genuine §1782: {self.stats['genuine_1782_cases']}")

    def generate_summary_report(self, results: List[Dict[str, Any]]):
        """Generate summary report."""
        report_path = Path("data/case_law/hybrid_processing_report.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Hybrid PDF Processing Report\\n\\n")
            f.write(f"**Date**: {self.stats['last_run']}\\n")
            f.write(f"**Total PDFs Processed**: {self.stats['total_pdfs_processed']}\\n")
            f.write(f"**Successful Extractions**: {self.stats['successful_extractions']}\\n")
            f.write(f"**Failed Extractions**: {self.stats['failed_extractions']}\\n")
            f.write(f"**Genuine §1782 Cases**: {self.stats['genuine_1782_cases']}\\n\\n")

            f.write("## Processed Cases\\n")
            for result in results:
                if result['status'] == 'success':
                    f.write(f"- **{result['features']['filename']}**\\n")
                    f.write(f"  - Sector: {result['features']['sector']}\\n")
                    f.write(f"  - Text Length: {result['features']['text_length']} chars\\n")
                    f.write(f"  - Intel Cited: {result['features']['intel_cited']}\\n")
                    f.write(f"  - Local Precedent: {result['features']['local_precedent_count']}\\n\\n")

        logger.info(f"📄 Report saved to: {report_path}")

    def create_manual_download_guide(self):
        """Create a guide for manual PDF downloads."""
        guide_path = Path("data/case_law/MANUAL_DOWNLOAD_GUIDE.md")

        guide_content = """# Manual PDF Download Guide

## 🎯 Strategy: Manual Discovery + Automated Processing

Since automated scraping is blocked by AWS WAF, we'll use a **hybrid approach**:

1. **Manual Discovery**: Use CourtListener interface to find cases
2. **Manual Download**: Download FREE PDFs manually
3. **Automated Processing**: Extract text and features automatically

## 📋 Step-by-Step Process

### Step 1: Find Cases
1. Go to: https://www.courtlistener.com/recap/search/
2. Enter query: `"28 usc 1782 discovery"`
3. ✅ Check "Only show results with PDFs"
4. Click Search

### Step 2: Download FREE PDFs
1. Click on case names to open docket pages
2. Look for **"Download PDF"** buttons (FREE)
3. **Skip** "Buy on PACER" buttons (PAID)
4. Save PDFs to: `data/recap_petitions_manual/`

### Step 3: Automated Processing
```bash
py scripts/hybrid_pdf_processor.py
```

## 🎯 Target Cases

Based on your screenshots, focus on:
- **EFG Bank AG** (D. Conn. 2025) - Has FREE PDFs
- **Navios South American Logistics** (S.D.N.Y. 2024) - Has FREE PDFs
- **HMD Global Oy** cases - Multiple FREE PDFs

## 📊 Expected Results

- **Target**: 100+ FREE PDFs
- **Processing**: Automated text extraction
- **Analysis**: Feature extraction and §1782 filtering
- **Model**: Retrain with expanded dataset

## ⏱️ Timeline

- **Manual Downloads**: 2-3 hours
- **Automated Processing**: 30 minutes
- **Model Retraining**: 1 hour

**This hybrid approach gives us the best of both worlds!** 🚀
"""

        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        logger.info(f"📄 Manual download guide saved to: {guide_path}")

def main():
    processor = HybridPDFProcessor()
    processor.stats['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Create manual download guide
    processor.create_manual_download_guide()

    # Process any existing PDFs
    processor.process_all_pdfs()

if __name__ == "__main__":
    main()
