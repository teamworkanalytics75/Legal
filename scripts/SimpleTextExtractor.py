#!/usr/bin/env python3
"""
Simple Text Extraction for 1782 PDFs
====================================

Step 1: Extract text from all readable PDFs
Step 2: Save extracted text to files
Step 3: Basic analysis and reporting
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        # Return text if substantial content found
        if len(text.strip()) > 100:
            return text.strip()
        else:
            logger.warning(f"⚠️ Minimal text extracted from {pdf_path.name}: {len(text)} chars")
            return None

    except Exception as e:
        logger.error(f"❌ Error extracting text from {pdf_path.name}: {e}")
        return None

def process_pdfs():
    """Process all PDFs and extract text."""
    pdf_dir = Path("data/case_law/1782_recap_api_pdfs")
    output_dir = Path("data/case_law/extracted_text")
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"📁 Found {len(pdf_files)} PDF files")

    results = {
        'total_files': len(pdf_files),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'extracted_files': [],
        'failed_files': []
    }

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")

        # Extract text
        text = extract_text_from_pdf(pdf_file)

        if text:
            # Save extracted text
            text_file = output_dir / f"{pdf_file.stem}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            results['successful_extractions'] += 1
            results['extracted_files'].append({
                'pdf_file': pdf_file.name,
                'text_file': text_file.name,
                'text_length': len(text),
                'word_count': len(text.split())
            })

            logger.info(f"✅ Extracted {len(text)} chars from {pdf_file.name}")
        else:
            results['failed_extractions'] += 1
            results['failed_files'].append(pdf_file.name)
            logger.warning(f"❌ Failed to extract text from {pdf_file.name}")

        # Progress update every 25 files
        if i % 25 == 0:
            logger.info(f"📊 Progress: {i}/{len(pdf_files)} files processed")

    # Save results summary
    results_file = output_dir / "extraction_summary.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEXT EXTRACTION COMPLETE!")
    print(f"✅ Successfully extracted: {results['successful_extractions']}/{results['total_files']} files")
    print(f"❌ Failed extractions: {results['failed_extractions']} files")

    if results['successful_extractions'] > 0:
        avg_length = sum(f['text_length'] for f in results['extracted_files']) / results['successful_extractions']
        print(f"📄 Average text length: {avg_length:.0f} characters")
        print(f"📁 Extracted text saved to: {output_dir}")

    return results

if __name__ == "__main__":
    print("🚀 Starting Simple Text Extraction for 1782 PDFs")
    print("=" * 60)

    results = process_pdfs()

    print("\n🎯 Next steps:")
    print("1. Review extracted text files")
    print("2. Run basic NLP analysis")
    print("3. Identify patterns in 1782 cases")
