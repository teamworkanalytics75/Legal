#!/usr/bin/env python3
"""
OCR Setup and Processing for Remaining 7 PDFs
=============================================

This script will:
1. Guide you through installing Tesseract OCR
2. Process the 7 failed PDFs with OCR
3. Extract text and add to the analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
import PyPDF2
import fitz  # PyMuPDF

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_tesseract_installation():
    """Check if Tesseract OCR is installed."""
    try:
        import pytesseract

        # Try to find Tesseract in common locations
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "tesseract"  # If it's in PATH
        ]

        tesseract_found = False
        for path in tesseract_paths:
            try:
                if path == "tesseract":
                    pytesseract.pytesseract.tesseract_cmd = path
                else:
                    pytesseract.pytesseract.tesseract_cmd = path

                version = pytesseract.get_tesseract_version()
                logger.info(f"✅ Tesseract OCR found: {version} at {path}")
                tesseract_found = True
                break
            except:
                continue

        if not tesseract_found:
            logger.error("❌ Tesseract OCR not found in any common location")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Tesseract OCR not found: {e}")
        return False

def install_tesseract_instructions():
    """Provide instructions for installing Tesseract OCR."""
    print("\n" + "=" * 60)
    print("🔧 TESSERACT OCR INSTALLATION REQUIRED")
    print("=" * 60)
    print("\n📥 To install Tesseract OCR on Windows:")
    print("\n1. Download the installer from:")
    print("   https://github.com/UB-Mannheim/tesseract/releases")
    print("\n2. Download the latest Windows installer:")
    print("   tesseract-ocr-w64-setup-5.3.3.20231005.exe")
    print("\n3. Run the installer and follow the setup wizard")
    print("   - Install to default location: C:\\Program Files\\Tesseract-OCR")
    print("   - Make sure to add Tesseract to PATH during installation")
    print("\n4. Restart your terminal/command prompt")
    print("\n5. Run this script again")
    print("\n" + "=" * 60)

def get_failed_pdfs() -> List[str]:
    """Get the list of failed PDFs that need OCR."""
    summary_file = Path("data/case_law/extracted_text/extraction_summary.json")

    if not summary_file.exists():
        logger.error("❌ Extraction summary not found")
        return []

    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('failed_files', [])

def extract_text_with_ocr(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        ocr_text = ""

        # Process first 3 pages (most important content)
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]

            # Convert page to image with good resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))

            # Perform OCR with optimized settings
            page_text = pytesseract.image_to_string(
                image,
                config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-[]{}"\' '
            )
            ocr_text += page_text + "\n"

        doc.close()

        if len(ocr_text.strip()) > 50:
            logger.info(f"✅ OCR successful: {pdf_path.name} ({len(ocr_text)} chars)")
            return ocr_text.strip()
        else:
            logger.warning(f"⚠️ OCR produced minimal text: {pdf_path.name}")
            return None

    except Exception as e:
        logger.error(f"❌ OCR failed for {pdf_path.name}: {e}")
        return None

def process_failed_pdfs():
    """Process the 7 failed PDFs with OCR."""
    failed_pdfs = get_failed_pdfs()

    if not failed_pdfs:
        logger.info("✅ No failed PDFs found")
        return

    logger.info(f"📄 Found {len(failed_pdfs)} PDFs that need OCR")

    pdf_dir = Path("data/case_law/1782_recap_api_pdfs")
    output_dir = Path("data/case_law/extracted_text")

    results = {
        'ocr_successful': 0,
        'ocr_failed': 0,
        'processed_files': []
    }

    for pdf_name in failed_pdfs:
        pdf_path = pdf_dir / pdf_name
        logger.info(f"🔍 Processing with OCR: {pdf_name}")

        if not pdf_path.exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            results['ocr_failed'] += 1
            continue

        # Extract text with OCR
        text = extract_text_with_ocr(pdf_path)

        if text:
            # Save extracted text
            text_file = output_dir / f"{pdf_path.stem}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

            results['ocr_successful'] += 1
            results['processed_files'].append({
                'pdf_file': pdf_name,
                'text_file': text_file.name,
                'text_length': len(text),
                'word_count': len(text.split()),
                'method': 'OCR'
            })

            logger.info(f"✅ OCR completed: {pdf_name}")
        else:
            results['ocr_failed'] += 1
            logger.warning(f"❌ OCR failed: {pdf_name}")

    # Save OCR results
    ocr_results_file = output_dir / "ocr_results.json"
    with open(ocr_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 OCR results saved to: {ocr_results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 OCR PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {results['ocr_successful']}/{len(failed_pdfs)} files")
    print(f"❌ Failed: {results['ocr_failed']} files")

    if results['ocr_successful'] > 0:
        print(f"📁 OCR text files saved to: {output_dir}")
        print("🎯 Ready to re-run NLP analysis with OCR results!")

def main():
    """Main execution function."""
    print("🚀 OCR Processing for Remaining 7 PDFs")
    print("=" * 60)

    # Check if Tesseract is installed
    if not check_tesseract_installation():
        install_tesseract_instructions()
        return

    # Process failed PDFs
    process_failed_pdfs()

if __name__ == "__main__":
    main()
