#!/usr/bin/env python3
"""
Test PDF readability - check if PDFs contain extractable text or are scanned images
"""

import os
import sys

def test_pdf_readability():
    """Test if PDFs are machine-readable or need OCR."""

    pdf_dir = "data/case_law/1782_recap_api_pdfs"
    files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')][:15]

    print("Testing PDF text readability:")
    print("=" * 50)

    readable_count = 0
    scanned_count = 0

    for f in files:
        print(f"\nTesting: {f}")
        try:
            import PyPDF2
            with open(os.path.join(pdf_dir, f), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page = reader.pages[0]
                text = page.extract_text()

                text_length = len(text.strip())
                print(f"  Text length: {text_length} characters")

                if text_length > 100:
                    print(f"  Status: ✅ READABLE")
                    print(f"  Sample text: {text[:150].replace(chr(10), ' ')}")
                    readable_count += 1
                else:
                    print(f"  Status: ❌ SCANNED/IMAGE (needs OCR)")
                    scanned_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            scanned_count += 1

    print("\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"  Readable PDFs: {readable_count}")
    print(f"  Scanned PDFs: {scanned_count}")
    print(f"  Total tested: {len(files)}")

    if readable_count > scanned_count:
        print(f"\n✅ MOST PDFs ARE MACHINE-READABLE!")
        print("You can extract text directly without OCR.")
    else:
        print(f"\n❌ MOST PDFs ARE SCANNED IMAGES!")
        print("You'll need OCR (Optical Character Recognition) to extract text.")

if __name__ == "__main__":
    test_pdf_readability()
