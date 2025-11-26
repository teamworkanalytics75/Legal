#!/usr/bin/env python3
"""
Test script for CourtListener PDF Downloader
Tests basic functionality without downloading large amounts of data
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from courtlistener_pdf_downloader import CourtListenerPDFDownloader

def test_database_connection():
    """Test database connection."""
    print("🔍 Testing database connection...")

    try:
        downloader = CourtListenerPDFDownloader()
        cases = downloader.get_cases_from_database(limit=5)
        print(f"✅ Database connection successful - found {len(cases)} cases")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_browser_setup():
    """Test browser setup."""
    print("🔍 Testing browser setup...")

    try:
        downloader = CourtListenerPDFDownloader()
        downloader.setup_browser()

        # Test basic navigation
        downloader.page.goto("https://www.courtlistener.com")
        title = downloader.page.title()
        print(f"✅ Browser setup successful - page title: {title}")

        # Close browser
        downloader.browser.close()
        return True
    except Exception as e:
        print(f"❌ Browser setup failed: {e}")
        return False

def test_single_case_download():
    """Test downloading PDFs for a single case."""
    print("🔍 Testing single case PDF download...")

    try:
        downloader = CourtListenerPDFDownloader()

        # Get one case from database
        cases = downloader.get_cases_from_database(limit=1)
        if not cases:
            print("❌ No cases found in database")
            return False

        case = cases[0]
        print(f"Testing with case: {case['case_name']}")

        # Setup browser
        downloader.setup_browser()

        # Search for the case
        docket_url = downloader.search_courtlistener_for_case(case)

        if docket_url:
            print(f"✅ Found case on CourtListener: {docket_url}")

            # Try to download PDFs
            pdf_paths = downloader.download_pdfs_from_docket(docket_url, case)
            print(f"✅ Downloaded {len(pdf_paths)} PDFs")

            # Update database
            downloader.update_database_with_pdfs(case, pdf_paths)
            print("✅ Database updated successfully")
        else:
            print("⚠️  Case not found on CourtListener (this is normal)")

        # Close browser
        downloader.browser.close()
        return True

    except Exception as e:
        print(f"❌ Single case download test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CourtListener PDF Downloader Test Suite")
    print("=" * 50)

    tests = [
        ("Database Connection", test_database_connection),
        ("Browser Setup", test_browser_setup),
        ("Single Case Download", test_single_case_download)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! PDF downloader is ready to use.")
        print("\nTo run the full downloader:")
        print("python scripts/courtlistener_pdf_downloader.py --topic 1782_discovery --limit 10")
    else:
        print("⚠️  Some tests failed. Please check the setup and try again.")

if __name__ == "__main__":
    main()
