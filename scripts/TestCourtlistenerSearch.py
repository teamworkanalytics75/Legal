#!/usr/bin/env python3
"""
Simple test script for CourtListener search functionality
Tests search without downloading PDFs
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from courtlistener_pdf_downloader import CourtListenerPDFDownloader

def test_courtlistener_search():
    """Test CourtListener search functionality."""
    print("🔍 Testing CourtListener search functionality...")

    try:
        downloader = CourtListenerPDFDownloader()

        # Setup browser
        downloader.setup_browser()

        # Test with a simple case name
        test_case = {
            'case_name': 'United States v. Simmons',
            'citation': '',
            'id': 999
        }

        print(f"Testing search for: {test_case['case_name']}")

        # Try to search for the case
        docket_url = downloader.search_courtlistener_for_case(test_case)

        if docket_url:
            print(f"✅ Successfully found case: {docket_url}")

            # Test navigating to the docket page
            print("Testing docket page navigation...")
            downloader.page.goto(docket_url)
            downloader.human_like_delay(2, 3)

            page_title = downloader.page.title()
            print(f"✅ Docket page loaded: {page_title}")

            # Check if there are any PDF links
            pdf_links = downloader.page.locator('a[href$=".pdf"]')
            count = pdf_links.count()
            print(f"✅ Found {count} PDF links on docket page")

            if count > 0:
                # Show first few PDF links
                for i in range(min(3, count)):
                    link = pdf_links.nth(i)
                    link_text = link.text_content().strip()
                    print(f"  - PDF {i+1}: {link_text}")
        else:
            print("❌ Case not found on CourtListener")

        # Close browser
        downloader.browser.close()
        return True

    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_direct_navigation():
    """Test direct navigation to CourtListener."""
    print("🔍 Testing direct navigation to CourtListener...")

    try:
        downloader = CourtListenerPDFDownloader()
        downloader.setup_browser()

        # Test direct navigation to RECAP search
        print(f"Navigating to: {downloader.recap_search_url}")
        downloader.page.goto(downloader.recap_search_url)
        downloader.human_like_delay(3, 5)

        page_title = downloader.page.title()
        print(f"✅ Page loaded: {page_title}")

        if "404" in page_title or "not found" in page_title.lower():
            print("❌ Got 404 error - URL might be wrong")
            return False

        # Check if search form exists
        search_box = downloader.page.locator('input[name="q"]')
        if search_box.is_visible():
            print("✅ Search form found")

            # Test a simple search
            search_box.fill("United States")
            downloader.human_like_delay(1, 2)

            search_button = downloader.page.locator('input[type="submit"]')
            search_button.click()
            print("✅ Search submitted")

            # Wait for results
            downloader.page.wait_for_selector('h2.case-name, .no-results', timeout=10000)
            downloader.human_like_delay(2, 3)

            # Check results
            if downloader.page.locator('.no-results').is_visible():
                print("⚠️  No results found")
            else:
                case_links = downloader.page.locator('h2.case-name a')
                count = case_links.count()
                print(f"✅ Found {count} search results")

                if count > 0:
                    # Show first few results
                    for i in range(min(3, count)):
                        link = case_links.nth(i)
                        link_text = link.text_content().strip()
                        print(f"  - Result {i+1}: {link_text}")
        else:
            print("❌ Search form not found")
            return False

        downloader.browser.close()
        return True

    except Exception as e:
        print(f"❌ Navigation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CourtListener Search Test Suite")
    print("=" * 50)

    tests = [
        ("Direct Navigation", test_direct_navigation),
        ("Case Search", test_courtlistener_search)
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
        print("🎉 All tests passed! Search functionality is working.")
    else:
        print("⚠️  Some tests failed. Check the CourtListener URL and search functionality.")

if __name__ == "__main__":
    main()
