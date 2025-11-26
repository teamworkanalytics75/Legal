#!/usr/bin/env python3
"""
Test refined RECAP search queries to get only genuine §1782 cases
"""

import requests
import json
import time

def test_search_query(query, description):
    """Test a specific search query and analyze results."""
    print(f"\n🔍 Testing: {description}")
    print(f"Query: '{query}'")

    url = "https://www.courtlistener.com/recap/search/"
    params = {
        'q': query,
        'stat_Precedential': 'on',
        'order_by': 'score desc',
        'format': 'json'
    }

    headers = {
        'User-Agent': 'MalcolmGrayson §1782 Research (malcolmgrayson00@gmail.com)'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                results = data.get('results', [])
                print(f"✓ Found {len(results)} results")

                # Analyze first few results
                for i, result in enumerate(results[:3]):
                    case_name = result.get('case_name', 'Unknown')
                    docket_number = result.get('docket_number', 'Unknown')
                    print(f"  {i+1}. {case_name} ({docket_number})")

                return len(results)
            except json.JSONDecodeError:
                print(f"✗ JSON decode error")
                return 0
        else:
            print(f"✗ HTTP error: {response.status_code}")
            return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        return 0

def main():
    """Test various search strategies."""
    print("🚀 Testing Refined RECAP Search Queries")
    print("=" * 60)

    # Test different search strategies
    search_tests = [
        # Original (too broad)
        ("28 usc 1782", "Original broad search"),

        # Exact phrase searches
        ('"28 U.S.C. § 1782"', "Exact phrase with section symbol"),
        ('"28 USC 1782"', "Exact phrase without section symbol"),
        ('"28 U.S.C. 1782"', "Exact phrase with periods"),

        # Broader capture phrases
        ('"Application for Judicial Assistance"', "Application phrase"),
        ('"foreign proceeding pursuant to 28"', "Foreign proceeding phrase"),

        # Combined searches
        ('"28 U.S.C. § 1782" OR "28 USC 1782"', "Combined exact phrases"),
        ('"Application for Judicial Assistance" AND 1782', "Application + 1782"),

        # More specific legal terms
        ('"Ex Parte Application" AND 1782', "Ex parte + 1782"),
        ('"Motion for Discovery" AND "foreign"', "Discovery motion + foreign"),
    ]

    results_summary = {}

    for query, description in search_tests:
        count = test_search_query(query, description)
        results_summary[description] = count
        time.sleep(2)  # Be polite between requests

    # Summary
    print(f"\n📊 Search Results Summary:")
    print("=" * 40)
    for description, count in results_summary.items():
        print(f"{count:4d} results - {description}")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("=" * 20)

    # Find the best searches
    sorted_results = sorted(results_summary.items(), key=lambda x: x[1], reverse=True)

    print(f"✅ Best searches (most results):")
    for desc, count in sorted_results[:3]:
        if count > 0:
            print(f"   • {desc}: {count} results")

    print(f"\n🎯 Recommended approach:")
    print(f"   1. Use exact phrase: \"28 U.S.C. § 1782\"")
    print(f"   2. Combine with: \"28 USC 1782\"")
    print(f"   3. Add broader capture: \"Application for Judicial Assistance\"")
    print(f"   4. Always check 'Only show results with PDFs'")

if __name__ == "__main__":
    main()
