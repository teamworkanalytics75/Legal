#!/usr/bin/env python3
"""
Test CourtListener Search Terms

Test different search terms to find actual §1782 cases.
"""

import requests
import json

def test_search_terms():
    """Test different search terms for §1782 cases."""

    api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
    headers = {
        'Authorization': f'Token {api_token}',
        'Content-Type': 'application/json'
    }

    base_url = "https://www.courtlistener.com/api/rest/v4/dockets/"

    # Test different search terms
    search_terms = [
        '"In re" AND "application"',
        '"In re" AND "discovery"',
        '"In re" AND "foreign"',
        '"In re" AND "1782"',
        '"In re" AND "judicial assistance"',
        '"In re" AND "international"',
        '"In re" AND "tribunal"',
        '"In re" AND "arbitration"',
        '"In re" AND "subpoena"',
        '"In re" AND "deposition"'
    ]

    for term in search_terms:
        print(f"\n🔍 Testing: {term}")

        try:
            params = {'q': term, 'format': 'json'}
            response = requests.get(base_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                print(f"✓ Found {len(results)} results")

                # Show first few results
                for i, result in enumerate(results[:3]):
                    case_name = result.get('case_name', 'Unknown')
                    docket_id = result.get('id', 'Unknown')
                    print(f"  {i+1}. {case_name} (ID: {docket_id})")

            else:
                print(f"❌ Error: {response.status_code}")

        except Exception as e:
            print(f"❌ Exception: {e}")

        # Be respectful to the API
        import time
        time.sleep(1)

if __name__ == "__main__":
    test_search_terms()
