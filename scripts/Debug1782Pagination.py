#!/usr/bin/env python3
"""
Debug CourtListener Pagination for §1782 Cases

This script makes direct API calls to CourtListener to debug why pagination
stops at 50 cases instead of continuing through 6000+ results.
"""

import sys
import json
import requests
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_courtlistener_pagination():
    """Debug CourtListener API pagination for §1782 cases."""

    print("="*80)
    print("🔍 PHASE 1: DEBUGGING COURTLISTENER PAGINATION")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # CourtListener API config
    base_url = "https://www.courtlistener.com/api/rest/v4"
    api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"

    headers = {
        'Authorization': f'Token {api_token}',
        'User-Agent': 'The MatrixLegalAI/2.0 (Research/Education)'
    }

    # Test query for §1782 cases
    search_params = {
        'q': 'usc 1782',  # Simple keyword search
        'type': 'o',      # Opinions
        'order_by': 'score desc'
    }

    print("📋 TEST 1: Initial API Call")
    print("-" * 60)
    print(f"Endpoint: {base_url}/search/")
    print(f"Query: {search_params}")
    print()

    try:
        # Make initial request
        print("🌐 Making API request...")
        response = requests.get(
            f"{base_url}/search/",
            params=search_params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Analyze response
        print("✅ Response received!")
        print()
        print("="*60)
        print("📊 RESPONSE ANALYSIS")
        print("="*60)

        # Key metrics
        total_count = data.get('count', 0)
        results = data.get('results', [])
        next_url = data.get('next')
        previous_url = data.get('previous')

        print(f"Total available: {total_count:,} cases")
        print(f"Results in page: {len(results)}")
        print(f"Next URL: {next_url}")
        print(f"Previous URL: {previous_url}")
        print()

        # Extract cursor
        if next_url:
            print("🔍 CURSOR EXTRACTION TEST")
            print("-" * 60)
            parsed_url = urlparse(next_url)
            query_params = parse_qs(parsed_url.query)
            cursor = query_params.get('cursor', [None])[0]

            print(f"Next URL: {next_url}")
            print(f"Extracted cursor: {cursor}")
            print()

            if cursor:
                print("✅ Cursor extraction successful!")
                print()

                # Test pagination with cursor
                print("📋 TEST 2: Follow-up Request with Cursor")
                print("-" * 60)

                page2_params = {
                    **search_params,
                    'cursor': cursor
                }

                print(f"Using cursor: {cursor[:50]}...")
                print("🌐 Making follow-up request...")

                response2 = requests.get(
                    f"{base_url}/search/",
                    params=page2_params,
                    headers=headers,
                    timeout=30
                )
                response2.raise_for_status()
                data2 = response2.json()

                results2 = data2.get('results', [])
                next_url2 = data2.get('next')

                print(f"✅ Page 2 received: {len(results2)} results")
                print(f"Next URL exists: {bool(next_url2)}")
                print()

                # Check if we got different results
                if results and results2:
                    case1_id = results[0].get('cluster_id', results[0].get('id'))
                    case2_id = results2[0].get('cluster_id', results2[0].get('id'))

                    print("🔍 PAGINATION VERIFICATION")
                    print("-" * 60)
                    print(f"Page 1 first case ID: {case1_id}")
                    print(f"Page 2 first case ID: {case2_id}")

                    if case1_id != case2_id:
                        print("✅ Pagination working! Different cases on page 2")
                    else:
                        print("❌ Pagination issue! Same case on both pages")
                    print()

                # Estimate total pages
                print("📊 PAGINATION STATISTICS")
                print("-" * 60)
                results_per_page = len(results)
                if results_per_page > 0:
                    estimated_pages = total_count / results_per_page
                    print(f"Results per page: {results_per_page}")
                    print(f"Estimated total pages: {estimated_pages:.0f}")
                    print(f"To get all {total_count:,} cases, need to walk through ~{estimated_pages:.0f} pages")
                print()

            else:
                print("❌ Cursor extraction failed!")
                print("Problem: Could not extract cursor from next URL")
                print()
        else:
            print("❌ No 'next' URL in response!")
            print(f"Problem: Only {len(results)} results returned, no pagination")
            print()

        # Show sample result structure
        if results:
            print("📄 SAMPLE RESULT STRUCTURE")
            print("-" * 60)
            sample = results[0]
            print(f"Keys in result: {list(sample.keys())}")
            print(f"Cluster ID: {sample.get('cluster_id')}")
            print(f"Case name: {sample.get('caseName', 'Unknown')}")
            print(f"Court: {sample.get('court', 'Unknown')}")
            print(f"Date filed: {sample.get('dateFiled', 'Unknown')}")
            print()

        # Summary
        print("="*80)
        print("📋 DIAGNOSTIC SUMMARY")
        print("="*80)

        if total_count > 5000:
            print(f"✅ CourtListener has {total_count:,} §1782 cases (6000+ confirmed!)")
        else:
            print(f"⚠️ Only {total_count:,} cases found (expected 6000+)")

        if next_url and cursor:
            print(f"✅ Pagination mechanism works (cursor-based)")
        else:
            print(f"❌ Pagination problem detected")

        if results_per_page > 0:
            print(f"✅ {results_per_page} results per page")

        print()
        print("💡 RECOMMENDATION:")
        if next_url and cursor:
            print("   Pagination is working! Need to walk through all pages.")
            print("   Fix: Ensure bulk_download() loops through all cursors.")
        else:
            print("   Pagination not working. Need to investigate API response.")

        print()
        print("✅ PHASE 1 COMPLETE: Pagination debugging finished")
        print("="*80)

        return {
            'total_count': total_count,
            'results_per_page': len(results),
            'has_next': bool(next_url),
            'cursor_works': bool(cursor) if next_url else False
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        logger.error(f"Error during pagination debug: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    debug_courtlistener_pagination()
