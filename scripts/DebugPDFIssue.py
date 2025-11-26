#!/usr/bin/env python3
"""
Debug script to investigate why we're not finding PDFs
"""

import json
import requests
from pathlib import Path

# Load configuration
with open("document_ingestion/courtlistener_config.json", 'r') as f:
    config = json.load(f)

base_url = config['api']['base_url']
headers = {
    'Authorization': f'Token {config["api"]["api_token"]}',
    'User-Agent': config["api"]["user_agent"]
}

def debug_opinion(opinion_id):
    """Debug a specific opinion to see what fields are available."""
    print(f"\n🔍 Debugging opinion {opinion_id}")

    url = f"{base_url}/opinions/{opinion_id}/"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Successfully retrieved opinion {opinion_id}")
        print(f"📄 Available fields: {list(data.keys())}")

        # Check for PDF-related fields
        pdf_fields = ['pdf_url', 'download_url', 'resource', 'filepath']
        for field in pdf_fields:
            if field in data:
                print(f"📎 {field}: {data[field]}")

        # Check for other important fields
        important_fields = ['id', 'caseName', 'absolute_url', 'resource_uri']
        for field in important_fields:
            if field in data:
                print(f"ℹ️  {field}: {data[field]}")

        return data
    else:
        print(f"❌ Failed to retrieve opinion {opinion_id}: {response.status_code}")
        return None

def debug_search_results():
    """Debug search results to see what we're getting."""
    print("🔍 Debugging search results...")

    # Search for 1782 cases
    search_url = f"{base_url}/search/"
    params = {
        'q': '28 USC 1782',
        'format': 'json',
        'order_by': 'dateFiled desc',
        'stat_Precedential': 'on',
        'page_size': 5
    }

    response = requests.get(search_url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"✅ Found {len(results)} search results")

        for i, result in enumerate(results[:3]):  # Check first 3 results
            print(f"\n📋 Result {i+1}: {result.get('caseName', 'Unknown')}")
            print(f"   Fields: {list(result.keys())}")

            # Check for opinions
            opinions = result.get('opinions', [])
            print(f"   Opinions: {len(opinions)}")

            for j, opinion in enumerate(opinions[:2]):  # Check first 2 opinions
                opinion_id = opinion.get('id')
                if opinion_id:
                    debug_opinion(opinion_id)
    else:
        print(f"❌ Search failed: {response.status_code}")

if __name__ == "__main__":
    debug_search_results()
