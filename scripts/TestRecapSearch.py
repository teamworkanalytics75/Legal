#!/usr/bin/env python3
"""
Test RECAP search to see what's happening
"""

import requests
import json

def test_recap_search():
    url = "https://www.courtlistener.com/recap/search/"
    params = {
        'q': '28 usc 1782',
        'stat_Precedential': 'on',
        'order_by': 'score desc',
        'format': 'json'
    }

    headers = {
        'User-Agent': 'MalcolmGrayson §1782 Research (malcolmgrayson00@gmail.com)'
    }

    print(f"Testing RECAP search...")
    print(f"URL: {url}")
    print(f"Params: {params}")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"JSON Response Keys: {list(data.keys())}")
                if 'results' in data:
                    print(f"Number of results: {len(data['results'])}")
                    if data['results']:
                        print(f"First result: {data['results'][0]}")
                else:
                    print(f"Full response: {data}")
            except json.JSONDecodeError:
                print(f"Response text (first 500 chars): {response.text[:500]}")
        else:
            print(f"Response text (first 500 chars): {response.text[:500]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_recap_search()
