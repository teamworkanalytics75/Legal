#!/usr/bin/env python3
"""
Debug script to find the correct RECAP search URL
"""

import requests
from bs4 import BeautifulSoup

def debug_recap_search():
    """Debug the RECAP search functionality."""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Try the main RECAP page
    print("1. Testing main RECAP page...")
    r = requests.get('https://www.courtlistener.com/recap/', headers=headers)
    print(f"   Status: {r.status_code}")

    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        forms = soup.find_all('form')
        print(f"   Found {len(forms)} forms")

        for i, form in enumerate(forms[:3]):
            action = form.get('action', 'No action')
            print(f"   Form {i+1} action: {action}")

            # Look for search inputs
            inputs = form.find_all('input')
            for inp in inputs:
                name = inp.get('name', 'No name')
                inp_type = inp.get('type', 'No type')
                print(f"     Input: {name} ({inp_type})")

    # Try different search URLs
    search_urls = [
        'https://www.courtlistener.com/recap/search/',
        'https://www.courtlistener.com/recap/search',
        'https://www.courtlistener.com/search/',
        'https://www.courtlistener.com/search'
    ]

    print("\n2. Testing different search URLs...")
    for url in search_urls:
        print(f"   Testing: {url}")
        r = requests.get(url, headers=headers)
        print(f"   Status: {r.status_code}")

        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            title = soup.find('title')
            if title:
                print(f"   Title: {title.get_text()}")

    # Try the exact URL from your screenshot
    print("\n3. Testing exact URL from screenshot...")
    screenshot_url = "https://www.courtlistener.com/recap/search/?q=28%20usc%201782&available_only=on&order_by=dateFiled%20desc"
    r = requests.get(screenshot_url, headers=headers)
    print(f"   Status: {r.status_code}")

    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title')
        if title:
            print(f"   Title: {title.get_text()}")

        # Look for case links
        case_links = soup.find_all('a', href=lambda x: x and '/docket/' in x)
        print(f"   Found {len(case_links)} case links")

if __name__ == "__main__":
    debug_recap_search()
