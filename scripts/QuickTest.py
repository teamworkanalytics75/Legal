#!/usr/bin/env python3
"""Quick scraper test - 5 articles with faster rate limiting"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "case_law" / "scripts"))

from harvard_crimson_scraper import HarvardCrimsonScraper

# Create scraper with faster rate limiting for testing
scraper = HarvardCrimsonScraper()
scraper.rate_limit_delay = 1.0  # Faster for testing

print("\n" + "="*80)
print("QUICK TEST: Scraping 5 articles with live progress")
print("="*80 + "\n")

# Scrape 5 articles
articles = scraper.scrape_section('news', max_articles=5, batch_size=5, resume=True)

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print(f"Scraped {len(articles)} articles")
print("\nARTICLE SUMMARIES:")
print("="*80)

for i, article in enumerate(articles, 1):
    print(f"\n{i}. {article['title']}")
    print(f"   Score: {article['relevance_score']:.1f} | Threat: {article['threat_level']} | Sentiment: {article['sentiment']}")
    if article['content']:
        summary = article['content'][:150].replace('\n', ' ').strip()
        print(f"   Summary: {summary}...")

