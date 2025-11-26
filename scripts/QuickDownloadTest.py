"""Quick test to verify new search API works."""
import sys
sys.path.insert(0, 'document_ingestion')

from download_case_law import CourtListenerClient
import logging

logging.basicConfig(level=logging.INFO)

print("Testing CourtListener Search API v4...")
print("="*60)

client = CourtListenerClient()

# Test search
result = client.search_opinions(
    courts=['mass', 'ca1'],
    keywords=['pseudonym', 'doe plaintiff'],
    date_filed_after='2015-01-01',
    limit=20
)

if result and 'results' in result:
    print(f"[ok] Search successful!")
    print(f" Results in this page: {len(result['results'])}")
    print(f" Total available: {result.get('count', 'unknown')}")
    print(f" First case ID: {result['results'][0].get('id') if result['results'] else 'none'}")
else:
    print(" Search failed")
    print(f" Response: {result}")

print("="*60)

