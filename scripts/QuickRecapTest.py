#!/usr/bin/env python3
"""
Quick RECAP Test with Proper Headers

Test RECAP access with correct User-Agent header.
"""

import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_recap_access():
    """Test RECAP access with proper headers."""

    api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
    headers = {
        'Authorization': f'Token {api_token}',
        'Content-Type': 'application/json',
        'User-Agent': 'Malcolm-Grayson-Research/1.0 (malcolmgrayson00@gmail.com)'
    }

    # Test a few docket IDs from our cases
    test_docket_ids = [66708408, 69107196, 68061677, 69626904, 69051120]

    logger.info("🔍 Testing RECAP Access with Proper Headers")
    logger.info("=" * 60)

    for docket_id in test_docket_ids:
        logger.info(f"Testing docket {docket_id}...")

        try:
            # Test RECAP documents endpoint
            url = f"https://www.courtlistener.com/api/rest/v4/recap/"
            params = {'docket_id': docket_id, 'format': 'json'}

            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"✓ SUCCESS: Found {len(results)} RECAP documents for docket {docket_id}")

                # Show first few results
                for i, doc in enumerate(results[:3]):
                    doc_type = doc.get('document_type', 'Unknown')
                    doc_id = doc.get('id', 'Unknown')
                    logger.info(f"  {i+1}. Document {doc_id} ({doc_type})")

            elif response.status_code == 403:
                logger.warning(f"✗ 403 Forbidden for docket {docket_id}")
            else:
                logger.warning(f"✗ Error {response.status_code} for docket {docket_id}")

        except Exception as e:
            logger.error(f"✗ Exception for docket {docket_id}: {e}")

        # Be respectful to the API
        import time
        time.sleep(1)

    logger.info("\n✅ RECAP test complete!")

if __name__ == "__main__":
    test_recap_access()
