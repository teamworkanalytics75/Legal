#!/usr/bin/env python3
"""
Download National Security Sealing Motions from CourtListener

Downloads NS sealing motions to fill coverage gaps identified in coverage assessment.
Uses existing CourtListenerClient tools.

Target: 100+ additional NS sealing motions (50+ granted, 50+ denied)
Priority: D. Massachusetts (highest), then D.D.C., S.D.N.Y., etc.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

try:
    from DownloadCaseLaw import CourtListenerClient
except ImportError:
    # Try alternative import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from document_ingestion.DownloadCaseLaw import CourtListenerClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_ns_sealing_motions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Priority download queries based on coverage assessment
PRIORITY_QUERIES = [
    # Priority 1: D. Massachusetts NS sealing (HIGHEST - your jurisdiction)
    {
        "topic": "ns_sealing_dmass",
        "courts": ["mad"],  # D. Massachusetts
        "keywords": ["national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "HIGH",
        "description": "D. Massachusetts NS sealing motions (most relevant)"
    },

    # Priority 2: D.D.C. NS sealing (high volume)
    {
        "topic": "ns_sealing_ddc",
        "courts": ["dcd"],  # D.D.C.
        "keywords": ["national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "D.D.C. NS sealing motions (high NS case volume)"
    },

    # Priority 3: Recent NS sealing (2020+) - post-Trump proclamation
    {
        "topic": "ns_sealing_recent",
        "courts": None,  # All federal courts
        "keywords": ["national security", "motion to seal"],
        "date_after": "2020-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "Recent NS sealing motions (post-proclamation era)"
    },

    # Priority 4: Classified information sealing
    {
        "topic": "classified_sealing",
        "courts": None,
        "keywords": ["classified information", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "Classified information sealing motions"
    },

    # Priority 5: State secrets sealing
    {
        "topic": "state_secret_sealing",
        "courts": None,
        "keywords": ["state secret", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "State secret sealing motions"
    },

    # Priority 6: Foreign government + sealing
    {
        "topic": "foreign_gov_sealing",
        "courts": None,
        "keywords": ["foreign government", "motion to seal", "national security"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "Foreign government + NS sealing motions"
    },

    # Priority 7: S.D.N.Y. and E.D.N.Y. NS sealing
    {
        "topic": "ns_sealing_ny",
        "courts": ["nysd", "nyed"],  # S.D.N.Y. and E.D.N.Y.
        "keywords": ["national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "New York federal NS sealing motions"
    },
]


def download_ns_sealing_batch(query_config: Dict[str, Any], client: CourtListenerClient) -> Dict[str, Any]:
    """Download a single batch of NS sealing motions."""
    topic = query_config["topic"]
    description = query_config["description"]
    priority = query_config["priority"]

    logger.info("\n" + "=" * 80)
    logger.info(f"Downloading: {description}")
    logger.info(f"Topic: {topic} | Priority: {priority}")
    logger.info("=" * 80)

    try:
        cases = client.bulk_download(
            topic=topic,
            courts=query_config["courts"],
            keywords=query_config["keywords"],
            date_after=query_config["date_after"],
            max_results=query_config["max_results"],
            resume=True,  # Resume from checkpoint if available
            apply_filters=False,  # Don't filter for Section 1782 only
            include_non_precedential=False  # Only precedential opinions
        )

        logger.info(f"Downloaded {len(cases)} cases for {topic}")
        return {
            "topic": topic,
            "description": description,
            "priority": priority,
            "cases_downloaded": len(cases),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error downloading {topic}: {e}")
        return {
            "topic": topic,
            "description": description,
            "priority": priority,
            "cases_downloaded": 0,
            "status": "error",
            "error": str(e)
        }


def main():
    """Download all NS sealing motion batches."""
    logger.info("=" * 80)
    logger.info("NATIONAL SECURITY SEALING MOTIONS DOWNLOAD")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Target: Download 100+ additional NS sealing motions")
    logger.info("Priority: D. Massachusetts > D.D.C. > Recent (2020+) > Others")
    logger.info("")

    # Initialize client - try multiple possible config locations
    # Prioritize local_courtlistener_config.json which has the actual token
    possible_configs = [
        Path("case_law_data/local_courtlistener_config.json"),  # Has actual token
        Path("document_ingestion/CourtlistenerConfig.json"),
        Path("document_ingestion/courtlistener_config.json"),
    ]

    config_path = None
    for path in possible_configs:
        if path.exists():
            config_path = str(path)
            logger.info(f"Found config file: {config_path}")
            break

    if not config_path:
        logger.error(f"Config file not found. Tried: {[str(p) for p in possible_configs]}")
        logger.error("Please ensure courtlistener config exists in one of these locations")
        return

    try:
        client = CourtListenerClient(config_path=config_path, overnight_mode=False)  # Set to True for long downloads
        logger.info("CourtListener client initialized successfully")

        # Verify token is set (not just a placeholder)
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            token = config.get('api', {}).get('api_token', '')
            if token and not (token.startswith('${') and token.endswith('}')):
                logger.info("[OK] Valid API token found in config file")
            elif token.startswith('${') and token.endswith('}'):
                import os
                env_var = token[2:-1]
                if os.environ.get(env_var):
                    logger.info(f"[OK] API token found in environment variable: {env_var}")
                else:
                    logger.warning(f"[WARN] Token placeholder found but {env_var} not set")
            else:
                logger.warning("[WARN] No API token found - downloads may fail")

    except Exception as e:
        logger.error(f"Failed to initialize CourtListener client: {e}")
        logger.error("Check that the config file has a valid API token")
        return

    # Download each batch
    results = []
    total_downloaded = 0

    for query_config in PRIORITY_QUERIES:
        result = download_ns_sealing_batch(query_config, client)
        results.append(result)
        total_downloaded += result["cases_downloaded"]

        # Small delay between batches
        import time
        time.sleep(2)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    for result in results:
        status_icon = "[OK]" if result["status"] == "success" else "[ERROR]"
        logger.info(f"{status_icon} {result['description']}")
        logger.info(f"   Topic: {result['topic']}")
        logger.info(f"   Cases downloaded: {result['cases_downloaded']}")
        if result["status"] == "error":
            logger.info(f"   Error: {result.get('error', 'Unknown')}")
        logger.info("")

    logger.info(f"Total cases downloaded: {total_downloaded}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Import downloaded cases to database using case law ingestion pipeline")
    logger.info("2. Tag cases with tag_national_security = 1")
    logger.info("3. Classify outcomes (granted/denied) for CatBoost analysis")
    logger.info("4. Re-run coverage assessment to verify improved coverage")
    logger.info("")

    # Save results
    output_path = Path("outputs/ns_sealing_download_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path, 'w') as f:
        json.dump({
            "download_date": datetime.now().isoformat(),
            "total_cases_downloaded": total_downloaded,
            "batches": results
        }, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

