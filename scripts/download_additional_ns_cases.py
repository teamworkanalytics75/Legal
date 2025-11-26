#!/usr/bin/env python3
"""
Download additional National Security cases with expanded search terms.

This script searches for NS cases using additional keywords and broader criteria
to find cases we may have missed in the initial download.
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
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from document_ingestion.DownloadCaseLaw import CourtListenerClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_additional_ns_cases.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Additional NS search queries - PRIORITIZED for your case context:
# - American federally funded Ivy League university
# - Endangerment of US citizen abroad
# - Censorship of speech
# - Foreign government interference
ADDITIONAL_NS_QUERIES = [
    # PRIORITY 1: Academic institutions + federal funding + NS (HIGHEST RELEVANCE)
    {
        "topic": "university_federal_funding_ns",
        "courts": None,
        "keywords": ["university", "federal funding", "national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGHEST",
        "description": "Universities + federal funding + NS sealing motions"
    },

    # PRIORITY 2: US citizen endangerment abroad + NS
    {
        "topic": "us_citizen_endangerment_ns",
        "courts": None,
        "keywords": ["US citizen", "endangerment", "abroad", "national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGHEST",
        "description": "US citizen endangerment abroad + NS sealing"
    },

    # PRIORITY 3: Speech censorship + NS
    {
        "topic": "speech_censorship_ns",
        "courts": None,
        "keywords": ["censorship", "speech", "free speech", "national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGHEST",
        "description": "Speech censorship + NS sealing motions"
    },

    # PRIORITY 4: Academic institutions + foreign influence + NS
    {
        "topic": "academic_foreign_influence_ns",
        "courts": None,
        "keywords": ["academic", "foreign influence", "foreign government", "national security", "motion to seal"],
        "date_after": "2018-01-01",  # Post-China Initiative era
        "max_results": 100,
        "priority": "HIGHEST",
        "description": "Academic institutions + foreign influence + NS sealing"
    },

    # PRIORITY 5: Research security + sealing (universities + NS)
    {
        "topic": "research_security_sealing",
        "courts": None,
        "keywords": ["research security", "academic research", "university", "motion to seal", "national security"],
        "date_after": "2018-01-01",  # Research security became prominent post-China Initiative
        "max_results": 100,
        "priority": "HIGH",
        "description": "Research security + NS sealing motions (universities)"
    },

    # PRIORITY 6: Harvard/Ivy League specific (if any)
    {
        "topic": "harvard_ivy_ns_sealing",
        "courts": None,
        "keywords": ["Harvard", "Ivy League", "national security", "motion to seal"],
        "date_after": "2000-01-01",
        "max_results": 50,
        "priority": "HIGH",
        "description": "Harvard/Ivy League + NS sealing motions"
    },

    # PRIORITY 7: Foreign government + US citizen + NS
    {
        "topic": "foreign_gov_us_citizen_ns",
        "courts": None,
        "keywords": ["foreign government", "US citizen", "American citizen", "national security", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "Foreign government + US citizen endangerment + NS"
    },

    # PRIORITY 8: Pseudonym + NS (endangerment context)
    {
        "topic": "ns_pseudonym_endangerment",
        "courts": None,
        "keywords": ["national security", "motion to proceed under pseudonym", "endangerment", "safety"],
        "date_after": "2010-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "NS-related pseudonym motions (endangerment context)"
    },

    # PRIORITY 9: D. Mass - academic/university + NS
    {
        "topic": "dmass_academic_ns",
        "courts": ["mad"],
        "keywords": ["university", "academic", "national security", "motion to seal"],
        "date_after": "2000-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "D. Mass academic/university + NS sealing motions"
    },

    # PRIORITY 10: FARA + academic institutions (foreign influence reporting)
    {
        "topic": "fara_academic_ns",
        "courts": None,
        "keywords": ["FARA", "Foreign Agents Registration Act", "university", "academic", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "HIGH",
        "description": "FARA + academic institutions + NS sealing"
    },

    # PRIORITY 11: More recent cases (2022+) - your case timeframe
    {
        "topic": "ns_sealing_very_recent",
        "courts": None,
        "keywords": ["national security", "motion to seal"],
        "date_after": "2022-01-01",
        "max_results": 100,
        "priority": "HIGH",
        "description": "Very recent NS sealing motions (2022+)"
    },

    # PRIORITY 12: D. Mass - broader historical search
    {
        "topic": "ns_sealing_dmass_historical",
        "courts": ["mad"],
        "keywords": ["national security", "motion to seal"],
        "date_after": "2000-01-01",  # Earlier than our 2010 cutoff
        "max_results": 100,
        "priority": "MEDIUM",
        "description": "D. Mass NS sealing (2000-2010, historical)"
    },

    # Lower priority - other NS categories
    {
        "topic": "fisa_sealing",
        "courts": None,
        "keywords": ["FISA", "Foreign Intelligence Surveillance Act", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "FISA-related sealing motions"
    },

    {
        "topic": "export_control_sealing",
        "courts": None,
        "keywords": ["export control", "ITAR", "EAR", "motion to seal"],
        "date_after": "2010-01-01",
        "max_results": 50,
        "priority": "MEDIUM",
        "description": "Export control + sealing motions"
    },
]


def download_ns_batch(query_config: Dict[str, Any], client: CourtListenerClient) -> Dict[str, Any]:
    """Download a single batch of NS cases."""
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
    """Download additional NS cases with expanded search terms."""
    logger.info("=" * 80)
    logger.info("ADDITIONAL NATIONAL SECURITY CASES DOWNLOAD")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Target: Download additional NS cases using expanded keywords")
    logger.info("Focus: FISA, export control, CFIUS, security clearance, pseudonym motions")
    logger.info("")

    # Initialize client
    possible_configs = [
        Path("case_law_data/local_courtlistener_config.json"),
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
        return

    try:
        client = CourtListenerClient(config_path=config_path, overnight_mode=False)
        logger.info("CourtListener client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CourtListener client: {e}")
        return

    # Download each batch
    results = []
    total_downloaded = 0

    for query_config in ADDITIONAL_NS_QUERIES:
        result = download_ns_batch(query_config, client)
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
    logger.info("1. Import downloaded cases using: python case_law_data/scripts/ingest_ns_sealing_cases.py")
    logger.info("2. Tag cases with tag_national_security = 1")
    logger.info("3. Analyze new cases for patterns")
    logger.info("")

    # Save results
    output_path = Path("outputs/additional_ns_download_results.json")
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

