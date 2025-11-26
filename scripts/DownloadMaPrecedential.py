#!/usr/bin/env python3
"""
Download Massachusetts Precedential Cases for Motion Strategy
Focus: Pseudonyms/Sealing and Standing Orders

This script downloads ONLY:
- MA state courts (SJC, Appeals Court, Superior Court)
- Precedential opinions (binding/persuasive for MA judges)
- Cases from 2000-present (modern internet era)
"""

import sys
import logging
from pathlib import Path
from typing import Dict
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListenerClient directly to avoid module init issues
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "Agents_1782_ML_Dataset" / "document_ingestion" / "DownloadCaseLaw.py"
)
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)
CourtListenerClient = download_module.CourtListenerClient

# Prefer the repo-scoped config that includes nationwide jurisdictions/storage.
LOCAL_CONFIG_PATH = (
    Path(__file__).parent.parent / "case_law_data" / "local_courtlistener_config.json"
)
FALLBACK_CONFIG_PATH = (
    Path(__file__).parent.parent
    / "Agents_1782_ML_Dataset"
    / "document_ingestion"
    / "CourtlistenerConfig.json"
)

CONFIG_PATH = LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else FALLBACK_CONFIG_PATH

# Setup logging with beautiful output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

if CONFIG_PATH == LOCAL_CONFIG_PATH:
    logger.info("Using local CourtListener config: %s", CONFIG_PATH)
else:
    logger.warning(
        "Local CourtListener config missing; using fallback config at %s",
        CONFIG_PATH
    )


def download_ma_precedential_for_motion(
    *,
    test_run: bool = True,
    pseudonym_target: int | None = None,
    standing_target: int | None = None,
    date_after: str = "2000-01-01",
    jurisdiction_sets: list[str] | None = None,
    topic_prefix: str = "ma_precedential",
    include_non_precedential: bool = False,
) -> Dict[str, int]:
    """
    Download MA precedential cases for motion strategy.

    Args:
        test_run: If True, download 500 cases (250 per topic).
                  If False, download 5000 cases (2500 per topic).

    Returns:
        Dictionary with download statistics.
    """
    logger.info("="*70)
    logger.info("MA PRECEDENTIAL CASES DOWNLOAD")
    logger.info("For Motion to Seal/Pseudonym Strategy")
    logger.info("="*70)

    # Initialize client
    logger.info("\nInitializing CourtListener client...")
    client = CourtListenerClient(config_path=str(CONFIG_PATH))

    # Resolve courts based on jurisdiction sets
    jurisdictions = client.config['filters'].get('jurisdictions', {})
    if jurisdiction_sets is None:
        jurisdiction_sets = ['massachusetts_state']

    courts: list[str] = []
    for key in jurisdiction_sets:
        values = jurisdictions.get(key)
        if not values:
            logger.warning(f"  - Unknown jurisdiction set '{key}' - skipping")
            continue
        courts.extend(values)

    if not courts:
        courts = ['mass', 'massappct', 'masssuperct']
        logger.warning("No valid jurisdiction sets supplied. Falling back to Massachusetts state courts.")

    courts = sorted(set(courts))

    logger.info(f"Target courts: {', '.join(courts)}")

    # Topics from config
    topics_config = client.config['filters']['topics']

    # Calculate download sizes
    default_size = 250 if test_run else 2500
    pseudonym_target = pseudonym_target or default_size
    standing_target = standing_target or default_size

    logger.info(f"\nTarget download: {pseudonym_target + standing_target} cases")
    logger.info(f"  - Pseudonyms/Sealing: {pseudonym_target}")
    logger.info(f"  - Standing Orders: {standing_target}")
    logger.info(f"  - Date range: {date_after} to present")
    logger.info(f"  - Include non-precedential: {'YES' if include_non_precedential else 'NO'}")
    logger.info(f"  - Storage prefix: {topic_prefix}\n")

    results = {}

    # ========== Topic 1: Pseudonyms & Sealing ==========
    logger.info("\n" + "="*70)
    logger.info("TOPIC 1: PSEUDONYMS & SEALING")
    logger.info("="*70)
    logger.info("Keywords: pseudonym, doe plaintiff, seal, protective order, anonymity")
    logger.info(f"Target: {pseudonym_target} cases\n")

    try:
        pseudonym_cases = client.bulk_download(
            topic=f'{topic_prefix}/pseudonyms_sealing',
            courts=courts,
            keywords=topics_config['pseudonyms_sealing'],
            date_after=date_after,
            max_results=pseudonym_target,
            include_non_precedential=include_non_precedential,
            apply_filters=False  # disable 1782 filter for MA motions
        )

        results['pseudonyms_sealing'] = len(pseudonym_cases)
        logger.info(f"\n[OK] Successfully downloaded {len(pseudonym_cases)} pseudonym/sealing cases")

    except Exception as e:
        logger.error(f"\n[ERROR] Error downloading pseudonym cases: {e}")
        results['pseudonyms_sealing'] = 0

    # ========== Topic 2: Standing Orders ==========
    logger.info("\n" + "="*70)
    logger.info("TOPIC 2: STANDING ORDERS")
    logger.info("="*70)
    logger.info("Keywords: standing order, local rule, district court order")
    logger.info(f"Target: {standing_target} cases\n")

    try:
        standing_cases = client.bulk_download(
            topic=f'{topic_prefix}/standing_orders',
            courts=courts,
            keywords=topics_config['standing_orders'],
            date_after=date_after,
            max_results=standing_target,
            include_non_precedential=include_non_precedential,
            apply_filters=False  # disable 1782 filter for MA motions
        )

        results['standing_orders'] = len(standing_cases)
        logger.info(f"\n[OK] Successfully downloaded {len(standing_cases)} standing order cases")

    except Exception as e:
        logger.error(f"\n[ERROR] Error downloading standing order cases: {e}")
        results['standing_orders'] = 0

    # ========== Summary ==========
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD COMPLETE - SUMMARY")
    logger.info("="*70)

    total_downloaded = results.get('pseudonyms_sealing', 0) + results.get('standing_orders', 0)
    results['total'] = total_downloaded

    logger.info(f"Pseudonyms/Sealing: {results.get('pseudonyms_sealing', 0)} cases")
    logger.info(f"Standing Orders:    {results.get('standing_orders', 0)} cases")
    logger.info(f"TOTAL:              {total_downloaded} cases")

    # Show storage location
    storage_path = Path(client.config['storage']['local_base_dir'])
    logger.info(f"\nFiles saved to: {storage_path}")
    logger.info(f"  - Pseudonyms: {storage_path / topic_prefix / 'pseudonyms_sealing'}")
    logger.info(f"  - Standing:   {storage_path / topic_prefix / 'standing_orders'}")
    logger.info(f"  - Checkpoints: {storage_path / '.checkpoints'}")

    # Next steps
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("1. Build database:")
    logger.info("   cd Agents_1782_ML_Dataset")
    logger.info("   python ../case_law_data/create_ma_database.py")
    logger.info("")
    logger.info("2. Run analysis:")
    logger.info("   ..\\.venv\\Scripts\\python.exe scripts/09_ma_motion_analysis.py")
    logger.info("="*70)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MA precedential cases for motion strategy"
    )
    parser.add_argument(
        '--date-after',
        default='2000-01-01',
        help='Restrict cases to those filed on/after this date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full download (5000 cases instead of 500 test run)'
    )
    parser.add_argument(
        '--pseudonym-count',
        type=int,
        help='Override target count for pseudonyms/sealing cases'
    )
    parser.add_argument(
        '--standing-count',
        type=int,
        help='Override target count for standing order cases'
    )
    parser.add_argument(
        '--topic-prefix',
        help='Base directory prefix for storage (default: ma_precedential or ma_non_precedential when including unpublished cases)'
    )
    parser.add_argument(
        '--jurisdiction-set',
        action='append',
        help='Jurisdiction set key from courtlistener_config filters (can be provided multiple times)'
    )
    parser.add_argument(
        '--include-non-precedential',
        action='store_true',
        help='Include non-precedential opinions'
    )

    args = parser.parse_args()

    # Run download
    test_run = not args.full

    default_target = 250 if test_run else 2500
    desired_pseudonym = args.pseudonym_count if args.pseudonym_count is not None else default_target
    desired_standing = args.standing_count if args.standing_count is not None else default_target
    total_target = desired_pseudonym + desired_standing

    topic_prefix = args.topic_prefix
    if topic_prefix is None:
        topic_prefix = 'ma_non_precedential' if args.include_non_precedential else 'ma_precedential'

    mode_label = "TEST RUN" if test_run else "FULL RUN"
    print(f"\n[{mode_label}] Targeting {total_target} cases "
          f"({desired_pseudonym} pseudonym/sealing + {desired_standing} standing orders)")
    if test_run:
        print("   Use --full for larger default targets or override counts with --pseudonym-count/--standing-count\n")

    results = download_ma_precedential_for_motion(
        test_run=test_run,
        pseudonym_target=args.pseudonym_count,
        standing_target=args.standing_count,
        date_after=args.date_after,
        jurisdiction_sets=args.jurisdiction_set,
        topic_prefix=topic_prefix,
        include_non_precedential=args.include_non_precedential,
    )

    # Exit with appropriate code
    if results['total'] == 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
