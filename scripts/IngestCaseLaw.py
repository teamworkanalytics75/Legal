"""
The Matrix Case Law Download Pipeline

Complete workflow: Download -> Import to MySQL -> Backup to Google Drive
With checkpoint/resume every 50 cases for safety.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / "document_ingestion"))

from download_case_law import download_massachusetts_topics, CourtListenerClient # type: ignore
from update_db_with_cases import import_all_topics, get_case_count, create_case_law_table # type: ignore
from db_setup import create_database_connection, initialize_database # type: ignore

try:
    from google_drive_backup import backup_all_topics # type: ignore
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("case_law_pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_system() -> bool:
    """Initialize database."""
    logger.info("Initializing The Matrix system...")

    if not initialize_database():
        return False

    connection = create_database_connection(include_db=True)
    if not connection:
        return False

    success = create_case_law_table(connection)
    connection.close()

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="The Matrix Case Law Pipeline")

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--import", dest="import_db", action="store_true")
    parser.add_argument("--backup-to-gdrive", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--topics", nargs="+")
    parser.add_argument("--max-per-topic", type=int, default=3000)
    parser.add_argument("--date-after", default="2000-01-01")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    setup_logging(args.verbose)

    print("\n" + "="*60)
    print("WITCHWEB CASE LAW PIPELINE")
    print("="*60 + "\n")

    if args.init:
        if initialize_system():
            print("Database initialized successfully!")
        return

    if not initialize_system():
        logger.error("System initialization failed")
        sys.exit(1)

    try:
        if args.download:
            logger.info("="*60)
            logger.info("PHASE 1: DOWNLOADING")
            logger.info("="*60)

            download_massachusetts_topics(
                topics=args.topics,
                date_after=args.date_after,
                max_per_topic=args.max_per_topic
            )

        if args.import_db:
            logger.info("="*60)
            logger.info("PHASE 2: IMPORTING TO DATABASE")
            logger.info("="*60)

            client = CourtListenerClient()
            import_all_topics(client.local_dir)

        if args.backup_to_gdrive and GDRIVE_AVAILABLE:
            logger.info("="*60)
            logger.info("PHASE 3: GOOGLE DRIVE BACKUP")
            logger.info("="*60)

            client = CourtListenerClient()
            backup_all_topics(client.local_dir)

        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted - progress saved!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

