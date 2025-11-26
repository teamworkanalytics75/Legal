#!/usr/bin/env python3
"""
Apply the existing MaximumAdjudicationSystem heuristic/NLP pipeline to the
federal motions database so we can replace the simple keyword labels with the
more nuanced outcomes it produces.

This script:
  1. Loads `case_law_data/ma_federal_motions.db`
  2. Ensures the `cases` table has `adjudicated_outcome` and
     `adjudicated_confidence` columns
  3. Runs the adjudicator over each case's text
  4. Writes the outcome + confidence back into the database
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict

import importlib.util
import sys

# Reuse the existing maximum adjudicator implementation.
MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT))
MAX_ADJUDICATOR_PATH = Path('scripts/maximum_case_adjudicator.py').resolve()

spec = importlib.util.spec_from_file_location(
    "maximum_case_adjudicator", MAX_ADJUDICATOR_PATH
)
maximum_case_adjudicator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maximum_case_adjudicator)  # type: ignore[attr-defined]

MaximumAdjudicationSystem = maximum_case_adjudicator.MaximumAdjudicationSystem


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_columns(conn: sqlite3.Connection) -> None:
    """Add adjudication columns if they don't already exist."""

    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(cases)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "adjudicated_outcome" not in existing_cols:
        logger.info("Adding adjudicated_outcome column")
        conn.execute("ALTER TABLE cases ADD COLUMN adjudicated_outcome TEXT")

    if "adjudicated_confidence" not in existing_cols:
        logger.info("Adding adjudicated_confidence column")
        conn.execute("ALTER TABLE cases ADD COLUMN adjudicated_confidence REAL")

    conn.commit()


def main() -> None:
    db_path = Path("case_law_data/ma_federal_motions.db").resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found at {db_path}")

    logger.info(f"Preparing to adjudicate cases in {db_path}")

    adjudicator = MaximumAdjudicationSystem()

    with sqlite3.connect(db_path) as conn:
        ensure_columns(conn)

        cursor = conn.cursor()
        cursor.execute(
            "SELECT cluster_id, cleaned_text, adjudicated_outcome FROM cases"
        )

        total = 0
        updated = 0

        for cluster_id, cleaned_text, existing_outcome in cursor.fetchall():
            total += 1

            if existing_outcome:
                continue  # already adjudicated

            case_payload: Dict[str, object] = {
                "cluster_id": cluster_id,
                "opinion_text": cleaned_text or "",
            }

            analysis = adjudicator.adjudicate_case(case_payload)

            conn.execute(
                """
                UPDATE cases
                   SET adjudicated_outcome = ?,
                       adjudicated_confidence = ?
                 WHERE cluster_id = ?
                """,
                (analysis["outcome"], analysis["confidence"], cluster_id),
            )
            updated += 1

            if updated % 50 == 0:
                conn.commit()
                logger.info(f"Adjudicated {updated} cases so far...")

        conn.commit()

    logger.info(f"Adjudication complete. Total cases: {total}, updated: {updated}")


if __name__ == "__main__":
    main()
