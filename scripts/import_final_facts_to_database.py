#!/usr/bin/env python3
"""
Import the curated final facts CSV into the shared lawsuit_facts_database.db.

Workflow
--------
- Load the curated CSV (defaults to case_law_data/top_1000_facts_for_chatgpt_final.csv).
- Map each row into the fact_registry schema, preserving contextual metadata.
- Optionally back up and replace the existing fact_registry rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "case_law_data" / "top_1000_facts_for_chatgpt_final.csv"
DEFAULT_DB = REPO_ROOT / "case_law_data" / "lawsuit_facts_database.db"
EXTRACTION_METHOD = "final_facts_csv"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryEntry:
    """Immutable container for the final fact mapped to fact_registry schema."""

    fact_id: str
    fact_type: str
    fact_value: str
    description: str
    source_doc: str
    extraction_method: str
    confidence: float
    metadata: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the final curated facts CSV into fact_registry.\n"
            "\n"
            "Steps:\n"
            "  - Read the CSV (605 facts) and normalize field names\n"
            "  - Map each record into fact_registry plus metadata JSON\n"
            "  - Replace or append to the current lawsuit_facts_database.db\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to case_law_data/top_1000_facts_for_chatgpt_final.csv",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB,
        help="Path to case_law_data/lawsuit_facts_database.db",
    )
    parser.add_argument(
        "--mode",
        choices=("replace", "append"),
        default="replace",
        help="replace (default) clears fact_registry before inserting, append preserves existing rows",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip automatic timestamped backups of the existing database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging smaller batches",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    LOGGER.info("Loaded %d row(s) from %s", len(rows), csv_path)
    return rows


def sanitize(value: Optional[str]) -> str:
    return (value or "").strip()


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "unspecified"


def infer_fact_type(row: Dict[str, str]) -> str:
    event_type = sanitize(row.get("eventtype"))
    if event_type:
        return event_type
    actor_role = sanitize(row.get("actorrole"))
    subject = sanitize(row.get("subject"))
    if actor_role:
        return f"actor_role:{slugify(actor_role)}"
    if subject:
        return f"subject:{slugify(subject)}"
    return "unspecified"


def build_source_doc(row: Dict[str, str]) -> str:
    evidence = sanitize(row.get("evidencetype"))
    classification = sanitize(row.get("classificationfixed_v3"))
    subject = sanitize(row.get("subject"))
    pieces = []
    if evidence:
        pieces.append(f"Evidence={evidence}")
    if classification:
        pieces.append(f"Classification={classification}")
    if subject:
        pieces.append(f"Subject={subject}")
    return " | ".join(pieces) if pieces else "final_facts_dataset"


def parse_float(value: Optional[str]) -> Optional[float]:
    cleaned = sanitize(value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        LOGGER.debug("Unable to parse float from %s", value)
        return None


def build_metadata(row: Dict[str, str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    mapping = {
        "truth_status": "truthstatus",
        "causal_salience_reason": "causal_salience_reason",
        "event_date": "eventdate",
        "event_location": "eventlocation",
        "manual_salience_tier": "manual_salience_tier",
        "new_salience_score": "new_salience_score",
        "public_exposure": "publicexposure",
        "safety_risk": "safetyrisk",
        "subject": "subject",
        "actor_role": "actorrole",
    }
    for target, source in mapping.items():
        value = sanitize(row.get(source))
        if value:
            metadata[target] = value

    causal_score = parse_float(row.get("causal_salience_score"))
    if causal_score is not None:
        metadata["causal_salience_score"] = causal_score

    event_type = sanitize(row.get("eventtype"))
    if event_type:
        metadata["event_type_raw"] = event_type

    classification = sanitize(row.get("classificationfixed_v3"))
    if classification:
        metadata["classification"] = classification

    return metadata


def derive_confidence(metadata: Dict[str, Any]) -> float:
    score = metadata.get("causal_salience_score")
    if isinstance(score, (int, float)):
        # Keep confidence inside CatBoost-friendly bounds.
        return max(0.1, min(0.99, float(score)))
    return 0.9


FACT_REGISTRY_DDL = """
    CREATE TABLE IF NOT EXISTS fact_registry (
        fact_id TEXT PRIMARY KEY,
        fact_type TEXT,
        fact_value TEXT,
        description TEXT,
        source_doc TEXT,
        extraction_method TEXT,
        confidence REAL,
        metadata TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
"""


def ensure_fact_registry_schema(conn: sqlite3.Connection) -> None:
    conn.execute(FACT_REGISTRY_DDL)
    cursor = conn.execute("PRAGMA table_info(fact_registry)")
    columns = {row[1] for row in cursor.fetchall()}

    if "fact_id" not in columns:
        LOGGER.info("Migrating fact_registry to the new schema with fact_id column")
        conn.execute("ALTER TABLE fact_registry RENAME TO fact_registry_legacy")
        conn.execute(FACT_REGISTRY_DDL)
        conn.execute(
            """
            INSERT INTO fact_registry
            (fact_id, fact_type, fact_value, description, source_doc, extraction_method, confidence, metadata, created_at)
            SELECT
                COALESCE(CAST(id AS TEXT), 'legacy_' || ROWID) AS fact_id,
                fact_type,
                fact_value,
                description,
                source_doc,
                extraction_method,
                confidence,
                metadata,
                created_at
            FROM fact_registry_legacy
            """
        )
        conn.execute("DROP TABLE fact_registry_legacy")
        cursor = conn.execute("PRAGMA table_info(fact_registry)")
        columns = {row[1] for row in cursor.fetchall()}

    if "description" not in columns:
        conn.execute("ALTER TABLE fact_registry ADD COLUMN description TEXT")
    if "metadata" not in columns:
        conn.execute("ALTER TABLE fact_registry ADD COLUMN metadata TEXT")


def backup_database(database_path: Path) -> Optional[Path]:
    if not database_path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"{database_path.stem}_backup_{timestamp}{database_path.suffix}"
    backup_path = database_path.with_name(backup_name)
    shutil.copy2(database_path, backup_path)
    LOGGER.info("Backup created at %s", backup_path)
    return backup_path


def prepare_registry_entries(rows: Iterable[Dict[str, str]]) -> List[RegistryEntry]:
    entries: List[RegistryEntry] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, 1):
        fact_id = sanitize(row.get("factid"))
        if not fact_id:
            raise ValueError(f"Row {idx} missing factid")
        if fact_id in seen_ids:
            raise ValueError(f"Duplicate factid detected: {fact_id}")

        proposition = sanitize(row.get("proposition"))
        if not proposition:
            raise ValueError(f"Row {idx} missing proposition")

        fact_type = infer_fact_type(row)
        source_doc = build_source_doc(row)
        metadata = build_metadata(row)
        metadata_json = json.dumps(metadata, sort_keys=True) if metadata else None
        confidence = derive_confidence(metadata)

        entries.append(
            RegistryEntry(
                fact_id=fact_id,
                fact_type=fact_type,
                fact_value=proposition,
                description=proposition,
                source_doc=source_doc,
                extraction_method=EXTRACTION_METHOD,
                confidence=confidence,
                metadata=metadata_json,
            )
        )
        seen_ids.add(fact_id)
    LOGGER.info("Prepared %d registry entries", len(entries))
    return entries


def insert_entries(
    conn: sqlite3.Connection,
    entries: Sequence[RegistryEntry],
    mode: str,
) -> int:
    ensure_fact_registry_schema(conn)
    if mode == "replace":
        LOGGER.info("Clearing existing fact_registry rows")
        conn.execute("DELETE FROM fact_registry")

    sql = """
        INSERT OR REPLACE INTO fact_registry
        (fact_id, fact_type, fact_value, description, source_doc, extraction_method, confidence, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    payload = [
        (
            entry.fact_id,
            entry.fact_type,
            entry.fact_value,
            entry.description,
            entry.source_doc,
            entry.extraction_method,
            entry.confidence,
            entry.metadata,
        )
        for entry in entries
    ]
    conn.executemany(sql, payload)
    conn.commit()
    LOGGER.info("Inserted %d row(s) into fact_registry", len(entries))
    return len(entries)


def validate_import(conn: sqlite3.Connection, expected_count: int) -> None:
    cur = conn.execute("SELECT COUNT(*) FROM fact_registry")
    total = cur.fetchone()[0]
    LOGGER.info("fact_registry now contains %d row(s)", total)
    if total < expected_count:
        LOGGER.warning(
            "Expected at least %d row(s) but found %d. Verify --mode setting.",
            expected_count,
            total,
        )

    sample = conn.execute(
        "SELECT fact_id, fact_type, substr(fact_value, 1, 80) FROM fact_registry ORDER BY fact_id LIMIT 5"
    ).fetchall()
    if sample:
        LOGGER.info("Sample entries:")
        for fact_id, fact_type, snippet in sample:
            LOGGER.info(" - %s | %s | %s", fact_id, fact_type, snippet)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    rows = load_csv_rows(args.csv, limit=args.limit)
    entries = prepare_registry_entries(rows)
    args.database.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_backup:
        backup_database(args.database)

    with sqlite3.connect(args.database) as conn:
        inserted = insert_entries(conn, entries, args.mode)
        validate_import(conn, inserted if args.mode == "replace" else len(entries))

    LOGGER.info("Import complete")


if __name__ == "__main__":
    main()
