from __future__ import annotations

import json
import sqlite3
import textwrap
from types import SimpleNamespace

import pytest


@pytest.fixture
def personal_facts_fixture(tmp_path):
    """
    Realistic personal facts bundle for integration tests.

    Creates:
    - Personal corpus directory with HK Statement + OGC email excerpts
    - lawsuit_facts_extracted.json containing fact blocks + aliases
    - lawsuit_facts_database.db SQLite database with document_facts/legal_implications
    """
    corpus_dir = tmp_path / "lawsuit_source_documents"
    corpus_dir.mkdir()

    (corpus_dir / "HK Statement.txt").write_text(
        textwrap.dedent(
            """
            Hong Kong Statement of Claim filed on June 2, 2025 details defamation and harassment.
            It cites the June 4, 2025 arrests as retaliation for exposing privacy breaches.
            """
        ).strip(),
        encoding="utf-8",
    )
    (corpus_dir / "OGC Email.txt").write_text(
        textwrap.dedent(
            """
            Office of General Counsel email dated April 7, 2025 threatens the plaintiff for reporting
            privacy breaches. A follow-up on April 18, 2025 refuses to correct the defamation.
            """
        ).strip(),
        encoding="utf-8",
    )
    (corpus_dir / "Timeline.txt").write_text(
        "June 2025 arrests escalated retaliation and harassment across Harvard systems.",
        encoding="utf-8",
    )

    fact_blocks = {
        "hk_allegation_defamation": (
            "The Hong Kong Statement of Claim filed June 2, 2025 documents defamation, harassment, "
            "and privacy breaches stemming from Harvard's disclosures."
        ),
        "harvard_retaliation_events": (
            "Harvard escalated retaliation on June 4, 2025 by coordinating arrests after the HK Statement."
        ),
        "ogc_email_1_threat": (
            "April 7, 2025 OGC email admits privacy breaches yet threatens plaintiff for escalating."
        ),
        "ogc_email_2_non_response": (
            "April 18, 2025 OGC follow-up refuses to retract defamatory statements."
        ),
        "privacy_leak_events": "Harvard leaked sealed HK Statement exhibits causing privacy harm.",
        "safety_concerns": "June 2025 arrests and surveillance raise safety and retaliation concerns.",
    }
    extracted_facts = {
        "timeline": [
            "April 2025 OGC emails show institutional retaliation.",
            "June 2025 arrests targeted the whistleblower for exposing privacy leaks.",
        ]
    }
    case_insights_payload = {
        "reference_id": "fixture_case",
        "summary": "Fixture insights for personal corpus validation.",
        "jurisdiction": "D. Mass.",
        "case_style": "Doe v. Harvard",
        "fact_blocks": fact_blocks,
        "extracted_facts": extracted_facts,
        "aliases": {
            "hk_statement": ["Hong Kong Statement of Claim", "HK Statement"],
            "ogc_emails": ["Office of General Counsel", "Harvard OGC emails"],
        },
    }
    lawsuit_facts_extracted_path = tmp_path / "lawsuit_facts_extracted.json"
    lawsuit_facts_extracted_path.write_text(json.dumps(case_insights_payload, indent=2), encoding="utf-8")
    # Also create old name for backward compatibility
    case_insights_path = tmp_path / "case_insights.json"
    case_insights_path.write_text(json.dumps(case_insights_payload, indent=2), encoding="utf-8")

    lawsuit_facts_db_path = tmp_path / "lawsuit_facts_database.db"
    conn = sqlite3.connect(lawsuit_facts_db_path)
    # Create new table names
    conn.execute(
        """
        CREATE TABLE document_facts (
            fact_id TEXT PRIMARY KEY,
            source_doc TEXT,
            fact_type TEXT,
            fact_text TEXT,
            citation TEXT,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE legal_implications (
            implication_id TEXT PRIMARY KEY,
            derived_from TEXT,
            legal_implication TEXT,
            supporting_fact_ids TEXT,
            jurisdiction_assumption TEXT,
            rationale TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO document_facts (fact_id, source_doc, fact_type, fact_text, citation, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "fact-hk-1",
            "Hong Kong Statement of Claim",
            "hk_statement",
            "HK Statement confirms defamation and privacy breaches tied to June 2025 arrests.",
            "HK Statement",
            0.95,
        ),
    )
    conn.execute(
        """
        INSERT INTO legal_implications (
            implication_id, derived_from, legal_implication, supporting_fact_ids,
            jurisdiction_assumption, rationale
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "implication-retaliation-1",
            "fact-hk-1",
            "April 7 and April 18 OGC emails establish retaliation and privacy torts.",
            "fact-hk-1",
            "D. Mass.",
            "Combines OGC correspondence with HK arrests timeline.",
        ),
    )
    # Also create old tables for backward compatibility
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_facts (
            truth_id TEXT PRIMARY KEY,
            source_doc TEXT,
            fact_type TEXT,
            fact_text TEXT,
            citation TEXT,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS derived_truths (
            truth_id TEXT PRIMARY KEY,
            derived_from TEXT,
            legal_implication TEXT,
            supporting_fact_ids TEXT,
            jurisdiction_assumption TEXT,
            rationale TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO source_facts (truth_id, source_doc, fact_type, fact_text, citation, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "truth-hk-1",
            "Hong Kong Statement of Claim",
            "hk_statement",
            "HK Statement confirms defamation and privacy breaches tied to June 2025 arrests.",
            "HK Statement",
            0.95,
        ),
    )
    conn.execute(
        """
        INSERT INTO derived_truths (
            truth_id, derived_from, legal_implication, supporting_fact_ids,
            jurisdiction_assumption, rationale
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "derived-retaliation-1",
            "truth-hk-1",
            "April 7 and April 18 OGC emails establish retaliation and privacy torts.",
            "truth-hk-1",
            "D. Mass.",
            "Combines OGC correspondence with HK arrests timeline.",
        ),
    )
    conn.commit()
    conn.close()

    personal_facts = {
        "fact_blocks": fact_blocks,
        "aliases": case_insights_payload["aliases"],
    }

    evidence_payload = {
        "fact_blocks": fact_blocks,
        "extracted_facts": extracted_facts,
    }

    # Also create old database name for backward compatibility
    truths_db_path = tmp_path / "truths.db"
    import shutil
    shutil.copy(lawsuit_facts_db_path, truths_db_path)
    
    return SimpleNamespace(
        corpus_dir=corpus_dir,
        case_insights_path=case_insights_path,  # Old name for backward compatibility
        lawsuit_facts_extracted_path=lawsuit_facts_extracted_path,  # New name
        truths_db_path=truths_db_path,  # Old name for backward compatibility
        lawsuit_facts_db_path=lawsuit_facts_db_path,  # New name
        fact_blocks=fact_blocks,
        extracted_facts=extracted_facts,
        personal_facts=personal_facts,
        evidence_payload=evidence_payload,
    )
