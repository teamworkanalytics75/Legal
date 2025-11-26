#!/usr/bin/env python3
"""
Build precedent database from LangChain capability seeding results.

Extracts precedents from seeded capabilities and stores them with
capability-based tags for future reference.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def create_precedent_database(db_path: Path) -> None:
    """Create precedent database with capability-based schema."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create precedents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS precedents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            precedent_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT NOT NULL,
            capability_tags TEXT,  -- JSON array of capability tags
            source_agent TEXT,
            source_query TEXT,
            sql_pattern TEXT,
            example_result TEXT,
            jurisdiction TEXT,
            case_type TEXT,
            relevance_score REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create capability_tags table for normalized tags
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS capability_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_name TEXT UNIQUE NOT NULL,
            tag_category TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create precedent_capability_mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS precedent_capability_mapping (
            precedent_id TEXT NOT NULL,
            capability_tag TEXT NOT NULL,
            relevance_weight REAL DEFAULT 1.0,
            PRIMARY KEY (precedent_id, capability_tag),
            FOREIGN KEY (precedent_id) REFERENCES precedents(precedent_id),
            FOREIGN KEY (capability_tag) REFERENCES capability_tags(tag_name)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedents_category ON precedents(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedents_jurisdiction ON precedents(jurisdiction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedents_relevance ON precedents(relevance_score)")

    conn.commit()
    conn.close()

    print(f"Created precedent database at {db_path}")


def extract_precedents_from_capabilities(capabilities_file: Path) -> List[Dict[str, Any]]:
    """Extract precedents from capability seeding results."""

    with open(capabilities_file, 'r', encoding='utf-8') as f:
        capabilities = json.load(f)

    precedents = []

    # Extract universal skills as precedents
    universal_skills = capabilities.get("universal_skills", [])
    for skill in universal_skills:
        precedent = {
            "precedent_id": f"universal_{skill['id']}",
            "title": f"Universal Skill: {skill['id']}",
            "description": skill["nl"],
            "category": "universal_skill",
            "capability_tags": json.dumps([skill["category"], "universal"]),
            "source_agent": "UniversalAgent",
            "source_query": skill["nl"],
            "sql_pattern": extract_sql_pattern(skill["nl"]),
            "example_result": f"Example result for {skill['id']}",
            "jurisdiction": "general",
            "case_type": "procedural",
            "relevance_score": 1.0
        }
        precedents.append(precedent)

    # Extract agent-specific precedents
    agents = capabilities.get("agents", {})
    for agent_name, agent_data in agents.items():
        agent_info = agent_data.get("agent_info", {})

        # Job primitives as precedents
        primitives = agent_data.get("job_primitives", [])
        for primitive in primitives:
            precedent = {
                "precedent_id": f"{agent_name}_{primitive['name']}",
                "title": f"{agent_name}: {primitive['name']}",
                "description": primitive["nl"],
                "category": "job_primitive",
                "capability_tags": json.dumps([primitive.get("category", "generic"), agent_name.lower()]),
                "source_agent": agent_name,
                "source_query": primitive["nl"],
                "sql_pattern": extract_sql_pattern(primitive["nl"]),
                "example_result": f"Example result for {primitive['name']}",
                "jurisdiction": "massachusetts",  # Default for this case
                "case_type": "legal_analysis",
                "relevance_score": 0.8
            }
            precedents.append(precedent)

        # Domain anchors as precedents
        anchors = agent_data.get("domain_anchors", [])
        for i, anchor in enumerate(anchors):
            precedent = {
                "precedent_id": f"{agent_name}_anchor_{i+1}",
                "title": f"{agent_name}: Domain Anchor {i+1}",
                "description": anchor["nl"],
                "category": "domain_anchor",
                "capability_tags": json.dumps(["domain_specific", agent_name.lower()]),
                "source_agent": agent_name,
                "source_query": anchor["nl"],
                "sql_pattern": extract_sql_pattern(anchor["nl"]),
                "example_result": f"Domain-specific result for {agent_name}",
                "jurisdiction": "massachusetts",
                "case_type": "legal_analysis",
                "relevance_score": 0.6
            }
            precedents.append(precedent)

    return precedents


def extract_sql_pattern(nl_query: str) -> str:
    """Extract SQL pattern from natural language query."""
    # Simple pattern extraction - in production this would be more sophisticated
    if "list all tables" in nl_query.lower():
        return "SELECT * FROM information_schema.tables"
    elif "count" in nl_query.lower():
        return "SELECT COUNT(*) FROM {table}"
    elif "find documents" in nl_query.lower():
        return "SELECT * FROM cleaned_documents WHERE content LIKE '%{phrase}%'"
    elif "between" in nl_query.lower() and "date" in nl_query.lower():
        return "SELECT * FROM {table} WHERE date_column BETWEEN '{start}' AND '{end}'"
    else:
        return "SELECT * FROM {table} WHERE {condition}"


def insert_precedents(db_path: Path, precedents: List[Dict[str, Any]]) -> None:
    """Insert precedents into database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for precedent in precedents:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO precedents (
                    precedent_id, title, description, category, capability_tags,
                    source_agent, source_query, sql_pattern, example_result,
                    jurisdiction, case_type, relevance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                precedent["precedent_id"],
                precedent["title"],
                precedent["description"],
                precedent["category"],
                precedent["capability_tags"],
                precedent["source_agent"],
                precedent["source_query"],
                precedent["sql_pattern"],
                precedent["example_result"],
                precedent["jurisdiction"],
                precedent["case_type"],
                precedent["relevance_score"]
            ))
        except sqlite3.Error as e:
            print(f"Error inserting precedent {precedent['precedent_id']}: {e}")

    conn.commit()
    conn.close()

    print(f"Inserted {len(precedents)} precedents into database")


def create_capability_tags(db_path: Path) -> None:
    """Create capability tags from precedents."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define capability tags
    tags = [
        ("schema", "universal", "Database schema operations"),
        ("search", "universal", "Text search and filtering"),
        ("temporal", "universal", "Date and time operations"),
        ("data_quality", "universal", "Data cleaning and validation"),
        ("performance", "universal", "Query optimization and pagination"),
        ("audit", "universal", "Logging and cost tracking"),
        ("error_handling", "universal", "Resilient query processing"),
        ("citation", "job_specific", "Citation extraction and verification"),
        ("precedent", "job_specific", "Precedent finding and ranking"),
        ("fact_extraction", "job_specific", "Fact extraction and analysis"),
        ("writing", "job_specific", "Document writing and formatting"),
        ("quality", "job_specific", "Quality assurance and checking"),
        ("export", "job_specific", "Document export and formatting"),
        ("reference", "job_specific", "Statute and exhibit location"),
        ("domain_specific", "domain", "Domain-specific validation"),
        ("universal", "meta", "Universal capability"),
        ("generic", "meta", "Generic capability")
    ]

    for tag_name, tag_category, description in tags:
        cursor.execute("""
            INSERT OR IGNORE INTO capability_tags (tag_name, tag_category, description)
            VALUES (?, ?, ?)
        """, (tag_name, tag_category, description))

    conn.commit()
    conn.close()

    print(f"Created {len(tags)} capability tags")


def generate_precedent_report(db_path: Path) -> Dict[str, Any]:
    """Generate summary report of precedent database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get counts by category
    cursor.execute("SELECT category, COUNT(*) FROM precedents GROUP BY category")
    category_counts = dict(cursor.fetchall())

    # Get counts by jurisdiction
    cursor.execute("SELECT jurisdiction, COUNT(*) FROM precedents GROUP BY jurisdiction")
    jurisdiction_counts = dict(cursor.fetchall())

    # Get top capability tags
    cursor.execute("""
        SELECT capability_tags, COUNT(*) as count
        FROM precedents
        GROUP BY capability_tags
        ORDER BY count DESC
        LIMIT 10
    """)
    top_tags = cursor.fetchall()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM precedents")
    total_precedents = cursor.fetchone()[0]

    conn.close()

    report = {
        "total_precedents": total_precedents,
        "category_breakdown": category_counts,
        "jurisdiction_breakdown": jurisdiction_counts,
        "top_capability_tags": top_tags,
        "generated_at": datetime.utcnow().isoformat()
    }

    return report


def main():
    """Main execution function."""

    # Paths
    capabilities_file = Path("config/langchain_capability_seeds.json")
    db_path = Path("precedent_database.db")

    print("Building precedent database from capability seeding results...")

    # Create database
    create_precedent_database(db_path)

    # Extract precedents from capabilities
    precedents = extract_precedents_from_capabilities(capabilities_file)
    print(f"Extracted {len(precedents)} precedents from capabilities")

    # Insert precedents
    insert_precedents(db_path, precedents)

    # Create capability tags
    create_capability_tags(db_path)

    # Generate report
    report = generate_precedent_report(db_path)

    # Save report
    report_file = Path("precedent_database_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nPrecedent Database Summary:")
    print(f"Total precedents: {report['total_precedents']}")
    print(f"Categories: {report['category_breakdown']}")
    print(f"Jurisdictions: {report['jurisdiction_breakdown']}")
    print(f"Report saved to: {report_file}")

    return 0


if __name__ == "__main__":
    exit(main())
