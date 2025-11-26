#!/usr/bin/env python3
"""
LangChain Query System for Case Materials

Queries your databases about:
- Your lawsuit
- HK statement of claim
- 1782 draft materials
- Deep research papers
"""

import os
import sys
from pathlib import Path

# Add writer_agents to path
sys.path.insert(0, str(Path(__file__).parent / "writer_agents" / "code"))

from LangchainIntegration import LangChainSQLAgent
from agents import ModelConfig

def find_databases():
    """Find all available databases."""
    databases = {}

    # MySQL database (lawsuit docs)
    mysql_config = {
        "type": "mysql",
        "host": os.getenv("DOCUMENT_INGESTION_DB_HOST", "localhost"),
        "database": os.getenv("DOCUMENT_INGESTION_DB_NAME", "lawsuit_docs"),
        "path": None  # MySQL uses connection string
    }
    databases["lawsuit_docs"] = mysql_config

    # SQLite databases
    sqlite_dbs = [
        ("unified_corpus", "case_law_data/unified_corpus.db"),
        ("ma_federal_motions", "case_law_data/ma_federal_motions.db"),
        ("harvard_corpus", "case_law_data/harvard_corpus.db"),
        ("1782_discovery", "Agents_1782_ML_Dataset/data/case_law/1782_discovery/Section1782.db"),
        ("esuwiki", "databases/esuwiki.db"),  # EsuWiki research documents
    ]

    for name, rel_path in sqlite_dbs:
        db_path = Path(rel_path)
        if db_path.exists():
            databases[name] = {
                "type": "sqlite",
                "path": db_path.resolve(),
                "exists": True
            }
        else:
            databases[name] = {
                "type": "sqlite",
                "path": db_path,
                "exists": False
            }

    return databases

def create_query_agent(db_path: Path):
    """Create LangChain query agent for a SQLite database."""
    try:
        model_config = ModelConfig(model="gpt-4o-mini")
        agent = LangChainSQLAgent(db_path, model_config, verbose=True)
        return agent
    except Exception as e:
        print(f"ERROR: Failed to create agent: {e}")
        return None

def query_examples():
    """Show example queries for case materials."""
    examples = {
        "HK Statement of Claim": [
            "Find documents mentioning Hong Kong statement of claim",
            "Show me the Hong Kong legal proceedings",
            "What does the HK statement of claim say about the case?",
        ],
        "1782 Draft Materials": [
            "Find all Section 1782 discovery draft materials",
            "Show documents related to 1782 petition drafting",
            "What 1782 discovery materials do we have?",
        ],
        "Research Papers": [
            "Find deep research papers about the case",
            "Show research documents and analysis papers",
            "What research materials do we have?",
        ],
        "Lawsuit Documents": [
            "Find all documents related to my lawsuit",
            "Show me the key facts from the case documents",
            "What evidence do we have about Harvard and China?",
        ]
    }
    return examples

def main():
    """Main query interface."""
    print("="*70)
    print("LANGCHAIN CASE MATERIALS QUERY SYSTEM")
    print("="*70)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not set!")
        print("Set it with: $env:OPENAI_API_KEY='your-key' (PowerShell)")
        print("Or: export OPENAI_API_KEY='your-key' (Linux/Mac)")
        return

    # Find databases
    print("\n[1/3] Finding databases...")
    databases = find_databases()

    available = {k: v for k, v in databases.items() if v.get("exists", True)}
    print(f"\nFound {len(available)} available database(s):")
    for name, info in available.items():
        if info["type"] == "sqlite":
            print(f"  - {name}: {info['path']}")
        else:
            print(f"  - {name}: MySQL ({info['host']}/{info['database']})")

    if not available:
        print("\n[WARNING] No databases found!")
        return

    # Select database (for now, use unified_corpus or first SQLite)
    target_db = None
    for name in ["unified_corpus", "1782_discovery", "harvard_corpus"]:
        if name in available and available[name]["type"] == "sqlite":
            target_db = available[name]["path"]
            print(f"\n[2/3] Using database: {name}")
            break

    if not target_db:
        print("\n[ERROR] No SQLite database found to query")
        return

    # Create agent
    print("\n[3/3] Creating LangChain agent...")
    agent = create_query_agent(target_db)
    if not agent:
        return

    print(f"\nAgent ready! Found {len(agent.table_names)} table(s): {', '.join(agent.table_names)}")

    # Show example queries
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    examples = query_examples()
    for category, queries in examples.items():
        print(f"\n{category}:")
        for q in queries:
            print(f"  - {q}")

    # Interactive query loop
    print("\n" + "="*70)
    print("INTERACTIVE QUERY MODE")
    print("="*70)
    print("\nEnter queries (or 'quit' to exit):\n")

    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            print(f"\n[Querying...]")
            result = agent.query_evidence(query)

            if result.get("success"):
                print("\n[ANSWER]")
                print(result.get("answer", "No answer returned"))

                if result.get("sql_query"):
                    print(f"\n[SQL Generated]")
                    print(result["sql_query"])

                if result.get("table_data"):
                    print(f"\n[Data Found: {len(result['table_data'])} rows]")
            else:
                print(f"\n[ERROR] {result.get('error', 'Unknown error')}")

            print("\n" + "-"*70 + "\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")

if __name__ == "__main__":
    main()

