#!/usr/bin/env python3
"""
Convert PDF Analysis to SQL Database and Set Up LangChain Querying
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add the writer_agents code to path for LangChain integration
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROJECT_ROOT / "writer_agents" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

def create_1782_database():
    """Create SQLite database from PDF analysis results."""

    print("🗄️ Creating 1782 SQL Database...")

    # Load PDF analysis results
    with open("data/case_law/simple_pdf_analysis.json", 'r') as f:
        pdf_results = json.load(f)

    # Create database connection
    db_path = "data/case_law/1782_pdf_analysis.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id TEXT UNIQUE,
            case_name TEXT,
            text_length INTEGER,
            word_count INTEGER,
            page_count INTEGER,
            file_size INTEGER,
            outcome TEXT,
            outcome_confidence REAL,
            granted_count INTEGER,
            denied_count INTEGER,
            section_1782_mentions INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intel_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id TEXT,
            factor_name TEXT,
            detected BOOLEAN,
            FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id TEXT,
            citation_pattern TEXT,
            citation_count INTEGER,
            FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS case_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id TEXT,
            page_number INTEGER,
            page_text TEXT,
            FOREIGN KEY (cluster_id) REFERENCES cases (cluster_id)
        )
    ''')

    # Insert case data
    print("📝 Inserting case data...")
    for case in pdf_results:
        cursor.execute('''
            INSERT OR REPLACE INTO cases (
                cluster_id, case_name, text_length, word_count, page_count,
                file_size, outcome, outcome_confidence, granted_count, denied_count,
                section_1782_mentions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case['cluster_id'],
            case['case_name'],
            case['text_length'],
            case['word_count'],
            case['page_count'],
            case['file_size'],
            case['outcome']['outcome'],
            case['outcome']['confidence'],
            case['outcome']['granted_count'],
            case['outcome']['denied_count'],
            case['section_1782_mentions']
        ))

        # Insert Intel factors
        for factor, detected in case['intel_factors'].items():
            cursor.execute('''
                INSERT INTO intel_factors (cluster_id, factor_name, detected)
                VALUES (?, ?, ?)
            ''', (case['cluster_id'], factor, detected))

        # Insert citations
        for citation in case['citations']:
            cursor.execute('''
                INSERT INTO citations (cluster_id, citation_pattern, citation_count)
                VALUES (?, ?, ?)
            ''', (case['cluster_id'], citation['pattern'], citation['count']))

    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cases_outcome ON cases(outcome)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cases_cluster_id ON cases(cluster_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_intel_factors_cluster_id ON intel_factors(cluster_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_cluster_id ON citations(cluster_id)')

    conn.commit()

    # Get summary statistics
    cursor.execute('SELECT COUNT(*) FROM cases')
    case_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM intel_factors')
    factor_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM citations')
    citation_count = cursor.fetchone()[0]

    print(f"✅ Database created successfully!")
    print(f"   📊 Cases: {case_count}")
    print(f"   🔍 Intel Factors: {factor_count}")
    print(f"   📚 Citations: {citation_count}")
    print(f"   💾 Database: {db_path}")

    conn.close()
    return db_path

def setup_langchain_querying(db_path):
    """Set up LangChain SQL agent for querying the database."""

    print("\n🤖 Setting up LangChain SQL Agent...")

    try:
        from langchain_integration import LangChainSQLAgent
        from agents import ModelConfig

        # Initialize LangChain SQL agent
        model_config = ModelConfig(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )

        sql_agent = LangChainSQLAgent(
            database_path=db_path,
            model_config=model_config
        )

        print("✅ LangChain SQL Agent initialized!")

        # Test queries
        test_queries = [
            "How many cases were granted vs denied?",
            "Which Intel factors are most commonly discussed?",
            "What are the citation patterns in successful cases?",
            "Show me cases with Factor 2 (receptivity) analysis",
            "What's the average text length of granted cases?"
        ]

        print("\n🧪 Testing LangChain queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                result = sql_agent.query(query)
                print(f"   Result: {result[:200]}...")
            except Exception as e:
                print(f"   Error: {e}")

        return sql_agent

    except ImportError as e:
        print(f"❌ LangChain integration not available: {e}")
        print("   You can still query the database directly with SQL")
        return None

def create_query_examples():
    """Create example SQL queries for the 1782 database."""

    print("\n📋 Example SQL Queries:")
    print("="*50)

    queries = {
        "Outcome Analysis": [
            "SELECT outcome, COUNT(*) as count FROM cases GROUP BY outcome",
            "SELECT AVG(text_length) as avg_length FROM cases WHERE outcome = 'granted'",
            "SELECT AVG(text_length) as avg_length FROM cases WHERE outcome = 'denied'"
        ],
        "Intel Factor Analysis": [
            "SELECT factor_name, COUNT(*) as detected_count FROM intel_factors WHERE detected = 1 GROUP BY factor_name",
            "SELECT c.case_name, COUNT(if.factor_name) as factors_discussed FROM cases c JOIN intel_factors if ON c.cluster_id = if.cluster_id WHERE if.detected = 1 GROUP BY c.cluster_id",
            "SELECT c.outcome, COUNT(if.factor_name) as factors_discussed FROM cases c JOIN intel_factors if ON c.cluster_id = if.cluster_id WHERE if.detected = 1 GROUP BY c.outcome"
        ],
        "Citation Analysis": [
            "SELECT citation_pattern, SUM(citation_count) as total_mentions FROM citations GROUP BY citation_pattern ORDER BY total_mentions DESC",
            "SELECT c.case_name, SUM(ci.citation_count) as total_citations FROM cases c JOIN citations ci ON c.cluster_id = ci.cluster_id GROUP BY c.cluster_id ORDER BY total_citations DESC"
        ],
        "Success Patterns": [
            "SELECT c.case_name, c.outcome, c.text_length, COUNT(if.factor_name) as factors_discussed FROM cases c JOIN intel_factors if ON c.cluster_id = if.cluster_id WHERE if.detected = 1 GROUP BY c.cluster_id ORDER BY c.outcome",
            "SELECT c.outcome, AVG(c.section_1782_mentions) as avg_1782_mentions FROM cases c GROUP BY c.outcome"
        ]
    }

    for category, query_list in queries.items():
        print(f"\n{category}:")
        for query in query_list:
            print(f"  {query}")

    return queries

def main():
    """Main execution function."""
    print("🚀 PDF to SQL Conversion with LangChain Integration")
    print("="*60)

    # Step 1: Create SQL database
    db_path = create_1782_database()

    # Step 2: Set up LangChain querying
    sql_agent = setup_langchain_querying(db_path)

    # Step 3: Create query examples
    create_query_examples()

    print(f"\n✅ Setup Complete!")
    print(f"📁 Database: {db_path}")
    print(f"🤖 LangChain Agent: {'Available' if sql_agent else 'Not Available'}")
    print(f"\n💡 You can now query your 1782 PDF data with natural language!")

    if sql_agent:
        print(f"\n🎯 Example LangChain Query:")
        print(f"   'Show me all successful cases that discuss Factor 2'")
        print(f"   'What are the common patterns in denied cases?'")
        print(f"   'Which cases have the most Intel Corp citations?'")

if __name__ == "__main__":
    main()
