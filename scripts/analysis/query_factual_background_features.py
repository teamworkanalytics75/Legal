#!/usr/bin/env python3
"""
Factual Background Feature Analysis Query System

Queries databases to find features of successful factual background sections in:
1. Your HK statement of claim
2. Your 1782 draft
3. Successful motions to seal/pseudonym from case law
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Load OpenAI API key from config
def load_openai_api_key():
    """Load OpenAI API key from config file."""
    config_path = Path(__file__).parent / "config" / "OpenaiConfig.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get("openai_api_key") or config.get("OPENAI_API_KEY")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    return True
        except Exception as e:
            print(f"Warning: Could not load API key from config: {e}")
    return False

# Auto-load API key on import
load_openai_api_key()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "writer_agents" / "code"))
sys.path.insert(0, str(Path(__file__).parent / "document_ingestion"))

try:
    from LangchainIntegration import LangChainSQLAgent
    from agents import ModelConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from SimpleQueryAgent import SimpleLegalQueryAgent
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


class FactualBackgroundQuerySystem:
    """Query system specifically for factual background analysis."""

    def __init__(self):
        """Initialize factual background query system."""
        self.query_log = []
        self.query_results = {}
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize query agents for available databases."""
        print("\n[INITIALIZING] Setting up query agents...")

        # Try to load API key from config if not set
        if not os.getenv("OPENAI_API_KEY"):
            load_openai_api_key()

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"  [OK] OpenAI API key loaded ({api_key[:10]}...)")
        else:
            print("  [WARN] OPENAI_API_KEY not found in config or environment")
            return

        if not LANGCHAIN_AVAILABLE:
            print("  [SKIP] LangChain not available")
            if 'LANGCHAIN_ERROR' in globals():
                print(f"        Error: {LANGCHAIN_ERROR}")
            print("        Install: pip install langchain langchain-openai langchain-community")
            return

        if LANGCHAIN_AVAILABLE and api_key:
            model_config = ModelConfig(model="gpt-4o-mini")

            # Unified corpus (case law - motions to seal/pseudonym)
            db_path = Path("case_law_data/unified_corpus.db")
            if db_path.exists():
                try:
                    self.agents['unified_corpus'] = LangChainSQLAgent(db_path, model_config, verbose=False)
                except Exception as e:
                    print(f"Warning: Failed to initialize unified_corpus agent: {e}")

            # Exported lawsuit docs (HK statement, 1782 drafts)
            lawsuit_db = Path("case_law_data/lawsuit_docs_langchain.db")
            if lawsuit_db.exists():
                try:
                    self.agents['lawsuit_docs'] = LangChainSQLAgent(lawsuit_db, model_config, verbose=False)
                except Exception as e:
                    print(f"Warning: Failed to initialize lawsuit_docs agent: {e}")

        # MySQL fallback for lawsuit docs
        if MYSQL_AVAILABLE:
            try:
                self.agents['lawsuit_docs_mysql'] = SimpleLegalQueryAgent()
            except Exception as e:
                print(f"Warning: MySQL agent not available: {e}")

    def generate_factual_background_queries(self) -> List[Dict]:
        """Generate targeted queries for factual background analysis."""
        queries = [
            {
                "category": "own_documents",
                "title": "HK Statement of Claim - Factual Background",
                "queries": [
                    "Find the factual background section in the Hong Kong statement of claim",
                    "What does the HK statement of claim say about the factual background?",
                    "Extract the factual background section from Hong Kong statement of claim documents"
                ],
                "databases": ["lawsuit_docs", "lawsuit_docs_mysql"]
            },
            {
                "category": "own_documents",
                "title": "1782 Draft - Factual Background",
                "queries": [
                    "Find the factual background section in Section 1782 draft materials",
                    "What does the 1782 draft say about the factual background?",
                    "Extract factual background from 1782 discovery petition drafts"
                ],
                "databases": ["lawsuit_docs", "lawsuit_docs_mysql"]
            },
            {
                "category": "successful_examples",
                "title": "Successful Factual Backgrounds - Motions to Seal",
                "queries": [
                    "Find examples of factual background sections in successful motions to seal",
                    "What are the features of strong factual background sections in granted motions to seal?",
                    "Show me how factual background sections are structured in successful sealing motions"
                ],
                "databases": ["unified_corpus"]
            },
            {
                "category": "successful_examples",
                "title": "Successful Factual Backgrounds - Pseudonym Motions",
                "queries": [
                    "Find examples of factual background sections in successful motions to proceed under pseudonym",
                    "What makes a compelling factual background in granted pseudonym motions?",
                    "Show me factual background sections from successful pseudonym cases"
                ],
                "databases": ["unified_corpus"]
            },
            {
                "category": "analysis",
                "title": "Factual Background Best Practices",
                "queries": [
                    "What are the key features of successful factual background sections in legal motions?",
                    "How long should a factual background section be in a motion to seal or pseudonym?",
                    "What elements make a factual background section persuasive in sealing/pseudonym motions?",
                    "What language and structure patterns appear in successful factual background sections?",
                    "How do successful factual backgrounds establish the need for sealing or pseudonym protection?"
                ],
                "databases": ["unified_corpus"]
            },
            {
                "category": "analysis",
                "title": "Factual Background Word Count Analysis",
                "queries": [
                    "What is the typical word count range for factual background sections in successful motions?",
                    "Compare factual background length in granted versus denied motions to seal",
                    "What word count range correlates with successful factual backgrounds in pseudonym motions?"
                ],
                "databases": ["unified_corpus"]
            }
        ]
        return queries

    def query_database(self, query: str, database: str) -> Dict:
        """Query a specific database."""
        agent = self.agents.get(database)
        if not agent:
            return {"success": False, "error": f"Agent for {database} not available"}

        try:
            # Use LangChain for SQLite
            if isinstance(agent, LangChainSQLAgent):
                result = agent.query_evidence(query)
            # Use SimpleQueryAgent for MySQL
            elif hasattr(agent, 'search'):
                # For factual background, we want full answers, not just search
                if hasattr(agent, 'answer'):
                    answer = agent.answer(query)
                    result = {
                        "success": True,
                        "answer": answer if answer else f"Found documents matching: {query}",
                        "search_results": agent.search(query)
                    }
                else:
                    results = agent.search(query)
                    result = {
                        "success": len(results) > 0,
                        "answer": f"Found {len(results)} relevant documents",
                        "documents": results
                    }
            else:
                return {"success": False, "error": "Unknown agent type"}

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_all_queries(self):
        """Execute all factual background queries."""
        query_configs = self.generate_factual_background_queries()

        print(f"\nExecuting {len(query_configs)} query categories...\n")

        for config in query_configs:
            category = config['category']
            title = config['title']
            queries = config['queries']
            databases = config['databases']

            print(f"[{category.upper()}] {title}")
            print("-" * 70)

            category_results = []

            for query in queries:
                print(f"  Query: {query[:60]}...")

                # Try each database until we get a result
                result = None
                for db in databases:
                    result = self.query_database(query, db)
                    if result.get("success"):
                        print(f"    [OK] Got result from {db}")
                        break

                if not result or not result.get("success"):
                    print(f"    [SKIP] No results from any database")
                    result = {"success": False, "query": query}

                # Log query
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "title": title,
                    "query": query,
                    "databases_tried": databases,
                    "database_used": db if result.get("success") else None,
                    "success": result.get("success", False),
                    "has_answer": bool(result.get("answer")),
                    "answer_length": len(result.get("answer", ""))
                }
                self.query_log.append(log_entry)

                category_results.append({
                    "query": query,
                    "result": result
                })

            self.query_results[title] = category_results
            print()

        print(f"Completed {len(self.query_log)} queries\n")

    def save_results(self, output_path: Optional[Path] = None):
        """Save query results for ML and analysis."""
        if output_path is None:
            output_path = Path("case_law_data/query_logs") / f"factual_background_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "factual_background_features",
            "top_feature": {
                "name": "factual_background_word_count",
                "importance": 12.74,
                "description": "Word count of factual background section - top predictive feature"
            },
            "query_log": self.query_log,
            "results": {}
        }

        # Compile results
        for title, category_results in self.query_results.items():
            results_data["results"][title] = []
            for item in category_results:
                result = item["result"]
                if result.get("success") and result.get("answer"):
                    results_data["results"][title].append({
                        "query": item["query"],
                        "answer": result["answer"][:2000],  # First 2000 chars
                        "full_answer_length": len(result.get("answer", "")),
                        "timestamp": datetime.now().isoformat()
                    })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {output_path}")
        return output_path

    def compile_analysis_report(self) -> str:
        """Compile query results into an analysis report."""
        report = []
        report.append("# Factual Background Feature Analysis Report\n")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Focus**: Top predictive feature `factual_background_word_count` (Importance: 12.74)\n\n")

        report.append("## Executive Summary\n\n")
        report.append("This report analyzes factual background sections from:\n")
        report.append("1. Your HK statement of claim\n")
        report.append("2. Your 1782 draft materials\n")
        report.append("3. Successful motions to seal/pseudonym from case law\n\n")

        # Own documents
        report.append("## I. Your Documents - Factual Background Analysis\n\n")
        for title, category_results in self.query_results.items():
            if "HK Statement" in title or "1782 Draft" in title:
                report.append(f"### {title}\n\n")
                for item in category_results:
                    result = item["result"]
                    if result.get("success") and result.get("answer"):
                        report.append(f"**Query**: {item['query']}\n\n")
                        answer = result["answer"]
                        # Truncate long answers
                        if len(answer) > 1000:
                            answer = answer[:1000] + "...\n\n[Answer truncated - see full results in JSON]"
                        report.append(f"{answer}\n\n")

        # Successful examples
        report.append("## II. Successful Examples - Case Law Analysis\n\n")
        for title, category_results in self.query_results.items():
            if "Successful" in title or "Best Practices" in title:
                report.append(f"### {title}\n\n")
                for item in category_results:
                    result = item["result"]
                    if result.get("success") and result.get("answer"):
                        report.append(f"**Query**: {item['query']}\n\n")
                        answer = result["answer"]
                        if len(answer) > 1500:
                            answer = answer[:1500] + "...\n\n[Answer truncated]"
                        report.append(f"{answer}\n\n")

        # Word count analysis
        report.append("## III. Word Count Analysis\n\n")
        report.append("Based on the top predictive feature (`factual_background_word_count`), ")
        report.append("this section analyzes optimal length for factual background sections.\n\n")
        for title, category_results in self.query_results.items():
            if "Word Count" in title:
                report.append(f"### {title}\n\n")
                for item in category_results:
                    result = item["result"]
                    if result.get("success") and result.get("answer"):
                        report.append(f"**Query**: {item['query']}\n\n")
                        answer = result["answer"]
                        if len(answer) > 1000:
                            answer = answer[:1000] + "...\n\n[Answer truncated]"
                        report.append(f"{answer}\n\n")

        report.append("## IV. Recommendations\n\n")
        report.append("### For Your Master Draft:\n\n")
        report.append("1. **Length**: Ensure factual background section meets optimal word count range\n")
        report.append("2. **Structure**: Follow patterns from successful sealing/pseudonym motions\n")
        report.append("3. **Content**: Clearly establish need for protection based on facts\n")
        report.append("4. **Language**: Use persuasive, specific language that supports sealing/pseudonym arguments\n\n")

        return "\n".join(report)


def main():
    """Run factual background feature analysis."""
    print("="*70)
    print("FACTUAL BACKGROUND FEATURE ANALYSIS")
    print("="*70)
    print("\nAnalyzing factual background sections based on top predictive feature:")
    print("  Feature: factual_background_word_count")
    print("  Importance: 12.74 (highest)\n")

    system = FactualBackgroundQuerySystem()

    print(f"\nInitialized {len(system.agents)} query agent(s)\n")
    if not system.agents:
        print("\n[ERROR] No query agents available!")
        print("\nTo fix this:")
        print("\n  Option 1: Use LangChain (recommended)")
        print("    1. Set OPENAI_API_KEY:")
        print("       PowerShell: $env:OPENAI_API_KEY='your-key'")
        print("    2. Make sure databases exist:")
        print("       - case_law_data/unified_corpus.db")
        print("       - case_law_data/lawsuit_docs_langchain.db (run export script if missing)")
        print("\n  Option 2: Use MySQL directly")
        print("    1. Start MySQL server")
        print("    2. Run: python export_mysql_to_sqlite_for_langchain.py")
        return

    # Execute queries
    system.execute_all_queries()

    # Save results
    print("\n[SAVING] Saving query results for ML training...")
    results_path = system.save_results()

    # Compile report
    print("\n[COMPILING] Generating analysis report...")
    report = system.compile_analysis_report()

    # Save report
    report_path = Path("case_law_data/query_logs") / f"factual_background_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[SUCCESS] Analysis complete!")
    print(f"\n[RESULTS] JSON: {results_path}")
    print(f"[REPORT] Markdown: {report_path}")
    print("\nNext steps:")
    print("  1. Review the analysis report")
    print("  2. Feed findings into master draft")
    print("  3. Use query results to train ML models")


if __name__ == "__main__":
    main()

