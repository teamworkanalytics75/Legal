#!/usr/bin/env python3
"""
Helper script to query EsuWiki corpus and format results for explanation.

This script is designed to be used by AI assistants to query EsuWiki
and get well-formatted results for explanation to users.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from unified_case_materials_query import UnifiedCaseMaterialsQuery
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False

try:
    from convert_esuwiki_to_sql import query_database, DEFAULT_DB_PATH
    DIRECT_QUERY_AVAILABLE = True
except ImportError:
    DIRECT_QUERY_AVAILABLE = False


def query_esuwiki(question: str, use_unified: bool = True) -> Dict[str, Any]:
    """
    Query EsuWiki corpus with a question.

    Args:
        question: Natural language question about EsuWiki
        use_unified: Use unified query system (True) or direct SQL (False)

    Returns:
        Dictionary with query results and metadata
    """
    result = {
        "question": question,
        "success": False,
        "method": None,
        "answer": None,
        "documents": [],
        "sql": None,
        "error": None,
        "database": "esuwiki"
    }

    # Try unified query system first (best for natural language)
    if use_unified and UNIFIED_AVAILABLE:
        try:
            query_system = UnifiedCaseMaterialsQuery()

            # Explicitly query EsuWiki
            unified_result = query_system.query(question, database="esuwiki")

            if unified_result.get("success"):
                result.update({
                    "success": True,
                    "method": "unified_langchain",
                    "answer": unified_result.get("answer"),
                    "sql": unified_result.get("sql"),
                    "database": unified_result.get("database", "esuwiki"),
                    "source": unified_result.get("source")
                })
                return result
            else:
                # Don't set error yet, try direct SQL fallback
                pass
        except Exception as e:
            # Continue to fallback
            pass

    # Fallback to direct SQL query (always try this)
    if DIRECT_QUERY_AVAILABLE:
        try:
            db_path = DEFAULT_DB_PATH
            if not db_path.exists():
                result["error"] = f"Database not found: {db_path}"
                return result

            # Extract keywords for direct search
            keywords = question.lower().split()
            # Remove common words
            stop_words = {"what", "does", "the", "say", "about", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "a", "an", "and", "or", "but", "mention", "documents", "find", "show"}
            search_terms = [w for w in keywords if w not in stop_words and len(w) > 2]

            # Use first meaningful term for FTS (FTS5 works best with single terms or quoted phrases)
            if search_terms:
                # Try first term, or combine with OR for multiple terms
                if len(search_terms) == 1:
                    search_query = search_terms[0]
                else:
                    # Use OR syntax for FTS5: term1 OR term2 OR term3
                    search_query = " OR ".join(search_terms[:3])

                try:
                    direct_results = query_database(db_path, search_query, use_fts=True)
                except Exception as fts_error:
                    # If FTS fails, try LIKE search instead
                    try:
                        direct_results = query_database(db_path, search_query, use_fts=False)
                    except Exception as like_error:
                        result["error"] = f"FTS error: {fts_error}, LIKE error: {like_error}"
                        return result

                if direct_results:
                    result.update({
                        "success": True,
                        "method": "direct_sql_fts",
                        "documents": direct_results,
                        "answer": f"Found {len(direct_results)} document(s) matching your query."
                    })
                    return result
        except Exception as e:
            if not result.get("error"):
                result["error"] = f"Direct query error: {str(e)}"

    return result


def format_results_for_explanation(result: Dict[str, Any]) -> str:
    """
    Format query results into a comprehensive explanation.

    Args:
        result: Query result dictionary

    Returns:
        Formatted explanation string
    """
    output = []

    output.append("=" * 70)
    output.append("ESUWIKI QUERY RESULTS")
    output.append("=" * 70)
    output.append(f"\nQuestion: {result['question']}")
    output.append(f"Method: {result.get('method', 'unknown')}")
    output.append(f"Database: {result.get('database', 'esuwiki')}")
    output.append("")

    if not result["success"]:
        output.append("❌ Query Failed")
        output.append(f"Error: {result.get('error', 'Unknown error')}")
        return "\n".join(output)

    output.append("✅ Query Successful")
    output.append("")

    # Show answer if available
    if result.get("answer"):
        output.append("📋 Answer:")
        output.append("-" * 70)
        output.append(result["answer"])
        output.append("")

    # Show SQL query if available
    if result.get("sql"):
        output.append("🔍 SQL Query Used:")
        output.append("-" * 70)
        output.append(result["sql"])
        output.append("")

    # Show documents if available
    documents = result.get("documents", [])
    if documents:
        output.append(f"📄 Documents Found ({len(documents)}):")
        output.append("-" * 70)

        for i, doc in enumerate(documents, 1):
            file_name = doc.get("file_name", "Unknown")
            word_count = doc.get("word_count", 0)
            char_count = doc.get("char_count", 0)

            output.append(f"\n{i}. {file_name}")
            if word_count:
                output.append(f"   Words: {word_count:,} | Characters: {char_count:,}")

            # Show snippet if available
            if "snippet" in doc and doc["snippet"]:
                snippet = doc["snippet"][:300]
                output.append(f"   Preview: {snippet}...")

            # Show excerpt from text if available
            if "extracted_text" in doc and doc["extracted_text"]:
                text = doc["extracted_text"][:200]
                output.append(f"   Excerpt: {text}...")

    output.append("")
    output.append("=" * 70)

    return "\n".join(output)


def main():
    """Command line interface for testing."""
    if len(sys.argv) < 2:
        print("Usage: python query_esuwiki_helper.py 'Your question here'")
        print("\nExample:")
        print("  python query_esuwiki_helper.py 'What documents mention crackdowns?'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"\n🔍 Querying EsuWiki: {question}\n")

    result = query_esuwiki(question, use_unified=True)
    explanation = format_results_for_explanation(result)

    print(explanation)


if __name__ == "__main__":
    main()

