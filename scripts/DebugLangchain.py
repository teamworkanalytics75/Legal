#!/usr/bin/env python
"""Debug LangChain SQL agent query results."""

import os
from pathlib import Path
from writer_agents.code.langchain_integration import LangChainSQLAgent
from writer_agents.code.agents import ModelConfig

def debug_langchain_query():
    """Debug the LangChain query result structure."""
    print("DEBUGGING LANGCHAIN QUERY RESULTS")
    print("="*50)

    lawsuit_db_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
    if not lawsuit_db_path.exists():
        print(f"Database not found at {lawsuit_db_path}")
        return

    try:
        # Initialize agent
        model_config = ModelConfig(model="gpt-4o-mini")
        agent = LangChainSQLAgent(
            db_path=lawsuit_db_path,
            model_config=model_config,
            verbose=True
        )

        print(f"Database tables: {agent.table_names}")

        # Test simple query
        print("\nTesting simple query...")
        result = agent.query_evidence("What tables are in this database?")

        print(f"\nResult keys: {list(result.keys())}")
        print(f"Success: {result.get('success')}")
        print(f"Answer: {result.get('answer')}")
        print(f"Error: {result.get('error')}")

        # Test direct agent invocation
        print("\nTesting direct agent invocation...")
        try:
            direct_result = agent.agent.invoke({"input": "What tables are in this database?"})
            print(f"Direct result type: {type(direct_result)}")
            print(f"Direct result keys: {list(direct_result.keys())}")
            print(f"Output: {direct_result.get('output', 'No output key')}")
        except Exception as e:
            print(f"Direct invocation error: {e}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_langchain_query()
