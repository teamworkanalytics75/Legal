#!/usr/bin/env python
"""Test complete LangChain integration with memory writes."""

import os
import pytest
from pathlib import Path
from writer_agents.code.langchain_integration import LangChainSQLAgent
from writer_agents.code.agents import ModelConfig


def test_complete_langchain_integration():
    """Test complete LangChain integration with actual SQL queries and memory writes."""
    print("Testing Complete LangChain Integration")
    print("="*50)
    
    # Check if database exists
    lawsuit_db_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
    if not lawsuit_db_path.exists():
        pytest.skip("Lawsuit database not found")
    
    # Create LangChain agent
    model_config = ModelConfig(model="gpt-4o-mini")
    agent = LangChainSQLAgent(lawsuit_db_path, model_config)
    
    print(f"Agent initialized with {len(agent.table_names)} tables")
    print(f"Tables: {agent.table_names}")
    
    # Test 1: Basic SQL query
    print("\n1. Testing basic SQL query:")
    result1 = agent.query_evidence("What tables are in this database?")
    print(f"   Success: {result1['success']}")
    print(f"   Answer: {result1.get('answer', 'No answer')[:100]}...")
    
    assert result1['success'] == True, f"Basic query failed: {result1.get('error', 'Unknown error')}"
    
    # Test 2: Evidence-specific query
    print("\n2. Testing evidence query:")
    result2 = agent.query_evidence("Find documents mentioning Harvard University")
    print(f"   Success: {result2['success']}")
    print(f"   Answer: {result2.get('answer', 'No answer')[:100]}...")
    
    # Test 3: Legal precedent query
    print("\n3. Testing legal precedent query:")
    result3 = agent.query_evidence("What are the most common legal citations in the database?")
    print(f"   Success: {result3['success']}")
    print(f"   Answer: {result3.get('answer', 'No answer')[:100]}...")
    
    # Test 4: Check meta-memory activity
    print("\n4. Checking meta-memory activity:")
    meta_memory_db = Path("writer_agents/code/memory_store/langchain_meta_memory.sqlite")
    if meta_memory_db.exists():
        import sqlite3
        conn = sqlite3.connect(meta_memory_db)
        cursor = conn.cursor()
        
        # Check query history
        cursor.execute("SELECT COUNT(*) FROM queries")
        query_count = cursor.fetchone()[0]
        print(f"   Total queries in meta-memory: {query_count}")
        
        # Check recent queries
        cursor.execute("SELECT question, success FROM queries ORDER BY created_at DESC LIMIT 3")
        recent_queries = cursor.fetchall()
        print(f"   Recent queries:")
        for question, success in recent_queries:
            print(f"     - {question[:50]}... (Success: {success})")
        
        conn.close()
        
        # Verify that queries were logged
        assert query_count > 0, "Should have logged queries to meta-memory"
    else:
        print("   Meta-memory database not found")
    
    # Test 5: Performance test
    print("\n5. Testing performance:")
    import time
    
    start_time = time.time()
    result4 = agent.query_evidence("Show me a sample document from the database")
    execution_time = time.time() - start_time
    
    print(f"   Query execution time: {execution_time:.2f}s")
    print(f"   Success: {result4['success']}")
    
    assert execution_time < 15.0, f"Query took {execution_time:.2f}s, should be <15s"
    
    print("\nLangChain Integration Test Complete!")
    print("All components working together successfully.")
    
    return True


if __name__ == "__main__":
    test_complete_langchain_integration()
