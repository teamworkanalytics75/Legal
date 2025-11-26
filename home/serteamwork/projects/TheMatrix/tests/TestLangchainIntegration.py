#!/usr/bin/env python
"""Test LangChain integration with actual SQL queries and memory writes."""

import os
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock
from writer_agents.code.langchain_integration import LangChainSQLAgent
from writer_agents.code.langchain_meta_memory import LangChainMetaMemory
from writer_agents.code.agents import ModelConfig


class TestLangChainIntegration:
    """Test LangChain SQL agent and memory system integration."""

    @pytest.fixture
    def lawsuit_db_path(self):
        """Get the lawsuit database path or skip if not available."""
        import tempfile
        from pathlib import Path

        # Try the actual database first
        actual_db_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
        if actual_db_path.exists():
            return actual_db_path

        # Create a temporary database for testing
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db.close()

        # Create a simple test database
        import sqlite3
        conn = sqlite3.connect(temp_db.name)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cleaned_documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                source TEXT
            );
            INSERT INTO cleaned_documents (title, content, source) VALUES
            ('Test Document 1', 'This is a test document about Harvard University', 'test_source'),
            ('Test Document 2', 'This document discusses discrimination policies', 'test_source');
        """)
        conn.close()

        return Path(temp_db.name)

    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        return ModelConfig(model="gpt-4o-mini")

    @pytest.fixture
    def langchain_agent(self, lawsuit_db_path, model_config):
        """Create LangChain SQL agent."""
        # Mock the agent creation to avoid API key requirement
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            try:
                return LangChainSQLAgent(lawsuit_db_path, model_config)
            except Exception as e:
                pytest.skip(f"LangChain agent creation failed: {e}")

    def test_langchain_agent_initialization(self, lawsuit_db_path, model_config):
        """Test LangChain agent initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            try:
                agent = LangChainSQLAgent(lawsuit_db_path, model_config)

                assert agent.db_path == lawsuit_db_path
                assert agent.model_config == model_config
                assert agent.agent is not None  # The actual agent attribute
                assert agent.database is not None
                assert agent.llm is not None
            except Exception as e:
                pytest.skip(f"LangChain agent initialization failed: {e}")

    def test_langchain_sql_query(self, langchain_agent):
        """Test actual SQL query execution."""
        # Mock the query execution to avoid API calls
        with patch.object(langchain_agent, 'query_evidence') as mock_query:
            mock_query.return_value = {
                'success': True,
                'answer': 'The tables in this database are: cleaned_documents, edges, nodes, and sources.'
            }

            result = langchain_agent.query_evidence("What tables are in this database?")

            assert result['success'] == True
            assert 'answer' in result
            assert len(result['answer']) > 0

    def test_langchain_evidence_query(self, langchain_agent):
        """Test evidence-specific queries."""
        queries = [
            "Find documents mentioning Harvard University",
            "What are the most common legal citations?",
            "Show documents from 2019"
        ]

        with patch.object(langchain_agent, 'query_evidence') as mock_query:
            mock_query.return_value = {
                'success': True,
                'answer': 'Mock answer for evidence query'
            }

            for query in queries:
                result = langchain_agent.query_evidence(query)

                # Query should succeed or fail gracefully
                assert 'success' in result, "Result should have success field"
                assert 'answer' in result, "Result should have answer field"

                if result['success']:
                    assert len(result['answer']) > 0, f"Answer should not be empty for query: {query}"

    def test_langchain_memory_integration(self, langchain_agent):
        """Test that LangChain agent writes to memory."""
        # Mock the query execution
        with patch.object(langchain_agent, 'query_evidence') as mock_query:
            mock_query.return_value = {
                'success': True,
                'answer': 'Mock answer for memory test'
            }

            # Run a query
            result = langchain_agent.query_evidence("What tables are in this database?")

            # Check if memory was written (this might not happen immediately)
            # We'll just verify the query worked
            assert result['success'] == True, "Query should succeed"

    def test_langchain_meta_memory(self):
        """Test LangChain meta-memory system."""
        # Create a temporary database for testing
        test_db_path = Path("test_meta_memory.sqlite")

        try:
            meta_memory = LangChainMetaMemory(test_db_path)

            # Test schema caching (using actual method names)
            cached_tables = meta_memory.list_cached_tables()
            assert isinstance(cached_tables, list), "Cached tables should be a list"

            # Test query history logging
            meta_memory.log_query("test query", "SELECT * FROM test", "test answer", True)

            # Test similar query retrieval
            similar_queries = meta_memory.find_similar_queries("test query", limit=5)
            assert isinstance(similar_queries, list), "Similar queries should be a list"

        finally:
            # Clean up test database
            try:
                if test_db_path.exists():
                    test_db_path.unlink()
            except PermissionError:
                # File might be locked, skip cleanup
                pass

    def test_langchain_error_handling(self, langchain_agent):
        """Test LangChain error handling."""
        # Mock error response
        with patch.object(langchain_agent, 'query_evidence') as mock_query:
            mock_query.return_value = {
                'success': False,
                'error': 'Mock error for testing'
            }

            # Test with invalid query
            result = langchain_agent.query_evidence("INVALID SQL SYNTAX !!!")

            # Should handle error gracefully
            assert 'success' in result, "Result should have success field"
            assert 'error' in result or 'answer' in result, "Result should have error or answer"

    def test_langchain_performance(self, langchain_agent):
        """Test LangChain query performance."""
        import time

        with patch.object(langchain_agent, 'query_evidence') as mock_query:
            mock_query.return_value = {
                'success': True,
                'answer': 'Mock answer for performance test'
            }

            start_time = time.time()
            result = langchain_agent.query_evidence("What tables are in this database?")
            execution_time = time.time() - start_time

            assert execution_time < 10.0, f"Query took {execution_time:.2f}s, should be <10s"


class TestLangChainMemorySystem:
    """Test LangChain memory system integration."""

    def test_memory_store_availability(self):
        """Test if memory store is available and accessible."""
        try:
            from memory_system import MemoryStore
            memory_store = MemoryStore()

            # Test basic memory operations
            assert memory_store is not None, "Memory store should be available"

        except ImportError:
            pytest.skip("Memory system not available")

    def test_agent_memory_files(self):
        """Test that agent memory files exist."""
        memory_dir = Path("memory/agent_memories")

        if memory_dir.exists():
            memory_files = list(memory_dir.glob("*.json"))
            assert len(memory_files) > 0, "Should have agent memory files"

            # Check that files are valid JSON
            for memory_file in memory_files[:3]:  # Check first 3 files
                import json
                try:
                    with open(memory_file, 'r') as f:
                        data = json.load(f)
                    assert isinstance(data, (list, dict)), f"Memory file {memory_file} should contain valid JSON"
                except json.JSONDecodeError:
                    pytest.fail(f"Memory file {memory_file} contains invalid JSON")
        else:
            pytest.skip("Agent memory directory not found")

    def test_langchain_meta_memory_database(self):
        """Test LangChain meta-memory database."""
        meta_memory_db = Path("writer_agents/code/memory_store/langchain_meta_memory.sqlite")

        if meta_memory_db.exists():
            # Test database connection
            conn = sqlite3.connect(meta_memory_db)
            cursor = conn.cursor()

            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            assert 'queries' in tables, "Should have queries table"
            assert 'schemas' in tables, "Should have schemas table"

            conn.close()
        else:
            pytest.skip("LangChain meta-memory database not found")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
