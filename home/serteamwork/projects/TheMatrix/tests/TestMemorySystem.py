"""Comprehensive tests for agent memory system."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from writer_agents.code.memory_system import MemoryStore, AgentMemory
from writer_agents.code.extract_memories import MemoryBuilder
from writer_agents.code.embeddings import EmbeddingService
from writer_agents.code.agent_context_templates import get_agent_context


class TestEmbeddingService:
    """Test embedding generation."""

    def test_local_embedding(self):
        """Test local sentence-transformers embedding."""
        service = EmbeddingService(use_local=True)

        text = "This is a test sentence."
        embedding = service.embed(text)

        assert embedding.shape == (384,) # all-MiniLM-L6-v2 dims
        assert embedding.dtype == np.float32

    def test_batch_embedding(self):
        """Test batch embedding."""
        service = EmbeddingService(use_local=True)

        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = service.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(emb.shape == (384,) for emb in embeddings)

    def test_cost_estimation(self):
        """Test cost estimation."""
        service = EmbeddingService(use_local=True)

        # Local embeddings should be free
        cost = service.estimate_cost(100, 50)
        assert cost == 0.0

        # Test OpenAI cost estimation
        service_openai = EmbeddingService(use_local=False)
        cost_openai = service_openai.estimate_cost(100, 50)
        assert cost_openai > 0.0


class TestMemoryStore:
    """Test memory storage and retrieval."""

    def test_add_and_retrieve(self, tmp_path):
        """Test adding and retrieving memories."""
        store = MemoryStore(storage_path=tmp_path)

        memory = AgentMemory(
            agent_type="CitationFinderAgent",
            memory_id="test-001",
            summary="Found 5 citations in 0.3s with high accuracy",
            context={'citations': 5, 'duration': 0.3},
            embedding=None,
            source="test",
            timestamp=datetime.now()
        )

        store.add(memory)

        retrieved = store.retrieve(agent_type="CitationFinderAgent", k=1)

        assert len(retrieved) == 1
        assert retrieved[0].summary == memory.summary

    def test_similarity_search(self, tmp_path):
        """Test semantic similarity search."""
        store = MemoryStore(storage_path=tmp_path)

        memories = [
            AgentMemory(
                agent_type="OutlineBuilderAgent",
                memory_id=f"test-{i}",
                summary=text,
                context={},
                embedding=None,
                source="test",
                timestamp=datetime.now()
            )
            for i, text in enumerate([
                "Built clear 5-section outline with introduction",
                "Created outline with facts, analysis, conclusion sections",
                "Found citation in case law database", # Not relevant
            ])
        ]

        for mem in memories:
            store.add(mem)

        # Query for outline-related memories
        results = store.retrieve(
            agent_type="OutlineBuilderAgent",
            query="creating document outline with sections",
            k=2
        )

        assert len(results) == 2
        assert "outline" in results[0].summary.lower()
        assert "outline" in results[1].summary.lower()

    def test_agent_stats(self, tmp_path):
        """Test agent statistics."""
        store = MemoryStore(storage_path=tmp_path)

        # Add some memories
        for i in range(3):
            memory = AgentMemory(
                agent_type="CitationFinderAgent",
                memory_id=f"test-{i}",
                summary=f"Test memory {i}",
                context={},
                embedding=None,
                source="test",
                timestamp=datetime.now()
            )
            store.add(memory)

        stats = store.get_agent_stats("CitationFinderAgent")

        assert stats['total_memories'] == 3
        assert stats['sources']['test'] == 3
        assert stats['date_range'] is not None

    def test_persistence(self, tmp_path):
        """Test memory persistence to disk."""
        store = MemoryStore(storage_path=tmp_path)

        memory = AgentMemory(
            agent_type="TestAgent",
            memory_id="persist-test",
            summary="Test persistence",
            context={},
            embedding=None,
            source="test",
            timestamp=datetime.now()
        )

        store.add(memory)
        store.save()

        # Create new store and load
        store2 = MemoryStore(storage_path=tmp_path)

        retrieved = store2.retrieve(agent_type="TestAgent", k=1)
        assert len(retrieved) == 1
        assert retrieved[0].summary == "Test persistence"


class TestMemoryBuilder:
    """Test memory extraction from logs."""

    def test_extract_from_jobs_db(self):
        """Test extracting memories from job database."""
        # Mock job manager
        mock_job_manager = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()

        # Mock successful job data
        mock_cursor.fetchall.return_value = [
            {
                'agent_type': 'CitationFinderAgent',
                'job_id': 'job-001',
                'tokens_in': 100,
                'tokens_out': 50,
                'duration_seconds': 2.5,
                'result': '{"citations": ["42 U.S.C. Section 1983"]}'
            }
        ]

        mock_conn.execute.return_value = mock_cursor
        mock_job_manager._get_connection.return_value = mock_conn

        builder = MemoryBuilder()
        memories = builder.extract_from_jobs_db(mock_job_manager)

        assert len(memories) > 0
        assert all(isinstance(m, AgentMemory) for m in memories)
        assert all(m.agent_type for m in memories)

    def test_extract_from_artifacts(self, tmp_path):
        """Test extracting memories from artifacts."""
        # Create test artifacts directory
        artifacts_dir = tmp_path / "analysis_outputs"
        artifacts_dir.mkdir()

        reports_dir = artifacts_dir / "legal_reports"
        reports_dir.mkdir()

        # Create test report
        test_report = reports_dir / "test_report.md"
        test_report.write_text("""
# Legal Analysis Report

## Introduction
This is a test legal report with proper structure.

## Analysis
The case involves 42 U.S.C. Section 1983 claims against the defendant.

## Conclusion
Based on the evidence, the plaintiff has a strong case.
        """)

        builder = MemoryBuilder()
        memories = builder.extract_from_artifacts(artifacts_dir)

        assert len(memories) > 0
        assert any("High-quality legal report" in m.summary for m in memories)

    def test_quality_scoring(self):
        """Test result quality scoring."""
        builder = MemoryBuilder()

        # High quality result
        high_quality = {
            'agent_name': 'CitationFinderAgent',
            'result': 'Found 5 citations with proper formatting',
            'citations': ['42 U.S.C. Section 1983', 'Miranda v. Arizona'],
            'summary': 'Citation extraction completed successfully'
        }

        score = builder._score_result_quality(high_quality)
        assert score > 0.5

        # Low quality result
        low_quality = {
            'output': 'error'
        }

        score = builder._score_result_quality(low_quality)
        assert score < 0.5


class TestAgentContextTemplates:
    """Test agent context templates."""

    def test_get_agent_context(self):
        """Test getting agent context."""
        context = get_agent_context("CitationFinderAgent")

        assert context['project'] == "The Matrix - Bayesian Legal AI System"
        assert context['role'] == "Find raw citation strings in legal text using 6 regex patterns"
        assert context['phase'] == "Citation"
        assert context['deterministic'] == True
        assert context['cost'] == "$0.00"
        assert 'system_context' in context

    def test_unknown_agent_context(self):
        """Test getting context for unknown agent."""
        context = get_agent_context("UnknownAgent")

        assert context == {'system_context': context['system_context']}

    def test_all_agents_have_context(self):
        """Test that all known agents have context."""
        known_agents = [
            "CitationFinderAgent", "CitationNormalizerAgent", "CitationVerifierAgent",
            "CitationLocatorAgent", "CitationInserterAgent", "FactExtractorAgent",
            "PrecedentFinderAgent", "PrecedentRankerAgent", "PrecedentSummarizerAgent",
            "StatuteLocatorAgent", "ExhibitFetcherAgent", "OutlineBuilderAgent",
            "SectionWriterAgent", "ParagraphWriterAgent", "TransitionAgent",
            "GrammarFixerAgent", "StyleCheckerAgent", "LogicCheckerAgent",
            "ConsistencyCheckerAgent", "RedactionAgent", "ComplianceAgent",
            "ExpertQAAgent", "MarkdownExporterAgent", "DocxExporterAgent",
            "MetadataTaggerAgent"
        ]

        for agent in known_agents:
            context = get_agent_context(agent)
            assert context['project'] == "The Matrix - Bayesian Legal AI System"
            assert 'role' in context
            assert 'phase' in context


class TestMemoryIntegration:
    """Test memory integration with atomic agents."""

    @pytest.mark.asyncio
    async def test_agent_with_memory(self, tmp_path):
        """Test agent initialization with memory enabled."""
        # Create memory store with sample data
        store = MemoryStore(storage_path=tmp_path)
        store.add(AgentMemory(
            agent_type="CitationFinderAgent",
            memory_id="test-001",
            summary="Successfully found 8 citations using pattern matching",
            context={},
            embedding=None,
            source="test",
            timestamp=datetime.now()
        ))
        store.save()

        # Mock agent factory
        mock_factory = Mock()

        # Test agent initialization with memory
        from writer_agents.code.atomic_agent import AtomicAgent

        class TestAgent(AtomicAgent):
            duty = "Test duty"
            is_deterministic = True

        agent = TestAgent(
            mock_factory,
            enable_memory=True,
            memory_config={'k_neighbors': 3}
        )

        # Check that memories were loaded
        assert agent.memories is not None
        assert 'project_context' in agent.memories
        assert 'past_patterns' in agent.memories

    @pytest.mark.asyncio
    async def test_agent_without_memory(self):
        """Test agent can disable memory system."""
        mock_factory = Mock()

        from writer_agents.code.atomic_agent import AtomicAgent

        class TestAgent(AtomicAgent):
            duty = "Test duty"
            is_deterministic = True

        agent = TestAgent(
            mock_factory,
            enable_memory=False
        )

        assert agent.memories is None

    def test_system_message_with_memory(self):
        """Test system message generation with memory."""
        mock_factory = Mock()

        from writer_agents.code.atomic_agent import AtomicAgent

        class TestAgent(AtomicAgent):
            duty = "Test duty"
            is_deterministic = True

        agent = TestAgent(mock_factory, enable_memory=True)

        # Mock memories
        agent.memories = {
            'project_context': {
                'project': 'The Matrix',
                'role': 'Test role',
                'team_position': '1/5',
                'upstream': 'TestUpstream',
                'downstream': 'TestDownstream',
                'codename': 'TestCodename',
                'system_context': 'Test system context'
            },
            'past_patterns': [
                Mock(summary="Test pattern 1"),
                Mock(summary="Test pattern 2")
            ]
        }

        message = agent._build_system_message_with_memory()

        assert "The Matrix" in message
        assert "Test role" in message
        assert "1/5" in message
        assert "TestUpstream" in message
        assert "TestDownstream" in message
        assert "TestCodename" in message
        assert "Test pattern 1" in message
        assert "Test pattern 2" in message


# Fixtures

@pytest.fixture
def sample_jobs_db(tmp_path):
    """Create sample jobs.db for testing."""
    from writer_agents.code.job_persistence import JobManager

    db_path = tmp_path / "test_jobs.db"
    jm = JobManager(db_path)

    # Create sample successful job
    job_id = jm.create_job(
        phase="citation",
        agent_type="CitationFinderAgent",
        payload={'text': 'sample'},
        priority=0
    )

    jm.mark_completed(
        job_id,
        result={'citations': ['42 U.S.C. Section 1983']},
        tokens_in=100,
        tokens_out=50,
        duration=0.5
    )

    return db_path


if __name__ == "__main__":
    pytest.main([__file__])
