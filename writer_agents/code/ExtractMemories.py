"""Memory extraction from logs and artifacts.

Extracts learnings from jobs.db, analysis_outputs/, and lawsuit database to build
agent memories for enhanced task-specific intelligence.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from .memory_system import AgentMemory
except ImportError:
    from memory_system import AgentMemory


class MemoryBuilder:
    """Extract learnings from logs and artifacts."""

    def __init__(self, llm_summarizer: Optional[Callable] = None):
        """Initialize memory builder.

        Args:
            llm_summarizer: Optional LLM function for smart summarization
        """
        self.llm_summarizer = llm_summarizer

    def extract_from_jobs_db(
        self,
        job_manager,
        min_date: Optional[datetime] = None
    ) -> List[AgentMemory]:
        """Mine jobs.db for execution patterns.

        Args:
            job_manager: JobManager instance
            min_date: Optional minimum date filter

        Returns:
            List of extracted memories
        """
        memories = []

        # Query successful jobs by agent type
        conn = job_manager._get_connection()

        query = """
            SELECT
                j.type as agent_type,
                j.id as job_id,
                j.payload_ref,
                r.tokens_in,
                r.tokens_out,
                r.duration_seconds,
                a.uri_or_blob as result
            FROM jobs j
            JOIN runs r ON j.id = r.job_id
            LEFT JOIN artifacts a ON j.id = a.job_id AND a.kind = 'result'
            WHERE j.status = 'succeeded'
        """

        if min_date:
            query += f" AND j.created_at >= '{min_date.isoformat()}'"

        query += " ORDER BY j.created_at DESC LIMIT 1000"

        cursor = conn.execute(query)

        for row in cursor.fetchall():
            agent_type = row['agent_type']
            tokens_total = row['tokens_in'] + row['tokens_out']
            duration = row['duration_seconds']

            # Create success memory for efficient executions
            if tokens_total < 1000 and duration < 5.0:
                summary = (
                    f"{agent_type} completed efficiently: "
                    f"{tokens_total} tokens, {duration:.1f}s"
                )
                memories.append(AgentMemory(
                    agent_type=agent_type,
                    memory_id=str(uuid.uuid4()),
                    summary=summary,
                    context={
                        'job_id': row['job_id'],
                        'tokens': tokens_total,
                        'duration': duration,
                        'tokens_in': row['tokens_in'],
                        'tokens_out': row['tokens_out']
                    },
                    embedding=None, # Will be embedded later
                    source="job_db_success",
                    timestamp=datetime.now()
                ))

            # Create quality memory for high-quality outputs
            if row['result']:
                try:
                    result_data = json.loads(row['result'])
                    if self._is_high_quality_result(result_data):
                        summary = (
                            f"{agent_type} produced high-quality output: "
                            f"structured result with {len(result_data)} fields"
                        )
                        memories.append(AgentMemory(
                            agent_type=agent_type,
                            memory_id=str(uuid.uuid4()),
                            summary=summary,
                            context={
                                'job_id': row['job_id'],
                                'result_fields': list(result_data.keys()),
                                'quality_score': self._score_result_quality(result_data)
                            },
                            embedding=None,
                            source="job_db_quality",
                            timestamp=datetime.now()
                        ))
                except (json.JSONDecodeError, TypeError):
                    pass

        # Query failed jobs for error patterns
        error_query = """
            SELECT
                j.type as agent_type,
                r.error_message,
                COUNT(*) as error_count
            FROM jobs j
            JOIN runs r ON j.id = r.job_id
            WHERE j.status IN ('failed', 'dead')
        """

        if min_date:
            error_query += f" AND j.created_at >= '{min_date.isoformat()}'"

        error_query += """
            GROUP BY j.type, r.error_message
            HAVING error_count > 2
        """

        error_cursor = conn.execute(error_query)

        for row in error_cursor.fetchall():
            summary = (
                f"{row['agent_type']} common error (saw {row['error_count']}x): "
                f"{row['error_message'][:100]}"
            )
            memories.append(AgentMemory(
                agent_type=row['agent_type'],
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context={
                    'error': row['error_message'],
                    'count': row['error_count']
                },
                embedding=None,
                source="job_db_errors",
                timestamp=datetime.now()
            ))

        return memories

    def extract_from_artifacts(
        self,
        artifacts_dir: Path
    ) -> List[AgentMemory]:
        """Mine saved outputs for quality patterns.

        Args:
            artifacts_dir: Path to analysis_outputs directory

        Returns:
            List of extracted memories
        """
        memories = []

        # Scan legal_reports for successful patterns
        reports_dir = artifacts_dir / "legal_reports"
        if reports_dir.exists():
            for md_file in reports_dir.glob("**/*.md"):
                try:
                    content = md_file.read_text(encoding='utf-8')

                    # Check for quality indicators
                    has_citations = "U.S." in content or "F.Supp" in content or "Section " in content
                    word_count = len(content.split())
                    has_structure = "##" in content # Markdown headers
                    has_legal_terms = any(term in content.lower() for term in [
                        'plaintiff', 'defendant', 'court', 'jurisdiction', 'motion'
                    ])

                    if has_citations and word_count > 500 and has_structure and has_legal_terms:
                        summary = (
                            f"High-quality legal report generated: {word_count} words, "
                            f"proper citations, clear structure, legal terminology"
                        )
                        memories.append(AgentMemory(
                            agent_type="OutlineBuilderAgent", # Attribute to relevant agents
                            memory_id=str(uuid.uuid4()),
                            summary=summary,
                            context={
                                'file': str(md_file),
                                'word_count': word_count,
                                'has_citations': has_citations,
                                'has_structure': has_structure
                            },
                            embedding=None,
                            source="artifact_analysis",
                            timestamp=datetime.now()
                        ))
                except (UnicodeDecodeError, OSError):
                    continue

        # Scan analysis_results for successful patterns
        results_dir = artifacts_dir / "analysis_results"
        if results_dir.exists():
            for json_file in results_dir.glob("**/*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Check for successful analysis patterns
                    if self._is_successful_analysis(data):
                        summary = (
                            f"Successful analysis completed: "
                            f"{len(data)} data points, structured output"
                        )
                        memories.append(AgentMemory(
                            agent_type="FactExtractorAgent", # Attribute to relevant agents
                            memory_id=str(uuid.uuid4()),
                            summary=summary,
                            context={
                                'file': str(json_file),
                                'data_points': len(data),
                                'analysis_type': self._get_analysis_type(data)
                            },
                            embedding=None,
                            source="artifact_analysis",
                            timestamp=datetime.now()
                        ))
                except (json.JSONDecodeError, OSError):
                    continue

        return memories

    def _is_high_quality_result(self, result_data: Dict[str, Any]) -> bool:
        """Check if result data indicates high quality.

        Args:
            result_data: Result dictionary

        Returns:
            True if result appears high quality
        """
        # Check for structured output
        if not isinstance(result_data, dict):
            return False

        # Check for meaningful fields
        meaningful_fields = [
            'agent_name', 'result', 'output', 'citations', 'facts',
            'summary', 'analysis', 'recommendations'
        ]

        has_meaningful_fields = any(field in result_data for field in meaningful_fields)

        # Check for non-empty content
        has_content = any(
            isinstance(v, str) and len(v.strip()) > 10
            for v in result_data.values()
        )

        return has_meaningful_fields and has_content

    def _score_result_quality(self, result_data: Dict[str, Any]) -> float:
        """Score result quality from 0-1.

        Args:
            result_data: Result dictionary

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        # Structure bonus
        if isinstance(result_data, dict) and len(result_data) > 1:
            score += 0.3

        # Content bonus
        content_fields = ['result', 'output', 'summary', 'analysis']
        for field in content_fields:
            if field in result_data and isinstance(result_data[field], str):
                if len(result_data[field].strip()) > 50:
                    score += 0.2
                    break

        # Specificity bonus
        specific_fields = ['citations', 'facts', 'recommendations', 'agent_name']
        for field in specific_fields:
            if field in result_data:
                score += 0.1

        return min(score, 1.0)

    def _is_successful_analysis(self, data: Any) -> bool:
        """Check if analysis data indicates success.

        Args:
            data: Analysis data

        Returns:
            True if analysis appears successful
        """
        if isinstance(data, dict):
            # Check for analysis indicators
            success_indicators = [
                'entities', 'relationships', 'insights', 'summary',
                'metadata', 'results', 'findings'
            ]
            return any(indicator in data for indicator in success_indicators)

        if isinstance(data, list):
            # Check for non-empty list with structured items
            return len(data) > 0 and isinstance(data[0], dict)

        return False

    def _get_analysis_type(self, data: Dict[str, Any]) -> str:
        """Determine type of analysis from data.

        Args:
            data: Analysis data

        Returns:
            Analysis type string
        """
        if 'entities' in data:
            return 'entity_extraction'
        elif 'relationships' in data:
            return 'relationship_analysis'
        elif 'insights' in data:
            return 'insight_generation'
        elif 'summary' in data:
            return 'summarization'
        else:
            return 'general_analysis'

    def extract_from_lawsuit_db(
        self,
        db_path: str = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db",
        max_documents: int = 50
    ) -> List[AgentMemory]:
        """Extract memories from lawsuit database documents.

        Args:
            db_path: Path to lawsuit database
            max_documents: Maximum documents to process

        Returns:
            List of extracted memories
        """
        memories = []

        if not Path(db_path).exists():
            return memories

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Get sample documents
            cur.execute("""
                SELECT id, content, content_hash, date_ingested
                FROM cleaned_documents
                ORDER BY LENGTH(content) DESC
                LIMIT ?
            """, (max_documents,))

            documents = cur.fetchall()

            for doc_id, content, content_hash, date_ingested in documents:
                # Extract document patterns for different agent types
                doc_memories = self._extract_document_patterns(
                    doc_id, content, content_hash, date_ingested
                )
                memories.extend(doc_memories)

            conn.close()

        except Exception as e:
            print(f"Error extracting from lawsuit database: {e}")

        return memories

    def extract_from_analysis_results(
        self,
        analysis_dir: str = "analysis_outputs"
    ) -> List[AgentMemory]:
        """Extract memories from analysis results.

        Args:
            analysis_dir: Directory containing analysis results

        Returns:
            List of extracted memories
        """
        memories = []
        analysis_path = Path(analysis_dir)

        if not analysis_path.exists():
            return memories

        try:
            # Extract from entities
            entities_file = analysis_path / "complete_analysis_fast" / "entities_all.json"
            if entities_file.exists():
                entity_memories = self._extract_entity_patterns(entities_file)
                memories.extend(entity_memories)

            # Extract from BN evidence mapping
            bn_file = analysis_path / "analysis_results" / "bn_evidence_mapping.json"
            if bn_file.exists():
                bn_memories = self._extract_bn_patterns(bn_file)
                memories.extend(bn_memories)

            # Extract from autogen context
            autogen_file = analysis_path / "analysis_results" / "autogen_context.json"
            if autogen_file.exists():
                autogen_memories = self._extract_autogen_patterns(autogen_file)
                memories.extend(autogen_memories)

        except Exception as e:
            print(f"Error extracting from analysis results: {e}")

        return memories

    def _extract_document_patterns(
        self,
        doc_id: int,
        content: str,
        content_hash: str,
        date_ingested: str
    ) -> List[AgentMemory]:
        """Extract agent-specific patterns from a document.

        Args:
            doc_id: Document ID
            content: Document content
            content_hash: Content hash
            date_ingested: Ingestion date

        Returns:
            List of memories for different agent types
        """
        memories = []

        # Analyze document content for patterns
        content_lower = content.lower()
        doc_length = len(content)

        # Citation patterns
        if any(term in content_lower for term in ['citation', 'cite', 'reference', '§', 'u.s.c']):
            memories.append(AgentMemory(
                agent_type="CitationFinderAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"Document {doc_id} contains legal citations and references",
                context={
                    'doc_id': doc_id,
                    'content_length': doc_length,
                    'has_citations': True,
                    'content_hash': content_hash
                },
                embedding=None,
                source="lawsuit_db_citations",
                timestamp=datetime.now()
            ))

        # Fact extraction patterns
        if any(term in content_lower for term in ['fact', 'evidence', 'allegation', 'claim']):
            memories.append(AgentMemory(
                agent_type="FactExtractorAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"Document {doc_id} contains factual allegations and evidence",
                context={
                    'doc_id': doc_id,
                    'content_length': doc_length,
                    'has_facts': True,
                    'content_hash': content_hash
                },
                embedding=None,
                source="lawsuit_db_facts",
                timestamp=datetime.now()
            ))

        # Legal analysis patterns
        if any(term in content_lower for term in ['legal', 'law', 'statute', 'regulation', 'court']):
            memories.append(AgentMemory(
                agent_type="LegalAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"Document {doc_id} contains legal analysis and court references",
                context={
                    'doc_id': doc_id,
                    'content_length': doc_length,
                    'has_legal_content': True,
                    'content_hash': content_hash
                },
                embedding=None,
                source="lawsuit_db_legal",
                timestamp=datetime.now()
            ))

        # Research patterns
        if any(term in content_lower for term in ['research', 'study', 'analysis', 'investigation']):
            memories.append(AgentMemory(
                agent_type="ResearchAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"Document {doc_id} contains research and investigative content",
                context={
                    'doc_id': doc_id,
                    'content_length': doc_length,
                    'has_research': True,
                    'content_hash': content_hash
                },
                embedding=None,
                source="lawsuit_db_research",
                timestamp=datetime.now()
            ))

        # Writing patterns (long documents)
        if doc_length > 10000:  # Long documents likely need writing assistance
            memories.append(AgentMemory(
                agent_type="WriterAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"Document {doc_id} is a long-form document ({doc_length} chars) requiring structured writing",
                context={
                    'doc_id': doc_id,
                    'content_length': doc_length,
                    'is_long_form': True,
                    'content_hash': content_hash
                },
                embedding=None,
                source="lawsuit_db_writing",
                timestamp=datetime.now()
            ))

        return memories

    def _extract_entity_patterns(self, entities_file: Path) -> List[AgentMemory]:
        """Extract memories from entity analysis.

        Args:
            entities_file: Path to entities JSON file

        Returns:
            List of entity-based memories
        """
        memories = []

        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)

            # Extract top entities for different agent types
            if 'entities' in entities_data:
                entities = entities_data['entities']

                # Top entities for fact extraction
                top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:20]

                memories.append(AgentMemory(
                    agent_type="FactExtractorAgent",
                    memory_id=str(uuid.uuid4()),
                    summary=f"Top entities in lawsuit: {', '.join([e[0] for e in top_entities[:5]])}",
                    context={
                        'top_entities': top_entities[:10],
                        'total_entities': len(entities),
                        'source_file': str(entities_file)
                    },
                    embedding=None,
                    source="entity_analysis",
                    timestamp=datetime.now()
                ))

        except Exception as e:
            print(f"Error extracting entity patterns: {e}")

        return memories

    def _extract_bn_patterns(self, bn_file: Path) -> List[AgentMemory]:
        """Extract memories from Bayesian Network evidence mapping.

        Args:
            bn_file: Path to BN evidence mapping file

        Returns:
            List of BN-based memories
        """
        memories = []

        try:
            with open(bn_file, 'r', encoding='utf-8') as f:
                bn_data = json.load(f)

            # Extract BN node mappings for strategic agents
            if 'node_mappings' in bn_data:
                node_mappings = bn_data['node_mappings']

                memories.append(AgentMemory(
                    agent_type="StrategicPlannerAgent",
                    memory_id=str(uuid.uuid4()),
                    summary=f"BN evidence mapping: {len(node_mappings)} nodes with evidence",
                    context={
                        'node_mappings': node_mappings,
                        'source_file': str(bn_file)
                    },
                    embedding=None,
                    source="bn_evidence",
                    timestamp=datetime.now()
                ))

        except Exception as e:
            print(f"Error extracting BN patterns: {e}")

        return memories

    def _extract_autogen_patterns(self, autogen_file: Path) -> List[AgentMemory]:
        """Extract memories from autogen context.

        Args:
            autogen_file: Path to autogen context file

        Returns:
            List of autogen-based memories
        """
        memories = []

        try:
            with open(autogen_file, 'r', encoding='utf-8') as f:
                autogen_data = json.load(f)

            # Extract strategic insights
            if 'strategic_insights' in autogen_data:
                insights = autogen_data['strategic_insights']

                memories.append(AgentMemory(
                    agent_type="StrategicPlannerAgent",
                    memory_id=str(uuid.uuid4()),
                    summary=f"Strategic insights from lawsuit analysis: {len(insights)} key insights",
                    context={
                        'insights': insights,
                        'source_file': str(autogen_file)
                    },
                    embedding=None,
                    source="autogen_context",
                    timestamp=datetime.now()
                ))

            # Extract bridge concepts
            if 'bridge_concepts' in autogen_data:
                bridges = autogen_data['bridge_concepts']

                memories.append(AgentMemory(
                    agent_type="ResearchAgent",
                    memory_id=str(uuid.uuid4()),
                    summary=f"Bridge concepts identified: {len(bridges)} critical connectors",
                    context={
                        'bridge_concepts': bridges,
                        'source_file': str(autogen_file)
                    },
                    embedding=None,
                    source="autogen_context",
                    timestamp=datetime.now()
                ))

        except Exception as e:
            print(f"Error extracting autogen patterns: {e}")

        return memories
