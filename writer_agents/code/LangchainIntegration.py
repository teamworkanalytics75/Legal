"""LangChain integration module for The Matrix.

Provides LangChain SQLDatabaseToolkit integration for evidence retrieval
while preserving the existing atomic agent architecture and cost optimization.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from langchain.agents import AgentExecutor
except ImportError:
    # AgentExecutor may not be needed directly, create_sql_agent returns it
    AgentExecutor = None

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
    OLLAMA_CLASS = OllamaLLM
except ImportError:
    try:
        from langchain_community.llms import Ollama
        OLLAMA_AVAILABLE = True
        OLLAMA_CLASS = Ollama
    except ImportError:
        OLLAMA_AVAILABLE = False
        OLLAMA_CLASS = None

try:
    from .agents import AgentFactory, ModelConfig
except ImportError:
    from agents import AgentFactory, ModelConfig

try:
    from .langchain_meta_memory import LangChainMetaMemory
except ImportError:
    try:
        from langchain_meta_memory import LangChainMetaMemory
    except ImportError:
        # Meta memory is optional
        LangChainMetaMemory = None


class LangChainSQLAgent:
    """LangChain SQL agent wrapper for The Matrix integration.

    Provides evidence retrieval capabilities using LangChain's SQLDatabaseToolkit
    while maintaining compatibility with existing atomic agent patterns.

    Enhanced with self-reflexive learning: queries LexBank for similar past queries
    before execution and saves all query results to LexBank.
    """

    def __init__(
        self,
        db_path: Path,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = False,
        memory_path: Optional[Path] = None
    ):
        """Initialize LangChain SQL agent.

        Args:
            db_path: Path to SQLite database
            model_config: Model configuration (defaults to gpt-4o-mini or local model)
            verbose: Whether to enable verbose output
            memory_path: Optional override for meta-memory SQLite location
        """
        self.db_path = db_path
        self.model_config = model_config or ModelConfig(model="gpt-4o-mini", use_local=True)
        self.verbose = verbose
        self.meta_memory = LangChainMetaMemory(memory_path) if LangChainMetaMemory else None
        self.table_names: List[str] = []

        # Check if using local model
        use_local = getattr(self.model_config, 'use_local', False)
        if not use_local and not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI not available and use_local=False. Install langchain-openai or set use_local=True")
        if use_local and not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install langchain-community or set use_local=False")

        # Initialize LangChain components
        self._setup_agent()
        self._bootstrap_meta_memory()

    def _setup_agent(self) -> None:
        """Set up LangChain SQL agent components."""
        # Database connection
        uri = f"sqlite:///{self.db_path.as_posix()}"
        self.database = SQLDatabase.from_uri(uri)
        try:
            self.table_names = sorted(self.database.get_usable_table_names())
        except Exception:
            self.table_names = []

        # LLM configuration - support local or OpenAI
        use_local = getattr(self.model_config, 'use_local', False)
        if use_local and OLLAMA_AVAILABLE:
            # Use local Ollama model
            local_model = getattr(self.model_config, 'local_model', 'qwen2.5:14b')
            local_base_url = getattr(self.model_config, 'local_base_url', 'http://localhost:11434')
            if self.verbose:
                logger.info(f"Using local Ollama model: {local_model} at {local_base_url}")
            # Try new API first, fallback to old
            try:
                self.llm = OLLAMA_CLASS(
                    model=local_model,
                    base_url=local_base_url,
                    temperature=0.0  # Deterministic for evidence retrieval
                )
            except TypeError:
                # Old API without base_url parameter
                self.llm = OLLAMA_CLASS(
                    model=local_model,
                    temperature=0.0
                )
        elif OPENAI_AVAILABLE:
            # Use OpenAI
            if "OPENAI_API_KEY" not in os.environ:
                raise RuntimeError("OPENAI_API_KEY environment variable required when using OpenAI")
            if self.verbose:
                logger.info(f"Using OpenAI model: {self.model_config.model}")
            self.llm = ChatOpenAI(
                model=self.model_config.model,
                temperature=0.0,  # Deterministic for evidence retrieval
                max_tokens=4000
            )
        else:
            raise RuntimeError("No LLM available. Install langchain-openai or langchain-community")

        # SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.database, llm=self.llm)

        # Create SQL agent with increased limits for local LLMs
        use_local = getattr(self.model_config, 'use_local', False)
        max_iterations = 10 if use_local else 5  # More iterations for slower local LLMs
        max_execution_time = 300 if use_local else 60  # 5 minutes for local, 1 minute for OpenAI

        # Build agent executor kwargs
        agent_executor_kwargs = {
            "handle_parsing_errors": True
        }

        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=self.verbose,
            top_k=10,  # Limit results for cost control
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,  # Pass directly to create_sql_agent
            agent_executor_kwargs=agent_executor_kwargs,
        )

    def _bootstrap_meta_memory(self) -> None:
        """Populate meta-memory with schema snapshots if missing."""
        if not self.meta_memory or not self.table_names:
            return

        cached_tables = set(self.meta_memory.list_cached_tables())
        for table in self.table_names:
            try:
                schema_text = self.database.get_table_info([table])
            except Exception:
                schema_text = ""
            sample_rows = None
            escaped_table = table.replace('"', '""')
            try:
                sample_rows = self.database.run(f'SELECT * FROM "{escaped_table}" LIMIT 3;')
                if sample_rows and not isinstance(sample_rows, str):
                    sample_rows = str(sample_rows)
            except Exception:
                sample_rows = None

            # Refresh cache if new or schema info changed
            if schema_text and (table not in cached_tables or self.verbose):
                try:
                    self.meta_memory.upsert_schema(table, schema_text, sample_rows)
                except Exception as cache_error:
                    if self.verbose:
                        print(f"Meta-memory schema cache failed for {table}: {cache_error}")

    def _infer_candidate_tables(self, question: str) -> List[str]:
        """Heuristic to choose relevant tables based on the question text."""
        if not self.table_names:
            return []

        question_lower = question.lower()
        matches = [tbl for tbl in self.table_names if tbl.lower() in question_lower]
        if matches:
            return matches

        # Fallback to first few tables to avoid overloading prompt context
        return self.table_names[: min(5, len(self.table_names))]

    def _build_meta_context(self, question: str) -> str:
        """Construct context block from cached schema and prior queries."""
        if not self.meta_memory:
            return ""
        tables = self._infer_candidate_tables(question)
        try:
            return self.meta_memory.build_context_block(question, tables)
        except Exception as context_error:
            if self.verbose:
                print(f"Meta-memory context build failed: {context_error}")
            return ""

    @staticmethod
    def _extract_sql_from_steps(steps: Sequence[Any]) -> List[str]:
        sql_statements: List[str] = []
        for step in steps or []:
            if not isinstance(step, tuple) or not step:
                continue
            action = step[0]
            tool_input = getattr(action, "tool_input", None)
            tool_name = getattr(action, "tool", "")
            if isinstance(tool_input, str) and "select" in tool_input.lower():
                sql_statements.append(tool_input.strip())
            elif isinstance(tool_input, dict):
                statement = tool_input.get("query") or tool_input.get("sql")
                if isinstance(statement, str) and "select" in statement.lower():
                    sql_statements.append(statement.strip())
            elif isinstance(action, str) and "select" in action.lower():
                sql_statements.append(action.strip())
            elif hasattr(action, "tool") and isinstance(tool_input, str):
                if "sql" in tool_name.lower() and "select" in tool_input.lower():
                    sql_statements.append(tool_input.strip())
        return sql_statements

    def _log_query_attempt(
        self,
        question: str,
        sql: Optional[str],
        success: bool,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        token_cost: Optional[float] = None
    ) -> None:
        if not self.meta_memory:
            return
        try:
            self.meta_memory.log_query(
                question=question,
                sql=sql,
                result_summary=result_summary,
                success=success,
                error=error,
                token_cost=token_cost,
                model=self.model_config.model
            )
        except Exception as log_error:
            if self.verbose:
                print(f"Meta-memory logging failed: {log_error}")

    def query_evidence(
        self,
        question: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query database for evidence using natural language.

        Enhanced with self-reflexive learning: queries LexBank for similar past queries
        and incorporates successful patterns into the query.

        Args:
            question: Natural language question about the evidence
            context: Optional context to include in the query

        Returns:
            Dictionary with query results and metadata
        """
        # SELF-REFLEXIVE LEARNING: Query LexBank for similar past queries
        past_query_memories = []
        past_sql_patterns = []
        try:
            # Try to import MemoryIntegrationManager
            from MemoryIntegrationManager import get_memory_manager
            memory_manager = get_memory_manager()
            if memory_manager:
                past_query_memories = memory_manager.query_similar_operations(
                    component="DatabaseQueryExecutor",
                    query_text=question,
                    k=3,
                    memory_types=["query"]
                )

                # Extract successful SQL patterns from past queries
                for memory in past_query_memories:
                    result = memory.context.get('result', {})
                    if isinstance(result, dict):
                        sql = result.get('sql') or result.get('executed_sql')
                        if sql and result.get('success', True):
                            past_sql_patterns.append({
                                'sql': sql,
                                'question': result.get('question', ''),
                                'relevance': memory.relevance_score
                            })
        except ImportError:
            # MemoryIntegrationManager not available, continue without it
            pass
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to query past queries from LexBank: {e}")

        # Build enhanced question with context
        enhanced_question = question
        if context:
            enhanced_question = f"Context: {context}\n\nQuestion: {question}"

        # Add past successful SQL patterns to context if available
        if past_sql_patterns and self.verbose:
            pattern_context = "\n\nPast successful SQL patterns for similar queries:\n"
            for i, pattern in enumerate(past_sql_patterns[:2], 1):  # Top 2 patterns
                pattern_context += f"\n{i}. Query: {pattern['question'][:100]}...\n"
                pattern_context += f"   SQL: {pattern['sql'][:200]}...\n"
            enhanced_question = pattern_context + "\n" + enhanced_question

        meta_context = self._build_meta_context(question)
        if meta_context:
            enhanced_question = (
                "Cached context for reference:\n"
                f"{meta_context}\n\n"
                "Use the cached context above to craft accurate SQL.\n\n"
                f"{enhanced_question}"
            )

        try:
            # Execute LangChain agent
            result = self.agent.invoke({"input": enhanced_question})

            cost_estimate = self._estimate_cost(result)
            intermediate_steps = result.get("intermediate_steps", [])
            sql_statements = self._extract_sql_from_steps(intermediate_steps)
            executed_sql = sql_statements[-1] if sql_statements else None
            answer = result.get("output", "")

            self._log_query_attempt(
                question=question,
                sql=executed_sql,
                success=True,
                result_summary=answer[:400] if answer else None,
                token_cost=cost_estimate
            )

            result_dict = {
                "success": True,
                "answer": answer,
                "intermediate_steps": intermediate_steps,
                "query_type": "langchain_sql",
                "model": self.model_config.model,
                "cost_estimate": cost_estimate,
                "meta_context": meta_context,
                "executed_sql": executed_sql,
                "sql_history": sql_statements,
                "past_memories_used": len(past_query_memories)
            }

            # SELF-REFLEXIVE LEARNING: Save query result to LexBank via DatabaseQueryRecorder
            # (DatabaseQueryRecorder already saves to LexBank via EpisodicMemoryBank)
            # But also save enhanced metadata via MemoryIntegrationManager if available
            try:
                from MemoryIntegrationManager import get_memory_manager
                memory_manager = get_memory_manager()
                if memory_manager:
                    metadata = {
                        'model': self.model_config.model,
                        'cost_estimate': cost_estimate,
                        'past_memories_used': len(past_query_memories),
                        'sql_patterns_reused': len(past_sql_patterns)
                    }
                    memory_manager.save_operation_result(
                        agent_type="DatabaseQueryExecutor",
                        operation_type="langchain_query",
                        result=result_dict,
                        metadata=metadata,
                        query_text=question,
                        memory_type="query"
                    )
            except ImportError:
                pass
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to save query to LexBank: {e}")

            return result_dict

        except Exception as e:
            self._log_query_attempt(
                question=question,
                sql=None,
                success=False,
                error=str(e)
            )

            # Save failure to memory for learning
            try:
                from MemoryIntegrationManager import get_memory_manager
                memory_manager = get_memory_manager()
                if memory_manager:
                    metadata = {
                        'model': self.model_config.model,
                        'error': str(e),
                        'past_memories_used': len(past_query_memories)
                    }
                    memory_manager.save_operation_result(
                        agent_type="DatabaseQueryExecutor",
                        operation_type="langchain_query",
                        result={"success": False, "error": str(e)},
                        metadata=metadata,
                        query_text=question,
                        memory_type="query"
                    )
            except ImportError:
                pass
            except Exception:
                pass  # Best-effort save, don't break on failure

            return {
                "success": False,
                "error": str(e),
                "query_type": "langchain_sql",
                "model": self.model_config.model,
                "meta_context": meta_context,
                "past_memories_used": len(past_query_memories)
            }

    def _estimate_cost(self, result: Dict[str, Any]) -> float:
        """Estimate cost for LangChain query.

        Args:
            result: LangChain agent result

        Returns:
            Estimated cost in dollars
        """
        # Rough estimation based on model and typical usage
        if self.model_config.model == "gpt-4o-mini":
            # gpt-4o-mini: $0.15/1M input, $0.60/1M output
            # Typical evidence query: ~500 input tokens, ~200 output tokens
            input_cost = 500 * 0.15 / 1_000_000
            output_cost = 200 * 0.60 / 1_000_000
            return round(input_cost + output_cost, 6)

        return 0.001  # Default estimate

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the connected database.

        Returns:
            Dictionary with database metadata
        """
        try:
            # Get table information
            tables = self.database.get_usable_table_names()

            return {
                "database_path": str(self.db_path),
                "tables": tables,
                "connection_status": "connected",
                "agent_type": "langchain_sql"
            }
        except Exception as e:
            return {
                "database_path": str(self.db_path),
                "connection_status": "error",
                "error": str(e),
                "agent_type": "langchain_sql"
            }

    def query_timeline(self, entity: str, date_range: Tuple[str, str]) -> List[Dict]:
        """Query events involving entity within date range.

        Args:
            entity: Entity to search for (e.g., "Harvard", "OGC", "Xi")
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format

        Returns:
            List of dictionaries containing timeline events
        """
        try:
            start_date, end_date = date_range

            timeline_query = f"""
            Find all events, communications, or actions involving "{entity}" between {start_date} and {end_date}.

            Focus on:
            - Emails or communications with dates
            - Statements or announcements with dates
            - Legal actions or decisions with dates
            - Any temporal constraints or deadlines

            For each event found, extract:
            - The specific date
            - Event description
            - Other entities involved
            - Source or context
            - Event significance

            Organize results chronologically by date.
            """

            result = self.query_evidence(timeline_query)

            # Parse and structure the timeline results
            timeline_events = []
            if isinstance(result, str):
                # Extract structured timeline from result
                lines = result.split('\n')
                current_event = {}

                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_event:
                            timeline_events.append(current_event)
                            current_event = {}
                        continue

                    # Look for date patterns
                    import re
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\w+ \d{1,2}, \d{4})', line)
                    if date_match:
                        if current_event:
                            timeline_events.append(current_event)
                        current_event = {
                            "date": date_match.group(1),
                            "description": line,
                            "entity": entity,
                            "date_range": date_range
                        }
                    elif current_event and line.startswith('-'):
                        current_event["description"] += " " + line[1:].strip()

                if current_event:
                    timeline_events.append(current_event)

            return timeline_events

        except Exception as e:
            logger.error(f"Timeline query failed: {e}")
            return [{"error": str(e), "entity": entity, "date_range": date_range}]

    def find_contradictions(self, claim: str) -> List[Dict]:
        """Find evidence that contradicts a claim.

        Args:
            claim: The claim to analyze for contradictions

        Returns:
            List of dictionaries containing contradicting evidence
        """
        try:
            contradiction_query = f"""
            Find evidence that contradicts or challenges the following claim: "{claim}"

            Look for:
            - Direct contradictions to the claim
            - Evidence that undermines the claim
            - Alternative explanations or interpretations
            - Conflicting statements or facts
            - Evidence that suggests the opposite

            For each piece of contradicting evidence found:
            - Describe how it contradicts the claim
            - Provide the specific evidence
            - Note the source or context
            - Assess the strength of the contradiction

            Focus on factual contradictions rather than opinions.
            """

            result = self.query_evidence(contradiction_query)

            # Parse contradictions from result
            contradictions = []
            if isinstance(result, str):
                lines = result.split('\n')
                current_contradiction = {}

                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_contradiction:
                            contradictions.append(current_contradiction)
                            current_contradiction = {}
                        continue

                    if line.startswith('-') or line.startswith('•'):
                        if current_contradiction:
                            contradictions.append(current_contradiction)
                        current_contradiction = {
                            "contradiction": line[1:].strip(),
                            "claim": claim,
                            "strength": "moderate"  # Default strength
                        }
                    elif current_contradiction and "strength:" in line.lower():
                        # Extract strength assessment
                        import re
                        strength_match = re.search(r'strength[:\s]+(\w+)', line.lower())
                        if strength_match:
                            current_contradiction["strength"] = strength_match.group(1)

            if current_contradiction:
                contradictions.append(current_contradiction)

            return contradictions

        except Exception as e:
            logger.error(f"Contradiction search failed: {e}")
            return [{"error": str(e), "claim": claim}]

    def correlate_evidence(self, entities: List[str]) -> Dict[str, List[Dict]]:
        """Find documents mentioning multiple entities together.

        Args:
            entities: List of entities to find correlations for

        Returns:
            Dictionary mapping entity pairs to correlated evidence
        """
        try:
            entities_str = ", ".join(entities)

            correlation_query = f"""
            Find documents and evidence that mention multiple entities together: {entities_str}

            Look for:
            - Documents that mention multiple entities in the same context
            - Communications between different entities
            - Joint actions or collaborations
            - Relationships or interactions between entities
            - Evidence that connects multiple entities

            For each correlation found:
            - List which entities are mentioned together
            - Describe the relationship or interaction
            - Provide the specific evidence
            - Note the source document
            - Assess the strength of the correlation

            Focus on meaningful relationships, not just co-occurrence.
            """

            result = self.query_evidence(correlation_query)

            # Parse correlations from result
            correlations = {}

            # Initialize correlation dictionary for all entity pairs
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    pair_key = f"{entity1} ↔ {entity2}"
                    correlations[pair_key] = []

            if isinstance(result, str):
                lines = result.split('\n')
                current_correlation = {}

                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_correlation:
                            # Determine which entity pair this correlation belongs to
                            entities_mentioned = []
                            for entity in entities:
                                if entity.lower() in current_correlation.get("description", "").lower():
                                    entities_mentioned.append(entity)

                            if len(entities_mentioned) >= 2:
                                pair_key = f"{entities_mentioned[0]} ↔ {entities_mentioned[1]}"
                                if pair_key in correlations:
                                    correlations[pair_key].append(current_correlation)
                            current_correlation = {}
                        continue

                    if line.startswith('-') or line.startswith('•'):
                        if current_correlation:
                            # Add previous correlation
                            entities_mentioned = []
                            for entity in entities:
                                if entity.lower() in current_correlation.get("description", "").lower():
                                    entities_mentioned.append(entity)

                            if len(entities_mentioned) >= 2:
                                pair_key = f"{entities_mentioned[0]} ↔ {entities_mentioned[1]}"
                                if pair_key in correlations:
                                    correlations[pair_key].append(current_correlation)

                        current_correlation = {
                            "description": line[1:].strip(),
                            "strength": "moderate"  # Default strength
                        }
                    elif current_correlation and "strength:" in line.lower():
                        # Extract strength assessment
                        import re
                        strength_match = re.search(r'strength[:\s]+(\w+)', line.lower())
                        if strength_match:
                            current_correlation["strength"] = strength_match.group(1)

            # Add final correlation if exists
            if current_correlation:
                entities_mentioned = []
                for entity in entities:
                    if entity.lower() in current_correlation.get("description", "").lower():
                        entities_mentioned.append(entity)

                if len(entities_mentioned) >= 2:
                    pair_key = f"{entities_mentioned[0]} ↔ {entities_mentioned[1]}"
                    if pair_key in correlations:
                        correlations[pair_key].append(current_correlation)

            return correlations

        except Exception as e:
            logger.error(f"Evidence correlation failed: {e}")
            return {"error": str(e), "entities": entities}


class EvidenceRetrievalAgent:
    """Enhanced evidence retrieval agent using LangChain.

    Replaces manual SQLite queries with LangChain's natural language
    database querying capabilities.
    """

    def __init__(
        self,
        db_path: Path,
        factory: AgentFactory,
        enable_langchain: bool = True,
        fallback_to_manual: bool = True
    ):
        """Initialize evidence retrieval agent.

        Args:
            db_path: Path to SQLite database
            factory: Agent factory for LLM configuration
            enable_langchain: Whether to use LangChain (vs manual queries)
            fallback_to_manual: Whether to fallback to manual queries on LangChain failure
        """
        self.db_path = db_path
        self.factory = factory
        self.enable_langchain = enable_langchain
        self.fallback_to_manual = fallback_to_manual

        # Initialize LangChain agent if enabled
        self.langchain_agent: Optional[LangChainSQLAgent] = None
        self._langchain_queries: int = 0
        self._langchain_cost_estimate: float = 0.0
        if enable_langchain:
            try:
                self.langchain_agent = LangChainSQLAgent(
                    db_path=db_path,
                    model_config=factory._config,
                    verbose=False  # Set to True for debugging
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain agent: {e}")
                self.langchain_agent = None

    def search_evidence(
        self,
        query: str,
        limit: int = 10,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for evidence using LangChain or manual queries.

        Args:
            query: Search query (natural language for LangChain, keyword for manual)
            limit: Maximum number of results
            context: Optional context for the search

        Returns:
            Dictionary with search results
        """
        # Try LangChain first if available
        if self.langchain_agent and self.enable_langchain:
            try:
                result = self.langchain_agent.query_evidence(query, context)
                if result["success"]:
                    self._langchain_queries += 1
                    self._langchain_cost_estimate += float(result.get("cost_estimate", 0.0) or 0.0)
                    return result
            except Exception as e:
                print(f"LangChain query failed: {e}")

        # Fallback to manual queries if enabled
        if self.fallback_to_manual:
            return self._manual_search(query, limit)

        return {
            "success": False,
            "error": "Both LangChain and manual queries failed",
            "query_type": "failed"
        }

    def _manual_search(self, keyword: str, limit: int) -> Dict[str, Any]:
        """Fallback manual search using direct SQLite queries.

        Args:
            keyword: Search keyword
            limit: Maximum results

        Returns:
            Dictionary with search results
        """
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    id,
                    SUBSTR(content, 1, 500) as preview,
                    LENGTH(content) as doc_length
                FROM cleaned_documents
                WHERE content LIKE ?
                ORDER BY doc_length DESC
                LIMIT ?
            """, (f'%{keyword}%', limit))

            results = cursor.fetchall()
            conn.close()

            return {
                "success": True,
                "results": [
                    {
                        "id": row[0],
                        "preview": row[1],
                        "length": row[2]
                    }
                    for row in results
                ],
                "query_type": "manual_sqlite",
                "count": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_type": "manual_sqlite"
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the evidence retrieval agent.

        Returns:
            Dictionary with agent metadata
        """
        info = {
            "agent_type": "evidence_retrieval",
            "database_path": str(self.db_path),
            "langchain_enabled": self.enable_langchain,
            "langchain_available": self.langchain_agent is not None,
            "fallback_enabled": self.fallback_to_manual
        }

        if self.langchain_agent:
            info.update(self.langchain_agent.get_database_info())

        return info

    # ------------------------------------------------------------------ #
    # LangChain metrics helpers
    # ------------------------------------------------------------------ #

    def reset_langchain_metrics(self) -> None:
        """Reset accumulated LangChain usage metrics."""
        self._langchain_queries = 0
        self._langchain_cost_estimate = 0.0

    def get_langchain_metrics(self) -> Dict[str, Any]:
        """Return accumulated LangChain usage metrics."""
        return {
            "queries_count": self._langchain_queries,
            "cost_estimate": round(self._langchain_cost_estimate, 6),
        }


# Factory function for easy integration
def create_evidence_retrieval_agent(
    db_path: Path,
    factory: AgentFactory,
    **kwargs
) -> EvidenceRetrievalAgent:
    """Create evidence retrieval agent with LangChain integration.

    Args:
        db_path: Path to SQLite database
        factory: Agent factory for configuration
        **kwargs: Additional arguments for EvidenceRetrievalAgent

    Returns:
        Configured EvidenceRetrievalAgent instance
    """
    return EvidenceRetrievalAgent(db_path, factory, **kwargs)
