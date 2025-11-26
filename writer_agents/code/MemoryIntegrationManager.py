"""Memory Integration Manager - Central coordinator for LexBank (EpisodicMemoryBank) integration.

Provides singleton pattern for shared LexBank instance, decorators for automatic memory capture,
and cross-component memory queries to enable self-reflexive learning across all system components.
"""

from __future__ import annotations

import functools
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

try:
    from .EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryEntry, EpisodicMemoryRetriever
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryEntry, EpisodicMemoryRetriever

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class MemoryIntegrationManager:
    """
    Singleton manager for LexBank (EpisodicMemoryBank) integration.

    Provides centralized memory management, decorators for automatic capture,
    and cross-component memory queries for self-reflexive learning.
    """

    _instance: Optional['MemoryIntegrationManager'] = None
    _memory_store: Optional[EpisodicMemoryBank] = None
    _retriever: Optional[EpisodicMemoryRetriever] = None

    def __new__(cls) -> 'MemoryIntegrationManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        use_local_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize memory integration manager.

        Args:
            storage_path: Path to LexBank storage directory
            use_local_embeddings: Whether to use local embeddings
            embedding_model: Embedding model name
        """
        if self._initialized:
            return

        if storage_path is None:
            # Default to project root memory_store
            project_root = Path(__file__).parent.parent.parent
            storage_path = project_root / "memory_store" / "memory_snapshots"

        # Ensure storage directory exists
        storage_path = Path(storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        try:
            self._memory_store = EpisodicMemoryBank(
                storage_path=storage_path,
                use_local_embeddings=use_local_embeddings,
                embedding_model=embedding_model
            )
            self._retriever = EpisodicMemoryRetriever(self._memory_store)
            self._storage_path = storage_path
            logger.info(f"MemoryIntegrationManager initialized with storage: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryIntegrationManager: {e}")
            self._memory_store = None
            self._retriever = None

        self._initialized = True

    @classmethod
    def get_instance(
        cls,
        storage_path: Optional[Path] = None,
        use_local_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> 'MemoryIntegrationManager':
        """Get singleton instance of MemoryIntegrationManager.

        Automatically initializes on first call - no manual setup required.
        Uses default storage path if not provided.

        Args:
            storage_path: Optional storage path (only used on first initialization)
            use_local_embeddings: Whether to use local embeddings
            embedding_model: Embedding model name

        Returns:
            MemoryIntegrationManager instance (always succeeds, may have no-op memory store if init fails)
        """
        if cls._instance is None:
            cls._instance = cls(storage_path, use_local_embeddings, embedding_model)
        elif not cls._instance._initialized:
            # Re-initialize if needed
            cls._instance.__init__(storage_path, use_local_embeddings, embedding_model)
        return cls._instance

    def get_memory_store(self) -> Optional[EpisodicMemoryBank]:
        """Get the shared LexBank (EpisodicMemoryBank) instance.

        Returns:
            EpisodicMemoryBank instance or None if not initialized
        """
        return self._memory_store

    def get_retriever(self) -> Optional[EpisodicMemoryRetriever]:
        """Get the memory retriever instance.

        Returns:
            EpisodicMemoryRetriever instance or None if not initialized
        """
        return self._retriever

    def save_operation_result(
        self,
        agent_type: str,
        operation_type: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        memory_type: str = "execution"
    ) -> Optional[str]:
        """Save an operation result to LexBank.

        Args:
            agent_type: Type of agent/component (e.g., "CitationRetrievalPlugin")
            operation_type: Type of operation (e.g., "citation_search")
            result: Operation result (will be serialized to context)
            metadata: Additional metadata dictionary
            query_text: Optional query text for summary generation
            memory_type: Memory type (execution, query, conversation, edit, document, analysis, report)

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self._memory_store:
            logger.warning("Memory store not initialized, cannot save operation result")
            return None

        try:
            # Generate summary
            if query_text:
                summary = f"{operation_type}: {query_text[:200]}"
            elif isinstance(result, dict) and "recommended_citation" in result:
                summary = f"{operation_type}: Recommended {result.get('recommended_citation', 'N/A')}"
            elif isinstance(result, dict) and "answer" in result:
                summary = f"{operation_type}: {result.get('answer', '')[:200]}"
            else:
                summary = f"{operation_type} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Build context
            context: Dict[str, Any] = {
                "operation_type": operation_type,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }

            # Serialize result
            if isinstance(result, dict):
                context["result"] = result
            elif isinstance(result, (str, int, float, bool)):
                context["result"] = result
            else:
                # Try to convert to dict if possible
                try:
                    context["result"] = str(result)
                except Exception:
                    context["result"] = f"<non-serializable: {type(result).__name__}>"

            # Create memory entry
            memory = EpisodicMemoryEntry(
                agent_type=agent_type,
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context=context,
                source="memory_integration_manager",
                timestamp=datetime.now(),
                memory_type=memory_type
            )

            self._memory_store.add(memory)
            logger.debug(f"Saved operation result to LexBank: {agent_type}/{operation_type}")

            return memory.memory_id

        except Exception as e:
            logger.error(f"Error saving operation result to LexBank: {e}")
            return None

    def query_similar_operations(
        self,
        component: str,
        query_text: str,
        k: int = 5,
        memory_types: Optional[List[str]] = None,
        agent_types: Optional[List[str]] = None
    ) -> List[EpisodicMemoryEntry]:
        """Query LexBank for similar past operations.

        Args:
            component: Component name to search (or None for all)
            query_text: Query text for semantic similarity search
            k: Number of results to return
            memory_types: Optional list of memory types to filter by
            agent_types: Optional list of agent types to filter by (defaults to component)

        Returns:
            List of EpisodicMemoryEntry objects
        """
        if not self._retriever:
            logger.warning("Memory retriever not initialized, cannot query similar operations")
            return []

        try:
            # Default agent_types to component if provided
            if agent_types is None and component:
                agent_types = [component]

            if agent_types:
                # Search specific agent types
                all_results: List[EpisodicMemoryEntry] = []
                for agent_type in agent_types:
                    memories = self._memory_store.retrieve(
                        agent_type=agent_type,
                        query=query_text,
                        k=k,
                        memory_types=memory_types
                    )
                    all_results.extend(memories)

                # Sort by relevance and return top k
                all_results.sort(key=lambda x: x.relevance_score, reverse=True)
                return all_results[:k]
            else:
                # Use retriever for cross-agent search
                if hasattr(self._retriever, 'get_all_relevant_context'):
                    all_context = self._retriever.get_all_relevant_context(
                        query=query_text,
                        k=k,
                        include_types=memory_types or None
                    )
                    # Flatten and sort
                    flattened = []
                    for agent_type, memories in all_context.items():
                        flattened.extend(memories)
                    flattened.sort(key=lambda x: x.relevance_score, reverse=True)
                    return flattened[:k]
                else:
                    return []

        except Exception as e:
            logger.error(f"Error querying similar operations: {e}")
            return []

    def capture_memory(
        self,
        agent_type: str,
        memory_type: str = "execution",
        operation_type: Optional[str] = None,
        include_result: bool = True,
        include_args: bool = True
    ) -> Callable[[F], F]:
        """Decorator for automatic memory capture.

        Args:
            agent_type: Type of agent/component
            memory_type: Memory type (execution, query, conversation, edit, document, analysis, report)
            operation_type: Operation type identifier (defaults to function name)
            include_result: Whether to include function result in memory
            include_args: Whether to include function arguments in memory

        Returns:
            Decorator function
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                op_type = operation_type or func.__name__

                # Build metadata
                metadata: Dict[str, Any] = {
                    "function_name": func.__name__,
                    "module": func.__module__
                }

                if include_args:
                    # Serialize args and kwargs (safely)
                    try:
                        metadata["args"] = [str(arg)[:500] for arg in args[:5]]  # Limit args
                        metadata["kwargs"] = {k: str(v)[:500] for k, v in list(kwargs.items())[:10]}
                    except Exception:
                        metadata["args"] = f"<{len(args)} args>"
                        metadata["kwargs"] = f"<{len(kwargs)} kwargs>"

                # Extract query text from args/kwargs if present
                query_text = None
                if 'query' in kwargs:
                    query_text = str(kwargs['query'])
                elif 'query_text' in kwargs:
                    query_text = str(kwargs['query_text'])
                elif 'text' in kwargs:
                    query_text = str(kwargs['text'])
                elif args and isinstance(args[0], str):
                    query_text = args[0][:500]

                # Execute function
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = (datetime.now() - start_time).total_seconds()
                    metadata["execution_duration_seconds"] = duration
                    metadata["success"] = success
                    if error:
                        metadata["error"] = error

                # Save to memory
                if include_result and success:
                    self.save_operation_result(
                        agent_type=agent_type,
                        operation_type=op_type,
                        result=result,
                        metadata=metadata,
                        query_text=query_text,
                        memory_type=memory_type
                    )

                return result

            return wrapper  # type: ignore
        return decorator

    def save(self) -> None:
        """Persist memory store to disk."""
        if self._memory_store:
            try:
                self._memory_store.save()
                logger.debug("Memory store saved to disk")
            except Exception as e:
                logger.error(f"Error saving memory store: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self._memory_store:
            return {"error": "Memory store not initialized"}

        try:
            return self._memory_store.get_all_stats()
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}


# Convenience functions for easy access
def get_memory_manager() -> MemoryIntegrationManager:
    """Get the global MemoryIntegrationManager instance.

    Automatically initializes on first call - no setup required.
    This function is called automatically by components that need memory integration.

    Returns:
        MemoryIntegrationManager instance (always succeeds, gracefully handles failures)
    """
    return MemoryIntegrationManager.get_instance()


def get_memory_store() -> Optional[EpisodicMemoryBank]:
    """Get the shared LexBank instance.

    Automatically initializes MemoryIntegrationManager if needed.

    Returns:
        EpisodicMemoryBank instance or None if initialization failed
    """
    return get_memory_manager().get_memory_store()


def get_memory_retriever() -> Optional[EpisodicMemoryRetriever]:
    """Get the memory retriever instance.

    Automatically initializes MemoryIntegrationManager if needed.

    Returns:
        EpisodicMemoryRetriever instance or None if initialization failed
    """
    return get_memory_manager().get_retriever()

