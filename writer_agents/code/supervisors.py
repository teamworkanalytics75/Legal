"""Phase supervisors for atomic agent orchestration.

Each phase supervisor manages a pool of atomic agents for its workflow phase.
Supervisors handle spawning, execution, retry logic, and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

try:
    from .atomic_agent import AtomicAgent
    from .agents import AgentFactory, ModelConfig
    from .job_persistence import JobManager
    from .task_dag import Task, TaskDAG
    from .worker_pool import WorkerPool

    from .atomic_agents.citations import (
        CitationFinderAgent,
        CitationNormalizerAgent,
        CitationVerifierAgent,
        CitationLocatorAgent,
        CitationInserterAgent,
    )
    from .atomic_agents.research import (
        FactExtractorAgent,
        PrecedentFinderAgent,
        PrecedentRankerAgent,
        PrecedentSummarizerAgent,
        StatuteLocatorAgent,
        ExhibitFetcherAgent,
    )
    from .atomic_agents.enhanced_research import (
        EnhancedPrecedentFinderAgent,
        EnhancedFactExtractorAgent,
    )
    from .atomic_agents.drafting import (
        OutlineBuilderAgent,
        SectionWriterAgent,
        ParagraphWriterAgent,
        TransitionAgent,
    )
    from .atomic_agents.review import (
        GrammarFixerAgent,
        StyleCheckerAgent,
        LogicCheckerAgent,
        ConsistencyCheckerAgent,
        RedactionAgent,
        ComplianceAgent,
        ExpertQAAgent,
    )
    from .atomic_agents.output import (
        MarkdownExporterAgent,
        DocxExporterAgent,
        MetadataTaggerAgent,
    )
except ImportError:  # Fallback for legacy execution paths
    from atomic_agent import AtomicAgent
    from agents import AgentFactory, ModelConfig
    from job_persistence import JobManager
    from task_dag import Task, TaskDAG
    from worker_pool import WorkerPool

    from atomic_agents.citations import (
        CitationFinderAgent,
        CitationNormalizerAgent,
        CitationVerifierAgent,
        CitationLocatorAgent,
        CitationInserterAgent,
    )
    from atomic_agents.research import (
        FactExtractorAgent,
        PrecedentFinderAgent,
        PrecedentRankerAgent,
        PrecedentSummarizerAgent,
        StatuteLocatorAgent,
        ExhibitFetcherAgent,
    )
    from atomic_agents.enhanced_research import (
        EnhancedPrecedentFinderAgent,
        EnhancedFactExtractorAgent,
    )
    from atomic_agents.drafting import (
        OutlineBuilderAgent,
        SectionWriterAgent,
        ParagraphWriterAgent,
        TransitionAgent,
    )
    from atomic_agents.review import (
        GrammarFixerAgent,
        StyleCheckerAgent,
        LogicCheckerAgent,
        ConsistencyCheckerAgent,
        RedactionAgent,
        ComplianceAgent,
        ExpertQAAgent,
    )
    from atomic_agents.output import (
        MarkdownExporterAgent,
        DocxExporterAgent,
        MetadataTaggerAgent,
    )

logger = logging.getLogger(__name__)


class PhaseSupervisor(ABC):
    """Base supervisor for a workflow phase.

    Responsibilities:
    - Maintain worker pool for phase agents
    - Spawn/reap agents based on demand
    - Execute tasks with retry logic
    - Monitor progress and enforce budgets
    """

    phase_name: str = "abstract"

    def __init__(
        self,
        session: Any,
        job_manager: JobManager,
        budgets: Dict[str, Any],
        max_workers: int = 6,
        memory_config: Optional[Dict[str, Any]] = None, # NEW: Memory config
        session_id: Optional[str] = None,  # NEW: Session ID
        use_langchain: bool = False,  # NEW: LangChain toggle
        langchain_db_path: Optional[Path] = None,  # NEW: Database path
        premium_mode: bool = False,  # NEW: Premium mode toggle
        premium_model: str = "gpt-4o",  # NEW: Premium model
    ) -> None:
        """Initialize phase supervisor.

        Args:
            session: SQLite session for persistence
            job_manager: Job manager for tracking
            budgets: Budget constraints (tokens, time)
            max_workers: Maximum concurrent workers
            memory_config: Memory system configuration
            session_id: Optional session ID for conversation continuity
        """
        self.session = session
        self.job_manager = job_manager
        self.budgets = budgets
        self.memory_config = memory_config or {} # NEW: Store memory config
        self.session_id = session_id  # NEW: Store session ID
        self.use_langchain = use_langchain  # NEW: Store LangChain toggle
        self.langchain_db_path = langchain_db_path  # NEW: Store database path
        self.premium_mode = premium_mode  # NEW: Store premium mode
        self.premium_model = premium_model  # NEW: Store premium model
        self.worker_pool = WorkerPool(max_workers)
        self.agents: Dict[str, AtomicAgent] = {}
        self.agent_factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        self.tokens_used = 0
        self.time_used = 0.0

    def _get_model_for_agent(self, agent_type: str) -> str:
        """Get appropriate model for agent based on meta-category and premium mode.

        Args:
            agent_type: Agent class name

        Returns:
            Model name to use
        """
        try:
            from .agent_tiers import get_agent_meta_category
            meta_category = get_agent_meta_category(agent_type)

            if self.premium_mode and meta_category in ["completeness", "precision"]:
                return self.premium_model
            return "gpt-4o-mini"
        except ImportError:
            # Fallback if agent_tiers not available
            return "gpt-4o-mini"

    async def execute(self, task_dag: TaskDAG) -> Dict[str, Any]:
        """Execute phase with parallel workers on ready tasks.

        Args:
            task_dag: DAG of tasks for this phase

        Returns:
            Dictionary of task results
        """
        logger.info(f"{self.phase_name} phase starting: {task_dag}")
        start_time = time.time()

        # Execute tasks in waves based on dependencies
        while not task_dag.is_complete():
            # Get next ready tasks
            ready_tasks = task_dag.next_ready()

            if not ready_tasks:
                if task_dag.has_failures():
                    logger.error(f"{self.phase_name} phase has failures, stopping")
                    break

                # No ready tasks but not complete - wait for running tasks
                await asyncio.sleep(0.1)
                continue

            # Check budget before proceeding
            if self._is_over_budget():
                logger.warning(f"{self.phase_name} phase over budget, stopping")
                break

            # Execute batch of ready tasks
            batch_size = min(len(ready_tasks), self.worker_pool.available or 1)
            batch = ready_tasks[:batch_size]

            logger.info(f"Executing batch of {len(batch)} tasks")

            # Run batch in parallel
            results = await asyncio.gather(
                *[self._execute_task(task) for task in batch],
                return_exceptions=True
            )

            # Process results
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Task {task.id} failed: {result}")
                    task_dag.mark_failed(task.id)
                else:
                    task_dag.set_result(task.id, result)
                    task_dag.mark_done([task])

        elapsed = time.time() - start_time
        self.time_used += elapsed

        logger.info(
            f"{self.phase_name} phase complete: "
            f"{task_dag.get_stats()}, time={elapsed:.2f}s"
        )

        return task_dag.collect_outputs()

    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute single task with retry and monitoring.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        # Create job record with LangChain tracking
        job_id = self.job_manager.create_job(
            phase=self.phase_name,
            agent_type=task.agent_type,
            payload=task.input_data,
            priority=task.priority,
            budget_tokens=self.budgets.get('tokens_per_task'),
            budget_seconds=self.budgets.get('seconds_per_task'),
            session_id=self.session_id,
            langchain_enabled=self.use_langchain,
            langchain_queries_count=0,  # Will be updated after execution
            langchain_cost_estimate=0.0,  # Will be updated after execution
            premium_mode=self.premium_mode,  # NEW: Pass premium mode
            premium_agents_used=0,  # Will be updated after execution
            estimated_premium_cost=0.0,  # Will be updated after execution
        )

        # Get or spawn agent
        agent = self._get_or_spawn_agent(task.agent_type)

        # Reset LangChain metrics for this execution if supported
        if hasattr(agent, "reset_langchain_metrics"):
            try:
                agent.reset_langchain_metrics()  # type: ignore[attr-defined]
            except Exception as reset_error:
                logger.debug(f"Could not reset LangChain metrics: {reset_error}")

        # Set session context if available
        if self.session_id and hasattr(agent, 'set_session_context'):
            # Get session context from session manager
            try:
                try:
                    from .session_manager import SessionManager
                except ImportError:
                    from session_manager import SessionManager
                session_manager = SessionManager(self.job_manager.db_path)
                session_context = session_manager.get_session_context(
                    self.session_id,
                    max_interactions=10,
                    max_tokens=5000
                )
                if session_context:
                    agent.set_session_context(session_context)
            except Exception as e:
                logger.debug(f"Could not load session context: {e}")

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Mark task as running
                task.status = "running"

                # Execute agent
                start_time = time.time()
                result = await agent.execute(task.input_data)
                duration = time.time() - start_time

                # Track metrics
                metrics = agent.get_metrics()
                self.tokens_used += metrics.tokens_in + metrics.tokens_out

                # Extract LangChain metrics if available
                langchain_queries_count = 0
                langchain_cost_estimate = 0.0
                if hasattr(agent, 'get_langchain_metrics'):
                    langchain_metrics = agent.get_langchain_metrics()
                    langchain_queries_count = langchain_metrics.get('queries_count', 0)
                    langchain_cost_estimate = langchain_metrics.get('cost_estimate', 0.0)

                # Calculate premium metrics
                premium_agents_used = 0
                estimated_premium_cost = 0.0

                if self.premium_mode:
                    try:
                        from .agent_tiers import get_agent_meta_category
                        meta_category = get_agent_meta_category(task.agent_type)
                        if meta_category in ["completeness", "precision"]:
                            premium_agents_used = 1
                            # Estimate premium cost based on tokens used
                            # GPT-4o is ~33x more expensive than GPT-4o-mini
                            estimated_premium_cost = (metrics.tokens_in + metrics.tokens_out) * 0.00003  # Rough estimate
                    except ImportError:
                        pass

                # Mark job complete with LangChain and premium metrics
                self.job_manager.mark_completed(
                    job_id,
                    result,
                    tokens_in=metrics.tokens_in,
                    tokens_out=metrics.tokens_out,
                    duration=duration,
                    langchain_queries_count=langchain_queries_count,
                    langchain_cost_estimate=langchain_cost_estimate,
                    premium_agents_used=premium_agents_used,
                    estimated_premium_cost=estimated_premium_cost,
                )

                logger.debug(
                    f"Task {task.id} completed: "
                    f"tokens={metrics.tokens_in + metrics.tokens_out}, "
                    f"duration={duration:.2f}s"
                )

                return result

            except Exception as e:
                logger.warning(
                    f"Task {task.id} attempt {attempt + 1} failed: {e}"
                )

                self.job_manager.log_event(job_id, "retry", {
                    "attempt": attempt + 1,
                    "error": str(e),
                })

                if attempt == max_retries - 1:
                    # Final failure
                    self.job_manager.mark_failed(
                        job_id,
                        str(e),
                        retry_count=attempt + 1,
                    )
                    raise

                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Task {task.id} failed after {max_retries} attempts")

    def _get_or_spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Get existing agent or spawn new instance.

        Args:
            agent_type: Agent class name

        Returns:
            Agent instance
        """
        if agent_type not in self.agents:
            # Update agent factory with appropriate model for this agent
            model = self._get_model_for_agent(agent_type)
            self.agent_factory = AgentFactory(ModelConfig(model=model))
            self.agents[agent_type] = self._spawn_agent(agent_type)

        return self.agents[agent_type]

    @abstractmethod
    def _spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Spawn a new agent instance.

        Override in subclasses to provide phase-specific agents.

        Args:
            agent_type: Agent class name

        Returns:
            New agent instance
        """
        raise NotImplementedError

    def _is_over_budget(self) -> bool:
        """Check if phase has exceeded budgets.

        Returns:
            True if over any budget limit
        """
        if 'max_tokens' in self.budgets:
            if self.tokens_used > self.budgets['max_tokens']:
                return True

        if 'max_seconds' in self.budgets:
            if self.time_used > self.budgets['max_seconds']:
                return True

        return False

    async def close(self) -> None:
        """Close all agents and clean up resources."""
        await self.agent_factory.close()
        for agent in self.agents.values():
            agent.clear_cache()


class CitationSupervisor(PhaseSupervisor):
    """Supervisor for citation processing phase.

    Manages citation finding, normalization, verification, location, and insertion.
    All agents are deterministic (no LLM cost).
    """

    phase_name = "citation"

    def _spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Spawn citation agent.

        Args:
            agent_type: Agent class name

        Returns:
            Citation agent instance
        """
        agent_classes: Dict[str, Type[AtomicAgent]] = {
            'CitationFinderAgent': CitationFinderAgent,
            'CitationNormalizerAgent': CitationNormalizerAgent,
            'CitationVerifierAgent': CitationVerifierAgent,
            'CitationLocatorAgent': CitationLocatorAgent,
            'CitationInserterAgent': CitationInserterAgent,
        }

        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown citation agent type: {agent_type}")

        return agent_class(
            self.agent_factory,
            enable_memory=self.memory_config.get('enabled', True), # NEW: Pass memory config
            memory_config={
                'k_neighbors': self.memory_config.get('k_neighbors', 5),
                'max_tokens': self.memory_config.get('max_tokens', 500)
            }
        )


class ResearchSupervisor(PhaseSupervisor):
    """Supervisor for research phase.

    Manages fact extraction, precedent finding/ranking, statute location, exhibit fetching.
    Mix of deterministic and LLM-based agents.
    """

    phase_name = "research"

    def _spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Spawn research agent.

        Args:
            agent_type: Agent class name

        Returns:
            Research agent instance
        """
        # Choose agent classes based on LangChain configuration
        if self.use_langchain and self.langchain_db_path:
            agent_classes: Dict[str, Type[AtomicAgent]] = {
                'FactExtractorAgent': EnhancedFactExtractorAgent,
                'PrecedentFinderAgent': EnhancedPrecedentFinderAgent,
                'PrecedentRankerAgent': PrecedentRankerAgent,
                'PrecedentSummarizerAgent': PrecedentSummarizerAgent,
                'StatuteLocatorAgent': StatuteLocatorAgent,
                'ExhibitFetcherAgent': ExhibitFetcherAgent,
            }
        else:
            agent_classes: Dict[str, Type[AtomicAgent]] = {
                'FactExtractorAgent': FactExtractorAgent,
                'PrecedentFinderAgent': PrecedentFinderAgent,
                'PrecedentRankerAgent': PrecedentRankerAgent,
                'PrecedentSummarizerAgent': PrecedentSummarizerAgent,
                'StatuteLocatorAgent': StatuteLocatorAgent,
                'ExhibitFetcherAgent': ExhibitFetcherAgent,
            }

        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown research agent type: {agent_type}")

        logger.info(f"Spawning {agent_type} for research phase (LangChain: {self.use_langchain})")

        # Pass database path to enhanced agents
        if self.use_langchain and self.langchain_db_path and agent_type in ['FactExtractorAgent', 'PrecedentFinderAgent']:
            return agent_class(
                self.agent_factory,
                db_path=self.langchain_db_path,
                enable_memory=self.memory_config.get('enabled', True),
                memory_config={
                    'k_neighbors': self.memory_config.get('k_neighbors', 5),
                    'max_tokens': self.memory_config.get('max_tokens', 500)
                }
            )
        else:
            return agent_class(
                self.agent_factory,
                enable_memory=self.memory_config.get('enabled', True), # NEW: Pass memory config
                memory_config={
                    'k_neighbors': self.memory_config.get('k_neighbors', 5),
                    'max_tokens': self.memory_config.get('max_tokens', 500)
                }
            )


class DraftingSupervisor(PhaseSupervisor):
    """Supervisor for drafting phase.

    Manages outline building, section writing, paragraph writing, transitions.
    All agents are LLM-based.
    """

    phase_name = "drafting"

    def _spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Spawn drafting agent.

        Args:
            agent_type: Agent class name

        Returns:
            Drafting agent instance
        """
        agent_classes: Dict[str, Type[AtomicAgent]] = {
            'OutlineBuilderAgent': OutlineBuilderAgent,
            'SectionWriterAgent': SectionWriterAgent,
            'ParagraphWriterAgent': ParagraphWriterAgent,
            'TransitionAgent': TransitionAgent,
        }

        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown drafting agent type: {agent_type}")

        logger.info(f"Spawning {agent_type} for drafting phase")
        return agent_class(
            self.agent_factory,
            enable_memory=self.memory_config.get('enabled', True), # NEW: Pass memory config
            memory_config={
                'k_neighbors': self.memory_config.get('k_neighbors', 5),
                'max_tokens': self.memory_config.get('max_tokens', 500)
            }
        )


class QASupervisor(PhaseSupervisor):
    """Supervisor for QA phase.

    Manages grammar, style, logic, consistency, redaction, compliance, expert QA.
    Mix of deterministic and LLM-based agents.
    Enforces sequential execution for deterministic revision order.
    """

    phase_name = "qa"

    # QA tasks must be sequential for deterministic results
    QA_SEQUENCE = [
        'GrammarFixerAgent',
        'StyleCheckerAgent',
        'ConsistencyCheckerAgent',
        'RedactionAgent',
        'ComplianceAgent',
        'LogicCheckerAgent',
        'ExpertQAAgent',
    ]

    async def execute(self, task_dag: TaskDAG) -> Dict[str, Any]:
        """Execute QA tasks sequentially for deterministic revisions.

        Args:
            task_dag: DAG of QA tasks

        Returns:
            Dictionary of task results
        """
        logger.info(f"{self.phase_name} phase starting (sequential)")
        start_time = time.time()

        results = {}

        # Execute tasks in strict sequence
        for agent_type in self.QA_SEQUENCE:
            # Find task for this agent type
            task = None
            for t in task_dag.tasks.values():
                if t.agent_type == agent_type and t.status == "pending":
                    task = t
                    break

            if not task:
                continue # Agent not needed for this analysis

            # Execute task
            try:
                result = await self._execute_task(task)
                results[task.id] = result
                task_dag.set_result(task.id, result)
                task_dag.mark_done([task])
            except Exception as e:
                logger.error(f"QA task {task.id} failed: {e}")
                task_dag.mark_failed(task.id)
                break # Stop QA chain on failure

        elapsed = time.time() - start_time
        self.time_used += elapsed

        logger.info(
            f"{self.phase_name} phase complete: "
            f"{len(results)} tasks, time={elapsed:.2f}s"
        )

        return results

    def _spawn_agent(self, agent_type: str) -> AtomicAgent:
        """Spawn QA agent.

        Args:
            agent_type: Agent class name

        Returns:
            QA agent instance
        """
        agent_classes: Dict[str, Type[AtomicAgent]] = {
            'GrammarFixerAgent': GrammarFixerAgent,
            'StyleCheckerAgent': StyleCheckerAgent,
            'LogicCheckerAgent': LogicCheckerAgent,
            'ConsistencyCheckerAgent': ConsistencyCheckerAgent,
            'RedactionAgent': RedactionAgent,
            'ComplianceAgent': ComplianceAgent,
            'ExpertQAAgent': ExpertQAAgent,
        }

        agent_class = agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown QA agent type: {agent_type}")

        logger.info(f"Spawning {agent_type} for QA phase")
        return agent_class(
            self.agent_factory,
            enable_memory=self.memory_config.get('enabled', True), # NEW: Pass memory config
            memory_config={
                'k_neighbors': self.memory_config.get('k_neighbors', 5),
                'max_tokens': self.memory_config.get('max_tokens', 500)
            }
        )
