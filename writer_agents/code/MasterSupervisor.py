"""Master supervisor coordinating all workflow phases.

The MasterSupervisor is the top-level orchestrator for the atomic agent architecture.
It decomposes tasks, builds DAGs, and coordinates phase supervisors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow environment override
DEFAULT_USE_LOCAL = os.getenv("WITCHWEB_USE_LOCAL_EMBEDDINGS", "true").lower() == "true"

try:
    from .insights import CaseInsights
    from .job_persistence import JobManager
    from .session_manager import SessionManager
    from .supervisors import (
        CitationSupervisor,
        DraftingSupervisor,
        QASupervisor,
        ResearchSupervisor,
    )
    from .task_dag import TaskDAG
    from .task_decomposer import TaskDecomposer, TaskProfile
except ImportError:  # Fallback for legacy direct-path imports
    from insights import CaseInsights
    from job_persistence import JobManager
    from session_manager import SessionManager
    from supervisors import (
        CitationSupervisor,
        DraftingSupervisor,
        QASupervisor,
        ResearchSupervisor,
    )
    from task_dag import TaskDAG
    from task_decomposer import TaskDecomposer, TaskProfile

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """ML configuration for the system."""

    enabled: bool = True
    use_outcome_prediction: bool = True
    use_document_classification: bool = True
    use_pattern_recognition: bool = True
    model_version: str = "latest"
    confidence_threshold: float = 0.7

@dataclass
class BudgetConfig:
    """Budget configuration for workflow execution."""

    # Per-phase token budgets
    research_tokens: int = 50000
    drafting_tokens: int = 100000
    citation_tokens: int = 10000 # Mostly deterministic, low budget
    qa_tokens: int = 30000

    # Per-phase time budgets (seconds)
    research_seconds: float = 300.0 # 5 minutes
    drafting_seconds: float = 600.0 # 10 minutes
    citation_seconds: float = 120.0 # 2 minutes
    qa_seconds: float = 300.0 # 5 minutes

    # Per-task budgets
    tokens_per_task: Optional[int] = 5000
    seconds_per_task: Optional[float] = 60.0

    def get_phase_budgets(self, phase: str) -> Dict[str, Any]:
        """Get budget configuration for a phase.

        Args:
            phase: Phase name

        Returns:
            Budget dictionary
        """
        return {
            'max_tokens': getattr(self, f'{phase}_tokens', 10000),
            'max_seconds': getattr(self, f'{phase}_seconds', 300.0),
            'tokens_per_task': self.tokens_per_task,
            'seconds_per_task': self.seconds_per_task,
        }


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2" # Local model
    use_local_embeddings: bool = DEFAULT_USE_LOCAL
    k_neighbors: int = 5 # Top-k memories to retrieve
    max_memory_tokens: int = 500 # Limit context length
    refresh_after_runs: int = 0 # 0=manual, 1=every run, N=every N runs
    cache_embeddings: bool = True

@dataclass
class SessionConfig:
    """Session memory configuration."""
    enabled: bool = True
    default_expiry_days: int = 7
    max_context_interactions: int = 10
    max_context_tokens: int = 5000
    auto_expire: bool = True

@dataclass
class SupervisorConfig:
    """Configuration for MasterSupervisor."""

    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    max_workers_per_phase: int = 6
    db_path: str = "jobs.db"
    enable_research: bool = True
    enable_drafting: bool = True
    enable_citation: bool = True
    enable_qa: bool = True
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    ml_config: MLConfig = field(default_factory=MLConfig)
    session_config: SessionConfig = field(default_factory=SessionConfig)  # NEW

    # LangChain configuration
    enable_langchain: bool = True  # Changed from False
    langchain_db_path: Optional[Path] = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")
    langchain_fallback: bool = True

    # Premium mode configuration
    premium_mode: bool = False  # Enable premium models for high-value cases
    premium_model: str = "gpt-4o"
    standard_model: str = "gpt-4o-mini"


class MasterSupervisor:
    """Top-level supervisor coordinating all phases.

    Responsibilities:
    - Task decomposition and planning
    - Phase supervisor coordination
    - Budget enforcement
    - Progress monitoring
    - Result aggregation
    """

    def __init__(
        self,
        session: Any,
        config: Optional[SupervisorConfig] = None,
        session_id: Optional[str] = None,  # NEW: Session ID for conversation continuity
    ) -> None:
        """Initialize master supervisor.

        Args:
            session: SQLite session for persistence
            config: Supervisor configuration
            session_id: Optional session ID for conversation continuity
        """
        self.session = session
        self.config = config or SupervisorConfig()
        self.current_session_id = session_id  # NEW: Store session ID

        # Initialize job manager
        self.job_manager = JobManager(self.config.db_path)

        # Initialize session manager
        self.session_manager = SessionManager(
            self.config.db_path,
            self.config.session_config.default_expiry_days
        )

        # Initialize task decomposer
        self.task_decomposer = TaskDecomposer()

        # Initialize phase supervisors
        self.research = ResearchSupervisor(
            session,
            self.job_manager,
            self.config.budgets.get_phase_budgets('research'),
            max_workers=self.config.max_workers_per_phase,
            memory_config={ # NEW: Pass memory config
                'enabled': self.config.memory_config.enabled,
                'k_neighbors': self.config.memory_config.k_neighbors,
                'max_tokens': self.config.memory_config.max_memory_tokens,
                'use_local_embeddings': self.config.memory_config.use_local_embeddings, # NEW
                'embedding_model': self.config.memory_config.embedding_model # NEW
            },
            session_id=self.current_session_id,  # NEW: Pass session ID
            use_langchain=self.config.enable_langchain,  # NEW: Pass LangChain toggle
            langchain_db_path=self.config.langchain_db_path,  # NEW: Pass database path
            premium_mode=self.config.premium_mode,  # NEW: Pass premium mode
            premium_model=self.config.premium_model,  # NEW: Pass premium model
        )

        self.drafting = DraftingSupervisor(
            session,
            self.job_manager,
            self.config.budgets.get_phase_budgets('drafting'),
            max_workers=self.config.max_workers_per_phase,
            memory_config={ # NEW: Pass memory config
                'enabled': self.config.memory_config.enabled,
                'k_neighbors': self.config.memory_config.k_neighbors,
                'max_tokens': self.config.memory_config.max_memory_tokens,
                'use_local_embeddings': self.config.memory_config.use_local_embeddings, # NEW
                'embedding_model': self.config.memory_config.embedding_model # NEW
            },
            session_id=self.current_session_id,  # NEW: Pass session ID
            premium_mode=self.config.premium_mode,  # NEW: Pass premium mode
            premium_model=self.config.premium_model,  # NEW: Pass premium model
        )

        self.citation = CitationSupervisor(
            session,
            self.job_manager,
            self.config.budgets.get_phase_budgets('citation'),
            max_workers=self.config.max_workers_per_phase,
            memory_config={ # NEW: Pass memory config
                'enabled': self.config.memory_config.enabled,
                'k_neighbors': self.config.memory_config.k_neighbors,
                'max_tokens': self.config.memory_config.max_memory_tokens,
                'use_local_embeddings': self.config.memory_config.use_local_embeddings, # NEW
                'embedding_model': self.config.memory_config.embedding_model # NEW
            },
            session_id=self.current_session_id,  # NEW: Pass session ID
            premium_mode=self.config.premium_mode,  # NEW: Pass premium mode
            premium_model=self.config.premium_model,  # NEW: Pass premium model
        )

        self.qa = QASupervisor(
            session,
            self.job_manager,
            self.config.budgets.get_phase_budgets('qa'),
            max_workers=self.config.max_workers_per_phase,
            memory_config={ # NEW: Pass memory config
                'enabled': self.config.memory_config.enabled,
                'k_neighbors': self.config.memory_config.k_neighbors,
                'max_tokens': self.config.memory_config.max_memory_tokens,
                'use_local_embeddings': self.config.memory_config.use_local_embeddings, # NEW
                'embedding_model': self.config.memory_config.embedding_model # NEW
            },
            session_id=self.current_session_id,  # NEW: Pass session ID
            premium_mode=self.config.premium_mode,  # NEW: Pass premium mode
            premium_model=self.config.premium_model,  # NEW: Pass premium model
        )

        # Track execution
        self.task_profile: Optional[TaskProfile] = None
        self.task_dag: Optional[TaskDAG] = None

    async def run(
        self,
        insights: CaseInsights,
        summary: str,
    ) -> Dict[str, Any]:
        """Execute full writer pipeline with atomic agents.

        Args:
            insights: Bayesian network case insights
            summary: Case summary text

        Returns:
            Complete analysis results
        """
        logger.info("MasterSupervisor starting atomic agent pipeline")

        # 1. Decompose task and build DAG
        logger.info("Decomposing task...")
        self.task_profile = await self.task_decomposer.compute(
            insights,
            summary,
            self.session
        )

        base_context = getattr(self, "base_context", {}) if hasattr(self, "base_context") else {}
        if base_context:
            argument_evidence = base_context.get("argument_evidence", {}) if isinstance(base_context, dict) else {}
            evidence_hits = sum(1 for docs in argument_evidence.values() if docs)

            # Ensure drafting and research agents spawn when evidence is available
            if self.task_profile.estimated_sections == 0:
                self.task_profile.estimated_sections = max(3, min(6, evidence_hits + 2))

            section_writer_count = self.task_profile.spawn_policies.get('SectionWriterAgent', 0)
            if section_writer_count == 0:
                self.task_profile.spawn_policies['SectionWriterAgent'] = max(1, min(4, evidence_hits or 1))

            if self.task_profile.evidence_count > 0:
                self.task_profile.spawn_policies.setdefault('FactExtractorAgent', 1)
                self.task_profile.spawn_policies.setdefault('PrecedentFinderAgent', 1)

        logger.info(f"Task profile: {self.task_profile}")
        logger.info(f"Spawn policies: {self.task_profile.spawn_policies}")

        # Estimate cost
        cost_estimate = self.task_decomposer.estimate_cost(self.task_profile, self.config.premium_mode)
        logger.info(f"Estimated cost: ${cost_estimate['total_estimated_cost']:.4f}")
        if self.config.premium_mode:
            logger.info(f"Premium cost: ${cost_estimate['premium_cost']:.4f}, Standard cost: ${cost_estimate['standard_cost']:.4f}")

        # Build task DAG
        logger.info("Building task DAG...")
        self.task_dag = TaskDAG.from_profile(self.task_profile)
        logger.info(f"Task DAG: {self.task_dag}")

        # Inject base context for all tasks if provided
        if base_context:
            self.task_dag.inject_context(base_context)

        # 2. Execute phases in order
        results = {}

        # Phase 1: Research (parallel)
        if self.config.enable_research:
            logger.info("=== RESEARCH PHASE ===")
            research_dag = self.task_dag.subgraph("research")
            if len(research_dag.tasks) > 0:
                try:
                    research_results = await self.research.execute(research_dag)
                    results['research'] = research_results
                    logger.info(f"Research complete: {len(research_results)} results")
                except NotImplementedError as e:
                    logger.warning(f"Research phase not yet implemented: {e}")
                    results['research'] = {}
            else:
                logger.info("No research tasks needed")
                results['research'] = {}

        # Phase 2: Drafting (parallel)
        if self.config.enable_drafting:
            logger.info("=== DRAFTING PHASE ===")
            drafting_dag = self.task_dag.subgraph("drafting")
            if len(drafting_dag.tasks) > 0:
                # Inject research results as context
                drafting_dag.inject_context(results.get('research', {}))

                try:
                    draft_results = await self.drafting.execute(drafting_dag)
                    results['drafting'] = draft_results
                    logger.info(f"Drafting complete: {len(draft_results)} results")
                except NotImplementedError as e:
                    logger.warning(f"Drafting phase not yet implemented: {e}")
                    results['drafting'] = {'text': summary} # Use summary as draft
            else:
                logger.info("No drafting tasks needed")
                results['drafting'] = {'text': summary}

        # Phase 3: Citation (parallel, deterministic)
        if self.config.enable_citation:
            logger.info("=== CITATION PHASE ===")
            citation_dag = self.task_dag.subgraph("citation")
            if len(citation_dag.tasks) > 0:
                # Inject draft text as context
                citation_dag.inject_context({
                    'text': results.get('drafting', {}).get('text', summary)
                })

                citation_results = await self.citation.execute(citation_dag)
                results['citation'] = citation_results
                logger.info(f"Citation complete: {len(citation_results)} results")
            else:
                logger.info("No citation tasks needed")
                results['citation'] = {}

        # Phase 4: QA (sequential, deterministic)
        if self.config.enable_qa:
            logger.info("=== QA PHASE ===")
            qa_dag = self.task_dag.subgraph("qa")
            if len(qa_dag.tasks) > 0:
                # Inject citation results as context
                qa_dag.inject_context({
                    **results.get('citation', {}),
                    'text': results.get('citation', {}).get('output_text',
                                                             results.get('drafting', {}).get('text', summary))
                })

                try:
                    qa_results = await self.qa.execute(qa_dag)
                    results['qa'] = qa_results
                    logger.info(f"QA complete: {len(qa_results)} results")
                except NotImplementedError as e:
                    logger.warning(f"QA phase not yet implemented: {e}")
                    results['qa'] = {}
            else:
                logger.info("No QA tasks needed")
                results['qa'] = {}

        # 3. Aggregate results
        final_result = self._aggregate_results(results)

        logger.info("MasterSupervisor pipeline complete")
        logger.info(f"Total tokens used: research={self.research.tokens_used}, "
                   f"drafting={self.drafting.tokens_used}, "
                   f"citation={self.citation.tokens_used}, "
                   f"qa={self.qa.tokens_used}")

        return final_result

    def _aggregate_results(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all phases into final output.

        Args:
            phase_results: Dictionary of results by phase

        Returns:
            Aggregated final result
        """
        def _extract_text(phase_data: Any, candidate_keys: List[str]) -> str:
            if not isinstance(phase_data, dict):
                return ""
            for key in candidate_keys:
                value = phase_data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in phase_data.values():
                if isinstance(value, dict):
                    for key in candidate_keys:
                        nested = value.get(key) if isinstance(value, dict) else None
                        if isinstance(nested, str) and nested.strip():
                            return nested.strip()
            return ""

        drafting_data = phase_results.get('drafting', {})
        section_texts: List[str] = []
        if isinstance(drafting_data, dict):
            for item in drafting_data.values():
                if isinstance(item, dict):
                    section_text = item.get('section_text')
                    if isinstance(section_text, str) and section_text.strip():
                        section_texts.append(section_text.strip())

        final_text = _extract_text(phase_results.get('qa', {}), ['text', 'corrected_text'])
        if not final_text:
            final_text = _extract_text(phase_results.get('citation', {}), ['output_text'])
        if not final_text and section_texts:
            final_text = "\n\n".join(section_texts)
        if not final_text:
            final_text = _extract_text(drafting_data, ['text', 'section_text'])

        # Build complete result
        return {
            'final_text': final_text,
            'compiled_sections': section_texts,
            'phase_results': phase_results,
            'task_profile': {
                'evidence_count': self.task_profile.evidence_count if self.task_profile else 0,
                'complexity_score': self.task_profile.complexity_score if self.task_profile else 0,
                'report_length': self.task_profile.report_length_estimate if self.task_profile else 'unknown',
            },
            'execution_stats': {
                'research_tokens': self.research.tokens_used,
                'drafting_tokens': self.drafting.tokens_used,
                'citation_tokens': self.citation.tokens_used,
                'qa_tokens': self.qa.tokens_used,
                'total_tokens': (
                    self.research.tokens_used +
                    self.drafting.tokens_used +
                    self.citation.tokens_used +
                    self.qa.tokens_used
                ),
            },
            'metadata': {
                'agent_architecture': 'atomic',
                'supervisor': 'MasterSupervisor',
            },
        }

    async def execute_with_session(
        self,
        user_prompt: str,
        insights: CaseInsights,
        summary: str,
    ) -> Dict[str, Any]:
        """Execute workflow with session context and save interaction.

        Args:
            user_prompt: User's input prompt
            insights: Bayesian network case insights
            summary: Case summary text

        Returns:
            Complete analysis results with session context
        """
        # Load session context if session is active
        session_context = ""
        if self.current_session_id and self.config.session_config.enabled:
            session_context = self.session_manager.get_session_context(
                self.current_session_id,
                max_interactions=self.config.session_config.max_context_interactions,
                max_tokens=self.config.session_config.max_context_tokens
            )

        # Enhance insights with session context
        enhanced_insights = insights
        if session_context:
            # Add session context to insights for agents to use
            enhanced_insights = CaseInsights(
                **insights.__dict__,
                session_context=session_context
            )

        # Execute the workflow
        result = await self.run(enhanced_insights, summary)

        # Save interaction to session if session is active
        if self.current_session_id and self.config.session_config.enabled:
            # Calculate total tokens used
            total_tokens = result.get('execution_stats', {}).get('total_tokens', 0)

            # Create agent response summary
            agent_response = f"Completed analysis with {result.get('execution_stats', {}).get('total_tokens', 0)} tokens"

            # Add interaction to session
            self.session_manager.add_interaction(
                self.current_session_id,
                user_prompt,
                agent_response,
                result,
                total_tokens
            )

        return result

    def create_session(self, case_name: str) -> str:
        """Create a new session for conversation continuity.

        Args:
            case_name: Name/description of the case

        Returns:
            Session ID
        """
        if not self.config.session_config.enabled:
            raise ValueError("Session management is disabled in configuration")

        session_id = self.session_manager.create_session(case_name)
        self.current_session_id = session_id
        return session_id

    def get_session_context(self) -> str:
        """Get current session context.

        Returns:
            Formatted session context string
        """
        if not self.current_session_id or not self.config.session_config.enabled:
            return ""

        return self.session_manager.get_session_context(
            self.current_session_id,
            max_interactions=self.config.session_config.max_context_interactions,
            max_tokens=self.config.session_config.max_context_tokens
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.

        Returns:
            Session summary with job statistics
        """
        if not self.current_session_id:
            return {}

        return self.session_manager.get_session(session_id=self.current_session_id)

    async def close(self) -> None:
        """Close all supervisors and clean up resources."""
        await self.research.close()
        await self.drafting.close()
        await self.citation.close()
        await self.qa.close()
        self.job_manager.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
