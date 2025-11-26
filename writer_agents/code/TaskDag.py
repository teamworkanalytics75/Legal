"""Task Directed Acyclic Graph for dependency management.

Represents task dependencies and enables parallel scheduling of independent tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Task:
    """Represents a single atomic task in the workflow."""

    id: str
    agent_type: str
    input_data: Dict[str, Any]
    phase: str # research, drafting, citation, qa
    dependencies: Set[str] = field(default_factory=set) # Task IDs this depends on
    status: str = "pending" # pending, ready, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    priority: int = 0

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Task):
            return self.id == other.id
        return False


class TaskDAG:
    """Directed Acyclic Graph of tasks with dependency management.

    Enables parallel execution of independent tasks while respecting dependencies.
    """

    def __init__(self) -> None:
        """Initialize empty task DAG."""
        self.tasks: Dict[str, Task] = {}
        self.adjacency: Dict[str, Set[str]] = {} # task_id -> dependent task IDs

    def add_task(self, task: Task) -> None:
        """Add a task to the DAG.

        Args:
            task: Task to add
        """
        self.tasks[task.id] = task

        # Initialize adjacency for this task
        if task.id not in self.adjacency:
            self.adjacency[task.id] = set()

        # Add edges for dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.adjacency:
                self.adjacency[dep_id] = set()
            self.adjacency[dep_id].add(task.id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task or None if not found
        """
        return self.tasks.get(task_id)

    def roots(self) -> List[Task]:
        """Get all root tasks (no dependencies).

        Returns:
            List of tasks with no dependencies and pending/ready status
        """
        return [
            task for task in self.tasks.values()
            if not task.dependencies and task.status in ("pending", "ready")
        ]

    def next_ready(self) -> List[Task]:
        """Get all tasks that are ready to execute.

        A task is ready if:
        - All its dependencies are completed
        - It's not already running or completed

        Returns:
            List of ready tasks sorted by priority
        """
        ready = []

        for task in self.tasks.values():
            if task.status not in ("pending", "ready"):
                continue

            # Check if all dependencies are completed
            all_deps_done = all(
                self.tasks[dep_id].status == "completed"
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if all_deps_done:
                task.status = "ready"
                ready.append(task)

        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority, reverse=True)

        return ready

    def mark_running(self, task_id: str) -> None:
        """Mark a task as running.

        Args:
            task_id: Task identifier
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = "running"

    def mark_done(self, tasks: List[Task | str]) -> None:
        """Mark tasks as completed.

        Args:
            tasks: List of Task objects or task IDs
        """
        for task in tasks:
            task_id = task if isinstance(task, str) else task.id

            if task_id in self.tasks:
                self.tasks[task_id].status = "completed"

    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: Task identifier
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"

    def set_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Set result for a completed task.

        Args:
            task_id: Task identifier
            result: Task output data
        """
        if task_id in self.tasks:
            self.tasks[task_id].result = result

    def collect_outputs(self) -> Dict[str, Any]:
        """Collect all completed task results.

        Returns:
            Dictionary mapping task IDs to results
        """
        return {
            task_id: task.result
            for task_id, task in self.tasks.items()
            if task.status == "completed" and task.result is not None
        }

    def subgraph(self, phase: str) -> TaskDAG:
        """Extract subgraph for a specific phase.

        Args:
            phase: Phase name (research, drafting, citation, qa)

        Returns:
            New TaskDAG containing only tasks from that phase
        """
        subdag = TaskDAG()

        # Add tasks from this phase
        for task in self.tasks.values():
            if task.phase == phase:
                subdag.add_task(task)

        return subdag

    def inject_context(self, context: Dict[str, Any]) -> None:
        """Inject context from previous phase into task inputs.

        Args:
            context: Dictionary of results from previous phase
        """
        for task in self.tasks.values():
            if task.status == "pending":
                # Merge context into input_data
                task.input_data = {
                    **context,
                    **task.input_data,
                }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the DAG.

        Returns:
            Dictionary with counts by status and phase
        """
        stats = {
            'total_tasks': len(self.tasks),
            'by_status': {},
            'by_phase': {},
        }

        for task in self.tasks.values():
            # Count by status
            stats['by_status'][task.status] = \
                stats['by_status'].get(task.status, 0) + 1

            # Count by phase
            stats['by_phase'][task.phase] = \
                stats['by_phase'].get(task.phase, 0) + 1

        return stats

    def is_complete(self) -> bool:
        """Check if all tasks are completed.

        Returns:
            True if no pending or ready tasks remain
        """
        return all(
            task.status in ("completed", "failed")
            for task in self.tasks.values()
        )

    def has_failures(self) -> bool:
        """Check if any tasks failed.

        Returns:
            True if any task has failed status
        """
        return any(
            task.status == "failed"
            for task in self.tasks.values()
        )

    @classmethod
    def from_profile(cls, profile: Any) -> TaskDAG:
        """Build task DAG from a TaskProfile.

        Args:
            profile: TaskProfile with spawn policies

        Returns:
            Constructed TaskDAG
        """
        dag = cls()

        # Map agent types to phases
        phase_map = {
            # Research phase
            'FactExtractorAgent': 'research',
            'PrecedentFinderAgent': 'research',
            'PrecedentRankerAgent': 'research',
            'PrecedentSummarizerAgent': 'research',
            'StatuteLocatorAgent': 'research',
            'ExhibitFetcherAgent': 'research',

            # Drafting phase
            'OutlineBuilderAgent': 'drafting',
            'SectionWriterAgent': 'drafting',
            'ParagraphWriterAgent': 'drafting',
            'TransitionAgent': 'drafting',

            # Citation phase
            'CitationFinderAgent': 'citation',
            'CitationNormalizerAgent': 'citation',
            'CitationVerifierAgent': 'citation',
            'CitationLocatorAgent': 'citation',
            'CitationInserterAgent': 'citation',

            # QA phase
            'GrammarFixerAgent': 'qa',
            'StyleCheckerAgent': 'qa',
            'LogicCheckerAgent': 'qa',
            'ConsistencyCheckerAgent': 'qa',
            'RedactionAgent': 'qa',
            'ComplianceAgent': 'qa',
            'ExpertQAAgent': 'qa',

            # Output phase
            'MarkdownExporterAgent': 'output',
            'DocxExporterAgent': 'output',
            'MetadataTaggerAgent': 'output',
        }

        # Citation pipeline dependencies (sequential)
        citation_deps = {
            'CitationNormalizerAgent': {'CitationFinderAgent'},
            'CitationVerifierAgent': {'CitationNormalizerAgent'},
            'CitationLocatorAgent': {'CitationVerifierAgent'},
            'CitationInserterAgent': {'CitationLocatorAgent'},
        }

        # Create tasks from spawn policies
        task_counter = {}

        for agent_type, count in profile.spawn_policies.items():
            phase = phase_map.get(agent_type, 'unknown')

            for i in range(count):
                # Generate unique task ID
                task_counter[agent_type] = task_counter.get(agent_type, 0) + 1
                task_id = f"{agent_type}_{task_counter[agent_type]}"

                # Get dependencies for this agent type
                deps = citation_deps.get(agent_type, set())

                # Resolve dependency IDs (use first instance of each dependency)
                dep_ids = {f"{dep}_1" for dep in deps}

                # Create task
                task = Task(
                    id=task_id,
                    agent_type=agent_type,
                    input_data={}, # Will be populated at runtime
                    phase=phase,
                    dependencies=dep_ids,
                    priority=0,
                )

                dag.add_task(task)

        return dag

    def __repr__(self) -> str:
        """String representation of DAG."""
        stats = self.get_stats()
        return (
            f"TaskDAG(tasks={stats['total_tasks']}, "
            f"by_status={stats['by_status']})"
        )
