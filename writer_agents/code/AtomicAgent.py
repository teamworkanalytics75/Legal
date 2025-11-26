"""Base class for atomic single-duty micro-agents.

Each atomic agent has exactly one responsibility and can execute either
deterministically (using code/regex/DB) or via LLM calls.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from .agents import BaseAutoGenAgent, AgentFactory
except ImportError:
    from agents import BaseAutoGenAgent, AgentFactory


@dataclass
class AgentMetrics:
    """Execution metrics for an atomic agent."""

    tokens_in: int = 0
    tokens_out: int = 0
    duration_seconds: float = 0.0
    cache_hit: bool = False


class AtomicAgent(BaseAutoGenAgent, ABC):
    """Base class for single-duty micro-agents.

    Each atomic agent:
    - Has exactly one duty (verifiable by name and docstring)
    - Can be deterministic (no LLM) or LLM-based
    - Tracks its own execution metrics
    - Supports caching of results
    """

    # Class-level configuration (override in subclasses)
    duty: str = "Abstract duty - override in subclass"
    is_deterministic: bool = False
    cost_tier: str = "mini" # mini, standard, premium
    max_cost_per_run: float = 0.01 # Maximum expected cost in dollars

    # Meta-categorization for premium tiers
    meta_category: str = "standard"  # "completeness" | "precision" | "standard"
    model_tier: str = "mini"         # "mini" | "premium"
    output_strategy: str = "normal"   # "maximize" | "optimize" | "normal"

    # Premium mode configuration
    premium_max_tokens: int = 8000
    premium_temperature: float = 0.3
    standard_max_tokens: int = 2000
    standard_temperature: float = 0.0

    def __init__(
        self,
        factory: AgentFactory,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        enable_memory: bool = True, # NEW: Toggle memory system
        memory_config: Optional[Dict[str, Any]] = None, # NEW: Memory settings
    ) -> None:
        """Initialize atomic agent with optional memory.

        Args:
            factory: Agent factory for creating LLM client
            name: Agent name (defaults to class name)
            system_message: System prompt (defaults to duty)
            enable_memory: Whether to load memories for enhanced context
            memory_config: Memory configuration settings
        """
        agent_name = name or self.__class__.__name__

        # NEW: Load memories if enabled
        self.memories: Optional[Dict[str, Any]] = None
        if enable_memory:
            self.memories = self._load_memories(memory_config or {})

        # Build system message (now with memory)
        agent_message = system_message or self._build_system_message_with_memory()

        super().__init__(factory, agent_name, agent_message)

        self.metrics = AgentMetrics()
        self.cache: Dict[str, Any] = {}

    def _load_memories(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load project context + historical patterns for this agent.

        Args:
            config: Memory configuration settings

        Returns:
            Dictionary with project context and past patterns
        """
        try:
            from agent_context_templates import get_agent_context
            from memory_system import MemoryStore

            agent_type = self.__class__.__name__

            # Get static project context
            project_context = get_agent_context(agent_type)

            # Get dynamic historical patterns
            memory_store = MemoryStore(
                use_local_embeddings=config.get('use_local_embeddings', True), # NEW
                embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2') # NEW
            )
            k_neighbors = config.get('k_neighbors', 5)

            # Query: What has this agent type done successfully before?
            query = f"successful execution of {agent_type}"
            past_patterns = memory_store.retrieve(
                agent_type=agent_type,
                query=query,
                k=k_neighbors
            )

            return {
                'project_context': project_context,
                'past_patterns': past_patterns,
                'config': config
            }
        except ImportError as e:
            # Memory system not available - return empty context
            return {
                'project_context': {},
                'past_patterns': [],
                'config': config,
                'error': f"Memory system unavailable: {e}"
            }

    def _build_system_message_with_memory(self) -> str:
        """Build enhanced system message with memory context.

        Returns:
            Enhanced system message with project context and past patterns
        """
        # Base message
        base = (
            f"You are {self.__class__.__name__}. "
            f"Your single duty: {self.duty} "
            f"Focus only on this task. Do not expand scope. "
            f"Provide structured, minimal output."
        )

        # Add memory context if available
        if self.memories:
            context = self.memories['project_context']
            patterns = self.memories['past_patterns']

            # Project awareness
            if context.get('project'):
                base += f"\n\n**PROJECT:** {context['project']}"
            if context.get('role'):
                base += f"\n**YOUR ROLE:** {context['role']}"

            # Team awareness
            if 'team_position' in context:
                base += f"\n**TEAM:** {context['team_position']}"
            if 'upstream' in context:
                base += f"\n**RECEIVES FROM:** {context['upstream']}"
            if 'downstream' in context:
                base += f"\n**SENDS TO:** {context['downstream']}"

            # Character identity (if assigned)
            if 'codename' in context:
                base += f"\n**CODENAME:** {context['codename']}"

            # Historical experience
            if patterns:
                base += "\n\n**PAST EXPERIENCE** (use to improve your work):"
                for i, mem in enumerate(patterns[:3], 1): # Top 3
                    base += f"\n{i}. {mem.summary}"

            # System context
            if 'system_context' in context:
                base += f"\n\n{context['system_context']}"

        # Add session context if available
        if hasattr(self, 'session_context') and self.session_context:
            base += "\n\n**PREVIOUS INTERACTIONS:**"
            base += self.session_context

        return base

    def set_session_context(self, session_context: str) -> None:
        """Set session context for conversation continuity.

        Args:
            session_context: Formatted session context string
        """
        self.session_context = session_context

    def _build_system_message(self) -> str:
        """Build system message from duty description (legacy method)."""
        return (
            f"You are {self.__class__.__name__}. "
            f"Your single duty: {self.duty} "
            f"Focus only on this task. Do not expand scope. "
            f"Provide structured, minimal output."
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's single duty.

        Args:
            input_data: Task input data

        Returns:
            Task output data with agent_name field
        """
        # Check cache first
        cache_key = self._compute_cache_key(input_data)
        if cache_key in self.cache:
            self.metrics.cache_hit = True
            return {
                **self.cache[cache_key],
                'agent_name': self.__class__.__name__,
                'cached': True,
            }

        # Execute based on mode
        if self.is_deterministic:
            result = await self._deterministic_execution(input_data)
        else:
            result = await self._llm_execution(input_data)

        # Add metadata
        result['agent_name'] = self.__class__.__name__
        result['duty'] = self.duty
        result['cached'] = False

        # Cache result
        self.cache[cache_key] = result

        # Write memory if enabled
        if hasattr(self, 'memories') and self.memories is not None:
            memory = await self._write_memory(input_data, result)
            if memory:
                # Store memory in global store
                try:
                    from memory_system import MemoryStore
                    memory_store = MemoryStore()
                    memory_store.add(memory)
                except Exception:
                    pass  # Don't fail execution if memory storage fails

        return result

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using deterministic code (no LLM).

        Override in subclasses for deterministic agents.

        Args:
            input_data: Task input

        Returns:
            Task output
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is deterministic but _deterministic_execution not implemented"
        )

    async def _llm_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using LLM.

        Args:
            input_data: Task input

        Returns:
            Task output
        """
        # Build prompt
        prompt = self._build_prompt(input_data)

        # Track tokens (approximate - actual tracking would need API response)
        self.metrics.tokens_in += len(prompt.split()) * 1.3 # Rough estimate

        # Call LLM
        response = await self.run(prompt)

        # Track output tokens
        self.metrics.tokens_out += len(str(response).split()) * 1.3

        # Parse response
        parsed = self._parse_response(response)

        return parsed

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build LLM prompt from input data.

        Override in subclasses for custom prompts.

        Args:
            input_data: Task input

        Returns:
            Formatted prompt string
        """
        # Default prompt structure
        return (
            f"Task: {self.duty}\n\n"
            f"Input:\n{json.dumps(input_data, indent=2)}\n\n"
            f"Provide output as JSON with relevant fields for this task."
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured output.

        Override in subclasses for custom parsing.

        Args:
            response: Raw LLM response

        Returns:
            Parsed output dictionary
        """
        # Try to parse as JSON
        try:
            # Clean response (remove markdown code blocks)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove code block markers
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Return as-is in a wrapper
            return {
                'output': response,
                'parsed': False,
            }

    def _compute_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Compute cache key for input data.

        Args:
            input_data: Task input

        Returns:
            SHA-256 hash of input
        """
        # Serialize input deterministically
        serialized = json.dumps(input_data, sort_keys=True)

        # Hash it
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_metrics(self) -> AgentMetrics:
        """Get execution metrics.

        Returns:
            Current metrics
        """
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset execution metrics."""
        self.metrics = AgentMetrics()

    def clear_cache(self) -> None:
        """Clear result cache."""
        self.cache.clear()

    def _should_write_memory(self, result: Dict[str, Any]) -> bool:
        """Decide if execution is worth remembering.

        Args:
            result: Execution result

        Returns:
            True if execution should be remembered
        """
        # Always remember errors
        if result.get('error'):
            return True

        # Remember very fast executions
        if self.metrics.duration_seconds < 1.0:
            return True

        # Remember high-quality outputs
        if result.get('quality_score', 0) > 0.8:
            return True

        # Remember cache hits (efficiency)
        if self.metrics.cache_hit:
            return True

        # Remember successful executions (20% of the time)
        import random
        return random.random() < 0.2

    async def _write_memory(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> Optional['AgentMemory']:
        """Write memory for this execution.

        Args:
            input_data: Task input
            result: Execution result

        Returns:
            AgentMemory if written, None otherwise
        """
        # Check if we should write memory
        if not self._should_write_memory(result):
            return None

        try:
            from memory_system import AgentMemory
            from datetime import datetime
            import uuid

            # Deterministic agents: auto-generate summary (FREE)
            if self.is_deterministic:
                summary = self._auto_generate_memory_summary(result)
            else:
                # LLM agents: use structured template (CHEAP)
                summary = await self._llm_generate_memory_summary(input_data, result)

            if not summary:
                return None

            memory = AgentMemory(
                agent_type=self.__class__.__name__,
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context={
                    'input': input_data,
                    'result': result,
                    'metrics': {
                        'tokens_in': self.metrics.tokens_in,
                        'tokens_out': self.metrics.tokens_out,
                        'duration': self.metrics.duration_seconds,
                        'cache_hit': self.metrics.cache_hit
                    }
                },
                embedding=None,  # Will be computed by MemoryStore
                source="agent_execution",
                timestamp=datetime.now()
            )

            return memory

        except Exception as e:
            # Don't fail execution if memory writing fails
            return None

    def _auto_generate_memory_summary(self, result: Dict[str, Any]) -> str:
        """Auto-generate memory summary for deterministic agents (FREE).

        Args:
            result: Execution result

        Returns:
            Memory summary string
        """
        agent_name = self.__class__.__name__
        duration = self.metrics.duration_seconds

        # Extract key metrics from result
        if 'citations' in result:
            count = len(result['citations']) if isinstance(result['citations'], list) else 1
            return f"{agent_name} found {count} citations in {duration:.1f}s using regex patterns"

        elif 'outline' in result:
            sections = len(result['outline']) if isinstance(result['outline'], list) else 1
            return f"{agent_name} built {sections}-section outline in {duration:.1f}s with clear structure"

        elif 'facts' in result:
            count = len(result['facts']) if isinstance(result['facts'], list) else 1
            return f"{agent_name} extracted {count} facts in {duration:.1f}s from case documents"

        elif 'errors_fixed' in result:
            count = result['errors_fixed']
            return f"{agent_name} fixed {count} grammar errors in {duration:.1f}s"

        elif 'quality_score' in result:
            score = result['quality_score']
            return f"{agent_name} achieved {score:.2f} quality score in {duration:.1f}s"

        else:
            # Generic success pattern
            return f"{agent_name} completed successfully in {duration:.1f}s with {self.metrics.tokens_in + self.metrics.tokens_out} tokens"

    async def _llm_generate_memory_summary(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate memory summary using LLM (CHEAP).

        Args:
            input_data: Task input
            result: Execution result

        Returns:
            Memory summary string
        """
        # Truncate input/output to save tokens
        input_str = str(input_data)[:200] + "..." if len(str(input_data)) > 200 else str(input_data)
        result_str = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

        prompt = f"""Task: {self.duty}
Input: {input_str}
Result: {result_str}
Duration: {self.metrics.duration_seconds:.1f}s
Tokens: {self.metrics.tokens_in + self.metrics.tokens_out}

Write a 1-2 sentence memory capturing what was important about this execution.
Focus on: what worked well, what patterns you noticed, what made this successful.

Memory:"""

        try:
            response = await self.run(prompt)
            # Clean response
            summary = response.strip()
            if len(summary) > 200:  # Limit length
                summary = summary[:200] + "..."
            return summary
        except Exception:
            # Fallback to auto-generation if LLM fails
            return self._auto_generate_memory_summary(result)

    def estimate_cost(self) -> float:
        """Estimate cost for typical execution.

        Returns:
            Estimated cost in dollars
        """
        if self.is_deterministic:
            return 0.0

        # Cost based on tier
        tier_costs = {
            'mini': 0.0015, # gpt-4o-mini typical call
            'standard': 0.005, # gpt-4o typical call
            'premium': 0.02, # gpt-4o for complex tasks
        }

        return tier_costs.get(self.cost_tier, 0.001)


class DeterministicAgent(AtomicAgent):
    """Base class for deterministic agents (no LLM).

    These agents use only code, regex, database queries, etc.
    They have zero LLM cost.
    """

    is_deterministic = True
    cost_tier = "free"
    max_cost_per_run = 0.0

    async def _llm_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic agents should not call LLM."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is deterministic - should not call _llm_execution"
        )


class LLMAgent(AtomicAgent):
    """Base class for LLM-based agents.

    These agents use LLM calls and have associated costs.
    Use the most cost-effective model tier possible.
    """

    is_deterministic = False

    # Subclasses should override cost_tier and max_cost_per_run


# Convenience type alias
Agent = AtomicAgent
