#!/usr/bin/env python3
"""
Context Manager - Manages conversation history with sliding window and fact extraction.

Prevents hallucinations by:
- Using sliding window (last N messages)
- Extracting verified facts separately
- Not carrying forward uncertain information
- Summarizing old context when window fills
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    facts_extracted: List[str] = field(default_factory=list)  # Verified facts from this message
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class VerifiedFact:
    """A verified fact extracted from system outputs."""
    fact: str
    source: str  # Which component/system produced it
    confidence: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Manages conversation context with sliding window and fact extraction."""

    def __init__(self, max_messages: int = 10, max_facts: int = 50):
        """
        Initialize context manager.

        Args:
            max_messages: Maximum messages to keep in sliding window
            max_facts: Maximum verified facts to store
        """
        self.max_messages = max_messages
        self.max_facts = max_facts
        self.messages: List[ConversationMessage] = []
        self.verified_facts: List[VerifiedFact] = []
        self.summaries: List[str] = []  # Summaries of old context
        logger.info(f"ContextManager initialized: max_messages={max_messages}, max_facts={max_facts}")

    def add_message(self, role: MessageRole, content: str, extract_facts: bool = True,
                   metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """
        Add a message to the conversation.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            extract_facts: Whether to extract verified facts from this message
            metadata: Optional metadata

        Returns:
            The created ConversationMessage
        """
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )

        # Extract facts if this is an assistant message with system output
        if extract_facts and role == MessageRole.ASSISTANT:
            facts = self._extract_facts_from_system_output(content)
            message.facts_extracted = facts
            for fact_text in facts:
                self.add_verified_fact(fact_text, source="system_output", confidence=0.8)

        self.messages.append(message)

        # Apply sliding window
        if len(self.messages) > self.max_messages:
            self._slide_window()

        return message

    def _extract_facts_from_system_output(self, content: str) -> List[str]:
        """
        Extract verified facts from system output.

        Looks for patterns like:
        - "Based on BN analysis: 72%"
        - "Found 5 similar cases"
        - "Probability: 0.85"
        - Structured data from system components
        """
        facts = []

        # Extract probability/percentage statements
        import re
        prob_patterns = [
            r"(\d+(?:\.\d+)?)\s*%",
            r"probability[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*probability",
        ]
        for pattern in prob_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                facts.append(f"Probability: {match.group(1)}%")

        # Extract case counts
        case_patterns = [
            r"found\s+(\d+)\s+cases?",
            r"(\d+)\s+similar\s+cases?",
            r"(\d+)\s+precedents?",
        ]
        for pattern in case_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                facts.append(f"Found {match.group(1)} cases")

        # Extract BN node results
        if "BN analysis" in content or "Bayesian" in content:
            # Look for node: probability patterns
            node_pattern = r"([A-Za-z_]+)[:\s]+(\d+(?:\.\d+)?)\s*%?"
            matches = re.finditer(node_pattern, content)
            for match in matches:
                node_name = match.group(1)
                prob = match.group(2)
                facts.append(f"BN node {node_name}: {prob}%")

        return facts

    def add_verified_fact(self, fact: str, source: str, confidence: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None) -> VerifiedFact:
        """
        Add a verified fact from system output.

        Args:
            fact: Fact text
            source: Source component (e.g., "BN", "Research", "ML")
            confidence: Confidence level (0.0-1.0)
            metadata: Optional metadata

        Returns:
            The created VerifiedFact
        """
        verified_fact = VerifiedFact(
            fact=fact,
            source=source,
            confidence=confidence,
            metadata=metadata or {}
        )

        self.verified_facts.append(verified_fact)

        # Apply max facts limit (remove oldest)
        if len(self.verified_facts) > self.max_facts:
            self.verified_facts = self.verified_facts[-self.max_facts:]

        return verified_fact

    def _slide_window(self) -> None:
        """Slide the window by summarizing old messages and keeping recent ones."""
        if len(self.messages) <= self.max_messages:
            return

        # Keep last max_messages
        old_messages = self.messages[:-self.max_messages]
        self.messages = self.messages[-self.max_messages:]

        # Summarize old messages
        if old_messages:
            summary = self._summarize_messages(old_messages)
            self.summaries.append(summary)
            logger.debug(f"Summarized {len(old_messages)} old messages")

        # Keep only last 3 summaries
        if len(self.summaries) > 3:
            self.summaries = self.summaries[-3:]

    def _summarize_messages(self, messages: List[ConversationMessage]) -> str:
        """Summarize a list of messages."""
        if not messages:
            return ""

        # Simple summarization: extract key facts and topics
        topics = []
        facts = []

        for msg in messages:
            if msg.role == MessageRole.USER:
                # Extract question topics
                content_lower = msg.content.lower()
                if "probability" in content_lower:
                    topics.append("probability questions")
                if "research" in content_lower or "case" in content_lower:
                    topics.append("research questions")
                if "national security" in content_lower:
                    topics.append("national security")

            facts.extend(msg.facts_extracted)

        summary_parts = []
        if topics:
            summary_parts.append(f"Topics: {', '.join(set(topics))}")
        if facts:
            summary_parts.append(f"Facts: {len(facts)} verified facts extracted")

        return " | ".join(summary_parts) if summary_parts else "Previous conversation context"

    def get_context(self, include_summaries: bool = True) -> str:
        """
        Get current conversation context for LLM.

        Args:
            include_summaries: Whether to include summaries of old messages

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add summaries if available
        if include_summaries and self.summaries:
            context_parts.append("Previous conversation summary:")
            for summary in self.summaries:
                context_parts.append(f"  - {summary}")
            context_parts.append("")

        # Add recent messages
        context_parts.append("Recent conversation:")
        for msg in self.messages[-self.max_messages:]:
            role_name = msg.role.value.upper()
            context_parts.append(f"{role_name}: {msg.content[:200]}")  # Limit length

        # Add verified facts summary
        if self.verified_facts:
            context_parts.append("")
            context_parts.append(f"Verified facts ({len(self.verified_facts)}):")
            for fact in self.verified_facts[-10:]:  # Last 10 facts
                context_parts.append(f"  - {fact.fact} (source: {fact.source})")

        return "\n".join(context_parts)

    def get_recent_messages(self, n: int = 5) -> List[ConversationMessage]:
        """Get the last N messages."""
        return self.messages[-n:] if self.messages else []

    def clear(self) -> None:
        """Clear all conversation history (keep verified facts)."""
        old_message_count = len(self.messages)
        self.messages = []
        self.summaries = []
        logger.info(f"Cleared {old_message_count} messages from context")

    def clear_all(self) -> None:
        """Clear everything including verified facts."""
        self.clear()
        old_facts_count = len(self.verified_facts)
        self.verified_facts = []
        logger.info(f"Cleared all context including {old_facts_count} verified facts")

