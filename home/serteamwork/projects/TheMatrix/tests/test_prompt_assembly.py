"""Unit tests for prompt assembly helpers."""

import importlib
import sys

_MODULE_ALIASES = {
    "agents": "writer_agents.agents",
    "insights": "writer_agents.insights",
    "tasks": "writer_agents.tasks",
    "sk_config": "writer_agents.sk_config",
    "sk_plugins": "writer_agents.sk_plugins",
}

for alias, target in _MODULE_ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(target)

from writer_agents.code.WorkflowOrchestrator import (  # noqa: E402
    _format_structured_facts_block,
    _format_autogen_notes_block,
    _format_quality_constraints_block,
)


def test_format_structured_facts_block_from_dict():
    block = _format_structured_facts_block({"hk_security": "Harvard Club retaliation summary"})
    assert "KEY FACTS SUMMARY" in block
    assert "STRICT FACT CORPUS" in block
    assert "Harvard Club retaliation summary" in block


def test_format_structured_facts_block_default_message():
    block = _format_structured_facts_block("")
    assert "Structured facts unavailable" in block


def test_format_autogen_notes_block_default_message():
    block = _format_autogen_notes_block("")
    assert "AutoGen exploratory notes not available" in block


def test_format_quality_constraints_block_default_message():
    block = _format_quality_constraints_block("")
    assert "QUALITY CONSTRAINTS AND REQUIREMENTS" in block
    assert "Perfect Outline" in block
