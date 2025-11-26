"""Unit tests for fact payload guards feeding SK plugins."""

import importlib
import sys
import json

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

from writer_agents.code.WorkflowStrategyExecutor import ensure_fact_payload_fields  # noqa: E402


def test_ensure_fact_payload_fields_serializes_context():
    context = {
        "fact_key_summary": ["hk_retaliation_events", "privacy_leak_events"],
        "filtered_evidence": [{"node_id": "fact_block_hk_retaliation_events", "state": "text"}],
        "fact_filter_stats": {"dropped_count": 2},
        "structured_facts": "hk_retaliation_events: summary text",
    }

    payload = ensure_fact_payload_fields({"document": "motion text"}, context)

    fact_keys = json.loads(payload["fact_key_summary"])
    assert fact_keys == ["hk_retaliation_events", "privacy_leak_events"]

    filtered = json.loads(payload["filtered_evidence"])
    assert filtered[0]["node_id"] == "fact_block_hk_retaliation_events"

    stats = json.loads(payload["fact_filter_stats"])
    assert stats["dropped_count"] == 2
    assert "key_facts_summary" in payload


def test_ensure_fact_payload_fields_keeps_existing_values():
    variables = {"fact_key_summary": '["existing_key"]'}
    context = {"fact_key_summary": ["hk_retaliation_events"]}

    payload = ensure_fact_payload_fields(variables, context)
    assert payload["fact_key_summary"] == '["existing_key"]'










