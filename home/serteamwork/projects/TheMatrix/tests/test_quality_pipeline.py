import json

import pytest

from writer_agents.code.WorkflowOrchestrator import QualityGatePipeline


@pytest.mark.asyncio
async def test_quality_pipeline_injects_fact_payload(monkeypatch):
    pipeline = QualityGatePipeline(sk_kernel=None, executor=None)
    gate = {
        "name": "citation_validity",
        "plugin": "ValidationPlugin",
        "threshold": 1.0,
        "required": True,
        "description": "All citations must be properly formatted",
    }
    context = {
        "structured_facts": "### HK Allegations\n- Weiqi Zhang defamed Plaintiff.",
        "fact_key_summary": ["hk_allegation_defamation"],
        "filtered_evidence": [{"node_id": "fact_block_hk_allegation_defamation", "state": "Defamation evidence"}],
        "fact_filter_stats": {"dropped_count": 0},
    }

    captured = {}

    async def fake_invoke(plugin_name, function_name, variables):
        captured["plugin"] = plugin_name
        captured["function"] = function_name
        captured["variables"] = variables

        class Dummy:
            value = json.dumps({"score": 1.0})

        return Dummy()

    monkeypatch.setattr(pipeline, "_invoke_sk_function_safe", fake_invoke)
    result = await pipeline._run_gate(gate, "Document text", context)

    assert result["score"] == 1.0
    assert captured["plugin"] == "ValidationPlugin"
    payload = captured["variables"]
    fact_keys = json.loads(payload["fact_key_summary"])
    assert "hk_allegation_defamation" in fact_keys
    filtered_evidence = json.loads(payload["filtered_evidence"])
    assert filtered_evidence
    assert payload["key_facts_summary"]

