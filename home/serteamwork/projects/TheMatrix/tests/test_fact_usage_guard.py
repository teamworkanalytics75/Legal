import json

from writer_agents.scripts.diagnose_facts_issue import compute_fact_usage
from writer_agents.code.fact_payload_utils import (
    normalize_fact_keys,
    parse_filtered_evidence,
    parse_fact_filter_stats,
    payload_metric_snapshot,
)


def test_compute_fact_usage_success_case():
    motion = (
        "Harvard and the Hong Kong defendants defamed the plaintiff. "
        "On April 7 2025 OGC threatened sanctions, and on April 18 2025 they ignored the follow-up. "
        "In April 2025, Harvard's Office of General Counsel sent emails documenting retaliation and privacy violations. "
        "Defendant Weiqi Zhang, who runs Blue Oak Education, sought to reach 2 million students. "
        "A photograph depicting Xi Mingze appeared in the leaked materials. "
        "This privacy breach triggered harassment and retaliation, culminating in June 2025 arrests that escalated the pressure."
    )

    usage = compute_fact_usage(motion)

    assert usage["coverage"] == 1.0
    assert usage["missing"] == []


def test_compute_fact_usage_flags_missing_items():
    motion = "Harvard filed a motion with generic references to Hong Kong without specifics."

    usage = compute_fact_usage(motion)

    assert usage["coverage"] < 0.5
    assert "weiqi zhang" in usage["missing"]
    assert "blue oak" in usage["missing"]


def test_compute_fact_usage_flags_missing_dates():
    motion = (
        "Harvard mentioned retaliation but never referenced the April communications or any concrete dates. "
        "The narrative entirely skips the OGC threat timeline."
    )

    usage = compute_fact_usage(motion)

    assert "april_7_2025" in usage["missing"]
    assert "april_18_2025" in usage["missing"]


def test_payload_metric_snapshot_parses_json_sources():
    raw_payload = {
        "structured_facts": "### Harvard Retaliation\n- April 7 threat\n- April 18 refusal",
        "fact_key_summary": json.dumps(["hk_allegation_defamation", "harvard_retaliation_events"]),
        "filtered_evidence": json.dumps([{"node_id": "fact_block_hk", "state": "detail"}]),
        "fact_filter_stats": json.dumps({"dropped_count": 2}),
        "key_facts_summary": "- hk allegation\n- harvard retaliation",
    }

    fact_keys = normalize_fact_keys(raw_payload["fact_key_summary"])
    filtered_items = parse_filtered_evidence(raw_payload["filtered_evidence"])
    fact_filter_stats = parse_fact_filter_stats(raw_payload["fact_filter_stats"])
    metrics = payload_metric_snapshot(
        structured_text=raw_payload["structured_facts"],
        fact_keys=fact_keys,
        filtered_evidence=filtered_items,
        key_facts_summary=raw_payload["key_facts_summary"],
        fact_filter_stats=fact_filter_stats,
    )

    assert metrics["structured_facts_length"] == len(raw_payload["structured_facts"])
    assert metrics["fact_key_count"] == 2
    assert metrics["filtered_evidence"] == 1
    assert metrics["fact_filter_dropped"] == 2
    assert metrics["key_facts_summary_length"] == len(raw_payload["key_facts_summary"])


def test_normalize_fact_keys_handles_comma_strings():
    fact_keys = normalize_fact_keys("hk_defamation, harvard_retaliation ,,")
    assert fact_keys == ["hk_defamation", "harvard_retaliation"]

