import pytest

from writer_agents.config.fact_extraction_templates import get_fact_templates
from writer_agents.scripts.convert_to_truth_table import apply_templates

TEMPLATES = get_fact_templates()


@pytest.mark.parametrize(
    ("text", "expected_phrase", "expected_metadata"),
    [
        (
            "From: Malcolm Grayson <m@law.com> To: Harvard OGC <ogc@harvard.edu>",
            "Malcolm Grayson sent email to",
            {"EvidenceType": "Email"},
        ),
        (
            "Harvard OGC has not acknowledged receipt of my warning letter.",
            "did not respond or acknowledge",
            {"ActorRole": "Harvard"},
        ),
        (
            "I was prepared to delay filing if I received a timely response. I did not.",
            "did not respond to the communication",
            {"ActorRole": "Harvard"},
        ),
        (
            "Harvard Global Support Services said that safety risks remained severe.",
            "Harvard Global Support Services said",
            {},
        ),
        (
            "Malcolm Grayson stated that Harvard never replied.",
            "Malcolm Grayson said",
            {"EvidenceType": "Email"},
        ),
        (
            "Malcolm Grayson warned Harvard OGC that sanctions were impending.",
            "Malcolm Grayson warned",
            {"EvidenceType": "Email"},
        ),
        (
            "Vivien Chan & Co demanded that Harvard release the documents.",
            "Vivien Chan & Co demanded",
            {"EvidenceType": "Letter"},
        ),
        (
            "Subject: Harvard OGC non-response update",
            "Email subject",
            {"EvidenceType": "Email"},
        ),
        (
            "Vivien Chan set deadline of April 20, 2025 for Harvard to respond.",
            "set deadline",
            {"EvidenceType": "Email"},
        ),
        (
            "On April 7, 2025, Malcolm Grayson noted that Harvard OGC ignored him.",
            "Malcolm Grayson said",
            {"EvidenceType": "Email"},
        ),
        (
            "Your office never responded to the April 7 warning.",
            "did not respond or acknowledge",
            {"ActorRole": "Harvard"},
        ),
        (
            "Harvard Global Support Services warned Malcolm that travel risk was extreme.",
            "Harvard Global Support Services warned",
            {"EvidenceType": "Email"},
        ),
    ],
)
def test_apply_templates_extracts_communication_facts(text: str, expected_phrase: str, expected_metadata: dict) -> None:
    result = apply_templates(text, TEMPLATES)
    assert result is not None, f"No template matched: {text}"
    assert expected_phrase in result["proposition"]
    assert result["metadata"].get("EventType") == "Communication"
    for key, value in expected_metadata.items():
        assert result["metadata"].get(key) == value
