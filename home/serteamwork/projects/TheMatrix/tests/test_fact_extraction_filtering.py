import pytest

from writer_agents.scripts.convert_to_truth_table import should_filter


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "## Temporary note",
        "Merged from 12 similar facts",
        "Temporal extraction: auto",
        "NER: LABEL_PERSON",
        "relative_date placeholder",
        "On on April 1",
        "short",
        "== heading ==",
    ],
)
def test_should_filter_flags_noise_patterns(text: str) -> None:
    assert should_filter(text)


@pytest.mark.parametrize(
    "text",
    [
        "privacy",
        "Privacy breach",
        "torture",
        "PRC",
        "wechat exposure",
        "xi reference",
        "monkey article",
        "persecution risk",
    ],
)
def test_should_filter_preserves_legal_and_risk_terms(text: str) -> None:
    assert not should_filter(text)


def test_should_filter_respects_punctuation_for_short_text() -> None:
    assert not should_filter("Short statement.")


def test_should_filter_passes_valid_sentence() -> None:
    sentence = "Harvard OGC ignored the warning email on April 7, 2025."
    assert not should_filter(sentence)
