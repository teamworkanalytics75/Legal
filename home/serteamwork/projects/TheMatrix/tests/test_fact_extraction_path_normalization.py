import re

import pytest

from writer_agents.scripts.convert_to_truth_table import normalize_path


@pytest.mark.parametrize(
    ("raw_path", "expected_tokens"),
    [
        ("case_law/Harvard Email Thread.DOCX", ("harvard", "email", "thread")),
        (r"C:\\evidence\\OGC Emails\\Apr7_email.pdf", ("apr7", "email")),
        ("./Exhibit 6-L – Email?.docx", ("exhibit", "6", "email")),
        ("notes/Harvard/Communication (final).TXT", ("communication", "final")),
        ("facts/WeChat Resume Article.html", ("wechat", "resume", "article")),
        ("facts//WeChat-Resume-Article.HTML", ("wechat", "resume", "article")),
        ("CaseFiles/xi-slides.PDF", ("xi", "slides")),
        ("evidence//OGC_acknowledgment??.pdf", ("ogc", "acknowledgment")),
        ("reports\\safety\\Travel Risk Memo v2.md", ("travel", "risk", "memo", "v2")),
        ("Harvard - The Art of War/3. 1782/Email Export 2025.pdf", ("email", "export", "2025")),
    ],
)
def test_normalize_path_contains_expected_tokens(raw_path: str, expected_tokens: tuple[str, ...]) -> None:
    normalized = normalize_path(raw_path)
    assert normalized == normalized.strip()
    assert re.fullmatch(r"[a-z0-9 ]*", normalized)
    for token in expected_tokens:
        assert token in normalized


@pytest.mark.parametrize("raw_path", ["", None, "   "])
def test_normalize_path_handles_empty_inputs(raw_path: str | None) -> None:
    assert normalize_path(raw_path) == ""


def test_normalize_path_preserves_numeric_tokens() -> None:
    assert "123" in normalize_path("files/OGC Follow Up 123.docx")
