import pytest

from writer_agents.scripts.convert_to_truth_table import fuzzy_match


@pytest.mark.parametrize(
    ("path_a", "path_b"),
    [
        ("case_law/Exhibit 6-L – Email from Malcolm.docx", "exhibit 6 l email from malcolm"),
        ("OGC Email - April 7.pdf", "ogc email april 7"),
        ("Harvard/OGC/Travel Risk Memo.pdf", "travel risk memo"),
        ("WeChat Resume Article.HTML", "wechat resume article"),
        ("GSS Travel Warning (final).docx", "gss travel warning final"),
        ("communications/Harvard acknowledgment draft.docx", "harvard acknowledgment draft"),
        ("evidence/EmailThread_to_OGC.msg", "email thread to ogc"),
        ("analysis/OGC_MONKEY_ALERT.txt", "ogc monkey alert"),
    ],
)
def test_fuzzy_match_identifies_equivalent_paths(path_a: str, path_b: str) -> None:
    assert fuzzy_match(path_a, path_b)


@pytest.mark.parametrize(
    ("path_a", "path_b"),
    [
        ("email_thread.docx", "wechat_article.pdf"),
        ("ogc_acknowledgment.docx", "completely different memo"),
        ("xi_slides.pdf", "communications summary"),
        ("risk memo", "Harvard dinner invitation"),
    ],
)
def test_fuzzy_match_rejects_unrelated_paths(path_a: str, path_b: str) -> None:
    assert not fuzzy_match(path_a, path_b)


def test_fuzzy_match_honors_threshold_without_substring_overlap() -> None:
    assert not fuzzy_match("harvard ogc memo", "global support briefing", threshold=90)
    assert fuzzy_match("harvard ogc memo", "global support briefing", threshold=10)


def test_fuzzy_match_requires_non_empty_inputs() -> None:
    assert not fuzzy_match("", "anything")
    assert not fuzzy_match("value", "")
