import pytest

from writer_agents.scripts.convert_to_truth_table import (
    SUPPORTED_FILE_EXTENSIONS as CONVERT_EXTENSIONS,
)
from writer_agents.scripts.convert_to_truth_table import infer_evidence_type
from writer_agents.scripts.extract_facts_ml_enhanced import (
    SUPPORTED_FILE_EXTENSIONS as EXTRACT_EXTENSIONS,
)


def test_supported_extensions_align_across_modules() -> None:
    expected = {".txt", ".md", ".docx", ".html", ".pdf"}
    assert set(CONVERT_EXTENSIONS) == expected
    assert set(EXTRACT_EXTENSIONS) == expected
    assert CONVERT_EXTENSIONS == EXTRACT_EXTENSIONS


@pytest.mark.parametrize(
    ("source_doc", "text", "expected"),
    [
        ("emails/Harvard_OGC_email.txt", "", "Email"),
        ("Exhibit 7-L – Email from Malcolm.pdf", "Reference to Exhibit 7-L", "Email"),
        ("HK_statement_of_claim.pdf", "The Statement of Claim alleges...", "HKFiling"),
        ("motions/motion_to_seal.docx", "Motion to Seal regarding plaintiff", "USFiling"),
        ("translations/wechat_resume_article.html", "WeChat article translation", "WeChatArticle"),
        ("Certified Translation of Hong Kong order.pdf", "certified translation", "CertifiedTranslation"),
        ("case_docs/summary.txt", "This email confirms receipt.", "Email"),
        ("evidence/misc_notes.md", "Exhibit 8 describes the issue.", "Exhibit"),
        ("notes/general memo.md", "General summary", "Unknown"),
    ],
)
def test_infer_evidence_type_detects_known_file_types(source_doc: str, text: str, expected: str) -> None:
    assert infer_evidence_type(source_doc, text) == expected
