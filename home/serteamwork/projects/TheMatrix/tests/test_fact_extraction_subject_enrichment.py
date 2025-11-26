from writer_agents.scripts.convert_to_truth_table import infer_subject_from_context


def test_infer_subject_from_source_document_email_header() -> None:
    text = "Email header fragment"
    source_doc = "Exhibit 6-M – Email from Malcolm Grayson to Harvard OGC (18 Apr 2025).docx"
    source_excerpt = ""
    speaker = "Harvard"
    assert infer_subject_from_context(text, source_doc, source_excerpt, speaker) == "Malcolm Grayson"


def test_infer_subject_from_source_excerpt_from_line() -> None:
    text = ""
    source_doc = ""
    source_excerpt = "From: Vivien K. Chan <vkchan@law.com>\nTo: Harvard OGC"
    speaker = "Unknown"
    assert infer_subject_from_context(text, source_doc, source_excerpt, speaker) == "Vivien K. Chan"


def test_infer_subject_prefers_bilingual_parenthetical() -> None:
    text = ""
    source_doc = ""
    source_excerpt = "From: 王晓明 (Wang Xiaoming)"
    speaker = "Unknown"
    assert infer_subject_from_context(text, source_doc, source_excerpt, speaker) == "Wang Xiaoming"


def test_infer_subject_falls_back_to_specific_speaker() -> None:
    text = ""
    source_doc = ""
    source_excerpt = ""
    speaker = "Vivien Chan"
    assert infer_subject_from_context(text, source_doc, source_excerpt, speaker) == "Vivien Chan"
