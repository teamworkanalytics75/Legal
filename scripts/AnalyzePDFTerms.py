from pathlib import Path
from pdfminer.high_level import extract_text
import re
import json

KEYWORDS = [
    "1782",
    "foreign",
    "tribunal",
    "letters",
    "judicial assistance",
    "discovery",
    "intel",
    "aid",
    "application",
    "commission",
]

PHRASE_PATTERNS = {
    "28 u.s.c. 1782": re.compile(r"28\s+u\.?s\.?c\.?[^0-9]{0,8}1782", re.IGNORECASE),
    "section 1782": re.compile(r"section\s+1782", re.IGNORECASE),
    "foreign tribunal": re.compile(r"foreign\s+tribunal", re.IGNORECASE),
    "foreign proceeding": re.compile(r"foreign\s+proceeding", re.IGNORECASE),
    "letters rogatory": re.compile(r"letters?\s+rogatory", re.IGNORECASE),
    "judicial assistance": re.compile(r"judicial\s+assistance", re.IGNORECASE),
    "intel": re.compile(r"intel", re.IGNORECASE),
    "commission": re.compile(r"commission(er|\s+to)", re.IGNORECASE),
    "for use in": re.compile(r"for\s+use\s+in\s+[^\n]{0,80}foreign", re.IGNORECASE),
}

pdf_dir = Path('data/case_law/pdfs/canonical_full')
pdfs = sorted(pdf_dir.glob('*.pdf'))[:13]
report = []

for pdf in pdfs:
    entry = {"file": pdf.name, "sample_lines": [], "counts": {}, "error": None}
    try:
        text = extract_text(pdf)
    except Exception as exc:
        entry["error"] = str(exc)
        report.append(entry)
        continue

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sample_lines = []
    seen = set()
    for line in lines:
        lower = line.lower()
        if any(k in lower for k in KEYWORDS):
            if line not in seen:
                sample_lines.append(line)
                seen.add(line)
        if len(sample_lines) >= 8:
            break

    counts = {}
    for label, pattern in PHRASE_PATTERNS.items():
        hits = pattern.findall(text)
        if hits:
            counts[label] = len(hits)

    entry["sample_lines"] = sample_lines
    entry["counts"] = counts
    report.append(entry)

output_path = Path('data/case_law/logs/pdf_phrase_samples.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
