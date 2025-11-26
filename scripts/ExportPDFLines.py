import json
from pathlib import Path

report = json.loads(Path('data/case_law/logs/pdf_phrase_samples.json').read_text(encoding='utf-8'))
canonical_phrases = set()
for entry in report:
    for line in entry.get('sample_lines', []):
        canonical_phrases.add(line.strip())

lines_path = Path('data/case_law/logs/pdf_phrase_lines.txt')
lines_path.write_text('\n'.join(sorted(canonical_phrases)), encoding='utf-8')
print(f'Extracted {len(canonical_phrases)} sample lines to {lines_path}')
