import json
from pathlib import Path

report = json.loads(Path('data/case_law/logs/pdf_phrase_samples.json').read_text(encoding='utf-8'))
agg = {}
for entry in report:
    for k, v in entry.get('counts', {}).items():
        agg[k] = agg.get(k, 0) + v

Path('data/case_law/logs/pdf_phrase_aggregate.json').write_text(json.dumps(agg, indent=2), encoding='utf-8')
print(json.dumps(agg, indent=2))
