from pathlib import Path
import sys
path = Path('document_ingestion/download_case_law.py')
text = path.read_text(encoding='utf-8')
old_sig = """    def bulk_download(\n        self,\n        topic: str,\n        courts: List[str] = None,\n        keywords: List[str] = None,\n        date_after: str = None,\n        max_results: int = 10000,\n        resume: bool = True\n    ) -> List[Dict]:\n"""
new_sig = """    def_bulk_download_placeholder"""
