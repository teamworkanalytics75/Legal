import sqlite3
from pathlib import Path
DB = Path('case_law_data/harvard_cases.db')
conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS download_cursor (source TEXT PRIMARY KEY, cursor TEXT)')
conn.commit()
conn.close()
