import sqlite3
conn = sqlite3.connect(r'C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print("Tables in lawsuit.db:")
for t in tables:
    print(f"  - {t[0]}")
    cursor.execute(f'SELECT COUNT(*) FROM {t[0]}')
    count = cursor.fetchone()[0]
    print(f"    Rows: {count}")
conn.close()

