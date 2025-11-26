import sqlite3
import json

# Check database contents
conn = sqlite3.connect('harvard_crimson_intelligence.db')
cursor = conn.cursor()

# Get articles
cursor.execute('SELECT title, url, relevance_score, threat_level, sentiment, publish_date, content FROM articles ORDER BY id DESC LIMIT 20')
results = cursor.fetchall()

print('\n' + '='*80)
print('HARVARD CRIMSON ARTICLES IN DATABASE')
print('='*80 + '\n')

for i, r in enumerate(results, 1):
    title, url, score, threat, sentiment, date, content = r
    print(f'{i}. {title[:70]}')
    print(f'   URL: {url[:75]}')
    print(f'   Score: {score:.1f} | Threat: {threat} | Sentiment: {sentiment} | Date: {date}')

    # Extract first 200 chars of content as summary
    if content:
        summary = content[:200].replace('\n', ' ').strip()
        print(f'   Summary: {summary}...')
    print()

# Get total count
cursor.execute('SELECT COUNT(*) FROM articles')
total = cursor.fetchone()[0]

print('='*80)
print(f'TOTAL ARTICLES IN DATABASE: {total}')
print('='*80)

# Get statistics
cursor.execute('SELECT AVG(relevance_score) FROM articles')
avg_score = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM articles WHERE threat_level = "high"')
high_threat = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM articles WHERE relevance_score > 50')
high_relevance = cursor.fetchone()[0]

print(f'\nSTATISTICS:')
print(f'  - Average relevance score: {avg_score:.1f}')
print(f'  - High threat articles: {high_threat}')
print(f'  - High relevance articles (>50): {high_relevance}')

conn.close()

