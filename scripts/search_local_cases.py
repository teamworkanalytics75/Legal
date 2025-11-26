#!/usr/bin/env python3
"""
Search our local case law database for keywords related to Xi Mingze, torture, political crackdowns, etc.
"""

import sqlite3
from pathlib import Path
import pandas as pd

def search_local_cases():
    """Search unified_corpus.db for relevant keywords."""

    db_path = Path(__file__).parent.parent / "case_law_data" / "unified_corpus.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)

    # Search queries
    searches = [
        # Xi Mingze
        ("Xi Mingze", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%Xi Mingze%' OR cleaned_text LIKE '%Xi Mingzhe%'"),

        # Torture
        ("Torture", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%torture%'"),

        ("Torture of Minors", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%torture of minors%' OR cleaned_text LIKE '%torture minor%' OR cleaned_text LIKE '%child torture%'"),

        # Political Crackdown
        ("Political Crackdown", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%political crackdown%' OR cleaned_text LIKE '%political crackdowns%'"),

        # Authoritarian
        ("Authoritarian Regime", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%authoritarian regime%' OR cleaned_text LIKE '%authoritarian regimes%'"),

        # ESUWiki
        ("ESUWiki", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%ESUWiki%' OR cleaned_text LIKE '%ESU Wiki%' OR cleaned_text LIKE '%Eswiki%'"),

        # Harvard + China
        ("Harvard AND China", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE corpus_type = 'harvard' AND cleaned_text LIKE '%China%' OR corpus_type = 'china' AND cleaned_text LIKE '%Harvard%'"),

        # Spoliation
        ("Spoliation", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%spoliation%' OR cleaned_text LIKE '%spoliation of evidence%'"),

        # Litigation Hold
        ("Litigation Hold", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%litigation hold%'"),

        # 18 USC 1001
        ("18 USC 1001", "SELECT cluster_id, case_name, court, date_filed, corpus_type FROM cases WHERE cleaned_text LIKE '%18 U.S.C. 1001%' OR cleaned_text LIKE '%Section 1001%'"),
    ]

    print("\n" + "="*80)
    print("LOCAL CASE LAW DATABASE SEARCH")
    print("="*80)

    all_results = {}

    for search_name, query in searches:
        print(f"\n\n{'='*80}")
        print(f"SEARCH: {search_name}")
        print('='*80)

        try:
            df = pd.read_sql_query(query, conn)

            print(f"\nFound: {len(df)} cases")

            if len(df) > 0:
                print("\nSample results:")
                print(df.head(10).to_string(index=False))

                # Corpus breakdown
                if 'corpus_type' in df.columns:
                    corpus_counts = df['corpus_type'].value_counts()
                    print(f"\nCorpus breakdown:")
                    for corpus, count in corpus_counts.items():
                        print(f"  {corpus}: {count}")

                all_results[search_name] = df.to_dict('records')
            else:
                print("No results found")
                all_results[search_name] = []

        except Exception as e:
            print(f"Error: {e}")
            all_results[search_name] = []

    conn.close()

    # Save results
    output_path = Path(__file__).parent.parent / "case_law_data" / "exports" / "local_keyword_search_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON
    import json
    json_results = {}
    for search_name, results in all_results.items():
        json_results[search_name] = {
            'count': len(results),
            'cases': results[:20]  # Limit to top 20
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"Results saved to: {output_path}")
    print("="*80)

    # Summary
    print("\n" + "="*80)
    print("SEARCH SUMMARY")
    print("="*80)

    for search_name, data in json_results.items():
        print(f"\n{search_name}: {data['count']} cases")

    print("\n" + "="*80)


if __name__ == "__main__":
    search_local_cases()
