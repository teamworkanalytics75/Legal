#!/usr/bin/env python3
"""Create a 100-fact distilled version for ChatGPT ingestion."""

from pathlib import Path
import pandas as pd

V8_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v8_final.csv")
OUTPUT_TXT = Path("case_law_data/facts_v8_top100_distilled.txt")
OUTPUT_CSV = Path("case_law_data/facts_v8_top100_distilled.csv")


def create_distilled_version():
    """Create top 100 facts by causal salience score."""
    df = pd.read_csv(V8_INPUT, low_memory=False)
    
    # Sort by causal_salience_score
    if "causal_salience_score" in df.columns:
        df["_score"] = pd.to_numeric(df["causal_salience_score"], errors="coerce").fillna(0.0)
        df = df.sort_values("_score", ascending=False)
        df = df.drop(columns=["_score"])
    
    # Take top 100
    top100 = df.head(100).copy()
    
    # Select only FactID and PropositionClean_v2
    if "PropositionClean_v2" in top100.columns:
        clean_df = top100[["FactID", "PropositionClean_v2"]].copy()
        clean_df.columns = ["FactID", "Proposition"]
    else:
        clean_df = top100[["FactID", "Proposition"]].copy()
    
    # Create .txt version (one fact per line: FactID | Proposition)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("# Top 100 Facts by Causal Salience Score\n")
        f.write("# Format: FactID | Proposition\n\n")
        for idx, (_, row) in enumerate(clean_df.iterrows(), 1):
            factid = str(row["FactID"]).strip()
            prop = str(row["Proposition"]).strip()
            # Escape any pipes in the proposition
            prop = prop.replace("|", "｜")
            f.write(f"{idx}. {factid} | {prop}\n")
    
    # Create .csv version
    clean_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"✅ Created {OUTPUT_TXT} (100 facts)")
    print(f"✅ Created {OUTPUT_CSV} (100 facts)")


if __name__ == "__main__":
    create_distilled_version()

