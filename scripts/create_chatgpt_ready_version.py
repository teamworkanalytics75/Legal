#!/usr/bin/env python3
"""Create a ChatGPT-ready version of v6 facts (FactID + clean Proposition only)."""

from pathlib import Path
import pandas as pd

V8_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v8_final.csv")
OUTPUT_TXT = Path("case_law_data/facts_v8_chatgpt_ready.txt")
OUTPUT_CSV = Path("case_law_data/facts_v8_chatgpt_ready.csv")


def create_chatgpt_ready_version():
    """Create simplified version for ChatGPT ingestion."""
    df = pd.read_csv(V8_INPUT, low_memory=False)
    
    # Select only FactID and PropositionClean_v2
    if "PropositionClean_v2" in df.columns:
        clean_df = df[["FactID", "PropositionClean_v2"]].copy()
        clean_df.columns = ["FactID", "Proposition"]
    else:
        clean_df = df[["FactID", "Proposition"]].copy()
    
    # Remove any rows with empty propositions
    clean_df = clean_df[clean_df["Proposition"].notna() & (clean_df["Proposition"].str.strip() != "")]
    
    # Create .txt version (one fact per line: FactID | Proposition)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for _, row in clean_df.iterrows():
            factid = str(row["FactID"]).strip()
            prop = str(row["Proposition"]).strip()
            # Escape any pipes in the proposition
            prop = prop.replace("|", "｜")
            f.write(f"{factid} | {prop}\n")
    
    # Create .csv version
    clean_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"✅ Created {OUTPUT_TXT} ({len(clean_df)} facts)")
    print(f"✅ Created {OUTPUT_CSV} ({len(clean_df)} facts)")


if __name__ == "__main__":
    create_chatgpt_ready_version()

