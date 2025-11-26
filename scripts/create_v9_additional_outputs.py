#!/usr/bin/env python3
"""Create additional v9 outputs: top100, legal facts only, BN export, superclean."""

from pathlib import Path
import json
import pandas as pd

V9_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v9_final.csv")
OUTPUT_DIR = Path("case_law_data")


def load_v9() -> pd.DataFrame:
    """Load v9 CSV."""
    return pd.read_csv(V9_INPUT, low_memory=False)


def create_top100_distilled():
    """Create top 100 facts by causal salience."""
    df = load_v9()
    
    # Sort by causal_salience_score
    if 'causal_salience_score' in df.columns:
        df['_score'] = pd.to_numeric(df['causal_salience_score'], errors='coerce').fillna(0.0)
        df = df.sort_values('_score', ascending=False)
        df = df.drop(columns=['_score'])
    
    top100 = df.head(100)
    
    # Create .txt version
    output_txt = OUTPUT_DIR / "facts_v9_top100_distilled.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("# Top 100 Facts by Causal Salience Score\n")
        f.write("# Format: FactID | Proposition\n\n")
        for idx, (_, row) in enumerate(top100.iterrows(), 1):
            factid = str(row['factid']).strip()
            prop = str(row['proposition']).strip().replace('|', '｜')
            f.write(f"{idx}. {factid} | {prop}\n")
    
    # Create .csv version
    output_csv = OUTPUT_DIR / "facts_v9_top100_distilled.csv"
    top100.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✅ Created {output_txt} (100 facts)")
    print(f"✅ Created {output_csv} (100 facts)")


def create_legal_facts_only():
    """Filter out personal/narrative email text, keep only legal facts."""
    df = load_v9()
    
    # Patterns to exclude (personal/narrative email text)
    exclude_patterns = [
        r'^i\s+(write|am|have|will|would|should|think|believe|hope|wish)',
        r'^thank\s+you',
        r'^best\s+regards',
        r'^sincerely',
        r'^yours\s+truly',
        r'^please\s+(let|know|see|note)',
        r'^as\s+you\s+(know|see|can|may)',
        r'^i\s+look\s+forward',
        r'^i\s+appreciate',
        r'^feel\s+free',
        r'^if\s+you\s+have\s+any\s+questions',
        r'^let\s+me\s+know',
        r'^i\s+would\s+be\s+happy',
    ]
    
    # Filter out personal/narrative text
    mask = pd.Series([True] * len(df))
    for pattern in exclude_patterns:
        mask &= ~df['proposition'].str.match(pattern, case=False, na=False)
    
    legal_facts = df[mask].copy()
    
    # Create .txt version
    output_txt = OUTPUT_DIR / "facts_v9_legal_facts_only.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        for _, row in legal_facts.iterrows():
            factid = str(row['factid']).strip()
            prop = str(row['proposition']).strip().replace('|', '｜')
            f.write(f"{factid} | {prop}\n")
    
    # Create .csv version
    output_csv = OUTPUT_DIR / "facts_v9_legal_facts_only.csv"
    legal_facts.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✅ Created {output_txt} ({len(legal_facts)} facts)")
    print(f"✅ Created {output_csv} ({len(legal_facts)} facts)")


def create_bn_export():
    """Create BN export JSON for graph loading."""
    df = load_v9()
    
    # Build graph structure
    nodes = []
    edges = []
    
    for _, row in df.iterrows():
        factid = str(row['factid']).strip()
        prop = str(row['proposition']).strip()
        
        # Create node
        node = {
            'id': factid,
            'label': prop[:100] + ('...' if len(prop) > 100 else ''),
            'proposition': prop,
            'actor_role': str(row.get('actorrole', '')),
            'event_type': str(row.get('eventtype', '')),
            'event_date': str(row.get('eventdate', '')),
            'safety_risk': str(row.get('safetyrisk', '')),
            'public_exposure': str(row.get('publicexposure', '')),
            'causal_salience_score': float(row.get('causal_salience_score', 0.0)) if pd.notna(row.get('causal_salience_score')) else 0.0,
        }
        nodes.append(node)
        
        # Create edges based on causal relationships
        # (This is a simplified version - you may want to enhance based on your BN structure)
        if 'harvard' in prop.lower() and 'prc' in prop.lower():
            # Link Harvard actions to PRC outcomes
            edges.append({
                'source': factid,
                'target': 'PRC_HARM',
                'type': 'causes',
                'weight': node['causal_salience_score']
            })
    
    graph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_facts': len(df),
            'version': 'v9',
            'export_date': pd.Timestamp.now().isoformat()
        }
    }
    
    output_json = OUTPUT_DIR / "facts_v9_bn_export.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {output_json} ({len(nodes)} nodes, {len(edges)} edges)")


def create_superclean():
    """Create superclean version (150-300 facts max) - highest salience only."""
    df = load_v9()
    
    # Sort by causal_salience_score
    if 'causal_salience_score' in df.columns:
        df['_score'] = pd.to_numeric(df['causal_salience_score'], errors='coerce').fillna(0.0)
        df = df.sort_values('_score', ascending=False)
        df = df.drop(columns=['_score'])
    
    # Take top 250 facts (middle of 150-300 range)
    superclean = df.head(250)
    
    # Create .txt version
    output_txt = OUTPUT_DIR / "facts_v9_chatgpt_superclean.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("# Top 250 Facts by Causal Salience Score (Superclean)\n")
        f.write("# Format: FactID | Proposition\n\n")
        for idx, (_, row) in enumerate(superclean.iterrows(), 1):
            factid = str(row['factid']).strip()
            prop = str(row['proposition']).strip().replace('|', '｜')
            f.write(f"{idx}. {factid} | {prop}\n")
    
    # Create .csv version
    output_csv = OUTPUT_DIR / "facts_v9_chatgpt_superclean.csv"
    superclean.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"✅ Created {output_txt} (250 facts)")
    print(f"✅ Created {output_csv} (250 facts)")


def main():
    """Create all additional outputs."""
    print("Creating v9 additional outputs...\n")
    
    print("1. Creating top 100 distilled...")
    create_top100_distilled()
    
    print("\n2. Creating legal facts only...")
    create_legal_facts_only()
    
    print("\n3. Creating BN export JSON...")
    create_bn_export()
    
    print("\n4. Creating superclean version...")
    create_superclean()
    
    print("\n✅ All additional outputs created!")


if __name__ == "__main__":
    main()

