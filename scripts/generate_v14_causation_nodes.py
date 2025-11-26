#!/usr/bin/env python3
"""Generate v14 BN/causation node list for reasoning systems."""

from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

FINAL_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_final.csv")
OUTPUT_CSV = Path("case_law_data/final_causation_nodes.csv")
OUTPUT_TXT = Path("case_law_data/final_causation_nodes.txt")
REPORT_PATH = Path("reports/analysis_outputs/final_causation_nodes_report.md")


def load_facts() -> list[dict]:
    """Load final facts."""
    facts = []
    with open(FINAL_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def extract_causation_nodes(facts: list[dict]) -> list[dict]:
    """Extract causation nodes from facts."""
    nodes = []
    
    # Group by causal pathways
    pathways = {
        'harvard_action': [],
        'prc_response': [],
        'harvard_knowledge': [],
        'harvard_prc_pathway': [],
        'safety_risk': [],
        'public_exposure': [],
        'ogc_non_response': [],
        'correspondence_timeline': [],
    }
    
    for fact in facts:
        factid = str(fact.get('factid', '')).strip()
        prop = str(fact.get('proposition', '')).strip().lower()
        actorrole = str(fact.get('actorrole', '')).strip()
        eventdate = str(fact.get('eventdate', '')).strip()
        salience = float(fact.get('causal_salience_score', 0) or 0)
        
        # Classify by causal pathway
        node_type = None
        
        if factid.startswith('CORR_'):
            node_type = 'correspondence_timeline'
        elif 'ogc' in prop or 'general counsel' in prop:
            if 'did not respond' in prop or 'no response' in prop or 'silence' in prop:
                node_type = 'ogc_non_response'
            else:
                node_type = 'harvard_action'
        elif 'harvard' in prop and ('prc' in prop or 'china' in prop):
            if 'knew' in prop or 'knowledge' in prop or 'aware' in prop:
                node_type = 'harvard_knowledge'
            elif 'caused' in prop or 'led to' in prop or 'resulted in' in prop:
                node_type = 'harvard_prc_pathway'
            else:
                node_type = 'harvard_action'
        elif 'prc' in prop or 'china' in prop:
            if 'arrest' in prop or 'detention' in prop or 'torture' in prop:
                node_type = 'prc_response'
            elif 'risk' in prop or 'danger' in prop:
                node_type = 'safety_risk'
        elif 'published' in prop or 'circulated' in prop or 'wechat' in prop:
            node_type = 'public_exposure'
        
        if node_type:
            nodes.append({
                'factid': factid,
                'node_type': node_type,
                'proposition': str(fact.get('proposition', '')).strip(),
                'actorrole': actorrole,
                'eventdate': eventdate,
                'salience_score': salience,
                'subject': str(fact.get('subject', '')).strip(),
                'safetyrisk': str(fact.get('safetyrisk', '')).strip(),
            })
    
    return nodes


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING FINAL CAUSATION NODES")
    print("(Final version after 25 rounds of refinement)")
    print("="*80)
    
    print("\n1. Loading final facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    print("\n2. Extracting causation nodes...")
    nodes = extract_causation_nodes(facts)
    print(f"   Extracted {len(nodes)} causation nodes")
    
    # Count by type
    by_type = defaultdict(int)
    for node in nodes:
        by_type[node['node_type']] += 1
    
    print("\n3. Node type distribution:")
    for node_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"   {node_type}: {count}")
    
    print("\n4. Exporting causation nodes CSV...")
    if nodes:
        fieldnames = ['factid', 'node_type', 'proposition', 'actorrole', 'eventdate', 
                     'salience_score', 'subject', 'safetyrisk']
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(nodes)
        print(f"   ✅ Exported {OUTPUT_CSV}")
    
    print("\n5. Exporting causation nodes TXT...")
    lines = []
    lines.append("="*80)
    lines.append("V23 CAUSATION NODES - BAYESIAN NETWORK / REASONING SYSTEM")
    lines.append("="*80)
    lines.append("")
    lines.append("Causal pathway nodes extracted from v23 master database")
    lines.append(f"Total nodes: {len(nodes)}")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    # Group by node type
    nodes_by_type = defaultdict(list)
    for node in nodes:
        nodes_by_type[node['node_type']].append(node)
    
    for node_type in sorted(nodes_by_type.keys()):
        type_nodes = nodes_by_type[node_type]
        lines.append(f"## {node_type.upper().replace('_', ' ')} ({len(type_nodes)} nodes)")
        lines.append("")
        
        # Sort by salience
        type_nodes.sort(key=lambda x: x['salience_score'], reverse=True)
        
        for idx, node in enumerate(type_nodes[:20], 1):  # Top 20 per type
            lines.append(f"### Node {idx}: {node['factid']}")
            lines.append(f"Salience: {node['salience_score']:.3f}")
            lines.append(f"Proposition: {node['proposition'][:200]}...")
            lines.append(f"ActorRole: {node['actorrole']}")
            lines.append(f"EventDate: {node['eventdate']}")
            lines.append("")
        
        if len(type_nodes) > 20:
            lines.append(f"... and {len(type_nodes) - 20} more nodes of this type")
        lines.append("")
        lines.append("-"*80)
        lines.append("")
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n6. Writing report...")
    report = f"""# Final Causation Nodes Report

## Summary

- **Input**: `top_1000_facts_for_chatgpt_final.csv`
- **Output**: `final_causation_nodes.csv` and `final_causation_nodes.txt`
- **Total nodes extracted**: {len(nodes)}

## Node Type Distribution

"""
    
    for node_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        report += f"- **{node_type.replace('_', ' ').title()}**: {count} nodes\n"
    
    report += f"""
## Node Types

### Harvard Action Nodes
Facts describing Harvard's actions, publications, or statements.

### PRC Response Nodes
Facts describing PRC government responses, arrests, detentions, or censorship.

### Harvard Knowledge Nodes
Facts establishing Harvard's knowledge or awareness of risks.

### Harvard → PRC Pathway Nodes
Facts explicitly connecting Harvard actions to PRC outcomes.

### Safety Risk Nodes
Facts describing safety risks, dangers, or threats.

### Public Exposure Nodes
Facts about publication, circulation, or public visibility.

### OGC Non-Response Nodes
Facts about Harvard OGC's failure to respond.

### Correspondence Timeline Nodes
All CORR_xxx facts documenting the communication timeline.

## Files Generated

- **CSV**: `final_causation_nodes.csv`
- **TXT**: `final_causation_nodes.txt`

## Usage

These nodes can be used to build:
- Bayesian Networks for causal reasoning
- Causal DAGs (Directed Acyclic Graphs)
- Reasoning systems for legal argumentation
- Timeline visualization
- Causal chain analysis
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Final causation nodes generated")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"  📄 {OUTPUT_CSV}")
    print(f"  📄 {OUTPUT_TXT}")
    print(f"  📄 {REPORT_PATH}")


if __name__ == "__main__":
    main()


