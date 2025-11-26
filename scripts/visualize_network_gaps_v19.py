#!/usr/bin/env python3
"""
Visualize the v19 concept network and highlight gaps.
Generates an HTML visualization showing:
- Network structure
- Gap locations
- High-impact questions
"""

import json
from pathlib import Path

V19_NETWORK_JSON = Path("case_law_data/v19_concept_network.json")
V19_GAPS_CSV = Path("case_law_data/v19_network_gaps.csv")
V19_QUESTIONS_CSV = Path("case_law_data/v19_gap_bridging_questions.csv")
OUTPUT_HTML = Path("case_law_data/v19_network_visualization.html")


def generate_html_visualization():
    """Generate HTML visualization of network and gaps."""
    print("="*80)
    print("GENERATING NETWORK VISUALIZATION")
    print("="*80)
    print()
    
    # Load network
    print("1. Loading network data...")
    with open(V19_NETWORK_JSON, 'r', encoding='utf-8') as f:
        network = json.load(f)
    print(f"   Nodes: {len(network['nodes'])}")
    print(f"   Edges: {len(network['edges'])}")
    
    # Load gaps
    print("\n2. Loading gaps...")
    gaps = []
    if V19_GAPS_CSV.exists():
        import csv
        with open(V19_GAPS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            gaps = list(reader)
    print(f"   Gaps: {len(gaps)}")
    
    # Load questions
    print("\n3. Loading questions...")
    questions = []
    if V19_QUESTIONS_CSV.exists():
        import csv
        with open(V19_QUESTIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            questions = list(reader)
    print(f"   Questions: {len(questions)}")
    
    # Identify important nodes (for highlighting)
    important_nodes = {'Harvard', 'Harvard OGC', 'Harvard Club', 'PRC', 'EsuWiki', 'Plaintiff'}
    
    # Generate HTML
    print("\n4. Generating HTML visualization...")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>V19 Concept Network - Gap Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .gap-list {{
            margin: 20px 0;
        }}
        .gap-item {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .gap-item.high {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        .question-item {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .question-item.high {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        .impact-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            margin-left: 10px;
        }}
        .priority-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
            margin-right: 10px;
        }}
        .priority-high {{
            background: #dc3545;
            color: white;
        }}
        .priority-medium {{
            background: #ffc107;
            color: #333;
        }}
        .network-summary {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .node-list {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }}
        .node-item {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .node-item.important {{
            background: #fff3cd;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>V19 Concept Network - Gap Analysis Visualization</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(network['nodes'])}</div>
                <div class="stat-label">Concept Nodes</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(network['edges'])}</div>
                <div class="stat-label">Edges</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(gaps)}</div>
                <div class="stat-label">Structural Gaps</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(questions)}</div>
                <div class="stat-label">Gap-Bridging Questions</div>
            </div>
        </div>
        
        <div class="network-summary">
            <h2>Network Summary</h2>
            <p>The concept network was built from <strong>534 facts</strong>, extracting entities, topics, and relationships.</p>
            <p><strong>Key Entities:</strong></p>
            <div class="node-list">
"""
    
    # Add important nodes
    for node in sorted(important_nodes):
        if node in network['nodes']:
            node_data = network['nodes'][node]
            html += f'                <div class="node-item important">{node} ({node_data["count"]} facts)</div>\n'
    
    html += """            </div>
        </div>
        
        <h2>Structural Gaps Found</h2>
        <div class="gap-list">
"""
    
    # Add gaps
    high_priority = [g for g in gaps if g.get('priority', '').lower() == 'high']
    medium_priority = [g for g in gaps if g.get('priority', '').lower() == 'medium']
    
    for gap in high_priority + medium_priority:
        gap_type = gap.get('type', '')
        priority = gap.get('priority', 'medium').lower()
        description = gap.get('description', '')
        html += f"""            <div class="gap-item {priority}">
                <span class="priority-badge priority-{priority}">{priority.upper()}</span>
                <strong>{gap_type.replace('_', ' ').title()}</strong>
                <p>{description}</p>
            </div>
"""
    
    html += """        </div>
        
        <h2>Top Gap-Bridging Questions</h2>
        <p>Questions are sorted by priority and impact (how many gaps they help resolve).</p>
        <div class="gap-list">
"""
    
    # Add top questions
    for i, q in enumerate(questions[:10], 1):
        priority = q.get('priority', 'medium').lower()
        question = q.get('question', '')
        impact = q.get('impact', '0')
        rationale = q.get('rationale', '')
        html += f"""            <div class="question-item {priority}">
                <span class="priority-badge priority-{priority}">{priority.upper()}</span>
                <span class="impact-badge">Impact: {impact} gaps</span>
                <h3>{i}. {question}</h3>
                <p><em>{rationale}</em></p>
            </div>
"""
    
    html += """        </div>
        
        <h2>Network Structure</h2>
        <p><strong>Note:</strong> For interactive network visualization, use the JSON file with a graph visualization tool like:</p>
        <ul>
            <li><a href="https://gephi.org/">Gephi</a> (desktop application)</li>
            <li><a href="https://cytoscape.org/">Cytoscape</a> (desktop application)</li>
            <li><a href="https://observablehq.com/@d3/force-directed-graph">D3.js Force-Directed Graph</a> (web-based)</li>
            <li><a href="https://infranodus.com/">InfraNodus</a> (web-based, similar to this analysis)</li>
        </ul>
        <p>The network JSON file is available at: <code>case_law_data/v19_concept_network.json</code></p>
        
        <hr>
        <p style="color: #666; font-size: 0.9em;">
            Generated by network_gap_analysis_v19.py<br>
            Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""
    
    # Fix datetime import
    from datetime import datetime
    html = html.replace('{datetime.now().strftime(\'%Y-%m-%d %H:%M:%S\')}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    OUTPUT_HTML.write_text(html, encoding='utf-8')
    print(f"   ✅ Generated {OUTPUT_HTML}")
    
    print()
    print("="*80)
    print("VISUALIZATION GENERATED")
    print("="*80)
    print()
    print(f"✅ HTML visualization: {OUTPUT_HTML}")
    print()
    print("Open the HTML file in a browser to view:")
    print(f"  - Network statistics")
    print(f"  - Structural gaps (highlighted by priority)")
    print(f"  - Top gap-bridging questions (sorted by impact)")
    print()
    print("For interactive network graph visualization, use the JSON file with:")
    print("  - Gephi (https://gephi.org/)")
    print("  - Cytoscape (https://cytoscape.org/)")
    print("  - D3.js (web-based)")
    print()


if __name__ == "__main__":
    generate_html_visualization()

