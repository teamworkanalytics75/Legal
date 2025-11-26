#!/usr/bin/env python3
"""
Network-based gap analysis tool (InfraNodus-style).
Identifies structural gaps in arguments using:
- Concept network analysis
- NLP topic extraction
- Network density analysis
- Disconnected node detection
- Gap-bridging question generation
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

V25_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v25_final.csv")
OUTPUT_GAPS_CSV = Path("case_law_data/v25_network_gaps.csv")
OUTPUT_GAPS_TXT = Path("case_law_data/v25_network_gaps.txt")
OUTPUT_QUESTIONS_CSV = Path("case_law_data/v25_gap_bridging_questions.csv")
OUTPUT_QUESTIONS_TXT = Path("case_law_data/v25_gap_bridging_questions.txt")
OUTPUT_REPORT = Path("reports/analysis_outputs/v25_network_gap_analysis_report.md")
OUTPUT_NETWORK_JSON = Path("case_law_data/v25_concept_network.json")

# Key entities and concepts to track
KEY_ENTITIES = {
    'Harvard', 'Harvard Club', 'Harvard OGC', 'HAA', 'Harvard GSS',
    'Plaintiff', 'Malcolm Grayson', 'MJ Tang', 'Yi Wang', 'McGrath',
    'PRC', 'China', 'PRC Authorities', 'StateActor',
    'EsuWiki', 'Xi Mingze', 'Niu Tengyu',
    'WeChat', 'Zhihu', 'Baidu', 'Sohu', 'E-Canada',
    'Statement 1', 'Statement 2',
}

# Key topics/themes
KEY_TOPICS = {
    'knowledge', 'foreseeability', 'control', 'responsibility',
    'non-response', 'silence', 'litigation hold',
    'publication', 'amplification', 'republication',
    'safety risk', 'travel risk', 'arbitrary detention',
    'torture', 'arrest', 'harassment', 'doxxing',
    'correspondence', 'communication', 'email',
    'timeline', 'causation', 'harm',
}

# Relationship patterns
RELATIONSHIP_PATTERNS = [
    (r'(\w+)\s+(?:told|said|stated|wrote|emailed|contacted)\s+(\w+)', 'communication'),
    (r'(\w+)\s+(?:published|posted|released|issued)\s+(\w+)', 'publication'),
    (r'(\w+)\s+(?:warned|advised|informed)\s+(\w+)', 'warning'),
    (r'(\w+)\s+(?:did not|failed to|refused to)\s+(\w+)', 'non-action'),
    (r'(\w+)\s+(?:caused|led to|resulted in|triggered)\s+(\w+)', 'causation'),
    (r'(\w+)\s+(?:knew|was aware|should have known)\s+(\w+)', 'knowledge'),
]


def extract_entities(text: str) -> Set[str]:
    """Extract entities from text using patterns and keywords."""
    entities = set()
    text_lower = text.lower()
    
    # Check for key entities
    for entity in KEY_ENTITIES:
        if entity.lower() in text_lower:
            entities.add(entity)
    
    # Extract capitalized phrases (potential entities)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for cap in capitalized:
        if len(cap) > 3 and cap not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How']:
            entities.add(cap)
    
    return entities


def extract_topics(text: str) -> Set[str]:
    """Extract topics/themes from text."""
    topics = set()
    text_lower = text.lower()
    
    # Check for key topics
    for topic in KEY_TOPICS:
        if topic in text_lower:
            topics.add(topic)
    
    # Extract topic phrases
    topic_patterns = [
        r'\b(?:harvard|ogc|club|haa)\s+(?:knowledge|awareness|foreseeability)',
        r'\b(?:prc|china|chinese)\s+(?:risk|danger|threat|retaliation)',
        r'\b(?:non-?response|silence|did not respond)',
        r'\b(?:litigation\s+hold|spoliation|preservation)',
        r'\b(?:control|responsibility|authority|oversight)',
        r'\b(?:publication|amplification|republication|circulation)',
    ]
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            topics.add(match.strip())
    
    return topics


def extract_relationships(text: str, factid: str) -> List[Tuple[str, str, str]]:
    """Extract relationships between entities."""
    relationships = []
    
    entities = extract_entities(text)
    if len(entities) < 2:
        return relationships
    
    # Check relationship patterns
    for pattern, rel_type in RELATIONSHIP_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity1 = match.group(1)
            entity2 = match.group(2) if len(match.groups()) > 1 else None
            if entity1 and entity2:
                relationships.append((entity1, rel_type, entity2))
    
    # Create co-occurrence relationships
    entity_list = list(entities)
    for i, e1 in enumerate(entity_list):
        for e2 in entity_list[i+1:]:
            if e1 != e2:
                relationships.append((e1, 'co-occurs', e2))
    
    return relationships


def build_concept_network(facts: List[Dict]) -> Dict:
    """Build a concept network from facts."""
    nodes = defaultdict(lambda: {
        'count': 0,
        'factids': set(),
        'topics': set(),
        'salience_sum': 0.0,
    })
    
    edges = defaultdict(lambda: {
        'count': 0,
        'factids': set(),
        'types': set(),
    })
    
    # Process each fact
    for fact in facts:
        factid = str(fact.get('factid', '')).strip()
        proposition = str(fact.get('proposition', '')).strip()
        salience = float(fact.get('causal_salience_score', 0) or 0)
        
        if not proposition:
            continue
        
        # Extract entities and topics
        entities = extract_entities(proposition)
        topics = extract_topics(proposition)
        relationships = extract_relationships(proposition, factid)
        
        # Add nodes
        for entity in entities:
            nodes[entity]['count'] += 1
            nodes[entity]['factids'].add(factid)
            nodes[entity]['topics'].update(topics)
            nodes[entity]['salience_sum'] += salience
        
        # Add edges (relationships)
        for rel in relationships:
            e1, rel_type, e2 = rel
            edge_key = (e1, e2)
            edges[edge_key]['count'] += 1
            edges[edge_key]['factids'].add(factid)
            edges[edge_key]['types'].add(rel_type)
    
    # Calculate node metrics
    node_data = {}
    for node, data in nodes.items():
        node_data[node] = {
            'count': data['count'],
            'factids': list(data['factids']),
            'topics': list(data['topics']),
            'avg_salience': data['salience_sum'] / data['count'] if data['count'] > 0 else 0,
            'degree': 0,  # Will calculate from edges
        }
    
    # Calculate node degrees
    for (e1, e2), edge_data in edges.items():
        if e1 in node_data:
            node_data[e1]['degree'] += edge_data['count']
        if e2 in node_data:
            node_data[e2]['degree'] += edge_data['count']
    
    # Convert edges to list format
    edge_list = []
    for (e1, e2), edge_data in edges.items():
        edge_list.append({
            'source': e1,
            'target': e2,
            'count': edge_data['count'],
            'factids': list(edge_data['factids']),
            'types': list(edge_data['types']),
        })
    
    return {
        'nodes': node_data,
        'edges': edge_list,
    }


def find_structural_gaps(network: Dict) -> List[Dict]:
    """Find structural gaps in the network."""
    gaps = []
    nodes = network['nodes']
    edges = network['edges']
    
    # Build edge lookup
    edge_lookup = defaultdict(set)
    for edge in edges:
        e1, e2 = edge['source'], edge['target']
        edge_lookup[e1].add(e2)
        edge_lookup[e2].add(e1)
    
    # Gap 1: Isolated nodes (low degree, important entities)
    important_entities = {'Harvard', 'Harvard OGC', 'Harvard Club', 'PRC', 'EsuWiki', 'Plaintiff'}
    for node, data in nodes.items():
        if node in important_entities:
            degree = data['degree']
            if degree < 3:  # Low connectivity
                gaps.append({
                    'type': 'isolated_important_node',
                    'node': node,
                    'degree': degree,
                    'count': data['count'],
                    'priority': 'high',
                    'description': f"Important entity '{node}' has low connectivity (degree={degree})",
                })
    
    # Gap 2: Missing connections between important entities
    for i, e1 in enumerate(important_entities):
        for e2 in list(important_entities)[i+1:]:
            if e1 in nodes and e2 in nodes:
                # Check if they're connected
                connected = (e2 in edge_lookup.get(e1, set())) or (e1 in edge_lookup.get(e2, set()))
                if not connected:
                    gaps.append({
                        'type': 'missing_connection',
                        'node1': e1,
                        'node2': e2,
                        'priority': 'high',
                        'description': f"Missing connection between '{e1}' and '{e2}'",
                    })
    
    # Gap 3: Low-density topic clusters
    topic_clusters = defaultdict(list)
    for node, data in nodes.items():
        for topic in data['topics']:
            topic_clusters[topic].append(node)
    
    for topic, cluster_nodes in topic_clusters.items():
        if len(cluster_nodes) >= 2:
            # Count connections within cluster
            internal_connections = 0
            for i, n1 in enumerate(cluster_nodes):
                for n2 in cluster_nodes[i+1:]:
                    if n2 in edge_lookup.get(n1, set()):
                        internal_connections += 1
            
            # Calculate density
            max_possible = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
            density = internal_connections / max_possible if max_possible > 0 else 0
            
            if density < 0.3:  # Low density
                gaps.append({
                    'type': 'low_density_cluster',
                    'topic': topic,
                    'nodes': cluster_nodes,
                    'density': density,
                    'priority': 'medium',
                    'description': f"Topic '{topic}' has low network density ({density:.2%})",
                })
    
    # Gap 4: High-salience nodes with low connectivity
    for node, data in nodes.items():
        if data['avg_salience'] > 0.8 and data['degree'] < 5:
            gaps.append({
                'type': 'high_salience_low_connectivity',
                'node': node,
                'avg_salience': data['avg_salience'],
                'degree': data['degree'],
                'priority': 'medium',
                'description': f"High-salience node '{node}' (salience={data['avg_salience']:.2f}) has low connectivity (degree={data['degree']})",
            })
    
    return gaps


def generate_gap_bridging_questions(gaps: List[Dict], network: Dict) -> List[Dict]:
    """Generate questions to bridge identified gaps."""
    questions = []
    nodes = network['nodes']
    
    for gap in gaps:
        gap_type = gap['type']
        
        if gap_type == 'isolated_important_node':
            node = gap['node']
            question = {
                'gap_type': gap_type,
                'priority': gap['priority'],
                'question': f"What facts connect '{node}' to other key entities in the case?",
                'rationale': f"Node '{node}' is important but has low connectivity. Need more facts linking it to other entities.",
                'related_nodes': [node],
            }
            questions.append(question)
        
        elif gap_type == 'missing_connection':
            n1, n2 = gap['node1'], gap['node2']
            question = {
                'gap_type': gap_type,
                'priority': gap['priority'],
                'question': f"What is the relationship or connection between '{n1}' and '{n2}'?",
                'rationale': f"Missing connection between two important entities. Need facts showing their relationship.",
                'related_nodes': [n1, n2],
            }
            questions.append(question)
        
        elif gap_type == 'low_density_cluster':
            topic = gap['topic']
            cluster_nodes = gap['nodes']
            question = {
                'gap_type': gap_type,
                'priority': gap['priority'],
                'question': f"What additional facts would strengthen the '{topic}' narrative and connect related entities?",
                'rationale': f"Topic '{topic}' has low network density. Need more facts connecting entities in this cluster.",
                'related_nodes': cluster_nodes[:5],  # Limit to first 5
            }
            questions.append(question)
        
        elif gap_type == 'high_salience_low_connectivity':
            node = gap['node']
            question = {
                'gap_type': gap_type,
                'priority': gap['priority'],
                'question': f"What facts connect the high-salience entity '{node}' to the broader causal chain?",
                'rationale': f"High-salience but low connectivity suggests missing intermediate facts.",
                'related_nodes': [node],
            }
            questions.append(question)
    
    return questions


def calculate_question_impact(question: Dict, all_gaps: List[Dict]) -> int:
    """Calculate how many gaps a question would help resolve."""
    impact = 0
    question_nodes = set(question.get('related_nodes', []))
    
    for gap in all_gaps:
        gap_nodes = set()
        if 'node' in gap:
            gap_nodes.add(gap['node'])
        if 'node1' in gap:
            gap_nodes.add(gap['node1'])
        if 'node2' in gap:
            gap_nodes.add(gap['node2'])
        if 'nodes' in gap:
            gap_nodes.update(gap['nodes'])
        
        # If question nodes overlap with gap nodes, it helps resolve the gap
        if question_nodes & gap_nodes:
            impact += 1
    
    return impact


def calculate_quality_score(facts: List[Dict], network: Dict, gaps: List[Dict]) -> Dict:
    """Calculate overall quality score (0-100) for the fact database."""
    scores = {}
    
    # 1. Gap-based score (40 points max)
    # Fewer gaps = higher score
    total_gaps = len(gaps)
    high_priority_gaps = sum(1 for g in gaps if g.get('priority') == 'high')
    medium_priority_gaps = sum(1 for g in gaps if g.get('priority') == 'medium')
    low_priority_gaps = sum(1 for g in gaps if g.get('priority') == 'low')
    
    # Ideal: 0 gaps = 40 points
    # Penalty: -5 per high priority, -2 per medium, -1 per low
    gap_score = max(0, 40 - (high_priority_gaps * 5) - (medium_priority_gaps * 2) - (low_priority_gaps * 1))
    scores['gap_score'] = gap_score
    scores['gap_details'] = {
        'total': total_gaps,
        'high': high_priority_gaps,
        'medium': medium_priority_gaps,
        'low': low_priority_gaps,
    }
    
    # 2. Network density score (25 points max)
    # Calculate average cluster density
    nodes = network['nodes']
    edges = network['edges']
    
    # Build edge lookup
    edge_lookup = defaultdict(set)
    for edge in edges:
        e1, e2 = edge['source'], edge['target']
        edge_lookup[e1].add(e2)
        edge_lookup[e2].add(e1)
    
    # Key topic clusters
    key_topics = ['knowledge', 'harm', 'publication', 'email', 'silence', 'spoliation']
    topic_clusters = defaultdict(list)
    for node, data in nodes.items():
        for topic in data['topics']:
            if topic in key_topics:
                topic_clusters[topic].append(node)
    
    densities = []
    for topic, cluster_nodes in topic_clusters.items():
        if len(cluster_nodes) >= 2:
            internal_connections = 0
            for i, n1 in enumerate(cluster_nodes):
                for n2 in cluster_nodes[i+1:]:
                    if n2 in edge_lookup.get(n1, set()):
                        internal_connections += 1
            max_possible = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
            density = internal_connections / max_possible if max_possible > 0 else 0
            densities.append(density)
    
    avg_density = sum(densities) / len(densities) if densities else 0
    # Scale: 0.5+ density = 25 points, 0.3-0.5 = 15 points, <0.3 = 5 points
    if avg_density >= 0.5:
        density_score = 25
    elif avg_density >= 0.3:
        density_score = 15 + (avg_density - 0.3) * 50  # Linear interpolation
    else:
        density_score = 5 + (avg_density / 0.3) * 10
    
    scores['density_score'] = density_score
    scores['density_details'] = {
        'average': avg_density,
        'topic_densities': {topic: d for topic, d in zip(topic_clusters.keys(), densities)},
    }
    
    # 3. Coverage score (20 points max)
    # Check coverage of key entities and narratives
    important_entities = {'Harvard', 'Harvard OGC', 'Harvard Club', 'PRC', 'EsuWiki', 'Plaintiff', 'Defamation'}
    covered_entities = sum(1 for e in important_entities if e in nodes)
    coverage_ratio = covered_entities / len(important_entities)
    coverage_score = coverage_ratio * 20
    scores['coverage_score'] = coverage_score
    scores['coverage_details'] = {
        'covered': covered_entities,
        'total': len(important_entities),
        'ratio': coverage_ratio,
    }
    
    # 4. Connectivity score (15 points max)
    # Check if important entities are well-connected
    important_connected = 0
    for entity in important_entities:
        if entity in nodes:
            degree = nodes[entity]['degree']
            if degree >= 10:
                important_connected += 1
            elif degree >= 5:
                important_connected += 0.5
    
    connectivity_ratio = important_connected / len(important_entities) if important_entities else 0
    connectivity_score = connectivity_ratio * 15
    scores['connectivity_score'] = connectivity_score
    scores['connectivity_details'] = {
        'well_connected': important_connected,
        'total': len(important_entities),
        'ratio': connectivity_ratio,
    }
    
    # Total score
    total_score = gap_score + density_score + coverage_score + connectivity_score
    scores['total_score'] = total_score
    scores['max_score'] = 100
    
    # Calculate gap percentage (how much is left to fill)
    gap_percentage = max(0, 100 - total_score)
    scores['gap_percentage'] = gap_percentage
    scores['completeness_percentage'] = total_score
    
    return scores


def main():
    """Run network gap analysis."""
    print("="*80)
    print("NETWORK GAP ANALYSIS - V25 (InfraNodus-style)")
    print("="*80)
    print()
    
    # Load facts
    print("1. Loading v25 facts...")
    facts = []
    with open(V25_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts")
    
    # Build concept network
    print("\n2. Building concept network...")
    network = build_concept_network(facts)
    print(f"   Nodes: {len(network['nodes'])}")
    print(f"   Edges: {len(network['edges'])}")
    
    # Save network JSON
    OUTPUT_NETWORK_JSON.write_text(json.dumps(network, indent=2), encoding='utf-8')
    print(f"   ✅ Saved network to {OUTPUT_NETWORK_JSON}")
    
    # Find structural gaps
    print("\n3. Analyzing network structure for gaps...")
    gaps = find_structural_gaps(network)
    print(f"   Found {len(gaps)} structural gaps")
    
    # Generate gap-bridging questions
    print("\n4. Generating gap-bridging questions...")
    questions = generate_gap_bridging_questions(gaps, network)
    
    # Calculate impact for each question
    for question in questions:
        question['impact'] = calculate_question_impact(question, gaps)
    
    # Sort by priority and impact
    questions.sort(key=lambda q: (
        {'high': 0, 'medium': 1, 'low': 2}.get(q['priority'], 3),
        -q['impact']
    ))
    
    print(f"   Generated {len(questions)} gap-bridging questions")
    
    # Export gaps
    print("\n5. Exporting gaps...")
    with open(OUTPUT_GAPS_CSV, 'w', encoding='utf-8', newline='') as f:
        if gaps:
            # Collect all possible fieldnames
            all_fieldnames = set()
            for gap in gaps:
                all_fieldnames.update(gap.keys())
            fieldnames = ['type', 'priority', 'description', 'node', 'node1', 'node2', 'topic', 'density', 'degree', 'nodes', 'avg_salience']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for gap in gaps:
                row = gap.copy()
                # Convert lists to strings
                if 'nodes' in row and isinstance(row['nodes'], list):
                    row['nodes'] = ', '.join(str(n) for n in row['nodes'])
                writer.writerow(row)
    print(f"   ✅ Exported {OUTPUT_GAPS_CSV}")
    
    # Export questions
    print("\n6. Exporting gap-bridging questions...")
    with open(OUTPUT_QUESTIONS_CSV, 'w', encoding='utf-8', newline='') as f:
        if questions:
            writer = csv.DictWriter(f, fieldnames=['gap_type', 'priority', 'impact', 'question', 'rationale', 'related_nodes'])
            writer.writeheader()
            for q in questions:
                row = q.copy()
                row['related_nodes'] = ', '.join(row.get('related_nodes', []))
                writer.writerow(row)
    print(f"   ✅ Exported {OUTPUT_QUESTIONS_CSV}")
    
    # Generate text reports
    print("\n7. Generating text reports...")
    
    # Gaps report
    gaps_lines = []
    gaps_lines.append("="*80)
    gaps_lines.append("V25 NETWORK GAP ANALYSIS - STRUCTURAL GAPS")
    gaps_lines.append("="*80)
    gaps_lines.append("")
    gaps_lines.append(f"Total gaps found: {len(gaps)}")
    gaps_lines.append("")
    
    by_priority = defaultdict(list)
    for gap in gaps:
        by_priority[gap['priority']].append(gap)
    
    for priority in ['high', 'medium', 'low']:
        if priority in by_priority:
            gaps_lines.append(f"{priority.upper()} PRIORITY GAPS ({len(by_priority[priority])})")
            gaps_lines.append("-"*80)
            for gap in by_priority[priority]:
                gaps_lines.append(f"Type: {gap['type']}")
                gaps_lines.append(f"Description: {gap['description']}")
                gaps_lines.append("")
    
    OUTPUT_GAPS_TXT.write_text("\n".join(gaps_lines), encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_GAPS_TXT}")
    
    # Questions report
    questions_lines = []
    questions_lines.append("="*80)
    questions_lines.append("V25 GAP-BRIDGING QUESTIONS")
    questions_lines.append("="*80)
    questions_lines.append("")
    questions_lines.append(f"Total questions: {len(questions)}")
    questions_lines.append("")
    questions_lines.append("Questions are sorted by priority and impact (how many gaps they help resolve).")
    questions_lines.append("")
    
    for i, q in enumerate(questions[:20], 1):  # Top 20
        questions_lines.append(f"{i}. {q['question']}")
        questions_lines.append(f"   Priority: {q['priority']}")
        questions_lines.append(f"   Impact: Resolves {q['impact']} gaps")
        questions_lines.append(f"   Rationale: {q['rationale']}")
        questions_lines.append(f"   Related nodes: {', '.join(q.get('related_nodes', []))}")
        questions_lines.append("")
    
    OUTPUT_QUESTIONS_TXT.write_text("\n".join(questions_lines), encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_QUESTIONS_TXT}")
    
    # Generate summary report
    print("\n8. Generating summary report...")
    report = f"""# V25 Network Gap Analysis Report

## Summary

Network-based gap analysis (InfraNodus-style) of v25 facts dataset.

## Network Statistics

- **Total facts analyzed**: {len(facts)}
- **Concept nodes**: {len(network['nodes'])}
- **Concept edges**: {len(network['edges'])}
- **Structural gaps found**: {len(gaps)}
- **Gap-bridging questions generated**: {len(questions)}

## Gap Breakdown

"""
    
    for priority in ['high', 'medium', 'low']:
        if priority in by_priority:
            report += f"- **{priority.upper()} priority**: {len(by_priority[priority])} gaps\n"
    
    report += f"""
## Top Gap-Bridging Questions

"""
    
    for i, q in enumerate(questions[:10], 1):
        report += f"{i}. **{q['question']}**\n"
        report += f"   - Priority: {q['priority']}\n"
        report += f"   - Impact: Resolves {q['impact']} gaps\n"
        report += f"   - Rationale: {q['rationale']}\n\n"
    
    report += f"""## Files Generated

- **Network JSON**: `case_law_data/v25_concept_network.json`
- **Gaps CSV**: `case_law_data/v25_network_gaps.csv`
- **Gaps TXT**: `case_law_data/v25_network_gaps.txt`
- **Questions CSV**: `case_law_data/v25_gap_bridging_questions.csv`
- **Questions TXT**: `case_law_data/v25_gap_bridging_questions.txt`
- **Report**: `reports/analysis_outputs/v25_network_gap_analysis_report.md`

## Methodology

1. **Concept Network**: Built from entities, topics, and relationships extracted from facts
2. **Gap Detection**: Identified structural gaps (isolated nodes, missing connections, low-density clusters)
3. **Question Generation**: Generated questions to bridge identified gaps
4. **Impact Scoring**: Calculated how many gaps each question would help resolve

## Next Steps

1. Review top gap-bridging questions
2. Answer high-priority, high-impact questions first
3. Add new facts to fill identified gaps
4. Re-run analysis to verify gaps are filled
"""
    
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {OUTPUT_REPORT}")
    
    # Calculate quality score
    print("\n9. Calculating quality score...")
    quality_scores = calculate_quality_score(facts, network, gaps)
    
    # Add quality score to report
    quality_report = f"""

## Quality Score Analysis

### Overall Quality Score: **{quality_scores['total_score']:.1f}/100** ({quality_scores['completeness_percentage']:.1f}% complete)

**Gap Remaining: {quality_scores['gap_percentage']:.1f}%**

### Score Breakdown

1. **Gap Score: {quality_scores['gap_score']:.1f}/40**
   - Total gaps: {quality_scores['gap_details']['total']}
   - High priority: {quality_scores['gap_details']['high']}
   - Medium priority: {quality_scores['gap_details']['medium']}
   - Low priority: {quality_scores['gap_details']['low']}

2. **Network Density Score: {quality_scores['density_score']:.1f}/25**
   - Average cluster density: {quality_scores['density_details']['average']:.2%}
   - Topic densities:
"""
    for topic, density in quality_scores['density_details']['topic_densities'].items():
        quality_report += f"     - {topic}: {density:.2%}\n"
    
    quality_report += f"""
3. **Coverage Score: {quality_scores['coverage_score']:.1f}/20**
   - Key entities covered: {quality_scores['coverage_details']['covered']}/{quality_scores['coverage_details']['total']} ({quality_scores['coverage_details']['ratio']:.1%})

4. **Connectivity Score: {quality_scores['connectivity_score']:.1f}/15**
   - Well-connected entities: {quality_scores['connectivity_details']['well_connected']:.1f}/{quality_scores['connectivity_details']['total']} ({quality_scores['connectivity_details']['ratio']:.1%})

### Interpretation

- **90-100**: Excellent - Litigation-ready, minimal gaps
- **80-89**: Very Good - Strong dataset with minor improvements possible
- **70-79**: Good - Solid foundation, some gaps remain
- **60-69**: Fair - Functional but needs significant work
- **Below 60**: Needs substantial improvement

**Current Status**: {quality_scores['completeness_percentage']:.1f}% complete with {quality_scores['gap_percentage']:.1f}% gap remaining.
"""
    
    # Append to report
    full_report = report + quality_report
    OUTPUT_REPORT.write_text(full_report, encoding='utf-8')
    
    # Generate quality score summary file
    quality_summary_path = Path("reports/analysis_outputs/v25_quality_score.txt")
    quality_summary_path.parent.mkdir(parents=True, exist_ok=True)
    quality_summary = f"""V25 FACT DATABASE QUALITY SCORE
{'='*80}

OVERALL SCORE: {quality_scores['total_score']:.1f}/100
COMPLETENESS: {quality_scores['completeness_percentage']:.1f}%
GAP REMAINING: {quality_scores['gap_percentage']:.1f}%

{'='*80}
SCORE BREAKDOWN
{'='*80}

1. Gap Score: {quality_scores['gap_score']:.1f}/40
   - Total gaps: {quality_scores['gap_details']['total']}
   - High priority: {quality_scores['gap_details']['high']}
   - Medium priority: {quality_scores['gap_details']['medium']}
   - Low priority: {quality_scores['gap_details']['low']}

2. Network Density Score: {quality_scores['density_score']:.1f}/25
   - Average cluster density: {quality_scores['density_details']['average']:.2%}

3. Coverage Score: {quality_scores['coverage_score']:.1f}/20
   - Key entities covered: {quality_scores['coverage_details']['covered']}/{quality_scores['coverage_details']['total']}

4. Connectivity Score: {quality_scores['connectivity_score']:.1f}/15
   - Well-connected entities: {quality_scores['connectivity_details']['well_connected']:.1f}/{quality_scores['connectivity_details']['total']}

{'='*80}
INTERPRETATION
{'='*80}

90-100: Excellent - Litigation-ready, minimal gaps
80-89:  Very Good - Strong dataset with minor improvements possible
70-79:  Good - Solid foundation, some gaps remain
60-69:  Fair - Functional but needs significant work
Below 60: Needs substantial improvement

CURRENT STATUS: {quality_scores['completeness_percentage']:.1f}% complete
GAP REMAINING: {quality_scores['gap_percentage']:.1f}%
"""
    quality_summary_path.write_text(quality_summary, encoding='utf-8')
    print(f"   ✅ Written {quality_summary_path}")
    
    print()
    print("="*80)
    print("NETWORK GAP ANALYSIS COMPLETE")
    print("="*80)
    print()
    print(f"✅ Found {len(gaps)} structural gaps")
    print(f"✅ Generated {len(questions)} gap-bridging questions")
    print()
    print("="*80)
    print("QUALITY SCORE: {:.1f}/100 ({:.1f}% complete, {:.1f}% gap remaining)".format(
        quality_scores['total_score'],
        quality_scores['completeness_percentage'],
        quality_scores['gap_percentage']
    ))
    print("="*80)
    print()
    print("Score Breakdown:")
    print(f"  • Gap Score: {quality_scores['gap_score']:.1f}/40")
    print(f"  • Network Density: {quality_scores['density_score']:.1f}/25")
    print(f"  • Coverage: {quality_scores['coverage_score']:.1f}/20")
    print(f"  • Connectivity: {quality_scores['connectivity_score']:.1f}/15")
    print()
    print("Top 5 questions by impact:")
    for i, q in enumerate(questions[:5], 1):
        print(f"  {i}. [{q['priority']}] {q['question']} (impact: {q['impact']})")
    print()


if __name__ == "__main__":
    main()

