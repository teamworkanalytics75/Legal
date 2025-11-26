"""
run_graph_analysis.py
Run all Infranodus-style graph analysis algorithms.

Computes: betweenness centrality, PageRank, communities, bridges, gaps, etc.
"""

import pickle
import json
import networkx as nx
from pathlib import Path
from collections import defaultdict
import time

# Import your existing analyzer
from nlp_analysis.graph_analyzer import GraphAnalyzer

# Configuration
INPUT_FILE = "analysis_results/cooccurrence_graph.gpickle"
OUTPUT_DIR = "analysis_results"

def main():
    print("=" * 70)
    print("GRAPH ANALYSIS - Infranodus-Style Metrics")
    print("=" * 70)
    
    # Load graph
    print(f"\n[1/8] Loading co-occurrence graph...")
    with open(INPUT_FILE, 'rb') as f:
        G = pickle.load(f)
    
    print(f"[OK] Graph loaded:")
    print(f" Nodes: {G.number_of_nodes():,}")
    print(f" Edges: {G.number_of_edges():,}")
    print(f" Density: {nx.density(G):.4f}")
    
    # Convert to directed for some metrics
    G_directed = G.to_directed()
    analyzer = GraphAnalyzer(G_directed)
    
    results = {"graph_stats": {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G)
    }}
    
    # 1. Centrality Metrics
    print(f"\n[2/8] Computing centrality metrics...")
    start_t = time.time()
    
    print(f" - Degree centrality...")
    degree_cent = dict(G.degree())
    
    print(f" - Betweenness centrality (bridge concepts)...")
    betweenness = nx.betweenness_centrality(G)
    
    print(f" - PageRank (key concepts)...")
    pagerank = nx.pagerank(G)
    
    print(f" - Closeness centrality...")
    closeness = nx.closeness_centrality(G)
    
    dur = time.time() - start_t
    print(f"[OK] Centrality computed in {dur:.1f}s")
    
    # Top nodes by each metric
    results["centrality"] = {
        "top_by_degree": sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:30],
        "top_by_betweenness": sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:30],
        "top_by_pagerank": sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:30],
        "top_by_closeness": sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:30]
    }
    
    # 2. Community Detection
    print(f"\n[3/8] Detecting communities (Louvain algorithm)...")
    start_t = time.time()
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        num_communities = len(communities)
        dur = time.time() - start_t
        print(f"[OK] Found {num_communities} communities in {dur:.1f}s")
        
        # Get top entities in each community
        community_summaries = []
        for comm_id, members in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            # Top members by degree
            top_members = sorted(members, key=lambda n: degree_cent.get(n, 0), reverse=True)[:10]
            
            community_summaries.append({
                "community_id": comm_id,
                "size": len(members),
                "top_members": top_members
            })
        
        results["communities"] = {
            "num_communities": num_communities,
            "partition": partition,
            "top_communities": community_summaries
        }
        
    except ImportError:
        print("[Warning] python-louvain not installed, skipping community detection")
        results["communities"] = None
    
    # 3. Bridge Nodes (High Betweenness)
    print(f"\n[4/8] Finding bridge concepts...")
    bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    results["bridges"] = [{"entity": node, "betweenness": score} for node, score in bridges]
    print(f"[OK] Top 5 bridges: {[b[0] for b in bridges[:5]]}")
    
    # 4. Clustering Coefficient
    print(f"\n[5/8] Computing clustering coefficients...")
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering) if clustering else 0
    results["clustering"] = {
        "average": avg_clustering,
        "by_node": dict(sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:50])
    }
    print(f"[OK] Average clustering coefficient: {avg_clustering:.4f}")
    
    # 5. Connected Components
    print(f"\n[6/8] Analyzing connected components...")
    components = list(nx.connected_components(G))
    results["components"] = {
        "num_components": len(components),
        "largest_component_size": len(max(components, key=len)),
        "component_sizes": [len(c) for c in sorted(components, key=len, reverse=True)[:10]]
    }
    print(f"[OK] Found {len(components)} components (largest: {len(max(components, key=len))} nodes)")
    
    # 6. Structural Gaps
    print(f"\n[7/8] Finding structural gaps...")
    
    if results["communities"]:
        # Find weak connections between communities
        intercommunity_edges = []
        
        for u, v in G.edges():
            if partition[u] != partition[v]:
                weight = G[u][v].get('weight', 1)
                intercommunity_edges.append({
                    "from_node": u,
                    "to_node": v,
                    "from_community": partition[u],
                    "to_community": partition[v],
                    "weight": weight
                })
        
        # Sort by weight (weakest connections = biggest gaps)
        intercommunity_edges.sort(key=lambda x: x['weight'])
        
        results["structural_gaps"] = {
            "num_intercommunity_edges": len(intercommunity_edges),
            "weak_connections": intercommunity_edges[:50] # Top 50 weak bridges
        }
        
        print(f"[OK] Found {len(intercommunity_edges)} inter-community connections")
        print(f" Weakest bridges indicate structural gaps")
    else:
        results["structural_gaps"] = None
    
    # 7. Network Modularity
    print(f"\n[8/8] Computing network modularity...")
    if results["communities"]:
        modularity = community_louvain.modularity(partition, G)
        results["modularity"] = modularity
        print(f"[OK] Modularity: {modularity:.4f} (higher = more distinct communities)")
    else:
        results["modularity"] = None
    
    # Save all results
    print(f"\n[Saving] Writing analysis results...")
    output_path = Path(OUTPUT_DIR)
    
    metrics_file = output_path / "graph_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create human-readable summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("GRAPH ANALYSIS SUMMARY (Infranodus-style)")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"Network Statistics:")
    summary_lines.append(f" Nodes: {results['graph_stats']['num_nodes']:,}")
    summary_lines.append(f" Edges: {results['graph_stats']['num_edges']:,}")
    summary_lines.append(f" Density: {results['graph_stats']['density']:.4f}")
    summary_lines.append(f" Connected: {results['graph_stats']['is_connected']}")
    summary_lines.append("")
    
    if results.get("communities"):
        summary_lines.append(f"Community Structure:")
        summary_lines.append(f" Total communities: {results['communities']['num_communities']}")
        summary_lines.append(f" Modularity: {results.get('modularity', 0):.4f}")
        summary_lines.append(f" Largest 5 communities:")
        for comm in results['communities']['top_communities'][:5]:
            summary_lines.append(f" Community {comm['community_id']}: {comm['size']} entities")
            summary_lines.append(f" Top members: {', '.join(comm['top_members'][:5])}")
    
    summary_lines.append("")
    summary_lines.append(f"Bridge Concepts (Top 10 by Betweenness):")
    for bridge in results['bridges'][:10]:
        summary_lines.append(f" {bridge['entity']:40s} (score: {bridge['betweenness']:.6f})")
    
    summary_lines.append("")
    summary_lines.append(f"Key Concepts (Top 10 by PageRank):")
    for entity, score in results['centrality']['top_by_pagerank'][:10]:
        summary_lines.append(f" {entity:40s} (score: {score:.6f})")
    
    summary_lines.append("")
    summary_lines.append(f"Clustering:")
    summary_lines.append(f" Average clustering coefficient: {results['clustering']['average']:.4f}")
    summary_lines.append(f" (Higher = more tightly knit concepts)")
    
    if results.get("structural_gaps"):
        summary_lines.append("")
        summary_lines.append(f"Structural Gaps:")
        summary_lines.append(f" Inter-community connections: {results['structural_gaps']['num_intercommunity_edges']}")
        summary_lines.append(f" Weakest connections (indicate gaps):")
        for gap in results['structural_gaps']['weak_connections'][:5]:
            summary_lines.append(f" {gap['from_node']} <-> {gap['to_node']} (weight: {gap['weight']})")
    
    summary_text = "\n".join(summary_lines)
    
    summary_file = output_path / "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    print(f"\n[Saved]")
    print(f" {metrics_file}")
    print(f" {summary_file}")
    
    print("\n[Next] Run: py export_to_bn.py")

if __name__ == "__main__":
    main()
