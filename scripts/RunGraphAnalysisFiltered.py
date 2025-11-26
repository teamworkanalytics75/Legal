"""
run_graph_analysis_filtered.py
Run Infranodus-style analysis on filtered graph.

Handles large graphs by using approximation algorithms where needed.
"""

import pickle
import json
import networkx as nx
from pathlib import Path
from collections import defaultdict
import time

INPUT_FILE = "analysis_results/cooccurrence_graph_filtered.gpickle"
OUTPUT_DIR = "analysis_results"

def main():
    print("=" * 70)
    print("GRAPH ANALYSIS - Infranodus Algorithms")
    print("=" * 70)
    
    # Load filtered graph
    print(f"\n[1/7] Loading filtered graph...")
    with open(INPUT_FILE, 'rb') as f:
        G = pickle.load(f)
    
    print(f"[OK] Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    results = {
        "graph_stats": {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G)
        }
    }
    
    # 1. Degree Centrality (fast)
    print(f"\n[2/7] Computing degree centrality...")
    degree_cent = dict(G.degree())
    results["degree_centrality"] = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:50]
    print(f"[OK] Top 5: {[x[0] for x in results['degree_centrality'][:5]]}")
    
    # 2. PageRank (fast, ~1 minute)
    print(f"\n[3/7] Computing PageRank...")
    start_t = time.time()
    pagerank = nx.pagerank(G, max_iter=100)
    dur = time.time() - start_t
    results["pagerank"] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:50]
    print(f"[OK] Computed in {dur:.1f}s")
    print(f" Top 5: {[x[0] for x in results['pagerank'][:5]]}")
    
    # 3. Community Detection (fast with Louvain)
    print(f"\n[4/7] Detecting communities (Louvain)...")
    start_t = time.time()
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        
        # Group by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # Top communities with their key members
        community_summaries = []
        for comm_id, members in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
            top_members = sorted(members, key=lambda n: degree_cent.get(n, 0), reverse=True)[:15]
            community_summaries.append({
                "id": comm_id,
                "size": len(members),
                "top_entities": top_members
            })
        
        modularity = community_louvain.modularity(partition, G)
        
        dur = time.time() - start_t
        print(f"[OK] Found {len(communities)} communities in {dur:.1f}s")
        print(f" Modularity: {modularity:.4f}")
        print(f" Top 5 community sizes: {[len(communities[i]) for i in sorted(communities, key=lambda x: len(communities[x]), reverse=True)[:5]]}")
        
        results["communities"] = {
            "num_communities": len(communities),
            "modularity": modularity,
            "top_communities": community_summaries,
            "partition": partition
        }
        
    except ImportError:
        print("[Warning] python-louvain not available, skipping")
        results["communities"] = None
        partition = None
    
    # 4. Betweenness Centrality (SLOW on large graphs - use approximation)
    print(f"\n[5/7] Computing betweenness centrality...")
    print(f" (Using approximation for speed on {G.number_of_edges():,} edges)")
    start_t = time.time()
    
    # Use k=500 sample for approximation (much faster than exact)
    betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    
    dur = time.time() - start_t
    results["betweenness"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:50]
    print(f"[OK] Computed in {dur:.1f}s")
    print(f" Top 5 bridges: {[x[0] for x in results['betweenness'][:5]]}")
    
    # 5. Clustering Coefficient
    print(f"\n[6/7] Computing clustering coefficients...")
    start_t = time.time()
    clustering = nx.clustering(G)
    avg_clustering = sum(clustering.values()) / len(clustering)
    dur = time.time() - start_t
    
    results["clustering"] = {
        "average": avg_clustering,
        "top_clustered": sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:30]
    }
    print(f"[OK] Average clustering: {avg_clustering:.4f} (computed in {dur:.1f}s)")
    
    # 6. Structural Gaps (between communities)
    print(f"\n[7/7] Finding structural gaps...")
    
    if partition:
        # Find edges connecting different communities
        intercommunity_edges = []
        
        for u, v in G.edges():
            if partition[u] != partition[v]:
                weight = G[u][v].get('weight', 1)
                intercommunity_edges.append({
                    "from": u,
                    "to": v,
                    "from_community": partition[u],
                    "to_community": partition[v],
                    "weight": weight,
                    "betweenness": betweenness.get(u, 0) + betweenness.get(v, 0)
                })
        
        # Sort by weight (lowest = biggest gap)
        intercommunity_edges.sort(key=lambda x: x['weight'])
        
        # Identify isolated communities (few connections)
        community_connections = defaultdict(int)
        for edge in intercommunity_edges:
            community_connections[edge['from_community']] += 1
            community_connections[edge['to_community']] += 1
        
        isolated_communities = [
            comm_id for comm_id, conn_count in community_connections.items()
            if conn_count < 10
        ]
        
        results["structural_gaps"] = {
            "num_intercommunity_edges": len(intercommunity_edges),
            "weak_bridges": intercommunity_edges[:30],
            "isolated_communities": isolated_communities,
            "num_isolated": len(isolated_communities)
        }
        
        print(f"[OK] Found {len(intercommunity_edges)} inter-community edges")
        print(f" {len(isolated_communities)} isolated communities (weak integration)")
    else:
        results["structural_gaps"] = None
    
    # Save results
    print(f"\n[Saving] Writing analysis...")
    output_path = Path(OUTPUT_DIR)
    
    metrics_file = output_path / "graph_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create readable summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("INFRANODUS-STYLE ANALYSIS RESULTS")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append("Network Overview:")
    summary_lines.append(f" Entities (nodes): {results['graph_stats']['num_nodes']:,}")
    summary_lines.append(f" Connections (edges): {results['graph_stats']['num_edges']:,}")
    summary_lines.append(f" Network density: {results['graph_stats']['density']:.4f}")
    summary_lines.append(f" Average clustering: {results['clustering']['average']:.4f}")
    summary_lines.append("")
    
    if results.get("communities"):
        summary_lines.append(f"Topic Communities:")
        summary_lines.append(f" Total: {results['communities']['num_communities']}")
        summary_lines.append(f" Modularity: {results['communities']['modularity']:.4f} (0.3+ = well-defined)")
        summary_lines.append("")
        summary_lines.append(f" Largest communities:")
        for comm in results['communities']['top_communities'][:5]:
            summary_lines.append(f" Community {comm['id']}: {comm['size']} entities")
            summary_lines.append(f" Key concepts: {', '.join(comm['top_entities'][:8])}")
            summary_lines.append("")
    
    summary_lines.append("Bridge Concepts (connect different topics):")
    for entity, score in results['betweenness'][:15]:
        summary_lines.append(f" {entity:40s} (score: {score:.6f})")
    summary_lines.append("")
    
    summary_lines.append("Key Concepts (most influential by PageRank):")
    for entity, score in results['pagerank'][:15]:
        summary_lines.append(f" {entity:40s} (score: {score:.6f})")
    summary_lines.append("")
    
    if results.get("structural_gaps"):
        summary_lines.append(f"Structural Analysis:")
        summary_lines.append(f" Isolated communities: {results['structural_gaps']['num_isolated']}")
        summary_lines.append(f" Inter-community connections: {results['structural_gaps']['num_intercommunity_edges']}")
        summary_lines.append("")
        summary_lines.append(f" Weakest bridges (potential gaps in argumentation):")
        for gap in results['structural_gaps']['weak_bridges'][:8]:
            summary_lines.append(f" {gap['from']} <-> {gap['to']} (weight: {gap['weight']})")
    
    summary_text = "\n".join(summary_lines)
    
    summary_file = output_path / "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    print(f"\n[Complete] Files saved:")
    print(f" {metrics_file}")
    print(f" {summary_file}")
    
    print(f"\n[Next] Run: py export_to_bn.py")

if __name__ == "__main__":
    main()
