"""
validate_graph_metrics.py
Validate that the pruned co-occurrence graph meets target metrics.

Target metrics:
- Edge count: 250,000-400,000
- Density: 0.01-0.05
- Modularity: >= 0.30
- Clustering: 0.2-0.4
"""

import pickle
import json
import networkx as nx
from pathlib import Path

# Target ranges
TARGET_EDGES = (250_000, 400_000)
TARGET_DENSITY = (0.01, 0.05)
TARGET_MODULARITY = 0.30
TARGET_CLUSTERING = (0.2, 0.4)

GRAPH_FILE = "analysis_results/cooccurrence_graph_pruned.gpickle"
OUTPUT_FILE = "analysis_results/graph_metrics_fixed.json"


def check_metric(name: str, value: float, target_range: tuple = None, target_min: float = None) -> bool:
    """Check if metric meets target."""
    if target_range:
        ok = target_range[0] <= value <= target_range[1]
        status = "[OK]" if ok else "[FAIL]"
        print(f" {status} {name}: {value:.4f} (target: {target_range[0]:.4f}-{target_range[1]:.4f})")
        return ok
    elif target_min is not None:
        ok = value >= target_min
        status = "[OK]" if ok else "[FAIL]"
        print(f" {status} {name}: {value:.4f} (target: >= {target_min:.4f})")
        return ok
    else:
        print(f" * {name}: {value:.4f}")
        return True


def main():
    print("=" * 70)
    print("GRAPH METRICS VALIDATION")
    print("=" * 70)
    
    # Load graph
    print(f"\n[1/3] Loading graph: {GRAPH_FILE}")
    try:
        with open(GRAPH_FILE, 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Graph file not found: {GRAPH_FILE}")
        print(f"[Hint] Run: uv run python build_cooccurrence_graph_fixed.py")
        return
    
    print(f"[OK] Loaded graph with {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Compute metrics
    print(f"\n[2/3] Computing metrics...")
    
    all_ok = True
    
    # Basic stats
    print(f"\n--- Basic Statistics ---")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f" - Nodes: {num_nodes:,}")
    
    edges_ok = check_metric(
        "Edges", 
        num_edges, 
        target_range=TARGET_EDGES
    )
    all_ok = all_ok and edges_ok
    
    # Density
    print(f"\n--- Density ---")
    density = nx.density(G)
    density_ok = check_metric(
        "Density",
        density,
        target_range=TARGET_DENSITY
    )
    all_ok = all_ok and density_ok
    
    # Clustering
    print(f"\n--- Clustering ---")
    print(f" (Computing average clustering coefficient...)")
    clustering = nx.average_clustering(G)
    clustering_ok = check_metric(
        "Clustering coefficient",
        clustering,
        target_range=TARGET_CLUSTERING
    )
    all_ok = all_ok and clustering_ok
    
    # Modularity (requires community detection)
    print(f"\n--- Modularity ---")
    print(f" (Computing communities with Louvain...)")
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
        modularity_ok = check_metric(
            "Modularity",
            modularity,
            target_min=TARGET_MODULARITY
        )
        all_ok = all_ok and modularity_ok
        num_communities = len(set(partition.values()))
        print(f" - Communities detected: {num_communities}")
    except ImportError:
        print(f" WARNING python-louvain not installed, using greedy modularity")
        from networkx.algorithms import community as nx_community
        communities = nx_community.greedy_modularity_communities(G)
        modularity = nx_community.modularity(G, communities)
        modularity_ok = check_metric(
            "Modularity",
            modularity,
            target_min=TARGET_MODULARITY
        )
        all_ok = all_ok and modularity_ok
        print(f" - Communities detected: {len(communities)}")
    
    # Additional metrics
    print(f"\n--- Additional Metrics ---")
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    print(f" - Average degree: {avg_degree:.1f}")
    
    components = list(nx.connected_components(G))
    print(f" - Connected components: {len(components)}")
    largest_cc_size = len(max(components, key=len))
    print(f" - Largest component: {largest_cc_size:,} nodes ({100*largest_cc_size/num_nodes:.1f}%)")
    
    # Top nodes by degree
    print(f"\n--- Top 10 Nodes by Degree ---")
    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (node, degree) in enumerate(top_nodes, 1):
        print(f" {rank:2d}. {node:40s} (degree: {degree})")
    
    # Save metrics
    print(f"\n[3/3] Saving metrics to: {OUTPUT_FILE}")
    
    metrics = {
        "basic": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree
        },
        "structural": {
            "density": density,
            "clustering_coefficient": clustering,
            "modularity": modularity,
            "num_communities": num_communities if 'num_communities' in locals() else len(communities)
        },
        "connectivity": {
            "num_components": len(components),
            "largest_component_size": largest_cc_size,
            "largest_component_pct": 100 * largest_cc_size / num_nodes
        },
        "top_nodes": [
            {"node": node, "degree": degree}
            for node, degree in top_nodes
        ],
        "validation": {
            "edges_in_range": edges_ok,
            "density_in_range": density_ok,
            "clustering_in_range": clustering_ok,
            "modularity_ok": modularity_ok,
            "all_targets_met": all_ok
        },
        "targets": {
            "edge_range": list(TARGET_EDGES),
            "density_range": list(TARGET_DENSITY),
            "modularity_min": TARGET_MODULARITY,
            "clustering_range": list(TARGET_CLUSTERING)
        }
    }
    
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] Metrics saved")
    
    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if all_ok:
        print("[SUCCESS] ALL TARGETS MET!")
        print()
        print(" The graph now has:")
        print(f" - Reasonable density ({density:.4f})")
        print(f" - Good modularity ({modularity:.4f})")
        print(f" - Appropriate clustering ({clustering:.4f})")
        print(f" - {num_edges:,} meaningful edges")
        print()
        print(" Graph is ready for analysis and visualization.")
    else:
        print("[WARNING] SOME TARGETS NOT MET")
        print()
        print(" Adjust parameters in build_cooccurrence_graph_fixed.py:")
        if not edges_ok:
            if num_edges > TARGET_EDGES[1]:
                print(f" - Increase MIN_EDGE_WEIGHT (currently 2) to reduce edges")
            else:
                print(f" - Decrease MIN_EDGE_WEIGHT (currently 2) to add edges")
        if not density_ok:
            print(f" - Adjust edge filtering to change density")
        if not clustering_ok:
            print(f" - Clustering: {clustering:.4f} (may adjust with window size)")
        if not modularity_ok:
            print(f" - Modularity too low - increase edge filtering")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

