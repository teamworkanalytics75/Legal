"""
filter_graph.py
Filter co-occurrence graph to remove weak connections and reduce density.
"""

import pickle
import json
import networkx as nx
from pathlib import Path

INPUT_FILE = "analysis_results/cooccurrence_graph.gpickle"
OUTPUT_DIR = "analysis_results"

def main():
    print("=" * 70)
    print("GRAPH FILTERING - Remove Weak Connections")
    print("=" * 70)
    
    # Load graph
    print(f"\n[1/3] Loading dense graph...")
    with open(INPUT_FILE, 'rb') as f:
        G_dense = pickle.load(f)
    
    print(f"[Original Graph]")
    print(f" Nodes: {G_dense.number_of_nodes():,}")
    print(f" Edges: {G_dense.number_of_edges():,}")
    print(f" Density: {nx.density(G_dense):.4f}")
    
    # Strategy: Keep only meaningful connections
    print(f"\n[2/3] Applying filters...")
    
    # Filter 1: Remove edges with weight < 5
    # (entities must co-occur in 5+ documents to be meaningfully related)
    G_filtered = nx.Graph()
    
    print(f" Filter 1: Keeping only edges with weight >= 5...")
    strong_edges = [(u, v, data['weight']) for u, v, data in G_dense.edges(data=True) if data['weight'] >= 5]
    
    G_filtered.add_weighted_edges_from(strong_edges, weight='weight')
    
    print(f" Edges after filter 1: {G_filtered.number_of_edges():,}")
    print(f" Density: {nx.density(G_filtered):.4f}")
    
    # Filter 2: Keep only nodes with degree >= 3
    # (entities must connect to 3+ other entities to be included)
    print(f" Filter 2: Removing isolated nodes (degree < 3)...")
    nodes_to_keep = [n for n in G_filtered.nodes() if G_filtered.degree(n) >= 3]
    G_final = G_filtered.subgraph(nodes_to_keep).copy()
    
    print(f" Nodes after filter 2: {G_final.number_of_nodes():,}")
    print(f" Edges after filter 2: {G_final.number_of_edges():,}")
    print(f" Density: {nx.density(G_final):.4f}")
    
    # Filter 3: Keep only largest connected component
    print(f" Filter 3: Keeping largest connected component...")
    if nx.is_connected(G_final):
        G_clean = G_final
        print(f" Graph already connected")
    else:
        components = list(nx.connected_components(G_final))
        largest_cc = max(components, key=len)
        G_clean = G_final.subgraph(largest_cc).copy()
        print(f" Kept largest component: {len(largest_cc)} nodes")
    
    print(f"\n[Final Graph]")
    print(f" Nodes: {G_clean.number_of_nodes():,}")
    print(f" Edges: {G_clean.number_of_edges():,}")
    print(f" Density: {nx.density(G_clean):.4f}")
    
    # Save filtered graph
    print(f"\n[3/3] Saving filtered graph...")
    output_path = Path(OUTPUT_DIR)
    
    # Save NetworkX pickle
    with open(output_path / "cooccurrence_graph_filtered.gpickle", 'wb') as f:
        pickle.dump(G_clean, f)
    
    # Save JSON
    graph_data = {
        "nodes": [{"id": node} for node in G_clean.nodes()],
        "edges": [
            {"source": u, "target": v, "weight": G_clean[u][v]['weight']} 
            for u, v in G_clean.edges()
        ],
        "stats": {
            "num_nodes": G_clean.number_of_nodes(),
            "num_edges": G_clean.number_of_edges(),
            "density": nx.density(G_clean),
            "avg_degree": sum(dict(G_clean.degree()).values()) / G_clean.number_of_nodes()
        }
    }
    
    with open(output_path / "cooccurrence_graph_filtered.json", 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    # Save GEXF
    nx.write_gexf(G_clean, output_path / "cooccurrence_graph_filtered.gexf")
    
    print(f"[OK] Filtered graph saved")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("FILTERING SUMMARY")
    print("=" * 70)
    print(f" Original: {G_dense.number_of_nodes():,} nodes, {G_dense.number_of_edges():,} edges (density: {nx.density(G_dense):.4f})")
    print(f" Filtered: {G_clean.number_of_nodes():,} nodes, {G_clean.number_of_edges():,} edges (density: {nx.density(G_clean):.4f})")
    print(f" Removed: {G_dense.number_of_edges() - G_clean.number_of_edges():,} weak edges ({(1 - G_clean.number_of_edges()/G_dense.number_of_edges())*100:.1f}%)")
    print(f" ")
    print(f" Result: Graph now has reasonable density for analysis")
    print(f" ")
    print(f"[Next] Run: py run_graph_analysis_filtered.py")

if __name__ == "__main__":
    main()
