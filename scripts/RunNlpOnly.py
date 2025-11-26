"""
run_nlp_only.py
Run NLP analysis on already-cleaned documents in the database.
"""

import sqlite3
from nlp_analysis.pipeline import analyze_legal_documents
from pathlib import Path

# Configuration
DB_PATH = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"
OUTPUT_DIR = "complete_analysis"
TARGET_OUTCOME = "Lawsuit Success"

def main():
    """Run NLP analysis on existing database."""
    print("=" * 70)
    print("NLP ANALYSIS ON EXISTING DATABASE")
    print("=" * 70)
    
    # Load documents from database
    print(f"\n[Step 1] Loading documents from: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM cleaned_documents")
    documents = [row[0] for row in cur.fetchall()]
    conn.close()
    
    print(f"[OK] Loaded {len(documents)} documents")
    
    if not documents:
        print("[Error] No documents found in database!")
        return
    
    # Run NLP analysis
    print(f"\n[Step 2] Running NLP analysis...")
    print(f" This will process {len(documents)} documents")
    print(f" Expected time: ~10-20 minutes")
    print(f" Output directory: {Path(OUTPUT_DIR).absolute()}")
    
    results = analyze_legal_documents(
        documents,
        output_dir=OUTPUT_DIR,
        target_outcome=TARGET_OUTCOME
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n[Results Summary]")
    print(f" Knowledge Graph:")
    print(f" - Entities: {results['analysis']['overview']['num_nodes']}")
    print(f" - Relations: {results['analysis']['overview']['num_edges']}")
    print(f" - Communities: {results['analysis']['overview']['num_communities']}")
    
    print(f"\n Bayesian Network:")
    print(f" - Nodes: {results['bayesian_network']['summary']['num_nodes']}")
    print(f" - Edges: {results['bayesian_network']['summary']['num_edges']}")
    print(f" - Evidence: {results['bayesian_network']['summary']['total_evidence']}")
    
    print(f"\n[Output Files]")
    output_path = Path(OUTPUT_DIR)
    for file in output_path.glob("*"):
        size_kb = file.stat().st_size / 1024
        print(f" - {file.name} ({size_kb:.1f} KB)")
    
    print(f"\n[Next Steps]")
    print(f" 1. Open: {output_path / 'knowledge_graph_interactive.html'}")
    print(f" 2. Review: {output_path / 'bayesian_network.json'}")
    print(f" 3. Feed BN to PySMILE for probabilistic inference")


if __name__ == "__main__":
    main()
