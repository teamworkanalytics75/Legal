"""
run_nlp_batch.py
Run NLP analysis in small batches to prevent hangs.
"""

import sqlite3
from nlp_analysis.pipeline import analyze_legal_documents
from pathlib import Path
import json

# Configuration
DB_PATH = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"
OUTPUT_DIR = "complete_analysis"
TARGET_OUTCOME = "Lawsuit Success"
BATCH_SIZE = 10 # Process 10 documents at a time

def main():
    """Run NLP analysis in batches."""
    print("=" * 70)
    print("BATCH NLP ANALYSIS")
    print("=" * 70)
    
    # Load all documents from database
    print(f"\n[Step 1] Loading documents from: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM cleaned_documents")
    documents = [row[0] for row in cur.fetchall()]
    conn.close()
    
    total_docs = len(documents)
    print(f"[OK] Loaded {total_docs} documents")
    print(f"[Config] Batch size: {BATCH_SIZE}")
    print(f"[Config] Number of batches: {(total_docs + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    if not documents:
        print("[Error] No documents found in database!")
        return
    
    # Process in batches
    all_results = []
    
    for batch_idx in range(0, total_docs, BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, total_docs)
        batch = documents[batch_idx:batch_end]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        
        print("\n" + "=" * 70)
        print(f"BATCH {batch_num}/{total_batches}: Documents {batch_idx+1}-{batch_end}")
        print("=" * 70)
        
        try:
            # Run analysis on batch
            results = analyze_legal_documents(
                batch,
                output_dir=f"{OUTPUT_DIR}_batch_{batch_num}",
                target_outcome=TARGET_OUTCOME
            )
            
            all_results.append({
                'batch': batch_num,
                'docs_range': f"{batch_idx+1}-{batch_end}",
                'results': results
            })
            
            print(f"\n[[ok]] Batch {batch_num} complete!")
            print(f" Entities: {results['analysis']['overview']['num_nodes']}")
            print(f" Relations: {results['analysis']['overview']['num_edges']}")
            
        except Exception as e:
            print(f"\n[] Batch {batch_num} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    print("\n" + "=" * 70)
    print("MERGING BATCH RESULTS")
    print("=" * 70)
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    combined_path = output_path / "batch_results_combined.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[[ok]] All batches processed!")
    print(f" Completed batches: {len(all_results)}")
    print(f" Combined results: {combined_path}")
    print(f"\n[Note] Individual batch outputs in: {OUTPUT_DIR}_batch_N/")
    print(f"[Next] Merge all batch graphs for final analysis")


if __name__ == "__main__":
    main()
