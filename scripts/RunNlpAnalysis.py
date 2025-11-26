"""
run_nlp_production.py
Production NLP analysis with batching, checkpointing, and resume capability.

This script processes 218 documents in small batches to prevent hangs and
enables resuming from where it left off if interrupted.
"""

import sqlite3
import json
import time
from pathlib import Path
from nlp_analysis.pipeline import analyze_legal_documents

# === CONFIGURATION ===
DB_PATH = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"
OUTPUT_BASE = "complete_analysis"
BATCH_SIZE = 25 # Small batches for stability
TARGET_OUTCOME = "Lawsuit Success"

def get_completed_batches(output_base: str) -> set:
    """Check which batches have already been completed."""
    completed = set()
    base_path = Path(output_base)
    
    if not base_path.exists():
        return completed
    
    # Look for batch completion markers
    for batch_dir in base_path.glob("batch_*"):
        if batch_dir.is_dir():
            # Check if batch has outputs
            if (batch_dir / "knowledge_graph.json").exists():
                batch_num = int(batch_dir.name.split("_")[1])
                completed.add(batch_num)
    
    return completed

def save_batch_metadata(output_base: str, batch_num: int, 
                       doc_range: tuple, duration: float, 
                       success: bool, error: str = None):
    """Save metadata about batch processing."""
    meta_file = Path(output_base) / "batch_metadata.jsonl"
    meta = {
        "batch": batch_num,
        "doc_range": f"{doc_range[0]}-{doc_range[1]}",
        "duration_seconds": round(duration, 2),
        "success": success,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(meta_file, 'a') as f:
        f.write(json.dumps(meta) + "\n")

def main():
    print("=" * 80)
    print("PRODUCTION NLP ANALYSIS - BATCH MODE WITH RESUME")
    print("=" * 80)
    
    # Load all documents
    print(f"\n[1/5] Loading documents from database...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM cleaned_documents")
    documents = [row[0] for row in cur.fetchall()]
    conn.close()
    
    total_docs = len(documents)
    num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"[OK] Loaded {total_docs} documents")
    print(f"[Config] Batch size: {BATCH_SIZE} docs")
    print(f"[Config] Total batches: {num_batches}")
    
    # Check for completed batches
    print(f"\n[2/5] Checking for completed batches...")
    completed = get_completed_batches(OUTPUT_BASE)
    
    if completed:
        print(f"[Resume] Found {len(completed)} completed batches: {sorted(completed)}")
        print(f"[Resume] Will skip these and continue from batch {max(completed) + 1}")
    else:
        print(f"[Fresh Start] No previous batches found")
    
    # Create output directory
    Path(OUTPUT_BASE).mkdir(exist_ok=True)
    
    # Process batches
    print(f"\n[3/5] Processing batches...")
    print("=" * 80)
    
    processed_count = 0
    failed_count = 0
    
    for batch_idx in range(0, total_docs, BATCH_SIZE):
        batch_num = batch_idx // BATCH_SIZE + 1
        
        # Skip if already completed
        if batch_num in completed:
            print(f"[Skip] Batch {batch_num}/{num_batches} already completed")
            continue
        
        batch_end = min(batch_idx + BATCH_SIZE, total_docs)
        batch = documents[batch_idx:batch_end]
        doc_range = (batch_idx + 1, batch_end)
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}/{num_batches}: Documents {doc_range[0]}-{doc_range[1]}")
        print(f"{'='*80}")
        
        batch_output = f"{OUTPUT_BASE}/batch_{batch_num}"
        batch_start = time.time()
        
        try:
            # Process batch with timeout awareness
            print(f"[Processing] {len(batch)} documents in batch {batch_num}...")
            
            results = analyze_legal_documents(
                batch,
                output_dir=batch_output,
                target_outcome=TARGET_OUTCOME
            )
            
            batch_duration = time.time() - batch_start
            
            # Success!
            print(f"\n[[ok]] Batch {batch_num} COMPLETE in {batch_duration:.1f}s")
            print(f" Entities: {results['analysis']['overview']['num_nodes']}")
            print(f" Relations: {results['analysis']['overview']['num_edges']}")
            print(f" Evidence: {results['bayesian_network']['summary']['total_evidence']}")
            
            save_batch_metadata(OUTPUT_BASE, batch_num, doc_range, 
                              batch_duration, True)
            processed_count += 1
            
        except KeyboardInterrupt:
            print(f"\n[!] Interrupted by user. Batch {batch_num} incomplete.")
            print(f"[Resume] Run script again to continue from batch {batch_num}")
            break
            
        except Exception as e:
            batch_duration = time.time() - batch_start
            error_msg = str(e)
            
            print(f"\n[] Batch {batch_num} FAILED after {batch_duration:.1f}s")
            print(f" Error: {error_msg}")
            
            save_batch_metadata(OUTPUT_BASE, batch_num, doc_range,
                              batch_duration, False, error_msg)
            failed_count += 1
            
            # Continue with next batch instead of stopping
            print(f"[Continue] Moving to next batch...")
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f" Batches processed this run: {processed_count}")
    print(f" Batches failed: {failed_count}")
    print(f" Total completed batches: {len(completed) + processed_count}")
    print(f" Output directory: {Path(OUTPUT_BASE).absolute()}")
    
    # Check if all done
    all_completed = get_completed_batches(OUTPUT_BASE)
    if len(all_completed) == num_batches:
        print(f"\n[[ok][ok][ok]] ALL {num_batches} BATCHES COMPLETE!")
        print(f"\n[4/5] Next step: Merge batch results")
        print(f" Run: py merge_batch_results.py")
    else:
        remaining = num_batches - len(all_completed)
        print(f"\n[Remaining] {remaining} batches still need processing")
        print(f"[Resume] Run this script again to continue")
    
    # Show batch locations
    print(f"\n[5/5] Batch outputs:")
    for i in sorted(all_completed):
        batch_path = Path(OUTPUT_BASE) / f"batch_{i}"
        if batch_path.exists():
            files = list(batch_path.glob("*.json")) + list(batch_path.glob("*.html"))
            print(f" batch_{i}/ ({len(files)} files)")


if __name__ == "__main__":
    main()
