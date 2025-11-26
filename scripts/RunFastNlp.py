"""
run_nlp_ultra_fast.py
ULTRA-FAST MODE: Process only entities, skip expensive analysis.

This extracts just the entities and basic relations from documents,
skipping causal inference and Bayesian evidence extraction which are slow.
Use this to get through all 218 documents quickly, then do deeper
analysis on important docs later.
"""

import sqlite3
import json
import time
from pathlib import Path
import spacy

# Configuration
DB_PATH = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"
OUTPUT_DIR = "complete_analysis_fast"
CHUNK_SIZE = 10000 # Process 10K chars at a time

# Load minimal spaCy
print("[Init] Loading spaCy (NER only)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "lemmatizer"])
print("[OK] SpaCy loaded")

def extract_entities_fast(text: str) -> list:
    """Extract entities using chunked processing."""
    entities = []
    
    # Process in 10K character chunks
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE]
        doc = nlp(chunk)
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "offset": i + ent.start_char
            })
    
    return entities

def analyze_document_fast(text: str) -> dict:
    """Fast analysis - entities only."""
    entities = extract_entities_fast(text)
    
    # Deduplicate entities
    unique_ents = {}
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in unique_ents:
            unique_ents[key] = ent
    
    return {
        "entities": list(unique_ents.values()),
        "num_entities": len(unique_ents),
        "text_length": len(text)
    }

def main():
    print("=" * 70)
    print("ULTRA-FAST NLP - ENTITIES ONLY")
    print("=" * 70)
    
    # Load documents
    print(f"\n[1/3] Loading documents...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT content FROM cleaned_documents")
    documents = [row[0] for row in cur.fetchall()]
    conn.close()
    
    total_docs = len(documents)
    print(f"[OK] Loaded {total_docs} documents")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Process documents
    print(f"\n[2/3] Processing documents (fast mode)...")
    print("=" * 70)
    
    all_entities = {}
    doc_results = []
    total_start = time.time()
    
    for i, doc in enumerate(documents, 1):
        start_t = time.time()
        
        try:
            result = analyze_document_fast(doc)
            doc_results.append({
                "doc_id": i,
                "num_entities": result["num_entities"],
                "text_length": result["text_length"]
            })
            
            # Accumulate entities
            for ent in result["entities"]:
                key = (ent["text"], ent["label"])
                if key not in all_entities:
                    all_entities[key] = {
                        "text": ent["text"],
                        "label": ent["label"],
                        "count": 0,
                        "docs": []
                    }
                all_entities[key]["count"] += 1
                all_entities[key]["docs"].append(i)
            
            dur = time.time() - start_t
            
            if i % 10 == 0 or i == 1 or i == total_docs:
                print(f"[{i}/{total_docs}] {result['num_entities']} entities in {dur:.2f}s")
        
        except Exception as e:
            print(f"[] Doc {i} failed: {e}")
            continue
    
    total_dur = time.time() - total_start
    
    # Save results
    print(f"\n[3/3] Saving results...")
    
    entities_file = output_path / "entities_all.json"
    with open(entities_file, 'w') as f:
        json.dump(list(all_entities.values()), f, indent=2)
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_documents": total_docs,
            "total_entities": len(all_entities),
            "processing_time_seconds": round(total_dur, 2),
            "avg_time_per_doc": round(total_dur / total_docs, 2),
            "doc_results": doc_results
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f" Documents processed: {total_docs}")
    print(f" Unique entities found: {len(all_entities)}")
    print(f" Total time: {total_dur/60:.1f} minutes")
    print(f" Avg per document: {total_dur/total_docs:.2f} seconds")
    print(f"\nOutput files:")
    print(f" {entities_file}")
    print(f" {summary_file}")
    
    # Top entities
    print(f"\nTop 20 entities by frequency:")
    sorted_ents = sorted(all_entities.values(), key=lambda x: x["count"], reverse=True)
    for ent in sorted_ents[:20]:
        print(f" {ent['text']:30s} [{ent['label']:12s}] ({ent['count']} occurrences)")


if __name__ == "__main__":
    main()
