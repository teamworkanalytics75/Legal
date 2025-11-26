#!/usr/bin/env python3
"""
Quick runner script for national security analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add scripts to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from analyze_national_security_matter import analyze_national_security_matter

async def main():
    # Load statement of claim
    statement_path = Path(__file__).parent.parent / "case_law_data" / "tmp_corpus" / "Exhibit 2 — Certified Statement of Claim (Hong Kong, 2 Jun 2025).txt"

    if not statement_path.exists():
        print(f"ERROR: Statement file not found at {statement_path}")
        return

    with open(statement_path, 'r', encoding='utf-8') as f:
        statement_of_claim = f.read()

    print(f"Loaded statement of claim: {len(statement_of_claim)} characters")
    print("="*70)
    print("Running national security analysis...")
    print("="*70)

    # Run analysis with LOCAL LLMs (Ollama)
    print("Using LOCAL LLMs (Ollama) for both Autogen and Semantic Kernel")
    print("  Model: qwen2.5:14b")
    print("  Server: http://localhost:11434")
    print()

    results = await analyze_national_security_matter(
        statement_of_claim=statement_of_claim,
        jurisdiction="US Federal",
        autogen_model="qwen2.5:14b",  # Will be ignored if sk_use_local=True
        sk_use_local=True,  # Use local LLM for both Autogen and Semantic Kernel
        max_iterations=3
    )

    # Print results
    if results.get("success"):
        print("\n" + "="*70)
        print("NATIONAL SECURITY MATTER ANALYSIS - RESULTS")
        print("="*70)

        classification = results.get("national_security_classification", {})
        print(f"\nClassification Strength: {classification.get('classification_strength', 'unknown').upper()}")
        print(f"Confidence Level: {classification.get('confidence', 0):.1%}")
        print(f"\nAssessment: {classification.get('assessment', 'Analysis incomplete')}")

        print(f"\nNational Security Indicators Found: {classification.get('total_indicators', 0)}")
        if classification.get('indicators'):
            print(f"  - {', '.join(classification['indicators'][:10])}")

        print(f"\nLegal Frameworks Identified: {classification.get('total_frameworks', 0)}")
        if classification.get('legal_frameworks'):
            print(f"  - {', '.join(classification['legal_frameworks'])}")

        print("\n" + "-"*70)
        print("Full Analysis Document Preview:")
        print("-"*70)
        analysis_doc = results.get("analysis_document", "")
        print(analysis_doc[:1500] + "..." if len(analysis_doc) > 1500 else analysis_doc)

        # Save results
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        import json
        output_path = output_dir / "national_security_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        analysis_path = output_dir / "national_security_analysis.txt"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(analysis_doc)
        print(f"Analysis document saved to: {analysis_path}")

    else:
        print("\n" + "="*70)
        print("ERROR - Analysis Failed")
        print("="*70)
        print(f"Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())

