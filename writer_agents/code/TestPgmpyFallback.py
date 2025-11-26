"""Test suite for pgmpy-based Bayesian inference fallback.

This module tests the XDSL parser and pgmpy inference engine to ensure
they work correctly as a fallback when PySMILE is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_xdsl_parsing():
    """Test that XDSL files can be parsed correctly."""
    print("\n=== Test 1: XDSL Parsing ===")
    
    from writer_agents.xdsl_parser import parse_xdsl, validate_network
    
    # Find the XDSL model file.
    model_path = BASE_DIR / "experiments" / "WizardWeb1.1.3.xdsl"
    
    if not model_path.exists():
        print(f"x Model file not found: {model_path}")
        return False
    
    print(f" Parsing {model_path.name}...")
    
    try:
        network = parse_xdsl(model_path)
        print(f"[ok] Parsed successfully")
        print(f" Network ID: {network.network_id}")
        print(f" Nodes: {len(network.nodes)}")
        print(f" Edges: {len(network.edges)}")
        
        # Show some example nodes.
        print("\n Sample nodes:")
        for i, (node_id, node_info) in enumerate(list(network.nodes.items())[:5]):
            print(f" - {node_id}: {len(node_info.states)} states, {len(node_info.parents)} parents")
        
        # Validate the network.
        print("\n Validating network structure...")
        validate_network(network)
        print("[ok] Network validation passed")
        
        return True
        
    except Exception as exc:
        print(f"x Parsing failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_pgmpy_inference():
    """Test that pgmpy can build and run inference on the network."""
    print("\n=== Test 2: pgmpy Inference ===")
    
    try:
        from writer_agents.pgmpy_inference import HAVE_PGMPY
        
        if not HAVE_PGMPY:
            print("WARNING pgmpy not available - install with: pip install pgmpy")
            return False
        
        from writer_agents.pgmpy_inference import build_bn_from_xdsl, run_inference
        from writer_agents.xdsl_parser import parse_xdsl
        
        # Parse the network.
        model_path = BASE_DIR / "experiments" / "WizardWeb1.1.3.xdsl"
        print(f" Loading {model_path.name}...")
        network = parse_xdsl(model_path)
        
        # Build the Bayesian network.
        print(" Building pgmpy BayesianNetwork...")
        model = build_bn_from_xdsl(network)
        print(f"[ok] Network built with {len(model.nodes())} nodes")
        
        # Define test evidence.
        evidence = {
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
        }
        
        print(f"\n Running inference with evidence:")
        for node, state in evidence.items():
            print(f" - {node} = {state}")
        
        # Run inference.
        posterior_data = run_inference(model, evidence)
        
        print(f"\n[ok] Inference complete: {len(posterior_data)} posterior distributions computed")
        
        # Show some example posteriors.
        print("\n Sample posterior probabilities:")
        interesting_nodes = [
            "LegalSuccess_US",
            "LegalSuccess_HK", 
            "FinancialDamage",
            "ReputationalHarm",
        ]
        
        for node_id in interesting_nodes:
            if node_id in posterior_data:
                probs = posterior_data[node_id]
                prob_str = ", ".join(f"{state}: {prob:.3f}" for state, prob in probs.items())
                print(f" {node_id}: {prob_str}")
        
        return True
        
    except Exception as exc:
        print(f"x Inference failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test the full integration with run_pgmpy_inference."""
    print("\n=== Test 3: Full Integration ===")
    
    try:
        from writer_agents.pgmpy_inference import HAVE_PGMPY
        
        if not HAVE_PGMPY:
            print("WARNING pgmpy not available - skipping integration test")
            return False
        
        from writer_agents.pgmpy_inference import run_pgmpy_inference
        
        model_path = BASE_DIR / "experiments" / "WizardWeb1.1.3.xdsl"
        
        evidence = {
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
        }
        
        summary = "Test case for pgmpy inference integration"
        reference_id = "TEST-001"
        
        print(f" Running full inference pipeline...")
        insights, posterior_data = run_pgmpy_inference(
            model_path=model_path,
            evidence=evidence,
            summary=summary,
            reference_id=reference_id,
        )
        
        print(f"[ok] Integration successful")
        print(f" Reference ID: {insights.reference_id}")
        print(f" Summary: {insights.summary}")
        print(f" Posteriors: {len(insights.posteriors)}")
        print(f" Evidence items: {len(insights.evidence)}")
        
        # Show insights structure.
        print("\n Case Insights Structure:")
        print(f" jurisdiction: {insights.jurisdiction}")
        print(f" case_style: {insights.case_style}")
        
        if insights.posteriors:
            print(f"\n Sample posteriors:")
            for posterior in list(insights.posteriors)[:3]:
                prob_str = ", ".join(
                    f"{state}: {prob:.3f}"
                    for state, prob in posterior.probabilities.items()
                )
                print(f" - {posterior.node_id}: {prob_str}")
        
        return True
        
    except Exception as exc:
        print(f"x Integration test failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_mechanism():
    """Test the automatic fallback in bn_adapter."""
    print("\n=== Test 4: Automatic Fallback Mechanism ===")
    
    try:
        from writer_agents.bn_adapter import run_bn_inference_with_fallback
        from writer_agents.pgmpy_inference import HAVE_PGMPY
        
        if not HAVE_PGMPY:
            print("WARNING pgmpy not available - skipping fallback test")
            return False
        
        model_path = BASE_DIR / "experiments" / "WizardWeb1.1.3.xdsl"
        
        evidence = {
            "OGC_Email_Apr18_2025": "Sent",
        }
        
        summary = "Test case for automatic fallback"
        reference_id = "FALLBACK-001"
        
        print(f" Testing automatic fallback...")
        insights, posterior_data = run_bn_inference_with_fallback(
            model_path=model_path,
            evidence=evidence,
            summary=summary,
            reference_id=reference_id,
        )
        
        print(f"[ok] Fallback mechanism working")
        print(f" Computed {len(posterior_data)} posterior distributions")
        print(f" Insights reference: {insights.reference_id}")
        
        return True
        
    except Exception as exc:
        print(f"x Fallback test failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("pgmpy Bayesian Inference Fallback Test Suite")
    print("=" * 60)
    
    results = {
        "XDSL Parsing": test_xdsl_parsing(),
        "pgmpy Inference": test_pgmpy_inference(),
        "Full Integration": test_full_integration(),
        "Automatic Fallback": test_fallback_mechanism(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "[ok] PASS" if passed else "x FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print(" All tests passed!")
    else:
        print("WARNING Some tests failed")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

