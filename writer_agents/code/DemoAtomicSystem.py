"""Demo of the atomic agent architecture.

Demonstrates the citation pipeline with 5 deterministic agents.
Shows zero-cost processing with full auditability.
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_citation_pipeline():
    """Demonstrate the citation processing pipeline."""

    print("\n" + "=" * 80)
    print("ATOMIC AGENT ARCHITECTURE DEMO")
    print("Citation Pipeline - 100% Deterministic (Zero LLM Cost)")
    print("=" * 80 + "\n")

    # Sample legal text with citations
    sample_text = """
    The Supreme Court held in Brown v. Board of Education, 347 U.S. 483 (1954),
    that separate educational facilities are inherently unequal. This principle
    was further developed in Green v. County School Board, 391 U.S. 430 (1968).

    Under 42 U.S.C. Section 1983, individuals can sue for violations of their civil rights.
    The statute provides that "every person who, under color of any statute...
    shall be liable to the party injured."

    Federal courts apply Fed. R. Civ. P. 12(b)(6) when evaluating motions to dismiss.
    """

    print("Sample Input Text:")
    print("-" * 80)
    print(sample_text)
    print("-" * 80 + "\n")

    # Import atomic agents
    from .atomic_agents.citations import (
        CitationFinderAgent,
        CitationNormalizerAgent,
        CitationVerifierAgent,
        CitationLocatorAgent,
        CitationInserterAgent,
    )
    from .agents import AgentFactory, ModelConfig

    # Create agent factory
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))

    # Instantiate agents
    finder = CitationFinderAgent(factory)
    normalizer = CitationNormalizerAgent(factory)
    verifier = CitationVerifierAgent(factory)
    locator = CitationLocatorAgent(factory)
    inserter = CitationInserterAgent(factory)

    print(" Step 1: Finding Citations...")
    print("-" * 80)

    result1 = await finder.execute({'text': sample_text})

    print(f"Found {result1['count']} citations:")
    for cite in result1['citations']:
        print(f" - {cite['type']:20s} -> {cite['text']}")

    print(f"\nBy type: {result1['by_type']}")
    print(f"Cost: $0.00 (deterministic)\n")

    print(" Step 2: Normalizing Citations...")
    print("-" * 80)

    result2 = await normalizer.execute({'citations': result1['citations']})

    print(f"Normalized {len(result2['normalized_citations'])} citations")
    if result2['changes']:
        print(f"Changes made: {result2['change_count']}")
        for change in result2['changes'][:3]: # Show first 3
            print(f" - {change['original']} -> {change['normalized']}")
    else:
        print(" No changes needed (already normalized)")

    print(f"Cost: $0.00 (deterministic)\n")

    print(" Step 3: Verifying Citations...")
    print("-" * 80)

    result3 = await verifier.execute({
        'normalized_citations': result2['normalized_citations']
    })

    print(f"Verified: {result3['verified_count']}/{len(result3['verified_citations'])}")
    print(f"Verification rate: {result3['verification_rate']:.1%}")
    print(f"Note: Actual DB verification not implemented in demo")
    print(f"Cost: $0.00 (deterministic)\n")

    print(" Step 4: Locating Citation Sources...")
    print("-" * 80)

    result4 = await locator.execute({
        'verified_citations': result3['verified_citations']
    })

    print(f"Located: {result4['located_count']}/{len(result4['located_citations'])}")
    print(f"Location rate: {result4['location_rate']:.1%}")
    print(f"Note: Actual DB lookup not implemented in demo")
    print(f"Cost: $0.00 (deterministic)\n")

    print(" Step 5: Inserting Formatted Citations...")
    print("-" * 80)

    result5 = await inserter.execute({
        'text': sample_text,
        'located_citations': result4['located_citations']
    })

    print(f"Insertions made: {result5['insertion_count']}")
    print(f"Cost: $0.00 (deterministic)\n")

    print("Final Output:")
    print("-" * 80)
    print(result5['output_text'][:500] + "...")
    print("-" * 80 + "\n")

    # Clean up
    await factory.close()

    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nTotal Cost: $0.00")
    print(f"Total Agents: 5 (all deterministic)")
    print(f"Total Citations Processed: {result1['count']}")
    print(f"Execution: 100% deterministic (no LLM calls)")
    print(f"\nKey Benefits:")
    print(f" [ok] Zero cost (no API calls)")
    print(f" [ok] Instant execution (no network latency)")
    print(f" [ok] 100% reproducible (deterministic)")
    print(f" [ok] Fully auditable (all steps logged)")
    print(f" [ok] Easily testable (pure functions)")
    print("\n")


async def demo_full_system():
    """Demonstrate the full MasterSupervisor system."""

    print("\n" + "=" * 80)
    print("FULL ATOMIC AGENT SYSTEM DEMO")
    print("MasterSupervisor with TaskDecomposer")
    print("=" * 80 + "\n")

    from .master_supervisor import MasterSupervisor, SupervisorConfig
    from .insights import CaseInsights

    # Create a simple case insights object
    insights = CaseInsights(
        posteriors={'outcome': {'win': 0.65, 'lose': 0.35}},
        evidence={'email_sent': 'yes', 'awareness': 'direct'},
        summary="Legal case analysis"
    )

    summary = """
    This case involves a dispute under 42 U.S.C. Section 1983.
    The precedent established in Smith v. Jones, 123 F.3d 456 (2020),
    supports the plaintiff's position. The defendant argues that
    Brown v. State, 789 F. Supp. 2d 321 (2021), requires dismissal
    under Fed. R. Civ. P. 12(b)(6).
    """

    print("Input Summary:")
    print("-" * 80)
    print(summary)
    print("-" * 80 + "\n")

    # Configure supervisor (citation only for demo)
    config = SupervisorConfig(
        enable_research=False, # Not implemented yet
        enable_drafting=False, # Not implemented yet
        enable_citation=True, # Fully implemented
        enable_qa=False, # Not implemented yet
    )

    # Create and run supervisor
    async with MasterSupervisor(session=None, config=config) as supervisor:
        print("Starting MasterSupervisor...")

        result = await supervisor.run(insights, summary)

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")

        print(f"Task Profile:")
        print(f" Evidence Count: {result['task_profile']['evidence_count']}")
        print(f" Complexity: {result['task_profile']['complexity_score']}")
        print(f" Report Length: {result['task_profile']['report_length']}")
        print()

        print(f"Execution Stats:")
        print(f" Total Tokens: {result['execution_stats']['total_tokens']}")
        print(f" Citation Tokens: {result['execution_stats']['citation_tokens']}")
        print()

        if result.get('final_text'):
            print(f"Final Text Preview:")
            print("-" * 80)
            print(result['final_text'][:300] + "...")
            print("-" * 80)

        print("\n[ok] Demo complete!")


async def main():
    """Run all demos."""

    # Demo 1: Citation pipeline (working)
    await demo_citation_pipeline()

    # Wait a moment
    await asyncio.sleep(1)

    # Demo 2: Full system (partial - only citation implemented)
    await demo_full_system()


if __name__ == "__main__":
    asyncio.run(main())

