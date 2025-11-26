#!/usr/bin/env python3
"""Draft motion to seal using the new memory-aware system.

This script uses the WorkflowStrategyExecutor and EpisodicMemoryBank to:
1. Create a motion to seal with full memory integration
2. Leverage past editing patterns and document metadata
3. Store all activity in the unified memory system
4. Create document in Google Docs with full tracking
"""

import asyncio
import logging
import os
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "writer_agents" / "code"))


async def draft_motion_with_memory():
    """Draft a motion to seal using the memory-aware system."""

    print("\n" + "="*70)
    print("📝 MOTION TO SEAL DRAFTING - WITH MEMORY SYSTEM")
    print("="*70)

    try:
        # Import the new system components
        from WorkflowStrategyExecutor import WorkflowStrategyExecutor, WorkflowStrategyConfig
        from EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryRetriever
        from insights import CaseInsights, Posterior, EvidenceItem
        from agents import ModelConfig
        from sk_config import SKConfig, create_sk_kernel

        print("\n✅ Imported all components")

        # 1. Initialize memory system
        print("\n📚 Initializing EpisodicMemoryBank...")
        memory_store = EpisodicMemoryBank(storage_path=Path("memory_store"))
        memory_retriever = EpisodicMemoryRetriever(memory_store)

        # Check if we have existing memories
        memory_count = len(memory_store)
        print(f"   Found {memory_count} existing memories")

        # 2. Retrieve relevant context from past work
        print("\n🔍 Searching for relevant context from past work...")
        relevant_context = memory_retriever.get_all_relevant_context(
            query="motion to seal document drafting patterns and edits",
            k=5,
            include_types=["edit", "document", "conversation"]
        )

        if relevant_context:
            print(f"   Found {len(relevant_context)} relevant memories")
            for idx, mem in enumerate(relevant_context, 1):
                print(f"   {idx}. {mem['memory_type']}: {mem['summary'][:80]}...")
        else:
            print("   No previous context found (this will be the first motion)")

        # 3. Create configuration with memory enabled
        print("\n⚙️  Creating configuration...")

        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try to load from file
            key_file = Path(".openai_api_key.txt")
            if key_file.exists():
                api_key = key_file.read_text(encoding="utf-8").strip()
            else:
                logger.warning("OpenAI API key not found. Using mock configuration.")
                api_key = "sk-mock-key-for-testing"

        # Create AutoGen configuration
        autogen_config = ModelConfig(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=4096
        )

        # Create SK configuration
        sk_config = SKConfig(
            model_name="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000,
            api_key=api_key
        )

        # Create workflow configuration
        config = WorkflowStrategyConfig(
            autogen_config=autogen_config,
            sk_config=sk_config,
            google_docs_enabled=True,
            google_drive_folder_id="1MZwep4pb9M52lSLLGQAd3quslA8A5iBu",
            google_docs_auto_share=True,
            google_docs_capture_version_history=True,
            google_docs_learning_enabled=True,
            memory_system_enabled=True,
            memory_storage_path="memory_store",
            memory_context_max_items=5,
            memory_context_types=["execution", "edit", "document", "query", "conversation"],
            enable_sk_planner=False  # Disable planner for now
        )

        print("   ✓ Configuration created with:")
        print(f"     - Memory system: Enabled ({memory_count} existing memories)")
        print(f"     - Google Docs: Enabled")
        print(f"     - Memory context: {relevant_context[:2] if relevant_context else 'None'}")

        # 4. Initialize WorkflowStrategyExecutor
        print("\n🤖 Initializing WorkflowStrategyExecutor...")
        orchestrator = WorkflowStrategyExecutor(config)

        # Note: The orchestrator should already have memory_store integration
        # via the new system

        # 5. Create case insights for the motion
        print("\n📋 Creating case insights for motion to seal...")

        insights = CaseInsights(
            reference_id="MOTION_TO_SEAL_001",
            summary="Motion to seal sensitive personal information under Federal Rule of Civil Procedure 5.2",
            posteriors=[
                Posterior(
                    element="Personal Information Privacy",
                    probability=0.95,
                    reasoning="Highly sensitive personal information requires protection from public disclosure",
                    evidence=[
                        EvidenceItem(
                            source="FRCP_5.2",
                            content="Federal Rule of Civil Procedure 5.2 permits courts to allow filing under seal when privacy or safety interests outweigh public access",
                            relevance_score=1.0
                        ),
                        EvidenceItem(
                            source="Existing_Memory",
                            content=("Past motions successfully sealed when extreme personal sensitivity exists"),
                            relevance_score=0.9
                        )
                    ]
                )
            ],
            evidence=[
                EvidenceItem(
                    source="Legal_Standard",
                    content="Federal courts have discretion to seal documents when necessary to protect privacy interests",
                    relevance_score=1.0
                ),
                EvidenceItem(
                    source="Memory_Context",
                    content=str(relevant_context) if relevant_context else "No prior context",
                    relevance_score=0.7
                )
            ],
            case_style="Federal District Court",
            jurisdiction="D. Mass."
        )

        print("   ✓ Case insights created")

        # 6. Run the workflow
        print("\n🚀 Running memory-aware workflow...")
        print("-" * 70)

        result = await orchestrator.run_hybrid_workflow(insights)

        print("-" * 70)

        # 7. Display results
        print("\n📊 RESULTS")
        print("="*70)

        if result:
            print(f"✅ Workflow completed successfully!")
            print(f"\n   Phase: {result.get('phase', 'unknown')}")
            print(f"   Iterations: {result.get('iterations', 'unknown')}")

            if 'document_url' in result:
                print(f"\n   📄 Document created: {result['document_url']}")

            if 'document_id' in result:
                print(f"   📝 Document ID: {result['document_id']}")

            print(f"\n   🧠 Memories stored: {len(memory_store)} total")

            # Show what was learned
            if relevant_context:
                print("\n   📚 Learned from previous work:")
                for mem in relevant_context[:3]:
                    print(f"      - {mem['summary'][:60]}...")
        else:
            print("⚠️  Workflow completed but no result returned")

        # 8. Query the memory system to see what was saved
        print("\n🔍 Querying memory system for newly created memories...")
        new_memories = memory_store.retrieve(
            agent_type="WorkflowStrategyExecutor",
            query="motion to seal drafting",
            k=3
        )

        if new_memories:
            print(f"   Found {len(new_memories)} new memories:")
            for mem in new_memories:
                print(f"      {mem.memory_type}: {mem.summary[:60]}...")

        print("\n" + "="*70)
        print("✅ MOTION DRAFTING COMPLETE")
        print("="*70)
        print("\n📝 Next steps:")
        print("   1. Check your Google Drive folder for the new document")
        print("   2. Review the document and customize as needed")
        print("   3. The system has learned from this drafting session")
        print("   4. Future motions will leverage these learnings")
        print()

        return result

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure you're running from the project root")
        return None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Failed to draft motion")
        return None


if __name__ == "__main__":
    result = asyncio.run(draft_motion_with_memory())

    if result:
        print("\n🎉 Success! Your motion has been drafted with full memory integration.")
    else:
        print("\n⚠️  Motion drafting failed. Check the logs for details.")

