#!/usr/bin/env python3
"""Manually trigger a live update to prove it works."""

import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))

from code.WorkflowOrchestrator import Conductor, WorkflowPhase, WorkflowState
from code.workflow_config import WorkflowStrategyConfig
from code.insights import CaseInsights
from code.agents import ModelConfig
from code.sk_config import SKConfig

async def manual_update():
    """Manually trigger live update."""

    DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

    print("Manually triggering live update...")
    print("=" * 80)

    # Create minimal config
    config = WorkflowStrategyConfig(
        autogen_config=ModelConfig(model="gpt-4o-mini"),
        sk_config=SKConfig(use_local=True),
        google_docs_enabled=True,
        google_docs_live_updates=True,
        master_draft_mode=True,
        master_draft_title="Motion for Seal and Pseudonym - Master Draft"
    )

    # Create conductor
    conductor = Conductor(config)

    # Wait for initialization
    await asyncio.sleep(2)

    # Create state with EXPLORE phase
    state = WorkflowState()
    state.google_doc_id = DOC_ID
    state.google_doc_url = f"https://docs.google.com/document/d/{DOC_ID}/edit"
    state.phase = WorkflowPhase.EXPLORE
    state.iteration = 0
    state.draft_result = None

    print(f"\nCalling _update_google_doc_live with:")
    print(f"  Phase: {state.phase.value}")
    print(f"  Doc ID: {state.google_doc_id}")
    print(f"  Draft result: {state.draft_result}")

    try:
        await conductor._update_google_doc_live(state, None)
        print("\n[SUCCESS] Live update completed!")
        print("\nCheck your Google Doc - it should now show:")
        print("  [LIVE UPDATE - EXPLORE - Iteration 0]")
        print("  [EXPLORE Phase - Iteration 0]")
        print("  Workflow is currently in the explore phase...")
    except Exception as e:
        print(f"\n[ERROR] Live update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(manual_update())
