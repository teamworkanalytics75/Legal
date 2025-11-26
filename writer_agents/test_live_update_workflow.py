#!/usr/bin/env python3
"""Test live update mechanism directly."""

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

async def test_live_update():
    """Test live update mechanism."""

    DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

    print("Testing live update mechanism...")
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

    # Create minimal state
    state = WorkflowState()
    state.google_doc_id = DOC_ID
    state.google_doc_url = f"https://docs.google.com/document/d/{DOC_ID}/edit"
    state.phase = WorkflowPhase.EXPLORE
    state.iteration = 0
    state.draft_result = None  # No draft yet, should show progress message

    print(f"\nState created:")
    print(f"  Phase: {state.phase.value}")
    print(f"  Doc ID: {state.google_doc_id}")
    print(f"  Draft result: {state.draft_result}")

    # Test live update
    print(f"\nCalling _update_google_doc_live...")
    try:
        await conductor._update_google_doc_live(state, None)
        print("[SUCCESS] Live update called successfully!")
    except Exception as e:
        print(f"[ERROR] Live update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_live_update())
