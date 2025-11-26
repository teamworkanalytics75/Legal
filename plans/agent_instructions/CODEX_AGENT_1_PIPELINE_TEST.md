# Codex Agent 1: Test Pipeline & Verify Offline Mode Fix

**Workstream:** Motion Generation Pipeline Testing  
**Status:** Blocked (waiting for local Ollama or network access during AutoGen phase)  
**Dependencies:** None (can work independently)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 1**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 1**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 1** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_1_PIPELINE_TEST.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_2_COMMIT_BLOCKING.md` - Agent 2's tasks  
- `CODEX_AGENT_3_REFINEMENT_VERIFICATION.md` - Agent 3's tasks
- `CODEX_AGENT_4_FACTS_AND_TIMEOUT.md` - Agent 4's tasks

---

## 🎯 Objective

Test the full pipeline with the offline mode fix and verify that `EmbeddingRetriever` no longer hangs on HuggingFace downloads.

---

## 📡 Status Update — 2025-11-15

- Ran the prescribed command twice (120 s & 600 s timeouts) and captured the latest run at `/tmp/agent1_pipeline_test.log`.
- Offline embedder logged immediately (`MATRIX_ENABLE_NETWORK_MODELS disabled; using offline embedder`) and no HuggingFace downloads were attempted (`/tmp/agent1_pipeline_test.log:14-15`).
- `EmbeddingRetriever` loaded the FAISS index instantly and never hung on model loads (`/tmp/agent1_pipeline_test.log:759-760`).
- Workflow stalled after `Invoking AutoGen exploration agent...` because the Ollama server is down and the fallback OpenAI call cannot reach the network, so the script blocks indefinitely (`/tmp/agent1_pipeline_test.log:775-779`).
- Next action: start the local Ollama server (`ollama serve` + `ollama run qwen2.5:14b`) or grant temporary network access so the OpenAI fallback can respond.

---

## 📋 Tasks

### Task 1: Run Full Pipeline Test with Offline Flags

**Command:**
```bash
cd /home/serteamwork/projects/TheMatrix
source .venv/bin/activate
HF_HUB_OFFLINE=1 MATRIX_ENABLE_NETWORK_MODELS=0 python writer_agents/scripts/generate_optimized_motion.py \
    --case-summary "Motion to seal and pseudonym for Section 1782 discovery case" \
    --max-iterations 5 \
    2>&1 | tee /tmp/agent1_pipeline_test.log
```

**Expected Behavior:**
- Pipeline should start without hanging
- `EmbeddingRetriever` should log "Loading BERT model in offline mode" or "Failed to load BERT model" (graceful fallback)
- Pipeline should complete or fail gracefully (not timeout)
- No repeated "Failed to resolve huggingface.co" errors

### Task 2: Verify EmbeddingRetriever Behavior

**File to check:** `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py` (lines 104-132)

**Check for:**
- Log message: "Loading BERT model in offline mode (local_files_only=True)"
- If model not cached: Should log warning and fall back to keyword search
- No hanging/blocking on network calls

**Grep for logs:**
```bash
grep -E "(offline mode|Failed to load BERT|local_files_only)" /tmp/agent1_pipeline_test.log
```

### Task 3: Verify Pipeline Completion

**Check pipeline output:**
- Did it complete? (look for "Motion Generation Complete" or final error)
- How long did it take? (should not timeout at 120s)
- What phase did it reach? (EXPLORE, DRAFT, VALIDATE, REFINE, COMMIT)

**Grep for completion:**
```bash
grep -E "(Motion Generation Complete|WorkflowPhase|iteration|phase)" /tmp/agent1_pipeline_test.log | tail -20
```

### Task 4: Report Results

**Report format:**
- ✅ Success: Pipeline completed without hanging
- ⚠️ Partial: Pipeline ran but encountered errors (list them)
- ❌ Failed: Pipeline still hangs or times out

**Include:**
- Pipeline completion status
- Any errors encountered
- Time taken
- Final phase reached
- Recommendations for fixes

---

## 📁 Key Files

- `writer_agents/scripts/generate_optimized_motion.py` - Main pipeline script
- `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py` - Embedding retriever (offline mode fix)
- `/tmp/agent1_pipeline_test.log` - Test output log

---

## ✅ Success Criteria

- [ ] Pipeline runs without hanging on HuggingFace downloads
- [x] `EmbeddingRetriever` gracefully falls back to keyword search if model not cached
- [ ] Pipeline completes or fails with clear error (not timeout)
- [x] Logs show offline mode is working correctly

---

## 🚨 Troubleshooting

**If pipeline still hangs:**
- Check if Ollama server is running: `curl http://localhost:11434/api/tags`
- Check if model is cached: `ls -la ~/.cache/huggingface/transformers/`
- Check environment variables: `echo $HF_HUB_OFFLINE $MATRIX_ENABLE_NETWORK_MODELS`

**If errors occur:**
- Note the error message and stack trace
- Check which phase failed
- Report to team for fixes

---

## 📝 Progress Tracking

- [x] Task 1: Pipeline test run
- [x] Task 2: EmbeddingRetriever verification
- [ ] Task 3: Pipeline completion check
- [x] Task 4: Results reported
