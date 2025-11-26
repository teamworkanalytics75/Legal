# Codex Agent 4: Verify Fact Registry & Add Timeout Wrapper

**Workstream:** Motion Generation Reliability & Validation  
**Status:** Ready to start  
**Dependencies:** None (can work independently)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 4**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 4**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 4** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_4_FACTS_AND_TIMEOUT.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_1_PIPELINE_TEST.md` - Agent 1's tasks
- `CODEX_AGENT_2_COMMIT_BLOCKING.md` - Agent 2's tasks  
- `CODEX_AGENT_3_REFINEMENT_VERIFICATION.md` - Agent 3's tasks

---

## 🎯 Objective

1. Verify fact registry completeness (all 12 fact types extracted and accessible)
2. Add timeout wrapper to model loading to prevent future hangs

---

## 📋 Tasks

### Task 1: Verify All 12+ Fact Types Are Extracted

**File:** `writer_agents/scripts/extract_fact_registry.py`

**Check extractors:**
1. `citizenship` - ✅ Already verified
2. `hk_statement` - Verify extractor exists
3. `ogc_emails` - Verify extractor exists
4. `date_april_7_2025` - Verify extractor exists
5. `date_april_18_2025` - Verify extractor exists
6. `date_june_2_2025` - Verify extractor exists
7. `date_june_4_2025` - Verify extractor exists
8. `allegation_defamation` - Verify extractor exists
9. `allegation_privacy_breach` - Verify extractor exists
10. `allegation_retaliation` - Verify extractor exists
11. `allegation_harassment` - Verify extractor exists
12. `timeline_april_ogc_emails` - Verify extractor exists
13. `timeline_june_2025_arrests` - Verify extractor exists

**List all extractors:**
```bash
grep -E "@register_fact_extractor|def extract_" writer_agents/scripts/extract_fact_registry.py
```

**Run extraction:**
```bash
python writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db
```

**Verify database:**
```bash
sqlite3 case_law_data/lawsuit_facts_database.db "SELECT DISTINCT fact_type FROM fact_registry;"
```

**Expected:** Should show all 12+ fact types

### Task 2: Verify CaseFactsProvider Can Access All Facts

**File:** `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`

**Check method:** `get_fact_registry()` or similar

**Verify:**
- Method reads from `fact_registry` table
- Returns all fact types, not just citizenship
- Facts are properly formatted for prompts

**Test:**
```python
from writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider
provider = CaseFactsProvider(...)
facts = provider.get_fact_registry()
print(f"Found {len(facts)} fact types: {list(facts.keys())}")
```

**Expected:** Should show all 12+ fact types

### Task 3: Verify ContradictionDetector Can Access All Facts

**File:** `writer_agents/code/validation/contradiction_detector.py`

**Check:**
- `ContradictionDetector` loads facts from database
- Can detect contradictions for all fact types
- Not just checking citizenship

**Test:**
```python
from writer_agents.code.validation.contradiction_detector import ContradictionDetector
detector = ContradictionDetector(database_path="...")
print(f"Loaded {len(detector.fact_registry)} fact types")
```

**Expected:** Should load all fact types from database

### Task 4: Verify Personal Facts Gate Works with Full Fact Set

**Check orchestrator:**
- Personal facts gate uses all facts from registry
- Validation checks all fact types, not just citizenship
- Missing facts are reported correctly

**Grep for fact usage:**
```bash
grep -E "fact_registry|get_fact_registry|personal_facts" writer_agents/code/WorkflowOrchestrator.py | head -20
```

### Task 5: Add Timeout Wrapper to Model Loading

**File:** `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py` (lines 104-132)

**Current issue:** Model loading can hang if network is slow/unavailable, even with `local_files_only=True`

**Solution:** Add timeout wrapper using `ThreadPoolExecutor` (cross-platform)

**Implementation:**

1. **Add import at top of file:**
```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
```

2. **Modify `_load_resources` method (around line 115):**
```python
# Load BERT model for query encoding
if TRANSFORMERS_AVAILABLE:
    try:
        hf_offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        local_files_only = hf_offline or not _NETWORK_MODELS_ENABLED
        if local_files_only:
            logger.info(
                "Loading BERT model in offline mode (local_files_only=True): %s",
                self.model_name,
            )

        # Add timeout to prevent hanging (5 seconds max)
        def load_model():
            return (
                AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=local_files_only,
                ),
                AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=local_files_only,
                )
            )
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_model)
                self._tokenizer, self._model = future.result(timeout=5.0)  # 5 second timeout
        except FutureTimeoutError:
            logger.warning("Model loading timed out after 5 seconds, falling back to keyword search")
            raise Exception("Model loading timed out")
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Model not available locally: {e}")
            raise Exception(f"Model not available locally: {e}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Loaded BERT model: {self.model_name} on {self._device}")
        self._semantic_enabled = True
    except Exception as e:
        logger.warning(f"Failed to load BERT model: {e}")
        self._model = None
        self._tokenizer = None
        self._fallback_reason = str(e)
```

### Task 6: Test Timeout Wrapper

**Test scenario:**
- Simulate slow model loading (or use non-existent model)
- Verify timeout triggers after 5 seconds
- Verify graceful fallback to keyword search

**Manual test:**
- Set `local_files_only=True` with non-existent model
- Verify timeout occurs and fallback works
- Check logs show timeout message

**Test command:**
```bash
HF_HUB_OFFLINE=1 MATRIX_ENABLE_NETWORK_MODELS=0 python -c "
from writer_agents.code.sk_plugins.FeaturePlugin.EmbeddingRetriever import EmbeddingRetriever
retriever = EmbeddingRetriever(model_name='non-existent-model-name')
# Should timeout or fail gracefully, not hang
"
```

---

## 📁 Key Files

- `writer_agents/scripts/extract_fact_registry.py` - Fact extraction (all 12+ extractors)
- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py` - Facts provider
- `writer_agents/code/validation/contradiction_detector.py` - Contradiction detector
- `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py` - Embedding retriever (add timeout)
- `case_law_data/lawsuit_facts_database.db` - Fact registry database

---

## ✅ Success Criteria

- [ ] All 12+ fact types are extracted and stored in database
- [ ] `CaseFactsProvider` can access all fact types
- [ ] `ContradictionDetector` can access all fact types
- [ ] Personal facts gate uses full fact set
- [ ] Timeout wrapper added to model loading (5 second timeout)
- [ ] Timeout gracefully falls back to keyword search
- [ ] No regression: existing functionality still works

---

## 🚨 Troubleshooting

**If fact types are missing:**
- Check extractor functions are registered correctly
- Verify source documents contain the facts
- Check database schema matches expected format

**If timeout doesn't work:**
- Verify `ThreadPoolExecutor` import is correct
- Check timeout value (5 seconds should be enough)
- Verify exception handling catches `FutureTimeoutError`

**If fallback doesn't work:**
- Check `_semantic_enabled` flag is set correctly
- Verify keyword search path is still functional
- Check logs for fallback messages

---

## 📝 Progress Tracking

- [ ] Task 1: Verify all fact types extracted
- [ ] Task 2: Verify CaseFactsProvider access
- [ ] Task 3: Verify ContradictionDetector access
- [ ] Task 4: Verify personal facts gate
- [ ] Task 5: Add timeout wrapper
- [ ] Task 6: Test timeout wrapper
