# Agent 4 – Workflow Testing & Embedding Fix

**Status:** Integration complete, workflow execution blocked by embedding download

## Goal

Unblock workflow execution to test coverage validation in real workflow.

## Problem

**File:** `writer_agents/code/embeddings.py` line 42

**Current (buggy):**
```python
self.model = SentenceTransformer(model_name) if allow_network or True else None
```

The `or True` makes it always try to load the model, even when network is disabled.

## Fix

**Replace lines 38-47 in `writer_agents/code/embeddings.py`:**

```python
if use_local:
    if allow_network:
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.mode = "local"
            logger.info(f"Using local embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Local sentence-transformers unavailable ({e}); falling back to network policy")
            self.use_local = False
    else:
        # Network disabled - use offline fallback immediately
        logger.info(f"MATRIX_ENABLE_NETWORK_MODELS disabled; using offline embedder (skipping {model_name})")
        self.model = None
        self.mode = "offline"
        self.use_local = False  # Prevent retry
```

## Test After Fix

```bash
# Test offline mode works
MATRIX_ENABLE_NETWORK_MODELS=0 HF_HUB_OFFLINE=1 python writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal sensitive information"

# Check logs for coverage metrics
grep -i "personal_facts_coverage" logs/*.log
grep -i "personal_facts_verification" logs/*.log
```

## Success Criteria

- ✅ Workflow runs without trying to download model
- ✅ Reaches validation phase
- ✅ Coverage metrics appear in logs
- ✅ Validation results include `personal_facts_verification`
