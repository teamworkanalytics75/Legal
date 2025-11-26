# 🚀 Version Management Quick Start

**Problem:** Google Doc is appending instead of replacing content.

**Solution:** ✅ Use master draft mode with automatic version backups (recommended approach)

---

## ✅ Recommended: Master Draft + Backups

**Why this is better:**
1. ✅ One clean document to work with
2. ✅ Automatic backups before each update
3. ✅ ML training data automatically saved
4. ✅ Can recover any previous version
5. ✅ Tracks document evolution over time

---

## 🔧 How to Use

### Enable Version Backups (Default: ON)

```python
from writer_agents.code.WorkflowOrchestrator import WorkflowStrategyConfig

config = WorkflowStrategyConfig(
    master_draft_mode=True,
    master_draft_title="Motion for Seal and Pseudonym - Master Draft",

    # Version management (enabled by default)
    enable_version_backups=True,           # ✅ Create backups before updates
    save_backups_for_ml=True,              # ✅ Save to ML training directory
    max_versions_to_keep=50                # Keep last 50 versions
)
```

### What Happens

**Every time you update the master draft:**
1. System captures current content from Google Docs
2. Creates backup: `outputs/master_drafts/versions/{doc_id}_{timestamp}.md`
3. Saves ML data: `outputs/ml_training_data/drafts/{doc_id}_{timestamp}.json`
4. **Replaces** master draft content (clears old, inserts new)
5. Master draft now shows latest version

**Result:**
- Master draft = always current ✅
- Backups = all previous versions ✅
- ML data = structured training files ✅

---

## 🐛 Fixing Append Issue

If content is appending instead of replacing:

**Check:**
1. The `update_document` method **should** clear first (line 178-188 in google_docs_bridge.py)
2. If it's not working, the `doc_length` calculation may be wrong

**Test:**
```python
# Run update and check if old content is deleted
await executor._update_existing_google_doc(deliverable, existing_doc, state)
# Should see: old content deleted, new content inserted
```

---

## 📁 Where Backups Are Saved

```
outputs/
├── master_drafts/
│   ├── master_draft.md              # Current version
│   └── versions/
│       ├── version_index.json       # Version registry
│       ├── {doc_id}_20251031_120000.md
│       └── {doc_id}_20251031_140000.md
└── ml_training_data/
    └── drafts/
        ├── {doc_id}_20251031_120000.json
        └── {doc_id}_20251031_140000.json
```

---

## 🎯 Summary

**Use:** One master draft + automatic version backups

**Benefits:**
- ✅ Clean workspace (one active doc)
- ✅ Full history preserved
- ✅ ML training data ready
- ✅ Easy to find current version
- ✅ Can recover any previous version

**Status:** ✅ Already implemented and enabled by default!

Your master draft will now:
- Replace content (not append)
- Create backups automatically
- Save ML training data
- Keep workspace clean

