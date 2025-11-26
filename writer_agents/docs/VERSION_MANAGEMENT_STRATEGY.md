# 📋 Version Management Strategy for Master Drafts

**Date:** 2025-10-31
**Recommendation:** ✅ One Master Draft + Versioned Backups

---

## 🎯 The Answer: Master Draft with Backups is Better

**Recommended Approach:**
- ✅ **One master draft** that gets updated/replaced
- ✅ **Automatic version backups** before each major update
- ✅ **ML training data** automatically saved
- ✅ **Clean workspace** (one active doc, not dozens)

**Why?**
1. **Better for ML training:** Version history shows document evolution
2. **Easier to use:** One document to work with
3. **Cleaner organization:** No clutter of multiple drafts
4. **ML insights:** Can learn from what changed between versions

---

## 🔧 How It Works

### Current Behavior

The `update_document` method in `google_docs_bridge.py` **does replace** content (not append):
- Line 178-188: Clears existing content first
- Line 190-234: Inserts new content
- **If content is appending, it's a bug in the update logic**

### Version Backup System

**Before each update:**
1. System captures current document content
2. Creates a backup with version ID: `{doc_id}_{timestamp}`
3. Saves to `outputs/master_drafts/versions/`
4. Saves ML training data to `outputs/ml_training_data/drafts/`
5. Then replaces the master draft with new content

**Result:**
- Master draft: Always current version
- Backups: All previous versions preserved
- ML data: Structured JSON files for training

---

## 📊 Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Master Draft + Backups** ✅ | • Clean workspace<br>• Easy to find current version<br>• Version history for ML<br>• Automatic organization | • Need backup system (already built!) |
| **New Draft Each Time** | • All versions visible<br>• Simple to implement | • Cluttered workspace<br>• Hard to find current version<br>• Poor for ML (no evolution tracking) |

---

## 🚀 Implementation

### Configuration

```python
config = WorkflowStrategyConfig(
    master_draft_mode=True,
    master_draft_title="Motion for Seal and Pseudonym - Master Draft",

    # Version management (enabled by default)
    enable_version_backups=True,           # Create backups before updates
    save_backups_for_ml=True,              # Save to ML training directory
    version_backup_directory="outputs/master_drafts/versions",
    max_versions_to_keep=50                # Keep last 50 versions
)
```

### What Happens on Update

1. **Current content is backed up:**
   - Saved to: `outputs/master_drafts/versions/{doc_id}_{timestamp}.md`
   - ML data: `outputs/ml_training_data/drafts/{doc_id}_{timestamp}.json`

2. **Master draft is replaced:**
   - Old content deleted (via Google Docs API)
   - New content inserted
   - Document stays in same location with same name

3. **Version tracking:**
   - All versions indexed in `version_index.json`
   - Can retrieve any previous version
   - ML can learn from version evolution

---

## 📁 File Structure

```
outputs/
├── master_drafts/
│   ├── master_draft.md              # Current version (auto-exported)
│   └── versions/
│       ├── version_index.json       # Version registry
│       ├── {doc_id}_20251031_120000.md
│       ├── {doc_id}_20251031_140000.md
│       └── ...
└── ml_training_data/
    └── drafts/
        ├── {doc_id}_20251031_120000.json
        ├── {doc_id}_20251031_140000.json
        └── ...
```

---

## 🧠 ML Training Benefits

**With versioned backups, you can train on:**
1. **Document evolution:** How drafts improve over iterations
2. **Edit patterns:** What changes correlate with better outcomes
3. **Quality progression:** How validation scores improve
4. **Iterative refinement:** Learning from revision cycles

**Example ML insights:**
- "Drafts that add more factual background between v1→v2 score higher"
- "Iterations that fix constraint violations improve success rates"
- "Version patterns that lead to high validation scores"

---

## ⚙️ Fixing the Append Issue

If content is appending instead of replacing, check:

1. **Google Docs API behavior:**
   - The `update_document` method should clear first (line 178-188)
   - If it's not working, the deleteContentRange may not be executing

2. **Test replacement:**
   ```python
   # Should see old content deleted, new content inserted
   await executor._update_existing_google_doc(deliverable, existing_doc, state)
   ```

3. **Check document length calculation:**
   - The `doc_length` calculation (line 169-174) may be wrong
   - If length is 0, content won't be deleted

---

## 🎯 Recommended Settings

**For active development:**
```python
enable_version_backups=True
save_backups_for_ml=True
max_versions_to_keep=50  # Keep last 50 iterations
```

**For production:**
```python
enable_version_backups=True
save_backups_for_ml=True
max_versions_to_keep=100  # Keep more history
```

**For testing:**
```python
enable_version_backups=False  # Skip backups during rapid iteration
```

---

## ✅ Summary

**Best Practice: One Master Draft + Version Backups**

1. ✅ **Master draft:** Single document, always current
2. ✅ **Automatic backups:** Before each update
3. ✅ **ML training:** Structured data for model improvement
4. ✅ **Clean workspace:** No clutter
5. ✅ **Full history:** Can recover any version

**The system is already set up for this!** Just enable `enable_version_backups=True` (it's the default).

---

**Status:** ✅ Implementation Complete
**Action:** Enable version backups in your config (already enabled by default)

