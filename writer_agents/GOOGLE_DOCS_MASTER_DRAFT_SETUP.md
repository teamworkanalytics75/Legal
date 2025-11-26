# 📝 Google Docs Master Draft Setup Guide

## ✅ Quick Status

**Script:** `writer_agents/scripts/update_master_draft_in_google_docs.py`
**Status:** ✅ Working
**Last Success:** November 7, 2025
**Document URL:** https://docs.google.com/document/d/1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE/edit?usp=drivesdk

---

## 🚀 Quick Start

### **1. Activate Virtual Environment**

```bash
cd /home/serteamwork/projects/TheMatrix
source .venv/bin/activate  # or your conda/env activation command
```

### **2. Install Required Packages**

```bash
pip install semantic-kernel autogen-agentchat autogen-ext google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

**Additional packages from requirements:**
```bash
pip install -r writer_agents/requirements_cuda.txt  # if available
```

### **3. Set Google Credentials**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json"
```

### **4. Run the Script**

```bash
python3 writer_agents/scripts/update_master_draft_in_google_docs.py
```

---

## 📋 What the Script Does

1. **Creates motion content** (~1,400 words)
2. **Finds or creates master draft** in Google Drive
3. **Updates the document** with new content
4. **Captures version history** for learning
5. **Exports markdown** to `outputs/master_drafts/master_draft.md`

---

## ⚙️ Configuration

### **Master Draft Settings**

The script is configured to:
- **Title:** "Motion for Seal and Pseudonym - Master Draft"
- **Drive Folder:** `1MZwep4pb9M52lSLLGQAd3quslA8A5iBu`
- **Mode:** Master draft mode (updates existing doc instead of creating new ones)
- **SK Config:** Uses OpenAI (not local Ollama) - see `use_local=False` in script

### **To Use Local Ollama Instead:**

If you have Ollama installed and want to use it:

1. Install Ollama connector:
   ```bash
   pip install semantic-kernel-connectors-ollama
   ```

2. Modify the script to use local:
   ```python
   sk_config = SKConfig(use_local=True)  # Instead of False
   ```

---

## 🔧 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'semantic_kernel'"**

**Solution:** Install the package:
```bash
pip install semantic-kernel
```

### **Issue: "Ollama connector not available"**

**Solution:** Either:
1. Install Ollama connector: `pip install semantic-kernel-connectors-ollama`
2. OR configure script to use OpenAI (already done - `use_local=False`)

### **Issue: "Google credentials not found"**

**Solution:** Set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/client_secret_*.json"
```

### **Issue: Import errors for `agents`, `insights`, `tasks`**

**Solution:** The `sys.path` adjustments in the script should handle this. If not, ensure you're running from the project root and the `writer_agents/code/` directory structure is intact.

**Note:** No symlinks needed - the script's `sys.path.insert()` calls handle module resolution.

---

## 📊 Expected Output

When successful, you should see:

```
================================================================================
UPDATING MASTER DRAFT IN GOOGLE DOCS
================================================================================

1. Creating motion content...
   [OK] Motion created (1398 words)

2. Configuring workflow...

3. Updating Google Docs master draft...
   [OK] Google Docs updated successfully
   [OK] Document URL: https://docs.google.com/document/d/...

4. Exporting to markdown...
   [OK] Markdown exported to outputs/master_drafts/master_draft.md

================================================================================
MASTER DRAFT UPDATE COMPLETE
================================================================================

View your motion at: https://docs.google.com/document/d/...
```

---

## 🔄 Running the Full Pipeline

For the complete optimized motion generation (with CatBoost analysis):

```bash
python3 writer_agents/scripts/generate_optimized_motion.py \
    --case-summary "Your case summary here" \
    --enable-google-docs \
    --google-drive-folder-id "1MZwep4pb9M52lSLLGQAd3quslA8A5iBu" \
    --master-draft-mode \
    --master-draft-title "Motion for Seal and Pseudonym - Master Draft"
```

This runs the full pipeline:
- CatBoost analysis
- Semantic Kernel enforcement
- Autogen writing
- Validation
- Google Docs update

---

## 📁 File Locations

- **Script:** `writer_agents/scripts/update_master_draft_in_google_docs.py`
- **Master Draft Doc:** https://docs.google.com/document/d/1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE/edit
- **Markdown Export:** `outputs/master_drafts/master_draft.md`
- **Version Backups:** `outputs/master_drafts/versions/`
- **ML Training Data:** `outputs/ml_training_data/drafts/`

---

## ✅ Verification Checklist

- [ ] Virtual environment activated
- [ ] All packages installed (`semantic-kernel`, `autogen-*`, `google-*`)
- [ ] `GOOGLE_APPLICATION_CREDENTIALS` environment variable set
- [ ] Script runs without import errors
- [ ] Google Doc updates successfully
- [ ] Can view document at provided URL

---

## 🎯 Key Points

1. **No symlinks needed** - `sys.path` adjustments handle imports
2. **Use OpenAI by default** - Script configured with `use_local=False` to avoid Ollama dependency
3. **Master draft mode** - Updates existing doc instead of creating new ones
4. **Version history** - Automatically captures revisions for learning
5. **Live updates** - Watch the document update in real-time in Google Drive

---

**Last Updated:** November 7, 2025
**Status:** ✅ Working
**Tested On:** WSL Ubuntu 24.04.3 LTS

