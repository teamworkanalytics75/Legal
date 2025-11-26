# Codex Agent 1: Fact Registry Population & Database Integration

**Workstream:** False Fact Prevention System  
**Status:** Ready to start  
**Dependencies:** None (can work independently)

---

## 🎯 Objective

Populate the fact registry database and verify that `CaseFactsProvider` and `ContradictionDetector` can read from it.

---

## 📋 Tasks

### Task 1: Run Fact Registry Extraction
**File:** `writer_agents/scripts/extract_fact_registry.py`

```bash
python writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db
```

**Expected Result:**
- Fact registry table populated with citizenship facts
- No errors during extraction

### Task 2: Verify Database Population
**File:** `writer_agents/scripts/verify_fact_registry_citizenship.py`

```bash
python writer_agents/scripts/verify_fact_registry_citizenship.py \
    --database case_law_data/lawsuit_facts_database.db
```

**Expected Output:**
```
Found 1 citizenship fact(s) in case_law_data/lawsuit_facts_database.db:
  1. US citizen (source=...)
```

### Task 3: Verify CaseFactsProvider Integration
**File:** `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`

Test that `CaseFactsProvider.get_fact_registry()` returns the citizenship fact:

```python
from writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider

provider = CaseFactsProvider(
    lawsuit_facts_db_path="case_law_data/lawsuit_facts_database.db"
)
registry = provider.get_fact_registry("citizenship")
assert "US citizen" in registry.get("citizenship", [])
```

### Task 4: Verify ContradictionDetector Integration
**File:** `writer_agents/code/validation/contradiction_detector.py`

Test that `ContradictionDetector` can load facts from database:

```python
from writer_agents.code.validation.contradiction_detector import ContradictionDetector
from pathlib import Path

detector = ContradictionDetector(
    lawsuit_facts_db=Path("case_law_data/lawsuit_facts_database.db")
)
assert detector.fact_registry.get("citizenship") == "US citizen"
```

### Task 5: Verify Structured Facts Include Registry Data
Test that `CaseFactsProvider._build_structured_facts()` includes fact_registry data:

```python
provider = CaseFactsProvider(
    lawsuit_facts_db_path="case_law_data/lawsuit_facts_database.db"
)
facts = provider.format_facts_for_autogen()
assert "citizenship" in facts.lower() or "US citizen" in facts
```

---

## ✅ Success Criteria

- [ ] Fact registry table has citizenship entry
- [ ] `CaseFactsProvider.get_fact_registry()` returns `['US citizen']`
- [ ] Structured facts block includes registry data (citizenship: US citizen)
- [ ] `ContradictionDetector` reads canonical citizenship from database

---

## 📁 Key Files

- `writer_agents/scripts/extract_fact_registry.py` - Extraction script
- `writer_agents/scripts/verify_fact_registry_citizenship.py` - Verification helper
- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py` - Provider class
- `writer_agents/code/validation/contradiction_detector.py` - Detector class
- `case_law_data/lawsuit_facts_database.db` - Database file

---

## 🔍 What to Check

1. **Database exists:** `case_law_data/lawsuit_facts_database.db`
2. **Table exists:** `fact_registry` table with columns: `fact_id`, `fact_type`, `fact_value`, `source_doc`
3. **Data populated:** At least one citizenship fact entry
4. **Integration works:** Both CaseFactsProvider and ContradictionDetector can read it

---

## 📝 Notes

- You can work independently - no dependencies on other agents
- If source documents are missing, use the sample fixtures in `case_law_data/lawsuit_source_documents/sample_case.txt`
- Report any errors or missing dependencies immediately

---

## 🚀 When Complete

Mark tasks as complete and report:
- Database file location
- Number of facts extracted
- Verification test results
- Any issues encountered

