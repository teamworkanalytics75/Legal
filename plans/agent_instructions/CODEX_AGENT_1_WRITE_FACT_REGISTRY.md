# Codex Agent 1: Write Fact Registry Extraction & Integration Code

**Your Task:** Write code to extract explicit facts from source documents and integrate them into the system.

---

## 🎯 What to Build

### 1. Write Fact Registry Extraction Script

**File to create/modify:** `writer_agents/scripts/extract_fact_registry.py`

**Requirements:**
- Create a script that scans source documents in `case_law_data/lawsuit_source_documents/`
- Extract explicit facts (e.g., citizenship statements like "US citizen", "not a PRC citizen")
- Use regex patterns to find facts in text
- Store facts in SQLite database `case_law_data/lawsuit_facts_database.db` in a `fact_registry` table
- Support extensible fact extractors via decorator pattern: `@register_fact_extractor("fact_type")`

**Database Schema:**
```sql
CREATE TABLE IF NOT EXISTS fact_registry (
    fact_id TEXT PRIMARY KEY,
    fact_type TEXT,
    fact_value TEXT,
    source_doc TEXT,
    extraction_method TEXT,
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

**Example Code Structure:**
```python
@register_fact_extractor("citizenship")
def extract_citizenship_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract explicit citizenship statements from source documents."""
    # Use regex to find patterns like:
    # - "US citizen", "United States citizen", "American citizen"
    # - "not a PRC citizen", "not a citizen of PRC"
    # Return List[FactEntry] with normalized values
    pass
```

**CLI Interface:**
```bash
python writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db
```

### 2. Write CaseFactsProvider Integration

**File to modify:** `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`

**Requirements:**
- Add method `get_fact_registry(fact_type: Optional[str] = None) -> Dict[str, List[str]]`
- Method should query `fact_registry` table from `lawsuit_facts_database.db`
- Return dict like `{"citizenship": ["US citizen"], "date": ["April 7, 2025"], ...}`
- Modify `_build_structured_facts()` to merge fact_registry data into structured facts block
- Include fact_registry facts in the STRUCTURED FACTS section that gets passed to prompts

**Example:**
```python
def get_fact_registry(self, fact_type: Optional[str] = None) -> Dict[str, List[str]]:
    """Retrieve fact entries from the fact_registry table."""
    # Query database, return dict grouped by fact_type
    pass

def _build_structured_facts(self) -> str:
    # ... existing code ...
    # Add fact_registry facts:
    fact_registry = self.get_fact_registry()
    for fact_type, values in fact_registry.items():
        # Merge into structured facts
    pass
```

### 3. Write ContradictionDetector Database Loading

**File to modify:** `writer_agents/code/validation/contradiction_detector.py`

**Requirements:**
- Add method `_load_fact_registry(db_path: Optional[Path]) -> Dict[str, Any]`
- Method should read from `fact_registry` table in SQLite database
- Return dict like `{"citizenship": "US citizen", ...}`
- Handle case where database doesn't exist (return empty dict)
- Update `__init__` to call `_load_fact_registry()` if `fact_registry` not provided

**Example:**
```python
def _load_fact_registry(self, db_path: Optional[Path]) -> Dict[str, Any]:
    """Load fact registry from SQLite database."""
    if not db_path or not db_path.exists():
        return {}
    # Query fact_registry table, return dict
    pass
```

### 4. Write Verification Helper Script

**File to create:** `writer_agents/scripts/verify_fact_registry_citizenship.py`

**Requirements:**
- Simple CLI script to query and print citizenship facts from database
- Helpful for debugging and verification

**Example:**
```bash
python writer_agents/scripts/verify_fact_registry_citizenship.py \
    --database case_law_data/lawsuit_facts_database.db
```

---

## 📁 Files to Create/Modify

1. **Create:** `writer_agents/scripts/extract_fact_registry.py`
2. **Create:** `writer_agents/scripts/verify_fact_registry_citizenship.py`
3. **Modify:** `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
4. **Modify:** `writer_agents/code/validation/contradiction_detector.py`
5. **Modify:** `writer_agents/scripts/build_truth_database.py` (ensure fact_registry table is created)

---

## ✅ Success Criteria

- [ ] `extract_fact_registry.py` script extracts citizenship facts from source documents
- [ ] Facts are stored in `fact_registry` table in database
- [ ] `CaseFactsProvider.get_fact_registry()` returns facts from database
- [ ] `CaseFactsProvider._build_structured_facts()` includes fact_registry data
- [ ] `ContradictionDetector._load_fact_registry()` reads from database
- [ ] Verification script can query and display facts

---

## 🔍 Reference Files

- Look at `writer_agents/scripts/build_truth_database.py` for database schema patterns
- Look at `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py` for existing structured facts logic
- Use existing patterns for CLI argument parsing and error handling

---

## 📝 Notes

- Use pathlib.Path for file paths
- Handle missing files/directories gracefully
- Use logging for debug/info messages
- Follow existing code style (4-space indentation, type hints)
- Test with sample documents if real source documents aren't available

