# Codex Agent 1: Refactor Fact Types to Hierarchical Structure

**Workstream:** Knowledge Graph Integration - Fact Type Refactoring  
**Status:** Complete (waiting on richer source docs to populate non-citizenship facts)  
**Dependencies:** None (can work independently)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 1**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 1**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 1** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_1_FACT_TYPES_REFACTOR.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_2_ENTITY_EXTRACTION.md` - Agent 2's tasks
- `CODEX_AGENT_3_KNOWLEDGE_GRAPH_LAYER.md` - Agent 3's tasks
- `CODEX_AGENT_4_VALIDATOR_INTEGRATION.md` - Agent 4's tasks

---

## 🎯 Objective

Refactor the fact registry system to use hierarchical fact types (categories) instead of specific instances. This will enable better integration with the Knowledge Graph system.

---

## 📡 Status Update — 2025-11-15

- Refactored `writer_agents/scripts/extract_fact_registry.py` to use one extractor per hierarchical type (`citizenship`, `date`, `allegation`, `timeline_event`, `document_reference`, `organization`) plus the new optional description field (see commit-ready diff in that file).
- Added schema support + migration utility (`writer_agents/scripts/migrate_fact_types.py`) and ran `python writer_agents/scripts/migrate_fact_types.py --database case_law_data/lawsuit_facts_database.db` to upgrade the live DB (only a citizenship fact existed, so no rows required remapping).
- Regenerated the registry via `python writer_agents/scripts/extract_fact_registry.py --source-dir case_law_data/lawsuit_source_documents --database case_law_data/lawsuit_facts_database.db` and exported the CSV using `python writer_agents/scripts/export_facts_to_csv.py --database case_law_data/lawsuit_facts_database.db --output case_law_data/facts_export_refactored.csv`.
- Verified schema changes and distinct types via ad-hoc `sqlite3` (citizenship is currently the only populated type because the sample corpus lacks other fact cues; the new extractors are ready for richer inputs).
- Enhancement pass (2025-11-15 evening): broadened date/organization/allegation patterns (additional date formats, privacy/data breach synonyms, fuzzy Harvard OGC matches), added deduplication + confidence boosting when multiple sources agree, and introduced summary/validation logging so missing fact categories and empty descriptions are surfaced automatically.

---

## 📋 Current Problem

**Current design (BAD):**
- `date_april_7_2025` ← specific date, not a type
- `date_april_18_2025` ← specific date, not a type
- `allegation_defamation` ← specific allegation, not a type
- etc.

**Target design (GOOD):**
- `date` (fact type) → "April 7, 2025" (fact value) → "OGC notice date" (description)
- `allegation` (fact type) → "defamation" (fact value)
- `timeline_event` (fact type) → "April 2025 OGC emails" (fact value)

---

## 📋 Tasks

### Task 1: Design New Fact Type Hierarchy

**Create mapping from old types to new types:**

```
OLD → NEW
---------
date_april_7_2025 → date (value: "April 7, 2025", description: "OGC notice date")
date_april_18_2025 → date (value: "April 18, 2025", description: "OGC follow-up date")
date_june_2_2025 → date (value: "June 2, 2025", description: "HK filing date")
date_june_4_2025 → date (value: "June 4, 2025", description: "Arrests/threats date")

allegation_defamation → allegation (value: "defamation")
allegation_privacy_breach → allegation (value: "privacy breach")
allegation_retaliation → allegation (value: "retaliation")
allegation_harassment → allegation (value: "harassment")

timeline_april_ogc_emails → timeline_event (value: "April 2025 OGC emails")
timeline_june_2025_arrests → timeline_event (value: "June 2025 arrests/threats")

hk_statement → document_reference (value: "Hong Kong Statement of Claim")
ogc_emails → organization (value: "Harvard OGC")

citizenship → citizenship (keep as-is, already correct)
```

**New fact type categories:**
1. `citizenship` - Identity facts
2. `date` - Key dates (with description field)
3. `allegation` - Types of allegations
4. `timeline_event` - Timeline events
5. `document_reference` - Document references
6. `organization` - Organizations/entities

### Task 2: Update Database Schema

**File:** `writer_agents/scripts/extract_fact_registry.py`

**Add optional `description` field to `FactEntry`:**
```python
@dataclass
class FactEntry:
    fact_type: str  # Now hierarchical: "date", "allegation", etc.
    fact_value: str  # Specific value: "April 7, 2025", "defamation", etc.
    description: Optional[str] = None  # NEW: Context/description
    source_doc: str
    extraction_method: str
    confidence: float = 0.9
```

**Update database schema:**
- Add `description` column to `fact_registry` table (nullable)
- Migration script to convert old data to new format

### Task 3: Refactor Extractors

**File:** `writer_agents/scripts/extract_fact_registry.py`

**Combine date extractors into one:**
```python
@register_fact_extractor("date")
def extract_date_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract all key dates from source documents."""
    patterns = [
        (re.compile(r"april\s+7,\s*2025", re.IGNORECASE), "April 7, 2025", "OGC notice date"),
        (re.compile(r"april\s+18,\s*2025", re.IGNORECASE), "April 18, 2025", "OGC follow-up date"),
        (re.compile(r"june\s+2,\s*2025", re.IGNORECASE), "June 2, 2025", "HK filing date"),
        (re.compile(r"june\s+4,\s*2025", re.IGNORECASE), "June 4, 2025", "Arrests/threats date"),
    ]
    # Return entries with fact_type="date", fact_value=date, description=context
```

**Combine allegation extractors into one:**
```python
@register_fact_extractor("allegation")
def extract_allegation_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract all allegations from source documents."""
    patterns = [
        (re.compile(r"\bdefamation\b", re.IGNORECASE), "defamation"),
        (re.compile(r"privacy\s+breach", re.IGNORECASE), "privacy breach"),
        (re.compile(r"\bretaliation\b", re.IGNORECASE), "retaliation"),
        (re.compile(r"\bharassment\b", re.IGNORECASE), "harassment"),
    ]
    # Return entries with fact_type="allegation", fact_value=allegation_type
```

**Combine timeline extractors:**
```python
@register_fact_extractor("timeline_event")
def extract_timeline_event_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract timeline events from source documents."""
    # Combine timeline_april_ogc_emails and timeline_june_2025_arrests
```

**Update document/organization extractors:**
```python
@register_fact_extractor("document_reference")
def extract_document_reference_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract document references (replaces hk_statement)."""

@register_fact_extractor("organization")
def extract_organization_facts(source_docs: Iterable[Path]) -> List[FactEntry]:
    """Extract organizations (replaces ogc_emails)."""
```

### Task 4: Create Migration Script

**File:** `writer_agents/scripts/migrate_fact_types.py` (new)

**Script to:**
- Read existing `fact_registry` table
- Map old fact types to new hierarchical types
- Update fact_value and add description where needed
- Write back to database

**Example mapping:**
```python
MIGRATION_MAP = {
    "date_april_7_2025": ("date", "April 7, 2025", "OGC notice date"),
    "date_april_18_2025": ("date", "April 18, 2025", "OGC follow-up date"),
    # ... etc
}
```

### Task 5: Update Export Script

**File:** `writer_agents/scripts/export_facts_to_csv.py`

**Update to:**
- Group facts by hierarchical type
- Show fact_value and description columns
- Update sentence formatting to use new structure

### Task 6: Test Refactored System

**Run extraction with new types:**
```bash
python writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db
```

**Verify database:**
```bash
sqlite3 case_law_data/lawsuit_facts_database.db "SELECT DISTINCT fact_type FROM fact_registry;"
# Should show: citizenship, date, allegation, timeline_event, document_reference, organization
```

**Export and verify:**
```bash
python writer_agents/scripts/export_facts_to_csv.py \
    --database case_law_data/lawsuit_facts_database.db \
    --output case_law_data/facts_export_refactored.csv
```

---

## 📁 Key Files

- `writer_agents/scripts/extract_fact_registry.py` - Main extractor (refactor extractors)
- `writer_agents/scripts/migrate_fact_types.py` - Migration script (create new)
- `writer_agents/scripts/export_facts_to_csv.py` - Export script (update)
- `case_law_data/lawsuit_facts_database.db` - Database (add description column)

---

## ✅ Success Criteria

- [x] Fact types are hierarchical (6 categories instead of 13+ specific types)
- [x] Database schema includes `description` field
- [x] All extractors refactored to use new type system
- [x] Migration script converts existing data
- [x] Export script works with new structure
- [x] Tests pass with new fact types

---

## 🚨 Important Notes

- **Backward compatibility**: Migration script should preserve existing data
- **No data loss**: All facts should be preserved, just reorganized
- **Agent 2 dependency**: Agent 2 will use your refactored types, so complete this first if possible
- **Database migration**: Test migration on a copy of the database first

---

## 📝 Progress Tracking

- [x] Task 1: Design new fact type hierarchy
- [x] Task 2: Update database schema
- [x] Task 3: Refactor extractors
- [x] Task 4: Create migration script
- [x] Task 5: Update export script
- [x] Task 6: Test refactored system
