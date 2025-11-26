# Codex Agent 1: FactEngine Schema & Data Loading Foundation

**Workstream:** FactEngine Module Build - Foundation Layer  
**Status:** ✅ Complete — fact_engine/ package created with all foundation components  
**Dependencies:** schema/loader/builder foundation (self-contained)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 1**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read `CODEX_AGENTS_FACTENGINE_OVERVIEW.md` and all `CODEX_AGENT_*_FACTENGINE_*.md` files to understand the full context.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 1**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 1** and follow those.

**Your instruction file:** This file contains YOUR specific tasks. This is the source of truth.

---

## 🎯 Objective

Create the foundation layer for FactEngine: Pydantic schemas matching the truth table structure, CSV loaders, and a canonical truth table builder that cleans and normalizes facts.

---

## 📦 Tasks

### 1. Create fact_engine/ Package Structure

**Create directory:**
- `fact_engine/` (at repo root, parallel to `writer_agents/`, `nlp_analysis/`)

**Create files:**
- `fact_engine/__init__.py` - Package initialization with exports
- `fact_engine/schema.py` - Pydantic models (see below)
- `fact_engine/loader.py` - CSV loading functions
- `fact_engine/builder.py` - Canonical truth table builder

---

### 2. Implement Schema (schema.py)

**Create Pydantic models:**

```python
from pydantic import BaseModel
from typing import Optional, Dict

class Fact(BaseModel):
    """Canonical fact representation matching truth table CSV."""
    fact_id: str
    subject: Optional[str] = None
    verb: Optional[str] = None
    obj: Optional[str] = None  # 'object' is reserved in Python
    proposition: str
    
    event_type: Optional[str] = None
    event_date: Optional[str] = None  # ISO yyyy-mm-dd format
    event_location: Optional[str] = None
    
    actor_role: Optional[str] = None  # Plaintiff / Harvard / Court / PRCState / ThirdPartyHostile / NGO / ...
    speaker: Optional[str] = None  # Who is asserting this text
    truth_status: str  # "True", "Alleged", "HostileFalseClaim", "Disputed", "Unknown"
    
    evidence_type: Optional[str] = None  # Email / HKFiling / USFiling / Exhibit / WeChatArticle / NewsArticle / NGOReport / etc.
    source_document: Optional[str] = None
    source_excerpt: Optional[str] = None  # MUST be actual text from corpus, not LLM paraphrase
    extraction_method: Optional[str] = None  # TEMPLATE / OPENIE / NER / MANUAL / PIPELINE / AUTO_PROMOTE
    extraction_confidence: Optional[float] = None
    
    safety_risk: Optional[str] = None  # None / Low / Medium / High / Extreme
    public_exposure: Optional[str] = None  # Public / SemiPublic / NotPublic / Unknown
    risk_rationale: Optional[str] = None  # short explanation for sealing
    notes: Optional[str] = None

class FactImportance(BaseModel):
    """ML importance scores and SHAP explanations for a fact."""
    fact_id: str
    importance_score: float
    shap_values: Dict[str, float]  # feature_name -> shap value

class BNMapping(BaseModel):
    """Mapping from fact to Bayesian Network node."""
    fact_id: str
    bn_node: str
    mapping_weight: float  # contribution strength to node (0-1)
    mapping_confidence: float  # confidence in this mapping (0-1)
    mapping_source: str  # "RULE", "ML", "MANUAL"
    notes: Optional[str] = None

**Reference:** Match columns from `case_law_data/facts_truth_table.csv` (current v1)

---

### 3. Implement Loader (loader.py)

**Functions to implement:**

```python
from pathlib import Path
from typing import List
import pandas as pd
from .schema import Fact

def load_raw_facts(csv_path: Path) -> pd.DataFrame:
    """
    Load facts_ranked_for_sealing.csv.
    
    Args:
        csv_path: Path to facts_ranked_for_sealing.csv
        
    Returns:
        DataFrame with raw facts
    """
    # Read CSV, handle encoding, return DataFrame
    pass

def load_truth_facts(csv_path: Path) -> List[Fact]:
    """
    Load facts_truth_table.csv and convert to Fact objects.
    
    Args:
        csv_path: Path to facts_truth_table.csv
        
    Returns:
        List of Fact objects
    """
    # Read CSV, map columns to Fact model, return List[Fact]
    pass

def save_facts(facts: List[Fact], csv_path: Path) -> None:
    """
    Save List[Fact] to CSV matching truth table format.
    
    Args:
        facts: List of Fact objects
        csv_path: Output CSV path
    """
    # Convert Fact objects to DataFrame, write CSV
    pass
```

**Reference files:**
- Input: `case_law_data/facts_ranked_for_sealing.csv`
- Input: `case_law_data/facts_truth_table.csv`
- Column names should match existing CSV structure

---

### 4. Implement Builder (builder.py)

**Main function:**

```python
from typing import List
from pathlib import Path
from .schema import Fact
from .loader import load_raw_facts, load_truth_facts
import re

def build_canonical_truth_table(
    raw_csv_path: Path,
    truth_csv_path: Path,
    output_path: Path
) -> List[Fact]:
    """
    Build canonical truth table by cleaning and normalizing facts.
    
    CRITICAL: This function NEVER invents new facts - only cleans what's already there.
    New facts come from auto_promote.py.
    
    Steps:
    1. Load raw facts and truth facts
    2. Drop scaffolding rows (see rules below)
    3. Separate legal labels from facts (move to separate CSV)
    4. Normalize broken propositions using heuristics
    5. Fill speaker/actor_role/evidence_type from source patterns (see heuristics below)
    6. Return cleaned List[Fact]
    
    Args:
        raw_csv_path: Path to facts_ranked_for_sealing.csv
        truth_csv_path: Path to facts_truth_table.csv
        output_path: Path to write facts_truth_table_v2.csv
        
    Returns:
        List of cleaned Fact objects
    """
    pass
```

**Scaffolding filter rules:**
1. Drop rows whose `Proposition` starts with `"Entity:"`, `"Event "`, `"•"`, `"category sheet"`
2. Drop rows with `len(proposition.strip()) < 15` AND no obvious verb
   - Verb regex: `\b(is|was|were|has|have|had|sent|filed|published|received|arrested|charged|tortured|detained|reported|claimed|alleges?)\b`

**Legal label separation:**
- If `proposition` is basically just `"defamation"`, `"privacy breach"`, `"harassment"`, `"retaliation"` (single word/phrase, no context)
- Move these to `case_law_data/facts_claim_labels.csv` (separate file)
- In canonical table, only keep contextual facts like:
  - "The Hong Kong Statement of Claim pleads causes of action in defamation, privacy breach, harassment, and retaliation."

**Heuristics to implement:**

1. **Scaffolding detection:**
   - Starts with "Entity:", "Event ", "•", "category sheet"
   - Length < 10 chars and no verb patterns (regex: `\b(is|was|are|were|has|have|had|do|does|did|will|would|can|could|should|must)\b`)

2. **Proposition normalization:**
   - Remove leading bullets/trailing fragments
   - Ensure ends with period
   - Fix common typos (e.g., "i" → "I" at start of sentence)

3. **Speaker/Actor/Evidence inference (concrete patterns):**
   - `"From: Malcolm Grayson"` in `source_excerpt` → `speaker="Plaintiff"`, `actor_role="Plaintiff"`, `evidence_type="Email"`
   - Snippets containing `"IN THE HIGH COURT OF THE HONG KONG SPECIAL ADMINISTRATIVE REGION"` → `speaker="Plaintiff"`, `evidence_type="HKFiling"`
   - `"United States District Court for the District of Massachusetts"` → `speaker="Plaintiff"`, `evidence_type="USFiling"`
   - `"Harvard Club of Beijing"` / `"Harvard Club of Shanghai"` + WeChat text → `speaker="ThirdPartyHostile"`, `actor_role="ThirdPartyHostile"`, `evidence_type="WeChatArticle"`
   - `"Harvard"` or `"OGC"` or `"GSS"` in source → `actor_role="Harvard"`

**Reference:** Use existing `writer_agents/scripts/convert_to_truth_table.py` for patterns, but create cleaner version

---

### 5. Package Exports (__init__.py)

```python
"""FactEngine: Canonical fact management and ML importance analysis."""

from .schema import Fact, FactImportance, BNMapping
from .loader import load_raw_facts, load_truth_facts, save_facts
from .builder import build_canonical_truth_table

__all__ = [
    "Fact",
    "FactImportance", 
    "BNMapping",
    "load_raw_facts",
    "load_truth_facts",
    "save_facts",
    "build_canonical_truth_table",
]
```

---

## ✅ Success Criteria

- [x] Can import `from fact_engine import Fact, load_truth_facts, build_canonical_truth_table`
- [x] Can load `facts_truth_table.csv` and get `List[Fact]`
- [x] Can run `build_canonical_truth_table()` and produce cleaned CSV
- [x] Scaffolding rows are removed (filters implemented)
- [x] Speaker/actor_role/evidence_type filled via heuristics
- [x] Template registry and conversion pipeline integrated

---

## 🧪 Testing

Create simple test script:
```python
from fact_engine import load_truth_facts, build_canonical_truth_table
from pathlib import Path

# Test loader
facts = load_truth_facts(Path("case_law_data/facts_truth_table.csv"))
print(f"Loaded {len(facts)} facts")

# Test builder
# NOTE: Requires real raw CSV to produce meaningful output
cleaned = build_canonical_truth_table(
    Path("case_law_data/facts_ranked_for_sealing.csv"),
    Path("case_law_data/facts_truth_table.csv"),
    Path("case_law_data/facts_truth_table_v2.csv")
)
print(f"Cleaned to {len(cleaned)} facts")
```
