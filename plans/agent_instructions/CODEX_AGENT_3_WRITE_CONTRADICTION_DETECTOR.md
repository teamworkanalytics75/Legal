# Codex Agent 3: Write Contradiction Detection Code

**Your Task:** Write code to detect contradictions between generated motions and verified source facts.

---

## 🎯 What to Build

### 1. Create ContradictionDetector Class

**File to create:** `writer_agents/code/validation/contradiction_detector.py`

**Requirements:**
- Create `Contradiction` dataclass with fields: `claim`, `contradiction_type`, `severity`, `location`, `source_evidence`, `fact_type`
- Create `ContradictionDetector` class that:
  - Accepts `source_docs_dir`, `lawsuit_facts_db`, and `fact_registry` in `__init__`
  - Loads source documents from directory
  - Loads fact registry from database or dict
  - Has extensible validator registry via `register_validator()` method
  - Has `detect_contradictions(motion_text: str) -> List[Dict[str, Any]]` method

**Contradiction Types:**
- `DIRECT_CONTRADICTION`: Motion claims something that directly contradicts verified fact
- `INFERENCE`: Motion infers a fact that isn't explicitly stated
- `HALLUCINATION`: Motion claims something with no support in sources

**Severity Levels:**
- `critical`: Direct contradiction of verified fact
- `warning`: Unsupported inference

**Example Structure:**
```python
@dataclass
class Contradiction:
    claim: str
    contradiction_type: str  # DIRECT_CONTRADICTION, INFERENCE, HALLUCINATION
    severity: str  # critical, warning
    location: str
    source_evidence: Optional[str] = None
    fact_type: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        # Convert to dict for JSON serialization
        pass

class ContradictionDetector:
    def __init__(
        self,
        source_docs_dir: Optional[Path] = None,
        lawsuit_facts_db: Optional[Path] = None,
        fact_registry: Optional[Dict[str, Any]] = None,
    ):
        # Load source documents
        # Load fact registry
        # Initialize validator registry
        # Register built-in validators (e.g., citizenship)
        pass
    
    def register_validator(
        self,
        fact_type: str,
        validator: Callable[[str], List[Contradiction]],
    ):
        """Register a validator function for a specific fact type."""
        pass
    
    def detect_contradictions(self, motion_text: str) -> List[Dict[str, Any]]:
        """Run all validators and return contradiction reports."""
        pass
```

### 2. Implement Citizenship Contradiction Validator

**File:** `writer_agents/code/validation/contradiction_detector.py`

**Requirements:**
- Implement `_validate_citizenship_claims()` method
- Check if motion claims citizenship that contradicts fact_registry
- If fact_registry says "US citizen" but motion says "PRC citizen" → DIRECT_CONTRADICTION
- If no fact_registry but motion infers citizenship → INFERENCE
- Return list of `Contradiction` objects

**Example:**
```python
def _validate_citizenship_claims(self, motion_text: str) -> List[Contradiction]:
    """Check for citizenship contradictions."""
    contradictions = []
    
    # Get canonical citizenship from fact_registry
    canonical_fact = self._get_canonical_citizenship_fact()  # e.g., "US citizen"
    
    # Split motion into sentences and check each
    sentences = self._split_sentences(motion_text)
    for sentence in sentences:
        claim = self._extract_citizenship_claim(sentence)  # e.g., "PRC citizen"
        if claim and canonical_fact:
            if claim != canonical_fact:
                contradictions.append(Contradiction(
                    claim=claim,
                    contradiction_type="DIRECT_CONTRADICTION",
                    severity="critical",
                    location=sentence,
                    source_evidence=f"Verified fact: {canonical_fact}",
                    fact_type="citizenship",
                ))
        elif claim and not canonical_fact:
            # Inference without support
            contradictions.append(Contradiction(
                claim=claim,
                contradiction_type="INFERENCE",
                severity="warning",
                location=sentence,
                fact_type="citizenship",
            ))
    
    return contradictions
```

### 3. Implement Helper Methods

**File:** `writer_agents/code/validation/contradiction_detector.py`

**Requirements:**
- `_load_source_documents()`: Load text files from source_docs_dir
- `_load_fact_registry()`: Load from SQLite database or use provided dict
- `_get_canonical_citizenship_fact()`: Extract citizenship from fact_registry
- `_split_sentences()`: Split text into sentences
- `_extract_citizenship_claim()`: Extract citizenship claim from sentence using regex

### 4. Write Unit Tests

**File to create:** `tests/test_contradiction_detector.py`

**Requirements:**
- Test direct contradiction detection
- Test inference detection
- Test no contradiction when claim matches fact
- Test with mock fact_registry (in-memory dict)
- Test validator registration

**Example Tests:**
```python
def test_detects_direct_citizenship_contradiction():
    detector = ContradictionDetector(fact_registry={"citizenship": "US citizen"})
    motion = "The petitioner is a PRC citizen facing retaliation."
    contradictions = detector.detect_contradictions(motion)
    
    assert contradictions
    assert contradictions[0]["contradiction_type"] == "DIRECT_CONTRADICTION"
    assert contradictions[0]["severity"] == "critical"

def test_no_contradiction_when_claim_matches_fact():
    detector = ContradictionDetector(fact_registry={"citizenship": "US citizen"})
    motion = "The petitioner is a United States citizen seeking relief."
    contradictions = detector.detect_contradictions(motion)
    
    assert contradictions == []
```

---

## 📁 Files to Create/Modify

1. **Create:** `writer_agents/code/validation/contradiction_detector.py`
   - `Contradiction` dataclass
   - `ContradictionDetector` class
   - Citizenship validator
   - Helper methods

2. **Create:** `tests/test_contradiction_detector.py`
   - Unit tests for contradiction detection

---

## ✅ Success Criteria

- [ ] `ContradictionDetector` class created with extensible validator system
- [ ] Citizenship validator detects direct contradictions
- [ ] Citizenship validator detects unsupported inferences
- [ ] Validators can be registered via `register_validator()`
- [ ] `detect_contradictions()` returns list of dicts
- [ ] Unit tests pass

---

## 🔍 Reference Files

- Look at `writer_agents/code/validation/personal_facts_verifier.py` for similar validation patterns
- Use regex patterns similar to citizenship detection
- Follow existing error handling and logging patterns

---

## 📝 Notes

- Make validators extensible - others should be able to add validators for dates, locations, etc.
- Use case-insensitive matching for citizenship claims
- Handle edge cases (empty fact_registry, missing source docs, etc.)
- Log warnings for validator failures but continue processing

