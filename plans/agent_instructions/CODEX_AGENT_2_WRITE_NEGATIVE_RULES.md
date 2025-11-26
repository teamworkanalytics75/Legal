# Codex Agent 2: Write Negative Fact Rules & Validation Code

**Your Task:** Write code to detect and validate negative facts (facts that should NOT appear in generated motions).

---

## 🎯 What to Build

### 1. Extend FactRule to Support Negative Matching

**File to modify:** `writer_agents/code/validation/personal_facts_verifier.py`

**Requirements:**
- Add `is_negative: bool = False` field to `FactRule` dataclass
- Modify `FactRule.match()` method to return `Optional[Tuple[str, bool]]` where the bool indicates if it's a violation
- When `is_negative=True` and pattern matches, return `(snippet, True)` to indicate violation
- When `is_negative=False` and pattern matches, return `(snippet, False)` to indicate normal match

**Example:**
```python
@dataclass
class FactRule:
    name: str
    description: str
    patterns: Iterable[str]
    optional: bool = False
    is_negative: bool = False  # NEW FIELD
    compiled_patterns: List[Pattern[str]] = field(init=False)

    def match(self, text: str, original_text: str, aliases: Optional[Iterable[str]] = None) -> Optional[Tuple[str, bool]]:
        """Return (snippet, is_violation) where is_violation indicates negative rule hit."""
        # If pattern matches:
        #   - If is_negative=True: return (snippet, True)  # VIOLATION
        #   - If is_negative=False: return (snippet, False)  # NORMAL MATCH
        # If no match: return None
        pass
```

### 2. Create NEGATIVE_FACT_RULES Tuple

**File to modify:** `writer_agents/code/validation/personal_facts_verifier.py`

**Requirements:**
- Create `NEGATIVE_FACT_RULES: Tuple[FactRule, ...]` after `DEFAULT_FACT_RULES`
- Include at least:
  - `not_prc_citizen` rule: Detects "PRC citizen", "citizen of PRC", "PRC national" patterns
  - `not_wrong_court_location` rule: Detects fabricated court locations
- Make it extensible for future negative rules

**Example:**
```python
NEGATIVE_FACT_RULES: Tuple[FactRule, ...] = (
    FactRule(
        name="not_prc_citizen",
        description="Must NOT claim PRC citizenship when source documents state US citizenship",
        patterns=(
            r"home\s+country\s+of\s+prc",
            r"prc\s+citizen",
            r"citizen\s+of\s+prc",
            r"prc\s+national",
            r"national\s+of\s+prc",
        ),
        is_negative=True,
    ),
    FactRule(
        name="not_wrong_court_location",
        description="Must NOT relocate the courthouse/case venue to an unsupported city",
        patterns=(
            r"district\s+of\s+hong\s+kong",
            r"beijing\s+district\s+court",
        ),
        is_negative=True,
    ),
)
```

### 3. Update verify_motion_uses_personal_facts() Function

**File to modify:** `writer_agents/code/validation/personal_facts_verifier.py`

**Requirements:**
- Modify function signature to accept `negative_rules: Optional[Iterable[FactRule]] = None`
- Add logic to check negative rules after checking positive rules
- Collect violations (negative rule matches) in a separate list
- Return 4-tuple: `(is_valid, missing_fact_names, violations, details)`
- Update `details` dict to include `violations` list
- Set `is_valid = False` if any violations found

**Example:**
```python
def verify_motion_uses_personal_facts(
    motion_text: str,
    personal_corpus_facts: Optional[Dict[str, Any]] = None,
    required_rules: Optional[Iterable[FactRule]] = None,
    negative_rules: Optional[Iterable[FactRule]] = None,  # NEW PARAMETER
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:  # NEW RETURN: 4-tuple
    """
    Returns: (is_valid, missing_fact_names, violations, details)
    """
    # ... existing positive rule checking ...
    
    # NEW: Check negative rules
    violations: List[str] = []
    negative_rules = tuple(negative_rules) if negative_rules else NEGATIVE_FACT_RULES
    for rule in negative_rules:
        match_result = rule.match(normalized_text, original_text, aliases_map.get(rule.name))
        if match_result:
            snippet, is_violation = match_result
            if is_violation:  # Negative rule matched = violation
                violations.append(rule.name)
    
    # Update is_valid: False if violations or missing facts
    is_valid = len(missing) == 0 and not violations
    
    details = {
        # ... existing details ...
        "violations": violations,  # NEW
    }
    
    return is_valid, missing, violations, details  # NEW: 4-tuple
```

### 4. Write Unit Tests

**File to create/modify:** `tests/test_personal_facts_verifier.py`

**Requirements:**
- Test that negative rules detect violations
- Test that violations are returned in the 4-tuple
- Test that `is_valid=False` when violations present
- Test that violations work alongside missing facts

**Example Tests:**
```python
def test_negative_rule_detects_citizenship_violation():
    motion_text = "The plaintiff, a PRC citizen, seeks protection from retaliation."
    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, {})
    
    assert is_valid is False
    assert violations == ["not_prc_citizen"]
    assert details["violations"] == ["not_prc_citizen"]

def test_negative_rule_no_violation():
    motion_text = "The plaintiff, a US citizen, seeks protection."
    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, {})
    
    assert "not_prc_citizen" not in violations
```

---

## 📁 Files to Create/Modify

1. **Modify:** `writer_agents/code/validation/personal_facts_verifier.py`
   - Extend `FactRule` dataclass
   - Create `NEGATIVE_FACT_RULES` tuple
   - Update `verify_motion_uses_personal_facts()` function

2. **Create/Modify:** `tests/test_personal_facts_verifier.py`
   - Add tests for negative rule detection
   - Add tests for violation reporting

---

## ✅ Success Criteria

- [ ] `FactRule` supports `is_negative` field
- [ ] `FactRule.match()` returns violation indicator
- [ ] `NEGATIVE_FACT_RULES` tuple defined with at least 2 rules
- [ ] `verify_motion_uses_personal_facts()` returns 4-tuple with violations
- [ ] Function correctly identifies violations in test motions
- [ ] Unit tests pass

---

## 🔍 Reference Files

- Look at existing `DEFAULT_FACT_RULES` for pattern examples
- Follow existing regex pattern style
- Use same error handling patterns as existing code

---

## 📝 Notes

- Negative rules should be case-insensitive (use `re.IGNORECASE`)
- Violations should be reported even if positive facts are missing
- Make patterns specific enough to avoid false positives
- Document each negative rule's purpose clearly

