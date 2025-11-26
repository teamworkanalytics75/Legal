# Plan: Fix Feature Target Interpretation for Quality Constraints

## Problem
CatBoost feature targets are being interpreted too literally, causing incorrect instructions. For example:
- `protective_measures` should mean "**propose/request** a protective order" (offering one to protect discovery materials)
- But the system was treating it as "protective order exists" (claiming one already exists)

## Solution: Feature Type Categorization

### Feature Types

1. **PROPOSAL/REQUEST** - Things to be requested/proposed/offered
   - Examples: `protective_measures`, `protective_order` (when proposing)
   - Format: "**Request/propose/offer** X" or "Include a request for X"
   - Context: "Plaintiff respectfully requests: (1) a protective order..."

2. **MENTION** - Concepts to be mentioned/discussed
   - Examples: `mentions_privacy`, `mentions_safety`, `mentions_harassment`
   - Format: "**Mention/discuss** X" or "Include discussion of X"
   - Context: "The motion should discuss privacy concerns..."

3. **COUNT** - Number of instances/occurrences
   - Examples: `citation_count`, `privacy_harm_count`, `word_count`
   - Format: "**Include X+ instances** of Y" or "Target: X+ Y"
   - Context: "Include at least 5 citations..."

4. **STRUCTURAL** - Document structure elements
   - Examples: `paragraph_count`, `enumeration_depth`, `sentence_count`
   - Format: "**Use X structure**" or "Target: X paragraphs/sections"
   - Context: "Use nested enumeration (1. a. i.)..."

5. **FACTUAL** - Existing facts (use with extreme caution)
   - Examples: None currently, but could exist
   - Format: "**Note that** X exists" (only if verified)
   - Context: Should rarely be used; prefer proposal/mention types

## Implementation Steps

### Step 1: Create Feature Type Mapping
- Add `_categorize_feature_type()` method to `WorkflowOrchestrator`
- Map feature names to types based on:
  - Feature name patterns (e.g., `mentions_*` → MENTION, `*_count` → COUNT)
  - Rules JSON metadata (add `feature_type` field to rules)
  - Plugin descriptions/context

### Step 2: Update Quality Constraints Formatting
- Modify `_format_quality_constraints()` to use feature type categorization
- Format instructions contextually based on type:
  - PROPOSAL: "Request/propose/offer X to protect Y"
  - MENTION: "Mention/discuss X in the context of Y"
  - COUNT: "Include at least X instances of Y"
  - STRUCTURAL: "Use X structure (target: Y)"
  - FACTUAL: "Note that X exists" (with verification warning)

### Step 3: Add Feature Type Metadata to Rules
- Update rules JSON files to include `feature_type` field
- Start with `protective_measures_rules.json` as example
- Add to other key feature rules files

### Step 4: Update Plugin Rules
- Add `feature_type` to `protective_measures_plugin` rules
- Update other proposal/request type plugins
- Ensure descriptions clarify intent (request vs. claim)

### Step 5: Test and Validate
- Run motion generation with updated constraints
- Verify protective order is proposed (not claimed to exist)
- Check other feature types format correctly

## Feature Type Detection Logic

```python
def _categorize_feature_type(feature_name: str, rules: Dict = None) -> str:
    """Categorize feature type based on name patterns and rules metadata."""
    
    # Check rules metadata first (most reliable)
    if rules and rules.get('feature_type'):
        return rules['feature_type']
    
    # Pattern-based detection
    feature_lower = feature_name.lower()
    
    # PROPOSAL/REQUEST patterns
    if any(pattern in feature_lower for pattern in [
        'protective_measures', 'protective_order', 'proposal', 'request', 'offer'
    ]):
        return 'PROPOSAL'
    
    # MENTION patterns
    if feature_lower.startswith('mentions_'):
        return 'MENTION'
    
    # COUNT patterns
    if feature_lower.endswith('_count') or 'count' in feature_lower:
        return 'COUNT'
    
    # STRUCTURAL patterns
    if any(pattern in feature_lower for pattern in [
        'paragraph', 'sentence', 'enumeration', 'word_count', 'structure'
    ]):
        return 'STRUCTURAL'
    
    # Default to MENTION (safest fallback)
    return 'MENTION'
```

## Example Output

### Before (Incorrect):
```
CATBOOST FEATURE TARGETS:
  - Target: Protective Measures = 2.0
  - Target: Mentions Privacy = 1.9
  - Target: Citation Count = 5.0
```

### After (Correct):
```
CATBOOST FEATURE TARGETS (based on successful case analysis):
  - PROPOSAL: Request/propose at least 2 different protective measures (e.g., protective order, file under seal, confidentiality order) to protect discovery materials
  - MENTION: Mention/discuss privacy concerns at least 1.9 times throughout the motion
  - COUNT: Include at least 5 properly formatted case citations
```

## Files to Modify

1. `writer_agents/code/WorkflowOrchestrator.py`
   - Add `_categorize_feature_type()` method
   - Update `_format_quality_constraints()` to use categorization

2. `writer_agents/code/sk_plugins/rules/protective_measures_rules.json`
   - Add `"feature_type": "PROPOSAL"` field

3. Other rules files (as needed)
   - Add `feature_type` metadata where appropriate

## Success Criteria

- ✅ Protective order mentioned as a **proposal/request**, not an existing fact
- ✅ All feature targets formatted with appropriate context
- ✅ LLM receives clear, actionable instructions
- ✅ No false factual claims in generated motions

