# Agent 1 – Content Verification Function

**Source:** [AGENT_PLAN_2025-11-15.md](../AGENT_PLAN_2025-11-15.md#agent-1-content-verification-function)

## Goal

Create a function that verifies generated motions actually reference personal corpus facts from the user's lawsuit.

## Context

- Personal corpus location: `case_law_data/tmp_corpus/`
- Case insights location: `writer_agents/outputs/case_insights.json`
- Key documents: HK Statement, OGC emails, Harvard correspondence
- Key dates: April 7/18, 2025 (OGC emails), June 2/4, 2025 (HK Statement/arrests)

## Deliverables

1. **Create** `writer_agents/code/validation/personal_facts_verifier.py`
   - Implement `verify_motion_uses_personal_facts(motion_text: str, personal_corpus_facts: Dict) -> Tuple[bool, List[str], Dict[str, Any]]`
   - Return: `(is_valid, missing_facts, verification_details)`

2. **Verification checks:**
   - HK Statement mentions: "Hong Kong Statement of Claim" or "HK Statement"
   - OGC email references: "OGC" or "Office of General Counsel"
   - Specific dates: "April 7, 2025", "April 18, 2025", "June 2, 2025", "June 4, 2025"
   - Key allegations: "defamation", "privacy breach", "retaliation", "harassment"
   - Timeline events: June 2025 arrests, April 2025 OGC emails

3. **Create** `tests/test_personal_facts_verifier.py`
   - Unit tests with sample motions
   - Test detection of personal fact mentions
   - Test identification of missing facts

## Key Files

- `writer_agents/code/validation/personal_facts_verifier.py` (new)
- `tests/test_personal_facts_verifier.py` (new)

## Success Criteria

- ✅ Function detects when motion mentions personal facts
- ✅ Function identifies missing key facts
- ✅ Unit tests pass: `pytest tests/test_personal_facts_verifier.py -v`

## Dependencies

- None (foundation for Agents 3 & 4)

## Validation

```bash
# Run tests
pytest tests/test_personal_facts_verifier.py -v

# Manual test
python -c "from writer_agents.code.validation.personal_facts_verifier import verify_motion_uses_personal_facts; print(verify_motion_uses_personal_facts('test motion', {}))"
```
