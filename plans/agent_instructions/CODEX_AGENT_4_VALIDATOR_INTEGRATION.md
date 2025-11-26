# Codex Agent 4: Update Validators to Use Knowledge Graph

**Workstream:** Knowledge Graph Integration - Validator Updates  
**Status:** Ready to start  
**Dependencies:** Agent 3 (query layer) - but can design integration independently

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 4**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 4**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 4** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_4_VALIDATOR_INTEGRATION.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_1_FACT_TYPES_REFACTOR.md` - Agent 1's tasks
- `CODEX_AGENT_2_ENTITY_EXTRACTION.md` - Agent 2's tasks
- `CODEX_AGENT_3_KNOWLEDGE_GRAPH_LAYER.md` - Agent 3's tasks

---

## 🎯 Objective

Update `ContradictionDetector` and `personal_facts_verifier` to use the Knowledge Graph query interface for fact validation, contradiction detection, and semantic fact checking.

---

## 📋 Tasks

### Task 1: Update ContradictionDetector to Use Knowledge Graph

**File:** `writer_agents/code/validation/contradiction_detector.py`

**Add Knowledge Graph integration:**
```python
from writer_agents.code.validation.fact_graph_query import FactGraphQuery
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

class ContradictionDetector:
    def __init__(
        self,
        source_docs_dir: Optional[Path] = None,
        lawsuit_facts_db: Optional[Path] = None,
        fact_registry: Optional[Dict[str, Any]] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,  # NEW
        fact_graph_query: Optional[FactGraphQuery] = None,  # NEW
    ):
        # ... existing init ...
        self.kg = knowledge_graph
        self.fact_query = fact_graph_query or (
            FactGraphQuery(knowledge_graph) if knowledge_graph else None
        )
```

**Update `_validate_citizenship_claims` to use graph:**
```python
def _validate_citizenship_claims(self, motion_text: str) -> List[Contradiction]:
    """Validate citizenship claims using Knowledge Graph."""
    contradictions = []
    
    # Use fact_query to verify citizenship facts
    if self.fact_query:
        # Check if motion claims match graph facts
        # Use semantic search for paraphrased claims
        pass
    
    # Fallback to existing fact_registry check
    # ... existing code ...
```

**Add validators for other fact types:**
```python
def _validate_date_claims(self, motion_text: str) -> List[Contradiction]:
    """Validate date claims using Knowledge Graph."""
    # Use fact_query.get_all_facts_by_type("date")
    # Check if dates in motion match graph
    pass

def _validate_allegation_claims(self, motion_text: str) -> List[Contradiction]:
    """Validate allegation claims using Knowledge Graph."""
    # Similar to date validation
    pass
```

**Register new validators:**
```python
self.register_validator("date", self._validate_date_claims)
self.register_validator("allegation", self._validate_allegation_claims)
self.register_validator("timeline_event", self._validate_timeline_claims)
```

### Task 2: Update Personal Facts Verifier to Use Knowledge Graph

**File:** `writer_agents/code/validation/personal_facts_verifier.py`

**Add Knowledge Graph support:**
```python
from writer_agents.code.validation.fact_graph_query import FactGraphQuery

def verify_motion_uses_personal_facts(
    motion_text: str,
    personal_corpus_facts: Optional[Dict[str, Any]] = None,
    required_rules: Optional[Iterable[FactRule]] = None,
    negative_rules: Optional[Iterable[FactRule]] = None,
    fact_graph_query: Optional[FactGraphQuery] = None,  # NEW
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Verify motion uses personal facts, optionally using Knowledge Graph."""
    
    # If Knowledge Graph available, use it for verification
    if fact_graph_query:
        # Query graph for required facts
        # Use semantic search for fact matching
        # Check relationships between facts
        pass
    
    # Fallback to existing regex-based verification
    # ... existing code ...
```

### Task 3: Add Semantic Fact Matching

**File:** `writer_agents/code/validation/personal_facts_verifier.py`

**Use semantic search for fact matching:**
```python
def _match_fact_semantically(
    self,
    motion_text: str,
    fact_type: str,
    fact_value: str,
    fact_query: FactGraphQuery
) -> bool:
    """Match fact using semantic similarity over Knowledge Graph."""
    # Use fact_query.find_similar_facts() to find paraphrased facts
    # Check if motion text contains semantically similar facts
    pass
```

### Task 4: Update WorkflowOrchestrator to Load Knowledge Graph

**File:** `writer_agents/code/WorkflowOrchestrator.py`

**Add Knowledge Graph loading:**
```python
def _load_knowledge_graph(self) -> Optional[KnowledgeGraph]:
    """Load Knowledge Graph for fact validation."""
    graph_path = Path("case_law_data/facts_knowledge_graph.json")
    if graph_path.exists():
        try:
            from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
            kg = KnowledgeGraph()
            kg.load_from_file(str(graph_path))
            return kg
        except Exception as e:
            logger.warning(f"Failed to load Knowledge Graph: {e}")
    return None
```

**Pass to validators:**
```python
def _run_personal_facts_verifier(self, document: str, context: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    
    # Load Knowledge Graph if available
    kg = self._load_knowledge_graph()
    fact_query = FactGraphQuery(kg) if kg else None
    
    verifier_output = verify_motion_uses_personal_facts(
        document,
        personal_facts,
        negative_rules=NEGATIVE_FACT_RULES,
        fact_graph_query=fact_query,  # NEW
    )
```

**Update ContradictionDetector initialization:**
```python
contradiction_detector = ContradictionDetector(
    source_docs_dir=...,
    lawsuit_facts_db=...,
    knowledge_graph=kg,  # NEW
    fact_graph_query=fact_query,  # NEW
)
```

### Task 5: Add Graph-Based Fact Coverage Check

**File:** `writer_agents/code/validation/personal_facts_verifier.py`

**Check fact coverage using graph:**
```python
def check_fact_coverage_via_graph(
    motion_text: str,
    fact_query: FactGraphQuery
) -> Dict[str, Any]:
    """Check which facts from Knowledge Graph are present in motion."""
    coverage = {
        "citizenship": False,
        "dates": [],
        "allegations": [],
        "timeline_events": [],
        "documents": [],
        "organizations": [],
    }
    
    # Query graph for all fact types
    for fact_type in ["citizenship", "date", "allegation", "timeline_event", "document_reference", "organization"]:
        facts = fact_query.get_all_facts_by_type(fact_type)
        for fact in facts:
            # Check if motion contains this fact (semantically)
            if fact_query.verify_fact_in_text(motion_text, fact_type, fact["value"]):
                if fact_type == "citizenship":
                    coverage["citizenship"] = True
                else:
                    coverage[fact_type + "s"].append(fact["value"])
    
    return coverage
```

### Task 6: Update Tests to Use Knowledge Graph

**File:** `tests/test_contradiction_detector.py` or create new test file

**Add tests with Knowledge Graph:**
```python
def test_contradiction_detection_with_kg():
    """Test contradiction detection using Knowledge Graph."""
    kg = KnowledgeGraph()
    # Load test graph or create sample
    fact_query = FactGraphQuery(kg)
    
    detector = ContradictionDetector(
        knowledge_graph=kg,
        fact_graph_query=fact_query
    )
    
    # Test contradiction detection
    contradictions = detector.detect_contradictions("The plaintiff is a PRC citizen")
    assert len(contradictions) > 0
```

### Task 7: Add Fallback Logic

**Ensure backward compatibility:**
- If Knowledge Graph not available, fall back to existing fact_registry
- If graph query fails, use regex-based verification
- Log when using fallback vs graph

---

## 📁 Key Files

- `writer_agents/code/validation/contradiction_detector.py` - Contradiction detector (update)
- `writer_agents/code/validation/personal_facts_verifier.py` - Personal facts verifier (update)
- `writer_agents/code/WorkflowOrchestrator.py` - Workflow orchestrator (update)
- `writer_agents/code/WorkflowStrategyExecutor.py` - Workflow executor (update)
- `tests/test_contradiction_detector.py` - Tests (update or create)

---

## ✅ Success Criteria

- [ ] ContradictionDetector uses Knowledge Graph for validation
- [ ] Personal facts verifier uses Knowledge Graph for fact checking
- [ ] Semantic fact matching works for paraphrased facts
- [ ] WorkflowOrchestrator loads and passes Knowledge Graph to validators
- [ ] Graph-based fact coverage check implemented
- [ ] Tests updated to use Knowledge Graph
- [ ] Fallback logic works when graph unavailable
- [ ] All existing tests still pass

---

## 🚨 Important Notes

- **Backward compatibility**: Must work with or without Knowledge Graph
- **Agent 3 dependency**: Wait for Agent 3's query interface, or design integration points
- **Performance**: Graph queries may be slower - add caching/timeouts
- **Fallback**: Always have fallback to existing fact_registry system

---

## 📝 Progress Tracking

- [ ] Task 1: Update ContradictionDetector to use Knowledge Graph
- [ ] Task 2: Update personal facts verifier to use Knowledge Graph
- [ ] Task 3: Add semantic fact matching
- [ ] Task 4: Update WorkflowOrchestrator to load Knowledge Graph
- [ ] Task 5: Add graph-based fact coverage check
- [ ] Task 6: Update tests to use Knowledge Graph
- [ ] Task 7: Add fallback logic

