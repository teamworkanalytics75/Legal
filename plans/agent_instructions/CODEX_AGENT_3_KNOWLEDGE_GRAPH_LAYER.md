# Codex Agent 3: Build Knowledge Graph Integration Layer

**Workstream:** Knowledge Graph Integration - Query Layer  
**Status:** Ready to start  
**Dependencies:** Agent 2 (entity extraction) - but can design interface independently

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 3**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 3**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 3** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_3_KNOWLEDGE_GRAPH_LAYER.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_1_FACT_TYPES_REFACTOR.md` - Agent 1's tasks
- `CODEX_AGENT_2_ENTITY_EXTRACTION.md` - Agent 2's tasks
- `CODEX_AGENT_4_VALIDATOR_INTEGRATION.md` - Agent 4's tasks

---

## 🎯 Objective

Build a query interface layer that allows the fact validation system to query the Knowledge Graph for fact verification, contradiction detection, and semantic search.

---

## 📋 Tasks

### Task 1: Create FactGraphQuery Interface

**File:** `writer_agents/code/validation/fact_graph_query.py` (new)

**Create interface for querying Knowledge Graph for facts:**
```python
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from typing import List, Dict, Optional, Set

class FactGraphQuery:
    """Query interface for fact validation using Knowledge Graph."""
    
    def __init__(self, kg: KnowledgeGraph, fact_registry_db: Optional[Path] = None):
        self.kg = kg
        self.fact_registry_db = fact_registry_db
    
    def find_fact_entity(self, fact_type: str, fact_value: str) -> Optional[str]:
        """Find Knowledge Graph entity for a fact."""
        # Search for entity matching fact_type and fact_value
        pass
    
    def get_fact_relationships(self, fact_type: str, fact_value: str) -> List[Dict]:
        """Get all relationships for a fact entity."""
        entity_id = self.find_fact_entity(fact_type, fact_value)
        if entity_id:
            return self.kg.get_relations_for_entity(entity_id)
        return []
    
    def verify_fact_exists(self, fact_type: str, fact_value: str) -> bool:
        """Check if fact exists in Knowledge Graph."""
        return self.find_fact_entity(fact_type, fact_value) is not None
    
    def find_related_facts(self, fact_type: str, fact_value: str, max_depth: int = 2) -> Set[str]:
        """Find facts related to the given fact via graph traversal."""
        entity_id = self.find_fact_entity(fact_type, fact_value)
        if entity_id:
            neighbors = self.kg.get_entity_neighbors(entity_id, depth=max_depth)
            return {n for n in neighbors if n.startswith("fact:")}
        return set()
```

### Task 2: Add Semantic Search Over Graph Entities

**File:** `writer_agents/code/validation/fact_graph_query.py`

**Add semantic similarity search:**
```python
def find_similar_facts(
    self,
    query_text: str,
    fact_type: Optional[str] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """Find facts semantically similar to query text."""
    # Use EmbeddingRetriever or similar for semantic search
    # Search over fact entities in Knowledge Graph
    pass
```

**Integration with EmbeddingRetriever:**
- Use existing `EmbeddingRetriever` for semantic search
- Index Knowledge Graph entities
- Query for similar facts

### Task 3: Create Fact Validation Queries

**File:** `writer_agents/code/validation/fact_graph_query.py`

**Add validation-specific queries:**
```python
def validate_fact_claim(
    self,
    claim_text: str,
    expected_fact_type: Optional[str] = None
) -> Dict[str, Any]:
    """Validate a factual claim against Knowledge Graph."""
    # Extract entities from claim
    # Check if entities exist in graph
    # Check if relationships match
    # Return validation result
    pass

def detect_fact_contradictions(
    self,
    motion_text: str
) -> List[Dict[str, Any]]:
    """Detect contradictions in motion text using Knowledge Graph."""
    # Extract factual claims from motion
    # Check against Knowledge Graph
    # Return contradictions
    pass
```

### Task 4: Create Graph-to-Fact-Registry Adapter

**File:** `writer_agents/code/validation/kg_fact_adapter.py` (new)

**Adapter to load facts from Knowledge Graph into fact registry format:**
```python
class KGFactAdapter:
    """Adapter to convert Knowledge Graph entities to fact registry format."""
    
    def load_facts_from_graph(self, kg: KnowledgeGraph) -> List[FactEntry]:
        """Load facts from Knowledge Graph as FactEntry objects."""
        facts = []
        for node in kg.graph.nodes():
            if node.startswith("fact:"):
                # Parse fact:type:value format
                # Convert to FactEntry
                pass
        return facts
    
    def sync_graph_to_registry(
        self,
        kg: KnowledgeGraph,
        database_path: Path
    ) -> None:
        """Sync Knowledge Graph facts to fact_registry database."""
        facts = self.load_facts_from_graph(kg)
        # Save to database
        pass
```

### Task 5: Add Hierarchical Fact Queries

**File:** `writer_agents/code/validation/fact_graph_query.py`

**Query facts by hierarchical type:**
```python
def get_all_facts_by_type(self, fact_type: str) -> List[Dict[str, Any]]:
    """Get all facts of a given type from Knowledge Graph."""
    facts = []
    for node in self.kg.graph.nodes():
        if node.startswith(f"fact:{fact_type}:"):
            facts.append({
                "entity": node,
                "type": fact_type,
                "value": self._extract_value_from_entity(node),
                "relationships": self.kg.get_relations_for_entity(node)
            })
    return facts

def get_fact_hierarchy(self) -> Dict[str, List[str]]:
    """Get hierarchical structure of facts by type."""
    hierarchy = {}
    for node in self.kg.graph.nodes():
        if node.startswith("fact:"):
            fact_type = self._extract_type_from_entity(node)
            if fact_type not in hierarchy:
                hierarchy[fact_type] = []
            hierarchy[fact_type].append(self._extract_value_from_entity(node))
    return hierarchy
```

### Task 6: Create Query Performance Optimizations

**File:** `writer_agents/code/validation/fact_graph_query.py`

**Add caching and indexing:**
- Cache entity lookups
- Index facts by type for fast queries
- Pre-compute common queries

### Task 7: Test Query Interface

**Test script:**
```python
from writer_agents.code.validation.fact_graph_query import FactGraphQuery
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

kg = KnowledgeGraph()
kg.load_from_file("case_law_data/facts_knowledge_graph.json")

query = FactGraphQuery(kg)

# Test queries
assert query.verify_fact_exists("date", "April 7, 2025")
related = query.find_related_facts("date", "April 7, 2025")
similar = query.find_similar_facts("OGC email notice", fact_type="date")
```

---

## 📁 Key Files

- `nlp_analysis/code/KnowledgeGraph.py` - Knowledge Graph (use existing)
- `writer_agents/code/validation/fact_graph_query.py` - Query interface (create new)
- `writer_agents/code/validation/kg_fact_adapter.py` - Graph-to-registry adapter (create new)
- `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py` - For semantic search (use existing)

---

## ✅ Success Criteria

- [ ] FactGraphQuery interface created and tested
- [ ] Can query facts by type, value, relationships
- [ ] Semantic search over graph entities works
- [ ] Fact validation queries implemented
- [ ] Graph-to-registry adapter works
- [ ] Hierarchical fact queries work
- [ ] Performance optimizations added
- [ ] All queries tested and verified

---

## 🚨 Important Notes

- **Use existing KnowledgeGraph**: Don't recreate - use `nlp_analysis.code.KnowledgeGraph`
- **Agent 2 dependency**: Wait for Agent 2 to build the graph, or work with sample graph
- **Agent 4 dependency**: Agent 4 will use your query interface - design it well
- **Performance**: Graph queries can be slow - add caching/indexing

---

## 📝 Progress Tracking

- [ ] Task 1: Create FactGraphQuery interface
- [ ] Task 2: Add semantic search over graph entities
- [ ] Task 3: Create fact validation queries
- [ ] Task 4: Create graph-to-fact-registry adapter
- [ ] Task 5: Add hierarchical fact queries
- [ ] Task 6: Create query performance optimizations
- [ ] Task 7: Test query interface

