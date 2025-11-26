# Codex Agent 2: Integrate EntityRelationExtractor with Fact Registry

**Workstream:** Knowledge Graph Integration - Entity Extraction  
**Status:** Ready to start  
**Dependencies:** Agent 1 (fact type refactoring) - but can start on entity extraction independently

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 2**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 2**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 2** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_2_ENTITY_EXTRACTION.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_1_FACT_TYPES_REFACTOR.md` - Agent 1's tasks
- `CODEX_AGENT_3_KNOWLEDGE_GRAPH_LAYER.md` - Agent 3's tasks
- `CODEX_AGENT_4_VALIDATOR_INTEGRATION.md` - Agent 4's tasks

---

## 🎯 Objective

Integrate the existing `EntityRelationExtractor` and `KnowledgeGraph` systems with the fact registry extraction process. Extract entities and relationships from source documents and build a Knowledge Graph alongside the fact registry.

---

## 📋 Tasks

### Task 1: Create Knowledge Graph Builder for Fact Registry

**File:** `writer_agents/scripts/build_fact_knowledge_graph.py` (new)

**Create script that:**
1. Loads source documents from `case_law_data/lawsuit_source_documents/`
2. Uses `EntityRelationExtractor` to extract entities and relations
3. Builds `KnowledgeGraph` from extracted data
4. Saves graph to file (JSON/GEXF format)

**Structure:**
```python
from nlp_analysis.code.EntityRelationExtractor import EntityRelationExtractor
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from pathlib import Path

def build_fact_knowledge_graph(source_dir: Path, output_path: Path):
    extractor = EntityRelationExtractor()
    kg = KnowledgeGraph()
    
    for doc_path in source_dir.rglob("*.txt"):
        text = doc_path.read_text()
        entities = extractor.extract_entities(text)
        relations = extractor.extract_relations_patterns(text)
        kg.add_from_extractions(entities, relations)
    
    kg.save_to_file(str(output_path))
    return kg
```

### Task 2: Integrate with Fact Registry Extraction

**File:** `writer_agents/scripts/extract_fact_registry.py`

**Add option to build Knowledge Graph during extraction:**
```python
def extract_facts_and_build_graph(
    source_dir: Path,
    database_path: Path,
    graph_output_path: Optional[Path] = None
) -> Tuple[List[FactEntry], Optional[KnowledgeGraph]]:
    """Extract facts and optionally build Knowledge Graph."""
    facts = extract_all_facts(source_dir)
    save_to_database(facts, database_path)
    
    kg = None
    if graph_output_path:
        kg = build_fact_knowledge_graph(source_dir, graph_output_path)
    
    return facts, kg
```

### Task 3: Map Fact Registry to Knowledge Graph Entities

**File:** `writer_agents/scripts/map_facts_to_kg.py` (new)

**Create mapping between fact registry and Knowledge Graph:**
- Map fact entries to Knowledge Graph entities
- Create relationships between facts (e.g., "date" → "relates_to" → "allegation")
- Link facts to source document entities

**Example:**
```python
def map_facts_to_kg(facts: List[FactEntry], kg: KnowledgeGraph) -> KnowledgeGraph:
    """Map fact registry entries to Knowledge Graph entities."""
    for fact in facts:
        # Add fact as entity
        kg.add_entity(
            f"fact:{fact.fact_type}:{fact.fact_value}",
            entity_type=fact.fact_type,
            fact_id=fact.fact_id,
            source_doc=fact.source_doc
        )
        
        # Link to source document entity
        kg.add_relation(
            f"fact:{fact.fact_type}:{fact.fact_value}",
            "extracted_from",
            fact.source_doc
        )
    
    return kg
```

### Task 4: Add Legal-Specific Entity Patterns

**File:** `nlp_analysis/code/EntityRelationExtractor.py` (or create extension)

**Add patterns for lawsuit-specific entities:**
- OGC emails patterns
- HK Statement references
- Date patterns (April 7, 2025, etc.)
- Allegation patterns
- Timeline event patterns

**Or create:** `writer_agents/code/validation/lawsuit_entity_extractor.py` (new)

**Extend EntityRelationExtractor with lawsuit-specific patterns:**
```python
class LawsuitEntityExtractor(EntityRelationExtractor):
    """Extended extractor with lawsuit-specific entity patterns."""
    
    def __init__(self):
        super().__init__()
        self._add_lawsuit_patterns()
    
    def _add_lawsuit_patterns(self):
        """Add patterns for OGC, HK Statement, dates, allegations."""
        # Add custom patterns to spaCy entity ruler
```

### Task 5: Create Fact-to-Entity Relationship Mapper

**File:** `writer_agents/scripts/map_fact_relationships.py` (new)

**Create relationships between facts:**
- Dates → relate_to → Timeline events
- Allegations → relate_to → Dates
- Documents → relate_to → Organizations
- Timeline events → relate_to → Allegations

**Example:**
```python
def build_fact_relationships(facts: List[FactEntry], kg: KnowledgeGraph):
    """Build relationships between facts based on their types."""
    dates = [f for f in facts if f.fact_type == "date"]
    allegations = [f for f in facts if f.fact_type == "allegation"]
    timeline_events = [f for f in facts if f.fact_type == "timeline_event"]
    
    # Link dates to timeline events
    for date in dates:
        for event in timeline_events:
            if date.fact_value in event.fact_value or event.fact_value in date.fact_value:
                kg.add_relation(
                    f"fact:date:{date.fact_value}",
                    "relates_to",
                    f"fact:timeline_event:{event.fact_value}"
                )
```

### Task 6: Test Entity Extraction Integration

**Test script:**
```python
from writer_agents.scripts.build_fact_knowledge_graph import build_fact_knowledge_graph
from pathlib import Path

kg = build_fact_knowledge_graph(
    Path("case_law_data/lawsuit_source_documents"),
    Path("case_law_data/facts_knowledge_graph.json")
)

print(f"Entities: {kg.graph.number_of_nodes()}")
print(f"Relations: {kg.graph.number_of_edges()}")
print(kg.get_summary_stats())
```

---

## 📁 Key Files

- `nlp_analysis/code/EntityRelationExtractor.py` - Entity extractor (may extend)
- `nlp_analysis/code/KnowledgeGraph.py` - Knowledge Graph (use existing)
- `writer_agents/scripts/build_fact_knowledge_graph.py` - Graph builder (create new)
- `writer_agents/scripts/map_facts_to_kg.py` - Fact-to-KG mapper (create new)
- `writer_agents/scripts/map_fact_relationships.py` - Relationship mapper (create new)
- `writer_agents/code/validation/lawsuit_entity_extractor.py` - Lawsuit-specific extractor (optional, create new)

---

## ✅ Success Criteria

- [ ] Knowledge Graph built from source documents
- [ ] Entities extracted using EntityRelationExtractor
- [ ] Relationships extracted and added to graph
- [ ] Fact registry entries mapped to Knowledge Graph entities
- [ ] Fact-to-fact relationships created
- [ ] Graph saved to file (JSON/GEXF)
- [ ] Integration tested and verified

---

## 🚨 Important Notes

- **Use existing systems**: Don't recreate EntityRelationExtractor or KnowledgeGraph - use what exists
- **Agent 1 dependency**: Wait for Agent 1's refactored fact types if possible, or work with current types
- **Performance**: Entity extraction can be slow - consider caching or limiting document size
- **Agent 3 dependency**: Agent 3 will build the query layer on top of your graph

---

## 📝 Progress Tracking

- [ ] Task 1: Create Knowledge Graph builder script
- [ ] Task 2: Integrate with fact registry extraction
- [ ] Task 3: Map facts to Knowledge Graph entities
- [ ] Task 4: Add legal-specific entity patterns
- [ ] Task 5: Create fact-to-entity relationship mapper
- [ ] Task 6: Test entity extraction integration

