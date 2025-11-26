# Codex Agent 3: FactEngine Evidence Querying (SQL + Embeddings + KG)

**Workstream:** FactEngine Module Build - Query Layer  
**Status:** ✅ Complete — evidence_query.py implemented with all query wrappers  
**Dependencies:** None (can work independently, just needs to know interfaces)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 3**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read `CODEX_AGENTS_FACTENGINE_OVERVIEW.md` and all `CODEX_AGENT_*_FACTENGINE_*.md` files to understand the full context.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 3**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 3** and follow those.

**Your instruction file:** This file contains YOUR specific tasks. This is the source of truth.

---

## 🎯 Objective

Create wrappers for existing query systems (LangChain SQL agent, EmbeddingRetriever, Knowledge Graph) to enable intelligent corpus querying for fact discovery.

---

## 📦 Tasks

### 1. Create Evidence Query Module (evidence_query.py)

**File:** `fact_engine/evidence_query.py`

**Dependencies:**
- `writer_agents.code.LangchainIntegration.LangChainSQLAgent`
- `writer_agents.code.sk_plugins.FeaturePlugin.EmbeddingRetriever.EmbeddingRetriever`
- `nlp_analysis.code.KnowledgeGraph.KnowledgeGraph`
- `nlp_analysis.code.pipeline.NLPAnalysisPipeline` (optional)
- `pathlib.Path`

---

### 2. Implement SQL Query Wrapper

**Function:**

```python
from pathlib import Path
from typing import Dict, Optional
from writer_agents.code.LangchainIntegration import LangChainSQLAgent

def query_db_for_pattern(
    question: str,
    db_path: Optional[Path] = None,
    context: str = ""
) -> str:
    """
    Query database using LangChain SQL agent.
    
    Args:
        question: Natural language question about evidence
        db_path: Optional path to SQLite database (default: case_law_data/lawsuit.db)
        context: Optional context to include in query
        
    Returns:
        Answer string from SQL agent
    """
    if db_path is None:
        db_path = Path("case_law_data/lawsuit.db")
    
    agent = LangChainSQLAgent(db_path=db_path)
    result = agent.query_evidence(question, context)
    
    if result.get("success"):
        return result["answer"]
    else:
        return ""  # Return empty string on failure
```

**Reference:** `writer_agents/code/LangchainIntegration.py::LangChainSQLAgent.query_evidence()`

**Default DB path:** `case_law_data/unified_corpus.db` (or detect from config)

---

### 3. Implement Embedding Search Wrapper

**Function:**

```python
from pathlib import Path
from typing import List, Dict, Optional
from writer_agents.code.sk_plugins.FeaturePlugin.EmbeddingRetriever import EmbeddingRetriever, RetrievedChunk

def search_similar_passages(
    text: str,
    k: int = 10,
    db_path: Optional[Path] = None,
    faiss_path: Optional[Path] = None
) -> List[Dict]:
    """
    Search for semantically similar passages using embeddings.
    
    Args:
        text: Query text
        k: Number of results to return
        db_path: Optional path to embedding database
        faiss_path: Optional path to FAISS index
        
    Returns:
        List of dictionaries with:
        - "text": str (snippet text)
        - "score": float (similarity score)
        - "source": str (source file path)
        - "metadata": dict
    """
    if db_path is None:
        db_path = Path("case_law_data/results/personal_corpus_embeddings.db")
    if faiss_path is None:
        faiss_path = Path("case_law_data/results/personal_corpus_embeddings.faiss")
    
    retriever = EmbeddingRetriever(db_path=db_path, faiss_path=faiss_path)
    chunks = retriever.retrieve_relevant_chunks(query=text, top_k=k, min_score=0.3)
    
    return [
        {
            "text": chunk.text,
            "score": chunk.score,
            "source": str(chunk.source_file),
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
```

**Reference:** `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py::EmbeddingRetriever.retrieve_relevant_chunks()`

**Default paths:**
- `db_path`: `case_law_data/results/personal_corpus_embeddings.db`
- `faiss_path`: `case_law_data/results/personal_corpus_embeddings.faiss`

---

### 4. Implement KG Query Wrapper

**Function:**

```python
from pathlib import Path
from typing import Dict, Optional
import json
import networkx as nx

def load_kg(kg_path: Optional[Path] = None) -> nx.Graph:
    """
    Load knowledge graph from JSON file.
    
    Args:
        kg_path: Optional path to KG JSON (default: case_law_data/facts_knowledge_graph.json)
        
    Returns:
        NetworkX graph
    """
    if kg_path is None:
        kg_path = Path("case_law_data/facts_knowledge_graph.json")
    
    with open(kg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return nx.readwrite.json_graph.node_link_graph(data)

def query_kg_for_entity(
    entity: str,
    kg_path: Optional[Path] = None,
    max_neighbors: int = 10
) -> Dict:
    """
    Query knowledge graph for entity and its neighbors.
    
    Args:
        entity: Entity name to search for
        kg_path: Optional path to KG JSON file
        max_neighbors: Maximum number of neighbor nodes to return
        
    Returns:
        Dictionary with:
        - "entity": str (matched entity name)
        - "attributes": dict (node attributes from KG)
        - "neighbors": list of dicts (neighbor nodes with edges)
        - "found": bool
    """
    g = load_kg(kg_path)
    
    # Search for entity (case-insensitive, partial match)
    entity_lower = entity.lower()
    matched_node = None
    for node in g.nodes():
        if entity_lower in node.lower():
            matched_node = node
            break
    
    if not matched_node:
        return {"entity": entity, "found": False, "attributes": {}, "neighbors": []}
    
    # Get attributes
    attrs = g.nodes[matched_node]
    
    # Get neighbors
    neighbors = []
    for neighbor in list(g.neighbors(matched_node))[:max_neighbors]:
        edge_data = g.get_edge_data(matched_node, neighbor, {})
        neighbors.append({
            "node": neighbor,
            "edge": edge_data,
            "attributes": g.nodes[neighbor]
        })
    
    return {
        "entity": matched_node,
        "found": True,
        "attributes": attrs,
        "neighbors": neighbors
    }
```

**Reference:**
- `nlp_analysis/code/KnowledgeGraph.py::KnowledgeGraph`
- `case_law_data/facts_knowledge_graph.json`

**KG loading:**
```python
kg = KnowledgeGraph()
kg.load_from_file(str(kg_path))
# Search nodes
# Get neighbors via kg.graph.neighbors(node)
```

---

### 5. Implement High-Level Evidence Finder

**Function:**

```python
from typing import List, Dict, Optional
from pathlib import Path
from nlp_analysis.code.pipeline import NLPAnalysisPipeline

def find_evidence_for_hypothesis(
    hypothesis: str,
    db_path: Optional[Path] = None,
    embedding_db_path: Optional[Path] = None,
    kg_path: Optional[Path] = None,
    use_nlp_pipeline: bool = False
) -> List[Dict]:
    """
    Use SQL + embeddings + KG to gather candidate evidence for a hypothesis.
    
    Args:
        hypothesis: Hypothesis to find evidence for (e.g., "Xi Mingze + 2016–2019 lectures")
        db_path: Optional path to SQLite database
        embedding_db_path: Optional path to embedding database
        kg_path: Optional path to KG JSON
        use_nlp_pipeline: Whether to run NLPAnalysisPipeline on excerpts
        
    Returns:
        List of evidence dictionaries, each with:
        - "text": str (evidence snippet)
        - "source": str (document name)
        - "source_type": str ("sql", "embedding", "kg")
        - "score": float (relevance score)
        - "entities": list (if use_nlp_pipeline=True)
        - "relations": list (if use_nlp_pipeline=True)
    """
    results = []
    
    # 1. SQL query
    sql_result = query_db_for_pattern(
        f"Find passages mentioning {hypothesis}, return doc ids + snippets.",
        db_path
    )
    if sql_result.get("success"):
        # Parse SQL result into evidence dicts
        pass
    
    # 2. Embedding search
    embedding_results = search_similar_passages(hypothesis, top_k=10)
    for chunk in embedding_results:
        results.append({
            "text": chunk.text,
            "source": str(chunk.source_file),
            "source_type": "embedding",
            "score": chunk.score
        })
    
    # 3. KG query (extract entities from hypothesis first)
    # Extract entity names from hypothesis (simple keyword extraction)
    # Query KG for each entity
    # Get related facts/events
    
    # 4. Optional: Run NLP pipeline on top results
    if use_nlp_pipeline:
        from nlp_analysis.code.pipeline import NLPAnalysisPipeline
        pipeline = NLPAnalysisPipeline()
        for result in results[:5]:  # Top 5 only (expensive)
            analysis = pipeline.analyze_text(result["text"], resolve_coref=False)
            result["entities"] = analysis.get("entities", {}).get("entities", [])
            result["relations"] = analysis.get("causal", {}).get("causal_relations", [])
    
    return results
```

---

### 6. Add Helper for Query Template Generation

**Function:**

```python
def build_query_from_hypothesis(hypothesis: str) -> Dict[str, str]:
    """
    Convert hypothesis string into structured query components.
    
    Example:
        "Xi Mingze + 2016–2019 lectures" →
        {
            "entities": ["Xi Mingze"],
            "keywords": ["lecture", "lectures"],
            "date_range": {"start": "2016", "end": "2019"},
            "sql_query": "Xi Mingze AND (lecture OR lectures) AND (2016 OR 2017 OR 2018 OR 2019)"
        }
    
    Args:
        hypothesis: Hypothesis string
        
    Returns:
        Dictionary with query components
    """
    # Simple parsing:
    # - Extract entity names (capitalized words, known entities)
    # - Extract date ranges (YYYY-YYYY, YYYY)
    # - Extract keywords
    # - Build SQL-like query string
    pass
```

---

### 7. Add to Package Exports

**Update `fact_engine/__init__.py`:**
```python
from .evidence_query import (
    query_db_for_pattern,
    search_similar_passages,
    query_kg_for_entity,
    find_evidence_for_hypothesis,
    build_query_from_hypothesis
)
```

---

## ✅ Success Criteria

1. Can call `query_db_for_pattern()` and get SQL results
2. Can call `search_similar_passages()` and get embedding results
3. Can call `query_kg_for_entity()` and get KG neighbors
4. Can call `find_evidence_for_hypothesis()` and get combined results from all sources
5. Handles missing files gracefully (returns empty results, logs warnings)

---

## 🧪 Testing

Create test script:
```python
from fact_engine import find_evidence_for_hypothesis
from pathlib import Path

# Test hypothesis
hypothesis = "Xi Mingze + 2016–2019 lectures"

# Find evidence
results = find_evidence_for_hypothesis(
    hypothesis,
    db_path=Path("case_law_data/unified_corpus.db"),
    embedding_db_path=Path("case_law_data/results/personal_corpus_embeddings.db"),
    kg_path=Path("case_law_data/facts_knowledge_graph.json")
)

print(f"Found {len(results)} evidence snippets")
for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. [{result['source_type']}] {result['source']}")
    print(f"   Score: {result['score']:.3f}")
    print(f"   Text: {result['text'][:200]}...")
```

---

## 📝 Notes

- Handle ImportError gracefully if LangChain/EmbeddingRetriever/KG not available
- Default paths should be configurable via environment variables or config file
- SQL queries can be slow - consider caching or limiting results
- Embedding search is fast but requires FAISS index
- KG queries are fast but require JSON file
- NLP pipeline is expensive - only use on top results if `use_nlp_pipeline=True`

