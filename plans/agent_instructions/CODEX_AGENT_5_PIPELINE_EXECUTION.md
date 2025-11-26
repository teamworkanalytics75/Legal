# Codex Agent 5: Pipeline Execution & Verification

**Workstream:** End-to-End Testing & Pipeline Execution  
**Status:** Ready to execute when instructed  
**Dependencies:** Agents 1-4 must be complete

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 5**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context.
3. **ONLY execute when instructed**: You should ONLY run pipelines when explicitly told "Agent 5: run pipeline" or "ready for testing".
4. **Report results clearly**: Always report what you did, what succeeded, what failed, and any issues encountered.

**Your instruction file:** This file (`CODEX_AGENT_5_PIPELINE_EXECUTION.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context):**
- `CODEX_AGENT_1_FACT_TYPES_REFACTOR.md` - Fact type refactoring
- `CODEX_AGENT_2_ENTITY_EXTRACTION.md` - Entity extraction & KG builder
- `CODEX_AGENT_3_KNOWLEDGE_GRAPH_LAYER.md` - Query layer
- `CODEX_AGENT_4_VALIDATOR_INTEGRATION.md` - Validator integration

---

## 🎯 Objective

Execute end-to-end motion generation pipeline and verify it correctly uses the **master facts database** and **master knowledge graph**. The motion generation system is **read-only** from these master sources—it does NOT build or generate them at runtime.

---

## 🏗️ Architecture Overview

**CRITICAL: Master Database/Graph vs Motion Generation**

The system has two separate phases:

1. **Fact Maintenance** (separate operations, run when updating facts):
   - Build/update master Knowledge Graph from source documents
   - Extract facts into master SQLite database
   - These are **persistent artifacts** that exist independently

2. **Motion Generation** (reads from master sources):
   - Loads existing `case_law_data/lawsuit_facts_database.db`
   - Loads existing `case_law_data/facts_knowledge_graph.json`
   - Uses these as **read-only sources of truth**
   - Does NOT build or generate them at runtime

**Master Facts Database Location:**
- Database: `case_law_data/lawsuit_facts_database.db`
- Knowledge Graph: `case_law_data/facts_knowledge_graph.json`
- CSV Export: `case_law_data/facts_export.csv` (optional, for spreadsheet view)

---

## 📋 Pipeline Execution Tasks

### Phase 1: Verify Master Facts Database Exists (Required Before Motion Generation)

**Before running motion generation, verify the master database/graph exist:**

### Task 1: Pre-Flight Checks

**Verify master facts database and graph exist (motion generation requires these):**

```bash
# Check master database exists (REQUIRED for motion generation)
if [ -f "case_law_data/lawsuit_facts_database.db" ]; then
    echo "✅ Master database exists"
    python3 -c "
    import sqlite3
    conn = sqlite3.connect('case_law_data/lawsuit_facts_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM fact_registry')
    count = cursor.fetchone()[0]
    print(f'   Facts in database: {count}')
    cursor.execute('SELECT DISTINCT fact_type FROM fact_registry')
    types = [row[0] for row in cursor.fetchall()]
    print(f'   Fact types: {types}')
    conn.close()
    "
else
    echo "❌ Master database NOT FOUND - motion generation will fail"
    echo "   Run fact maintenance tasks first (see Phase 2)"
fi

# Check master knowledge graph exists (OPTIONAL but recommended)
if [ -f "case_law_data/facts_knowledge_graph.json" ]; then
    echo "✅ Master knowledge graph exists"
    python3 -c "
    import json
    with open('case_law_data/facts_knowledge_graph.json') as f:
        data = json.load(f)
    nodes = len(data.get('nodes', []))
    edges = len(data.get('edges', []))
    print(f'   Graph nodes: {nodes}, edges: {edges}')
    "
else
    echo "⚠️  Master knowledge graph NOT FOUND - motion will use database only"
fi

# Check Python environment
python3 --version
python3 -c "import sqlite3, networkx, spacy; print('✅ Dependencies OK')"
```

**Report:**
- [ ] Master database exists (REQUIRED)
- [ ] Master knowledge graph exists (OPTIONAL)
- [ ] Fact count in database
- [ ] Dependencies available

**If master database/graph are missing, run Phase 2 (Fact Maintenance) first.**

---

## 🔧 Phase 2: Fact Maintenance (Run Only When Updating Facts)

**These tasks build/update the master facts database and graph. Run these separately when you want to update facts, NOT as part of motion generation.**

### Task 2: Build Knowledge Graph (Maintenance Only)

**⚠️ This is a MAINTENANCE task. Only run when updating the master knowledge graph.**

**Execute the Knowledge Graph builder with enhanced features:**

```bash
# Check source documents exist first
ls -la case_law_data/lawsuit_source_documents/*.txt 2>/dev/null || \
ls -la case_law_data/tmp_corpus/*.txt 2>/dev/null || \
echo "❌ No source documents found"

# Build graph with entity merging and progress logging
python3 writer_agents/scripts/build_fact_knowledge_graph.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --output case_law_data/facts_knowledge_graph.json \
    --merge-similar \
    --merge-threshold 0.92 \
    --log-every 5
```

**Note:** Script will automatically fallback to `case_law_data/tmp_corpus` if `lawsuit_source_documents` is empty.

**Verify output:**
```bash
# Check graph file was created
ls -lh case_law_data/facts_knowledge_graph.json

# Verify graph structure
python3 -c "
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
kg = KnowledgeGraph()
kg.load_from_file('case_law_data/facts_knowledge_graph.json')
stats = kg.get_summary_stats()
print(f'✅ Entities: {stats[\"num_entities\"]}')
print(f'✅ Relations: {stats[\"num_relations\"]}')
print(f'✅ Entity Types: {stats[\"entity_types\"]}')
print(f'✅ Top Relations: {stats[\"top_relations\"][:5]}')
print(f'✅ Density: {stats[\"density\"]:.4f}')
print(f'✅ Connected: {stats[\"is_connected\"]}')
"
```

**Report:**
- [ ] Graph file created (size in MB)
- [ ] Entity count
- [ ] Relation count
- [ ] Entity types found
- [ ] Any errors or warnings

---

### Task 3: Extract Facts and Sync to Graph (Maintenance Only)

**⚠️ This is a MAINTENANCE task. Only run when updating the master facts database.**

**Run fact extraction to populate/update the master database:**

```bash
# Extract facts and optionally sync to graph
python3 writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db \
    --knowledge-graph-output case_law_data/facts_knowledge_graph.json
```

**Note:** Script will automatically fallback to `case_law_data/tmp_corpus` if `lawsuit_source_documents` is empty.

**Verify fact extraction:**
```bash
# Check database was populated
python3 -c "
import sqlite3
conn = sqlite3.connect('case_law_data/lawsuit_facts_database.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM fact_registry')
count = cursor.fetchone()[0]
print(f'✅ Facts in database: {count}')

cursor.execute('SELECT DISTINCT fact_type FROM fact_registry')
types = [row[0] for row in cursor.fetchall()]
print(f'✅ Fact types: {types}')
conn.close()
"
```

**Report:**
- [ ] Facts extracted (count)
- [ ] Fact types found (list all 6 categories)
- [ ] Graph file updated/created
- [ ] Any errors or warnings

---

### Task 4: Verify Graph Quality (Maintenance Only)

**⚠️ This is a MAINTENANCE task. Verify the master graph after building/updating it.**

**Check graph contains expected entities and relationships:**

```bash
# Comprehensive graph verification
python3 << 'PYTHON'
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from pathlib import Path

graph_path = Path("case_law_data/facts_knowledge_graph.json")
if not graph_path.exists():
    print("❌ Graph file not found")
    exit(1)

kg = KnowledgeGraph()
kg.load_from_file(str(graph_path))
stats = kg.get_summary_stats()

print("=" * 60)
print("KNOWLEDGE GRAPH QUALITY REPORT")
print("=" * 60)
print(f"Entities: {stats['num_entities']}")
print(f"Relations: {stats['num_relations']}")
print(f"Entity Types: {stats['entity_types']}")
print(f"Top Relations: {stats['top_relations'][:10]}")
print(f"Average Degree: {stats['avg_degree']:.2f}")
print(f"Density: {stats['density']:.4f}")
print(f"Connected: {stats['is_connected']}")

# Check for fact nodes
fact_nodes = [n for n in kg.graph.nodes() if str(n).startswith("fact:")]
print(f"\nFact Nodes: {len(fact_nodes)}")

# Check for lawsuit-specific entities
lawsuit_keywords = ["harvard", "ogc", "hong kong", "statement", "allegation", "date"]
found_keywords = []
for node in kg.graph.nodes():
    node_lower = str(node).lower()
    for keyword in lawsuit_keywords:
        if keyword in node_lower and keyword not in found_keywords:
            found_keywords.append(keyword)

print(f"Lawsuit Keywords Found: {found_keywords}")

# Check relationship types
relation_types = set()
for u, v, data in kg.graph.edges(data=True):
    rel = data.get("relation", "unknown")
    relation_types.add(rel)

print(f"Relationship Types: {sorted(relation_types)}")
print("=" * 60)
PYTHON
```

**Report:**
- [ ] Graph quality metrics (entities, relations, density)
- [ ] Fact nodes present
- [ ] Lawsuit-specific entities found
- [ ] Relationship types present (relates_to, references_date, etc.)
- [ ] Any quality issues

---

### Task 5: Test Query Interface (Maintenance Only)

**⚠️ This is a MAINTENANCE task. Test that queries work against the master graph.**

**Verify FactGraphQuery works correctly:**

```bash
# Test query interface
python3 << 'PYTHON'
from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from writer_agents.code.validation.fact_graph_query import FactGraphQuery
from pathlib import Path

graph_path = Path("case_law_data/facts_knowledge_graph.json")
if not graph_path.exists():
    print("❌ Graph file not found")
    exit(1)

kg = KnowledgeGraph()
kg.load_from_file(str(graph_path))
query = FactGraphQuery(kg)

print("=" * 60)
print("QUERY INTERFACE TEST")
print("=" * 60)

# Test 1: Get all facts by type
print("\n1. Testing get_all_facts_by_type('date'):")
date_facts = query.get_all_facts_by_type("date")
print(f"   Found {len(date_facts)} date facts")
if date_facts:
    print(f"   Example: {date_facts[0]}")

# Test 2: Get fact hierarchy
print("\n2. Testing get_fact_hierarchy():")
hierarchy = query.get_fact_hierarchy()
print(f"   Fact types: {list(hierarchy.keys())}")
for fact_type, values in hierarchy.items():
    print(f"   - {fact_type}: {len(values)} facts")

# Test 3: Semantic search
print("\n3. Testing find_similar_facts():")
similar = query.find_similar_facts("OGC email", top_k=3)
print(f"   Found {len(similar)} similar facts")
if similar:
    print(f"   Example: {similar[0].fact_value} (score: {similar[0].score:.2f})")

# Test 4: Verify fact exists
print("\n4. Testing verify_fact_exists():")
if date_facts:
    test_fact = date_facts[0]
    exists = query.verify_fact_exists(test_fact.get("type", "date"), test_fact.get("value", ""))
    print(f"   Fact exists: {exists}")

print("\n✅ Query interface tests complete")
print("=" * 60)
PYTHON
```

**Report:**
- [ ] Query interface loads successfully
- [ ] All query methods work
- [ ] Semantic search returns results
- [ ] Any query errors

---

## 🚀 Phase 3: Motion Generation (Reads from Master Database/Graph)

**Motion generation is READ-ONLY from the master database/graph. It does NOT build them.**

### Task 6: Run Motion Generation Pipeline

**Execute full motion generation. It will automatically load and use the master database/graph:**

```bash
# Verify master database exists first (REQUIRED)
if [ ! -f "case_law_data/lawsuit_facts_database.db" ]; then
    echo "❌ ERROR: Master database not found. Run Phase 2 (Fact Maintenance) first."
    exit 1
fi

# Run motion generation (reads from master database/graph)
python3 writer_agents/scripts/generate_optimized_motion.py \
    --case-summary "Motion to seal sensitive personal information" \
    --enable-google-docs 2>&1 | tee motion_generation.log
```

**What motion generation does:**
- ✅ Loads existing `case_law_data/lawsuit_facts_database.db` (read-only)
- ✅ Loads existing `case_law_data/facts_knowledge_graph.json` if present (read-only)
- ✅ Uses facts from database/graph for validation and prompts
- ❌ Does NOT build or generate the database/graph

**Monitor for these log messages:**
- `[FACTS] Loading facts from database: case_law_data/lawsuit_facts_database.db`
- `[FACTS] Knowledge Graph loaded successfully` (if graph exists)
- `[FACTS] Using graph-backed fact validation` (if graph exists)
- `[FACTS] Graph query results: X facts found`
- `[FACTS] Semantic search matched: Y facts`
- `[FACTS] ContradictionDetector using graph-backed facts`

**Check logs for errors:**
```bash
# Extract key log messages
grep -i "knowledge graph\|graph-backed\|fact.*graph\|graph.*query" motion_generation.log | tail -20
grep -i "error\|exception\|failed\|warning" motion_generation.log | tail -20
```

**Report:**
- [ ] Motion generation completed
- [ ] Knowledge Graph loaded (check logs)
- [ ] Graph-backed validation used (check logs)
- [ ] Any errors or warnings
- [ ] Motion quality (if generated)

---

### Task 7: Verify Validator Integration

**Test that validators correctly use the master database/graph (read-only):**

```bash
# Run validator tests
python3 -m pytest tests/test_personal_facts_verifier.py -v
python3 -m pytest tests/test_workflow_personal_facts_gate.py -v
python3 -m pytest tests/test_fact_graph_query.py -v 2>/dev/null || echo "Test file may not exist"
```

**Report:**
- [ ] All tests pass
- [ ] Validators use graph when available
- [ ] Fallback works when graph missing
- [ ] Any test failures

---

### Task 8: End-to-End Verification

**Create comprehensive verification report:**

```bash
# Generate verification report
python3 << 'PYTHON'
from pathlib import Path
import json
import sqlite3

print("=" * 60)
print("END-TO-END VERIFICATION REPORT")
print("=" * 60)

# 1. Check files exist
files_to_check = [
    "case_law_data/facts_knowledge_graph.json",
    "case_law_data/lawsuit_facts_database.db",
    "writer_agents/code/validation/fact_graph_query.py",
    "writer_agents/code/validation/kg_fact_adapter.py",
]

print("\n1. File Existence:")
for file_path in files_to_check:
    path = Path(file_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    size = f"({path.stat().st_size / 1024:.1f} KB)" if exists else ""
    print(f"   {status} {file_path} {size}")

# 2. Check database
print("\n2. Database Status:")
db_path = Path("case_law_data/lawsuit_facts_database.db")
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fact_registry")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT fact_type FROM fact_registry")
    types = [row[0] for row in cursor.fetchall()]
    conn.close()
    print(f"   ✅ Facts in database: {count}")
    print(f"   ✅ Fact types: {types}")
else:
    print("   ❌ Database not found")

# 3. Check graph
print("\n3. Knowledge Graph Status:")
graph_path = Path("case_law_data/facts_knowledge_graph.json")
if graph_path.exists():
    try:
        with open(graph_path) as f:
            graph_data = json.load(f)
        nodes = len(graph_data.get("nodes", []))
        edges = len(graph_data.get("edges", []))
        print(f"   ✅ Graph nodes: {nodes}")
        print(f"   ✅ Graph edges: {edges}")
    except Exception as e:
        print(f"   ❌ Graph file invalid: {e}")
else:
    print("   ❌ Graph file not found")

# 4. Check integration
print("\n4. Integration Status:")
integration_files = [
    "writer_agents/code/validation/contradiction_detector.py",
    "writer_agents/code/validation/personal_facts_verifier.py",
    "writer_agents/code/WorkflowOrchestrator.py",
]

for file_path in integration_files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        has_kg = "FactGraphQuery" in content or "knowledge_graph" in content
        status = "✅" if has_kg else "⚠️"
        print(f"   {status} {file_path} (KG integration: {has_kg})")
    else:
        print(f"   ❌ {file_path} not found")

print("\n" + "=" * 60)
PYTHON
```

**Report:**
- [ ] All files present
- [ ] Database populated
- [ ] Graph valid
- [ ] Integration complete
- [ ] Overall status (ready/not ready)

---

## 📊 Success Criteria

### Minimum Requirements
- [ ] Knowledge Graph file created and valid
- [ ] At least 10 entities in graph
- [ ] At least 5 relationships in graph
- [ ] Fact database populated with at least 1 fact per type
- [ ] Query interface works
- [ ] Validators can load and use graph
- [ ] Motion generation completes without errors
- [ ] All tests pass

### Quality Targets
- [ ] 50+ entities in graph
- [ ] 20+ relationships in graph
- [ ] All 6 fact types populated
- [ ] Lawsuit-specific entities found (Harvard, OGC, HK Statement)
- [ ] Temporal relationships present
- [ ] Graph-backed validation working in logs
- [ ] Semantic search returns relevant results

---

## 🚨 Troubleshooting

### If graph is empty or small:
1. Check source documents exist and contain text
2. Verify entity extractor is finding entities
3. Try lower merge threshold: `--merge-threshold 0.85`
4. Check logs for extraction errors

### If validators don't use graph:
1. Verify graph file exists and is valid JSON
2. Check logs for "Knowledge Graph loaded" message
3. Verify `FactGraphQuery` import succeeds
4. Check fallback is working (should still work without graph)

### If tests fail:
1. Check if graph file exists
2. Verify database is populated
3. Check test fixtures are correct
4. Run tests with `-v` flag for verbose output

### If motion generation fails:
1. Check all dependencies installed
2. Verify source documents exist
3. Check logs for specific error messages
4. Try without `--enable-google-docs` flag

---

## 📝 Reporting Format

**When reporting results, use this format:**

```
Agent 5 Pipeline Execution Report
==================================

Date: [timestamp]
Status: [SUCCESS / PARTIAL / FAILED]

Task 1: Pre-Flight Checks
- [✅/❌] Source documents: [count]
- [✅/❌] Database: [status]
- [✅/❌] Dependencies: [status]

Task 2: Build Knowledge Graph
- [✅/❌] Graph file created: [size]
- [✅/❌] Entities: [count]
- [✅/❌] Relations: [count]
- [⚠️] Warnings: [list]

Task 3: Extract Facts
- [✅/❌] Facts extracted: [count]
- [✅/❌] Fact types: [list]
- [⚠️] Issues: [list]

Task 4: Verify Graph Quality
- [✅/❌] Quality metrics: [summary]
- [✅/❌] Lawsuit entities: [found/not found]
- [⚠️] Quality issues: [list]

Task 5: Test Query Interface
- [✅/❌] Query interface: [working/not working]
- [✅/❌] Semantic search: [working/not working]
- [⚠️] Issues: [list]

Task 6: Run Motion Generation
- [✅/❌] Motion generated: [yes/no]
- [✅/❌] Graph-backed validation: [used/not used]
- [⚠️] Errors: [list]

Task 7: Verify Validator Integration
- [✅/❌] Tests pass: [count/total]
- [✅/❌] Graph integration: [working/not working]
- [⚠️] Failures: [list]

Task 8: End-to-End Verification
- [✅/❌] Overall status: [ready/not ready]
- [✅/❌] All components: [working/not working]
- [⚠️] Blockers: [list]

Summary:
[Brief summary of overall status and any critical issues]
```

---

## 🎯 When to Execute

### Motion Generation (Phase 3):
**Execute when:**
- Explicitly instructed: "Agent 5: run motion pipeline" or "ready for testing"
- Master database exists (`case_law_data/lawsuit_facts_database.db`)
- You want to generate a motion using existing facts

**Do NOT execute:**
- If master database is missing (run Phase 2 first)
- Without explicit instruction
- If there are known blocking issues

### Fact Maintenance (Phase 2):
**Execute when:**
- Explicitly instructed: "Agent 5: update facts" or "rebuild master database"
- Source documents have been updated
- You want to refresh the master facts database/graph

**Do NOT execute:**
- As part of motion generation (motion generation is read-only)
- If source documents are missing

---

## 📁 Key Files (Master Facts Database)

**Master Database/Graph (read by motion generation):**
- `case_law_data/lawsuit_facts_database.db` - **Master facts database (REQUIRED)**
- `case_law_data/facts_knowledge_graph.json` - **Master knowledge graph (OPTIONAL)**
- `case_law_data/facts_export.csv` - CSV export for spreadsheet view (optional)

**Motion Generation Output:**
- `motion_generation.log` - Motion generation logs
- `writer_agents/outputs/` - Generated motion documents

**Test Files:**
- Test output files in `tests/` directory

---

## ✅ Completion Checklist

### For Motion Generation (Phase 3):
- [ ] Master database exists and is populated
- [ ] Motion generation completed successfully
- [ ] Motion generation loaded master database/graph (check logs)
- [ ] Validators used master database/graph (check logs)
- [ ] Motion quality acceptable

### For Fact Maintenance (Phase 2):
- [ ] Knowledge Graph built/updated
- [ ] Facts extracted to master database
- [ ] Graph quality verified
- [ ] Query interface tested
- [ ] Master database/graph ready for motion generation

### Overall:
- [ ] Comprehensive report generated
- [ ] Issues documented
- [ ] Recommendations provided
- [ ] System status confirmed (ready/not ready)

