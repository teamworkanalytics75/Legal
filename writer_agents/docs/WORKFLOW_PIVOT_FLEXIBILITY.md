# 🔄 Workflow Pivot Flexibility Analysis

## How Flexible Can We Pivot Between Research → ML → Writing?

### 📊 Current Architecture

Your system has **3 distinct task modes** that can be coordinated:

```
┌─────────────────────────────────────────────────────────┐
│                    RESEARCH MODE                        │
│  - CaseLawResearcher (SK + LangChain)                  │
│  - Database queries (SQLite, MySQL)                    │
│  - Semantic search (Chroma)                            │
│  - CourtListener API                                    │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              MACHINE LEARNING MODE                      │
│  - CatBoost model (feature analysis)                   │
│  - SHAP importance (feature weights)                   │
│  - RefinementLoop (sub-coordinator)                    │
│  - Feature prediction & validation                     │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│                    WRITING MODE                         │
│  - AutoGen (exploration, review)                       │
│  - SK Functions (drafting, planning)                   │
│  - SK Plugins (quality enforcement)                    │
│  - Multi-model ensemble (parallel drafting)            │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Flexibility Levels

### **Level 1: Configuration-Based Pivoting** ⭐⭐⭐⭐⭐

**Easiest - Just flip config flags**

You can enable/disable entire workflow phases:

```python
config = WorkflowStrategyConfig(
    enable_sk_planner=False,      # Skip planning phase
    enable_quality_gates=False,    # Skip validation
    enable_autogen_review=False,   # Skip review
    max_iterations=1,              # Single pass (no refinement loops)
)
```

**What this enables:**
- ✅ **Research-only mode**: `RESEARCH → COMMIT` (skip writing)
- ✅ **Writing-only mode**: `DRAFT → COMMIT` (skip research)
- ✅ **ML-only mode**: `REFINE → VALIDATE` (skip research & writing)
- ✅ **Fast mode**: Disable all quality gates for speed

**Limitation:** Still follows linear phase order.

---

### **Level 2: Phase-Based Pivoting** ⭐⭐⭐⭐

**Moderate - Customize phase routing**

You can customize `WorkflowRouter.get_next_phase()` to skip phases:

```python
class CustomRouter(WorkflowRouter):
    def get_next_phase(self, current_phase, state):
        # Skip directly to ML analysis
        if current_phase == WorkflowPhase.EXPLORE:
            return WorkflowPhase.REFINE  # Jump to ML refinement

        # Or skip to writing
        if current_phase == WorkflowPhase.RESEARCH:
            return WorkflowPhase.DRAFT  # Skip planning

        return super().get_next_phase(current_phase, state)
```

**What this enables:**
- ✅ **Custom phase sequences**: `EXPLORE → REFINE → COMMIT` (skip research & writing)
- ✅ **Iterative loops**: `DRAFT → VALIDATE → REFINE → VALIDATE → ...`
- ✅ **Parallel execution**: Run multiple phases simultaneously

**Limitation:** Requires code changes to router.

---

### **Level 3: Direct Component Access** ⭐⭐⭐⭐⭐

**Most Flexible - Call components directly**

You can bypass the workflow entirely and call components directly:

```python
# Research-only task
conductor = Conductor(config)
research_results = await conductor.case_law_researcher.research_case_law(
    query="Harvard national security cases",
    top_k=20
)

# ML-only task
weak_features = await conductor.feature_orchestrator.analyze_draft(draft_text)
prediction = await conductor.feature_orchestrator.predict_draft_success(draft_text)

# Writing-only task
draft = await conductor.bridge.generate_draft(
    section="privacy_harm",
    context=research_results
)
```

**What this enables:**
- ✅ **Standalone research**: Query databases without workflow
- ✅ **Standalone ML analysis**: Analyze any text with CatBoost
- ✅ **Standalone writing**: Generate drafts without research/ML
- ✅ **Custom workflows**: Mix and match as needed

**No limitation:** Full programmatic control.

---

### **Level 4: Task-Specific Orchestration** ⭐⭐⭐

**Advanced - Create specialized orchestrators**

You could create lightweight orchestrators for specific tasks:

```python
# Research orchestrator
class ResearchOrchestrator:
    def __init__(self):
        self.researcher = CaseLawResearcher(...)
        self.catboost_analyzer = CatBoostAnalyzer(...)

    async def research_and_analyze(self, query):
        # Research + ML analysis pipeline
        results = await self.researcher.research_case_law(query)
        features = await self.catboost_analyzer.extract_features(results)
        return features

# Writing orchestrator
class WritingOrchestrator:
    def __init__(self):
        self.autogen = AgentFactory(...)
        self.sk_kernel = create_sk_kernel(...)

    async def write_with_context(self, context):
        # Writing-only pipeline
        draft = await self.autogen.generate_draft(context)
        return draft
```

**What this enables:**
- ✅ **Lightweight task-specific workflows**
- ✅ **Reduced overhead** (no full Conductor initialization)
- ✅ **Specialized optimization** for each task type

---

## 🔄 Current Pivot Capabilities

### **Research → ML Pivot** ✅ **Easy**

```python
# Research phase stores results in state
state.research_results = await conductor._execute_research_phase(...)

# ML phase can access results
weak_features = await conductor.feature_orchestrator.analyze_draft(
    draft_text,
    research_context=state.research_results  # ← Pass research results
)
```

**Status:** ✅ **Works** - Research results are stored in `WorkflowState` and can be passed to ML components.

---

### **ML → Writing Pivot** ✅ **Easy**

```python
# ML analysis identifies weak features
weak_features = await conductor.feature_orchestrator.analyze_draft(draft)

# Writing uses ML insights
improved_draft = await conductor.feature_orchestrator.strengthen_draft(
    draft,
    weak_features  # ← ML insights guide writing improvements
)
```

**Status:** ✅ **Works** - RefinementLoop coordinates ML analysis with SK plugins for writing improvements.

---

### **Writing → Research Pivot** ✅ **Easy**

```python
# Writing generates draft
draft = await conductor._execute_drafting_phase(...)

# Research can be triggered based on draft content
if needs_more_citations(draft):
    research_results = await conductor._execute_research_phase(...)
    # Update draft with research
```

**Status:** ✅ **Works** - Can re-enter RESEARCH phase from any point in workflow.

---

### **Parallel Execution** ⚠️ **Possible but not optimized**

```python
# Could run in parallel
import asyncio

research_task = conductor._execute_research_phase(...)
ml_task = conductor.feature_orchestrator.analyze_draft(...)
writing_task = conductor._execute_drafting_phase(...)

results = await asyncio.gather(research_task, ml_task, writing_task)
```

**Status:** ⚠️ **Possible** - Components are async, but not currently optimized for parallel execution.

---

## 🎯 Recommended Pivot Strategies

### **Strategy 1: Research-Heavy Workflow**

```python
config = WorkflowStrategyConfig(
    enable_sk_planner=False,      # Skip planning
    enable_quality_gates=True,     # Keep validation
    enable_autogen_review=False,   # Skip review
)

# Sequence: EXPLORE → RESEARCH → DRAFT → VALIDATE → REFINE → VALIDATE → COMMIT
# Focus: Deep research before any writing
```

### **Strategy 2: ML-Heavy Workflow**

```python
config = WorkflowStrategyConfig(
    enable_sk_planner=False,
    enable_quality_gates=True,
    enable_autogen_review=True,
    max_iterations=5,  # Multiple refinement loops
)

# Sequence: DRAFT → VALIDATE → REFINE → VALIDATE → REFINE → ...
# Focus: Iterative ML-driven improvements
```

### **Strategy 3: Writing-Heavy Workflow**

```python
config = WorkflowStrategyConfig(
    enable_sk_planner=True,
    enable_quality_gates=False,   # Skip validation
    enable_autogen_review=True,   # Keep review
    max_iterations=2,
)

# Sequence: EXPLORE → RESEARCH → PLAN → DRAFT → REVIEW → DRAFT → COMMIT
# Focus: Multiple writing iterations with review
```

### **Strategy 4: Fast Iteration Mode**

```python
config = WorkflowStrategyConfig(
    enable_sk_planner=False,
    enable_quality_gates=False,
    enable_autogen_review=False,
    max_iterations=1,
)

# Sequence: EXPLORE → RESEARCH → DRAFT → COMMIT
# Focus: Speed over quality
```

---

## 📈 Improving Flexibility

### **Current Limitations:**

1. **Linear phase order** - Can't easily reorder phases
2. **No parallel execution** - Phases run sequentially
3. **Router is hardcoded** - Requires code changes for custom routing
4. **State sharing** - Components share state, but not always optimally

### **Potential Enhancements:**

1. **Dynamic Phase Routing** - Configuration-driven phase sequences
   ```python
   config.custom_phase_sequence = [
       "RESEARCH",
       "ML_ANALYSIS",
       "DRAFT",
       "ML_VALIDATION",
       "COMMIT"
   ]
   ```

2. **Parallel Phase Execution** - Run independent phases simultaneously
   ```python
   config.parallel_phases = {
       "RESEARCH": ["CASE_LAW", "STATUTE_LOOKUP"],
       "ML_ANALYSIS": ["CATBOOST", "SHAP"]
   }
   ```

3. **Conditional Phase Routing** - Route based on intermediate results
   ```python
   if research_results.score < 0.7:
       next_phase = "MORE_RESEARCH"
   else:
       next_phase = "DRAFT"
   ```

4. **Task-Specific Orchestrators** - Lightweight specialized orchestrators
   ```python
   research_orchestrator = ResearchOrchestrator()
   ml_orchestrator = MLOrchestrator()
   writing_orchestrator = WritingOrchestrator()
   ```

---

## ✅ Summary

**Current Flexibility: ⭐⭐⭐⭐ (4/5)**

- ✅ **Easy config-based pivoting** - Enable/disable phases
- ✅ **Direct component access** - Bypass workflow entirely
- ✅ **Good state sharing** - Research → ML → Writing data flows
- ⚠️ **Linear phase order** - Can't easily reorder or parallelize
- ⚠️ **Router hardcoded** - Requires code changes for custom sequences

**Recommendation:** For most use cases, **Level 3 (Direct Component Access)** provides maximum flexibility. You can already call research, ML, and writing components independently without the full workflow.

**For advanced use cases:** Consider implementing dynamic phase routing or task-specific orchestrators.

