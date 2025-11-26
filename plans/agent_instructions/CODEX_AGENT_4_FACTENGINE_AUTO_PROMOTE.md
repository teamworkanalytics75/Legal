# Codex Agent 4: FactEngine Auto-Promotion & Integration

**Workstream:** FactEngine Module Build - Orchestration Layer  
**Status:** ✅ Complete — auto_promote.py, run_fact_engine.py, helpers.py implemented; all dependencies available  
**Dependencies:** Agents 1-3 (needs all components: schema, ML model, evidence query)
  - ✅ Agent 1 schema available
  - ✅ Agent 2 ML model complete (training/explanation ready)
  - ✅ Agent 3 evidence query complete (hypothesis search ready)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 4**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read `CODEX_AGENTS_FACTENGINE_OVERVIEW.md` and all `CODEX_AGENT_*_FACTENGINE_*.md` files to understand the full context.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 4**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 4** and follow those.

**Your instruction file:** This file contains YOUR specific tasks. This is the source of truth.

**Dependencies:** Wait for Agents 1-3 to complete, or start by importing their modules and handle ImportError gracefully.

---

## 🎯 Objective

Create the auto-promotion logic that uses SHAP insights to discover missing facts, and build the main entrypoint that orchestrates the full FactEngine pipeline. Also integrate with existing legal agents.

---

## 📦 Tasks

### 1. Create Auto-Promote Module (auto_promote.py)

**File:** `fact_engine/auto_promote.py`

**Dependencies:**
- `fact_engine.schema` (Fact, FactImportance)
- `fact_engine.ml_importance` (FactImportanceModel, get_global_importance)
- `fact_engine.evidence_query` (find_evidence_for_hypothesis)
- `nlp_analysis.code.EntityRelationExtractor` (for S-V-O extraction)
- `fuzzywuzzy` or `rapidfuzz` (for deduplication)
- `pandas`

---

### 2. Implement Feature Gap Detection

**Function:**

```python
from typing import List, Dict
from .schema import Fact, FactImportance
from .ml_importance import FactImportanceModel

def find_feature_gaps(
    facts: List[Fact],
    shap_summary: Dict[str, float],
    threshold: float = 0.1
) -> List[str]:
    """
    Find missing fact areas based on high-importance SHAP features.
    
    For each high-importance feature, check if all matching facts are captured.
    Return list of "feature hypotheses" (e.g., "Xi Mingze + 2016–2019 lectures").
    
    Args:
        facts: List of existing Fact objects
        shap_summary: Dictionary mapping feature_name -> mean_abs_shap_value
        threshold: Minimum SHAP value to consider "high importance"
        
    Returns:
        List of hypothesis strings
    """
    hypotheses = []
    
    # 1. Get top features (above threshold)
    top_features = {
        feat: val for feat, val in shap_summary.items() 
        if val >= threshold
    }
    
    # 2. For each feature, check coverage:
    #    - Entity features: Check if we have facts mentioning that entity
    #    - Date features: Check if we have facts in that date range
    #    - Keyword features: Check if we have facts with those keywords
    #    - Graph features: Check if we have facts related to those BN nodes
    
    # 3. Generate hypotheses for missing areas:
    #    - "Xi Mingze + 2016–2019" if entity=Xi_Mingze and date=2016-2019 are both high
    #    - "GSS + Level 2 advisory" if mentions_gss and mentions_level_2 are both high
    
    return hypotheses
```

**Example logic:**
- If `mentions_xi_mingze` and `event_date_2019` are both high → hypothesis: "Xi Mingze + 2019 events"
- If `mentions_gss` and `mentions_china_travel` are both high → hypothesis: "GSS + China travel advisory"

---

### 3. Implement Fact Proposal from Hypothesis

**Function:**

```python
from typing import List
from .schema import Fact
from .evidence_query import find_evidence_for_hypothesis
from nlp_analysis.code.EntityRelationExtractor import EntityRelationExtractor

def propose_new_facts_from_hypothesis(
    hypothesis: str,
    existing_facts: List[Fact],
    use_nlp_pipeline: bool = True
) -> List[Fact]:
    """
    Search corpus for evidence matching hypothesis and extract candidate facts.
    
    CRITICAL GUARDRAILS:
    1. Every new Fact MUST be grounded in an actual snippet from the corpus.
       - source_excerpt MUST be actual text from DB/embeddings, not LLM paraphrase
       - You can paraphrase into proposition, but keep underlying text as evidentiary anchor
    2. Require at least one "hard signal":
       - SQL/embedding hit AND entity/edge in KG for key entities (Xi, GSS, EsuWiki, etc.)
       - OR ≥ 2 independent corpus snippets matching the same pattern
    
    Steps:
    1. Call find_evidence_for_hypothesis() to get evidence snippets
    2. For each snippet, run S-V-O extraction:
       - Use EntityRelationExtractor or NLPAnalysisPipeline
       - Extract subject, verb, object
    3. Create Fact objects:
       - truth_status = "True" if procedural/structural (dates, "Email sent", "Statement filed")
       - truth_status = "Alleged" if claim in pleading
       - truth_status = "HostileFalseClaim" if from hostile article
    4. Fill source_document, source_excerpt (MUST be actual corpus text), evidence_type, speaker, actor_role
    
    Args:
        hypothesis: Hypothesis string (e.g., "Xi Mingze + 2016–2019 lectures")
        existing_facts: List of existing facts (for context)
        use_nlp_pipeline: Whether to use full NLP pipeline (slower but better)
        
    Returns:
        List of candidate Fact objects (all grounded in actual corpus snippets)
    """
    from .evidence_query import find_evidence_for_hypothesis
    from nlp_analysis.code.pipeline import NLPAnalysisPipeline
    from nlp_analysis.code.EntityRelationExtractor import EntityRelationExtractor
    
    new_facts = []
    
    # 1. Find evidence
    evidence = find_evidence_for_hypothesis(
        hypothesis,
        use_nlp_pipeline=use_nlp_pipeline
    )
    
    # Check guardrail: need at least 2 independent snippets OR 1 snippet + KG match
    if len(evidence) < 2:
        # Check if we have KG match for key entities
        from .evidence_query import query_kg_for_entity
        # Extract entity names from hypothesis (simple keyword extraction)
        # If no KG match, skip this hypothesis
        pass
    
    # 2. Extract facts from evidence
    if use_nlp_pipeline:
        pipeline = NLPAnalysisPipeline()
    else:
        extractor = EntityRelationExtractor()
    
    for ev in evidence:
        snippet_text = ev["text"]  # This is the actual corpus text - MUST preserve
        
        # Run entity/relation extraction
        if use_nlp_pipeline:
            analysis = pipeline.analyze_text(snippet_text, resolve_coref=False)
            entities = analysis.get("entities", {}).get("entities", [])
            relations = analysis.get("causal", {}).get("causal_relations", [])
        else:
            analysis = extractor.analyze_document(snippet_text)
            entities = analysis.get("entities", [])
            relations = analysis.get("relations", [])
        
        # Build propositions from relations
        for rel in relations:
            # rel is (subject, relation, object) or similar
            subject, verb, obj = rel[:3]
            proposition = f"{subject} {verb} {obj}."
            
            # Determine truth_status from source
            truth_status = "Alleged"  # default
            if "email" in ev["source"].lower():
                truth_status = "True"  # Emails are factual
            elif "wechat" in ev["source"].lower() or "monkey" in ev["source"].lower():
                truth_status = "HostileFalseClaim"
            
            # Create Fact - source_excerpt MUST be actual corpus text
            fact = Fact(
                fact_id=f"PROPOSED_{len(new_facts) + 1}",
                subject=subject,
                verb=verb,
                obj=obj,
                proposition=proposition,  # Can be paraphrased
                source_document=ev["source"],
                source_excerpt=snippet_text[:500],  # MUST be actual corpus text
                truth_status=truth_status,
                extraction_method="AUTO_PROMOTE",
                extraction_confidence=ev.get("score", 0.5)
            )
            new_facts.append(fact)
    
    return new_facts

**Simplified S-V-O extraction (if NLP pipeline too slow):**
- Use regex patterns: `(subject) (verb) (object)`
- Use dependency parsing (spaCy) for simple sentences
- Fallback to keyword extraction if no clear S-V-O

---

### 4. Implement Deduplication

**Function:**

```python
from typing import List, Tuple
from .schema import Fact
from rapidfuzz import fuzz

def deduplicate_facts(
    new_facts: List[Fact],
    existing_facts: List[Fact],
    similarity_threshold: float = 0.85
) -> Tuple[List[Fact], List[str]]:
    """
    Remove duplicates from new_facts by comparing to existing_facts.
    
    Uses fuzzy string similarity on propositions.
    
    Args:
        new_facts: List of candidate Fact objects
        existing_facts: List of existing Fact objects
        similarity_threshold: Minimum similarity to consider duplicate (0-1)
        
    Returns:
        Tuple of (deduplicated_new_facts, merged_fact_ids)
    """
    deduplicated = []
    merged_ids = []
    
    existing_propositions = {f.proposition.lower() for f in existing_facts}
    
    for new_fact in new_facts:
        is_duplicate = False
        
        # Check exact match (case-insensitive)
        if new_fact.proposition.lower() in existing_propositions:
            is_duplicate = True
        
        # Check fuzzy match
        if not is_duplicate:
            for existing_fact in existing_facts:
                similarity = fuzz.ratio(
                    new_fact.proposition.lower(),
                    existing_fact.proposition.lower()
                ) / 100.0
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    merged_ids.append(existing_fact.fact_id)
                    break
        
        if not is_duplicate:
            deduplicated.append(new_fact)
    
    return deduplicated, merged_ids
```

---

### 5. Implement Main Auto-Promote Pipeline

**Function:**

```python
from typing import List
from pathlib import Path
from .schema import Fact
from .ml_importance import FactImportanceModel
from .evidence_query import find_evidence_for_hypothesis

def auto_promote_missing_facts(
    facts: List[Fact],
    model: FactImportanceModel,
    max_new_facts: int = 50
) -> List[Fact]:
    """
    Main pipeline: Use SHAP to find gaps, search corpus, propose new facts.
    
    Steps:
    1. Get global SHAP importance from model
    2. Find feature gaps (missing areas)
    3. For each gap, propose new facts
    4. Deduplicate
    5. Assign safety_risk and public_exposure
    6. Return new facts
    
    Args:
        facts: List of existing Fact objects
        model: Trained FactImportanceModel
        max_new_facts: Maximum number of new facts to propose
        
    Returns:
        List of new Fact objects to add to canonical table
    """
    # 1. Get SHAP summary
    shap_summary = model.get_global_importance(facts)
    
    # 2. Find gaps
    hypotheses = find_feature_gaps(facts, shap_summary)
    
    # 3. Propose facts for each hypothesis
    all_new_facts = []
    for hypothesis in hypotheses[:10]:  # Limit to top 10 hypotheses
        proposed = propose_new_facts_from_hypothesis(hypothesis, facts)
        all_new_facts.extend(proposed)
        
        if len(all_new_facts) >= max_new_facts:
            break
    
    # 4. Deduplicate
    deduplicated, merged_ids = deduplicate_facts(all_new_facts, facts)
    
    # 5. Assign safety_risk and public_exposure (use heuristics)
    for fact in deduplicated:
        # Calculate safety_risk based on keywords
        prop_lower = fact.proposition.lower()
        if any(kw in prop_lower for kw in ["torture", "persecution"]):
            fact.safety_risk = "Extreme"
        elif "risk" in prop_lower or "threat" in prop_lower:
            fact.safety_risk = "High"
        elif any(kw in prop_lower for kw in ["prc", "china", "xi", "esuwiki"]):
            fact.safety_risk = "Medium"
        else:
            fact.safety_risk = "Low"
        
        # Calculate public_exposure from source
        source_lower = fact.source_document.lower() if fact.source_document else ""
        if "wechat" in source_lower or "article" in source_lower:
            fact.public_exposure = "Public"
        elif "email" in source_lower:
            fact.public_exposure = "NotPublic"
        else:
            fact.public_exposure = "SemiPublic"
    
    return deduplicated[:max_new_facts]
```

---

### 6. Create Sealing Index Builder

**Function:**

```python
from typing import List
import pandas as pd
from .schema import Fact, FactImportance

def build_sealing_index(
    facts: List[Fact],
    importances: List[FactImportance]
) -> pd.DataFrame:
    """
    Build sealing index CSV with facts ranked by importance for sealing arguments.
    
    Args:
        facts: List of Fact objects
        importances: List of FactImportance objects (from model.explain_facts())
        
    Returns:
        DataFrame with columns: fact_id, proposition, safety_risk, public_exposure, 
        importance_score, risk_rationale
    """
    # Join facts and importances on fact_id
    fact_dict = {f.fact_id: f for f in facts}
    imp_dict = {imp.fact_id: imp for imp in importances}
    
    rows = []
    for fact_id, fact in fact_dict.items():
        imp = imp_dict.get(fact_id)
        rows.append({
            "fact_id": fact_id,
            "proposition": fact.proposition,
            "safety_risk": fact.safety_risk or "Unknown",
            "public_exposure": fact.public_exposure or "Unknown",
            "importance_score": imp.importance_score if imp else 0.0,
            "risk_rationale": fact.risk_rationale or ""
        })
    
    df = pd.DataFrame(rows)
    # Sort by importance_score descending
    df = df.sort_values("importance_score", ascending=False)
    return df
```

**Output file:** `case_law_data/facts_sealing_index.csv`

---

### 7. Create BN Mapping Updater

**Function:**

```python
from typing import List
import pandas as pd
from .schema import Fact, BNMapping

def update_bn_mapping(facts: List[Fact]) -> pd.DataFrame:
    """
    Generate rule-based BN mappings for facts.
    
    Rules:
    - If proposition/source_excerpt contains "Monkey"/"Resume"/"WeChat" + racialized language
      → map to ART_Monkey_WeChat or ART_Resume_WeChat and HARM_Security (weight 0.8-1.0)
    - If about OGC letters (Apr 7, Apr 18, Aug 11, 2025)
      → map to COR_OGC_2025_04_07, COR_OGC_2025_04_18, COR_OGC_2025_08_11, HUB_Lawsuit
    
    Args:
        facts: List of Fact objects
        
    Returns:
        DataFrame with columns: fact_id, bn_node, mapping_weight, mapping_confidence, mapping_source, notes
    """
    mappings = []
    
    for fact in facts:
        prop_lower = fact.proposition.lower()
        source_lower = (fact.source_excerpt or "").lower()
        text_lower = prop_lower + " " + source_lower
        
        # Rule 1: Monkey/Resume/WeChat + racialized language
        if any(kw in text_lower for kw in ["monkey", "resume", "wechat"]):
            if "monkey" in text_lower:
                mappings.append(BNMapping(
                    fact_id=fact.fact_id,
                    bn_node="ART_Monkey_WeChat",
                    mapping_weight=0.9,
                    mapping_confidence=0.8,
                    mapping_source="RULE",
                    notes="Monkey WeChat content"
                ))
            if "resume" in text_lower:
                mappings.append(BNMapping(
                    fact_id=fact.fact_id,
                    bn_node="ART_Resume_WeChat",
                    mapping_weight=0.9,
                    mapping_confidence=0.8,
                    mapping_source="RULE",
                    notes="Resume WeChat content"
                ))
            # Also map to HARM_Security
            mappings.append(BNMapping(
                fact_id=fact.fact_id,
                bn_node="HARM_Security",
                mapping_weight=0.8,
                mapping_confidence=0.7,
                mapping_source="RULE",
                notes="WeChat content security risk"
            ))
        
        # Rule 2: OGC letters
        if "ogc" in text_lower and "2025" in text_lower:
            if "april 7" in text_lower or "apr 7" in text_lower:
                mappings.append(BNMapping(
                    fact_id=fact.fact_id,
                    bn_node="COR_OGC_2025_04_07",
                    mapping_weight=1.0,
                    mapping_confidence=0.9,
                    mapping_source="RULE",
                    notes="OGC letter April 7, 2025"
                ))
            if "april 18" in text_lower or "apr 18" in text_lower:
                mappings.append(BNMapping(
                    fact_id=fact.fact_id,
                    bn_node="COR_OGC_2025_04_18",
                    mapping_weight=1.0,
                    mapping_confidence=0.9,
                    mapping_source="RULE",
                    notes="OGC letter April 18, 2025"
                ))
            if "august 11" in text_lower or "aug 11" in text_lower:
                mappings.append(BNMapping(
                    fact_id=fact.fact_id,
                    bn_node="COR_OGC_2025_08_11",
                    mapping_weight=1.0,
                    mapping_confidence=0.9,
                    mapping_source="RULE",
                    notes="OGC letter August 11, 2025"
                ))
            # Also map to HUB_Lawsuit
            mappings.append(BNMapping(
                fact_id=fact.fact_id,
                bn_node="HUB_Lawsuit",
                mapping_weight=0.7,
                mapping_confidence=0.8,
                mapping_source="RULE",
                notes="OGC correspondence"
            ))
    
    # Convert to DataFrame
    df = pd.DataFrame([m.dict() for m in mappings])
    return df
```

**Output file:** `case_law_data/bn_node_lookup.csv`

---

### 8. Create Main Entrypoint (run_fact_engine.py)

**File:** `fact_engine/run_fact_engine.py`

**Script structure:**

```python
#!/usr/bin/env python3
"""Main entrypoint for FactEngine pipeline."""

from pathlib import Path
import argparse
import pandas as pd
from .builder import build_canonical_truth_table
from .loader import load_truth_facts, save_facts
from .ml_importance import FactImportanceModel, load_training_data
from .auto_promote import auto_promote_missing_facts, build_sealing_index, update_bn_mapping

def main():
    parser = argparse.ArgumentParser(description="Run FactEngine pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild canonical table")
    parser.add_argument("--train", action="store_true", help="Train ML model")
    parser.add_argument("--promote", action="store_true", help="Auto-promote missing facts")
    parser.add_argument("--output", type=Path, default=Path("case_law_data/facts_truth_table_v2.csv"))
    args = parser.parse_args()
    
    # 1. Build canonical table
    if args.rebuild:
        facts = build_canonical_truth_table(
            Path("case_law_data/facts_ranked_for_sealing.csv"),
            Path("case_law_data/facts_truth_table.csv"),
            args.output
        )
        save_facts(facts, args.output)
        print(f"Built canonical table with {len(facts)} facts")
    
    # 2. Train model
    if args.train:
        facts = load_truth_facts(args.output)
        labels_path = Path("case_law_data/facts_labels_sealing.csv")
        if labels_path.exists():
            train_facts, labels = load_training_data(args.output, labels_path)
        else:
            print("Warning: facts_labels_sealing.csv not found, using heuristics")
            # Fall back to heuristics
            train_facts, labels = prepare_training_data_heuristic(facts)
        
        model = FactImportanceModel()
        metrics = model.train(train_facts, labels)
        model.save_model(Path("case_law_data/models/fact_importance_model.cbm"))
        print(f"Trained model: accuracy={metrics['accuracy']:.3f}")
    
    # 3. Auto-promote
    if args.promote:
        facts = load_truth_facts(args.output)
        model_path = Path("case_law_data/models/fact_importance_model.cbm")
        if not model_path.exists():
            print("Error: Model not found. Train model first with --train")
            return
        
        model = FactImportanceModel(model_path)
        shap_summary = model.get_global_importance(facts)
        new_facts = auto_promote_missing_facts(facts, model, max_new_facts=50)
        
        # Write to separate file for manual review (don't auto-merge yet)
        save_facts(new_facts, Path("case_law_data/facts_truth_table_autogenerated.csv"))
        print(f"Promoted {len(new_facts)} new facts to facts_truth_table_autogenerated.csv")
        print("Review and merge manually before adding to canonical table")
    
    # 4. Build sealing index and BN mappings (always run if model exists)
    model_path = Path("case_law_data/models/fact_importance_model.cbm")
    if model_path.exists():
        facts = load_truth_facts(args.output)
        model = FactImportanceModel(model_path)
        importances = model.explain_facts(facts)
        
        # Build sealing index
        sealing_df = build_sealing_index(facts, importances)
        sealing_df.to_csv(Path("case_law_data/facts_sealing_index.csv"), index=False)
        print(f"Built sealing index with {len(sealing_df)} facts")
        
        # Update BN mappings
        bn_df = update_bn_mapping(facts)
        bn_df.to_csv(Path("case_law_data/bn_node_lookup.csv"), index=False)
        print(f"Updated BN mappings with {len(bn_df)} entries")

if __name__ == "__main__":
    main()
```

---

### 7. Create Helper for Legal Agents

**Function (add to `fact_engine/__init__.py` or new `fact_engine/helpers.py`):**

```python
from typing import List
from pathlib import Path
from .schema import Fact
from .ml_importance import FactImportanceModel

def get_top_facts_for_sealing(
    n: int = 50,
    facts_csv_path: Path = Path("case_law_data/facts_truth_table_v2.csv"),
    model_path: Path = Path("case_law_data/models/fact_importance_model.cbm")
) -> List[Fact]:
    """
    Return the n highest-importance facts for sealing & safety arguments.
    
    Args:
        n: Number of facts to return
        facts_csv_path: Path to canonical facts CSV
        model_path: Path to trained model
        
    Returns:
        List of Fact objects, sorted by importance_score (descending)
    """
    # Load facts
    facts = load_truth_facts(facts_csv_path)
    
    # Load model
    model = FactImportanceModel(model_path)
    
    # Get importance scores
    importances = model.explain_facts(facts)
    
    # Sort by importance_score
    fact_dict = {f.fact_id: f for f in facts}
    importance_dict = {imp.fact_id: imp.importance_score for imp in importances}
    
    sorted_facts = sorted(
        facts,
        key=lambda f: importance_dict.get(f.fact_id, 0.0),
        reverse=True
    )
    
    return sorted_facts[:n]
```

---

### 8. Integration with FactExtractorAgent

**Option A: Modify existing FactExtractorAgent**
- Update to read from `facts_truth_table_v2.csv` instead of ad-hoc NER
- Use `get_top_facts_for_sealing()` to get high-importance facts

**Option B: Create new FactSupervisorAgent**
- New agent that reads canonical fact table
- Provides facts to motion writers
- Validates facts used in motions

**File:** `writer_agents/code/FactSupervisorAgent.py` (if creating new agent)

---

## ✅ Success Criteria

1. Can run `auto_promote_missing_facts()` and get new candidate facts
2. Can run `run_fact_engine.py --rebuild --train --promote` end-to-end
3. Can call `get_top_facts_for_sealing(50)` and get sorted facts
4. New facts are deduplicated and have safety_risk/public_exposure assigned
5. Integration with legal agents works (FactExtractorAgent or FactSupervisorAgent)

---

## 🧪 Testing

Create test script:
```python
from fact_engine import (
    load_truth_facts,
    FactImportanceModel,
    auto_promote_missing_facts,
    get_top_facts_for_sealing
)
from pathlib import Path

# Load facts
facts = load_truth_facts(Path("case_law_data/facts_truth_table_v2.csv"))

# Load model
model = FactImportanceModel(Path("case_law_data/models/fact_importance_model.cbm"))

# Auto-promote
new_facts = auto_promote_missing_facts(facts, model, max_new_facts=10)
print(f"Proposed {len(new_facts)} new facts")

# Get top facts
top_facts = get_top_facts_for_sealing(20)
print(f"\nTop 20 facts for sealing:")
for i, fact in enumerate(top_facts, 1):
    print(f"{i}. {fact.proposition[:100]}...")
```

---

## 📝 Notes

- Auto-promotion can be slow (queries corpus multiple times) - consider caching
- Deduplication threshold (0.85) may need tuning
- S-V-O extraction quality depends on NLP pipeline - may need refinement
- Integration with agents should be optional (fallback to existing behavior if FactEngine not available)
- **CRITICAL**: Never create facts without grounding in actual corpus snippets
- Auto-generated facts go to separate CSV for manual review before merging

