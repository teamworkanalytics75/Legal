# Codex Agent 2: FactEngine ML Importance Model (CatBoost + SHAP)

**Workstream:** FactEngine Module Build - ML Layer  
**Status:** ✅ Complete — ml_importance.py implemented with CatBoost + SHAP integration  
**Dependencies:** Agent 1 (needs Fact schema from schema.py) ✅ Schema available

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 2**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read `CODEX_AGENTS_FACTENGINE_OVERVIEW.md` and all `CODEX_AGENT_*_FACTENGINE_*.md` files to understand the full context.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 2**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 2** and follow those.

**Your instruction file:** This file contains YOUR specific tasks. This is the source of truth.

**Dependency:** Wait for Agent 1 to create `fact_engine/schema.py` with `Fact` model, or start by importing it and handle ImportError gracefully.

---

## 🎯 Objective

Create a CatBoost + SHAP wrapper for fact importance scoring. Train models to predict which facts are important for sealing/safety arguments, and use SHAP to explain which features matter most.

---

## 📦 Tasks

### 1. Create ML Module (ml_importance.py)

**File:** `fact_engine/ml_importance.py`

**Dependencies:**
- `catboost` (CatBoostClassifier)
- `shap` (TreeExplainer)
- `pandas`, `numpy`
- `sklearn` (TfidfVectorizer, train_test_split)
- `networkx` (for graph features, if KG available)
- `fact_engine.schema` (Fact, FactImportance)

---

### 2. Implement Feature Engineering

**Function to create:**

```python
from typing import List, Dict
import pandas as pd
from .schema import Fact

def extract_features(facts: List[Fact], kg_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Extract features from facts for CatBoost training.
    
    Features:
    1. TF-IDF on proposition and source_excerpt (top 100 terms)
    2. Binary flags:
       - mentions_prc, mentions_china, mentions_xinjiang
       - mentions_esuwiki, mentions_niu_tengyu, mentions_xi
       - mentions_wechat, mentions_monkey, mentions_resume
       - mentions_harvard, mentions_haa, mentions_hcs, mentions_hcb, mentions_gss
    3. Event metadata (one-hot encoded):
       - event_type, evidence_type, actor_role, speaker, truth_status
    4. Date features (bucketed):
       - event_date_year, event_date_month, event_date_quarter
    5. Graph features (if KG available):
       - entity_degree (from KG), entity_closeness_to_harm_security
       - entity_closeness_to_hub_amplifiers, entity_closeness_to_hub_lawsuit
    
    Args:
        facts: List of Fact objects
        kg_path: Optional path to knowledge graph JSON
        
    Returns:
        DataFrame with features (one row per fact)
    """
    pass
```

**Reference:** See existing CatBoost scripts:
- `case_law_data/scripts/run_catboost_shap_party.py`
- `case_law_data/scripts/catboost_structure_analysis.py`
- `case_law_data/scripts/complete_recap_matching_analysis.py`

**Graph features:** Load KG from `case_law_data/facts_knowledge_graph.json` or `nlp_analysis/code/KnowledgeGraph.py`

---

### 3. Implement FactImportanceModel Class

**Class structure:**

```python
from catboost import CatBoostClassifier
import shap
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from .schema import Fact, FactImportance

class FactImportanceModel:
    """CatBoost model wrapper for fact importance scoring with SHAP."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize model.
        
        Args:
            model_path: Optional path to saved CatBoost model (.cbm file)
        """
        self.model = None
        self.feature_names = []
        self.shap_explainer = None
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def train(
        self,
        facts: List[Fact],
        labels: List[int],  # 1 = sealing_critical, 0 = not
        test_size: float = 0.2,
        model_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Train CatBoost model on facts.
        
        Steps:
        1. Extract features using extract_features()
        2. Split train/test
        3. Train CatBoostClassifier
        4. Evaluate (accuracy, precision, recall, f1)
        5. Store model and feature names
        
        Args:
            facts: List of Fact objects
            labels: List of binary labels (1 = sealing_critical, 0 = not)
            test_size: Fraction for test set
            model_params: Optional CatBoost parameters
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        # 1. Extract features
        X = extract_features(facts)
        
        # 2. Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 3. Train CatBoostClassifier
        # 4. Evaluate
        # 5. Store model and feature names
        pass
    
    def explain_facts(self, facts: List[Fact]) -> List[FactImportance]:
        """
        Get SHAP explanations for facts.
        
        Args:
            facts: List of Fact objects
            
        Returns:
            List of FactImportance objects with importance_score and shap_values
        """
        # 1. Extract features
        # 2. Get predictions
        # 3. Compute SHAP values
        # 4. Return List[FactImportance]
        pass
    
    def get_global_importance(self, facts: List[Fact]) -> Dict[str, float]:
        """
        Get global feature importance (mean absolute SHAP across all facts).
        
        Steps:
        1. Extract features for all facts
        2. Get SHAP values (if available) or fall back to model feature_importances_
        3. Compute mean absolute SHAP per feature
        4. Return sorted dictionary (feature_name -> mean_abs_shap_value)
        
        Note: If SHAP is not available, use model.get_feature_importance() as fallback.
        
        Args:
            facts: List of Fact objects
            
        Returns:
            Dictionary mapping feature_name -> mean_abs_shap_value (or feature_importance if SHAP unavailable)
        """
        # 1. Extract features
        X = extract_features(facts)
        
        # 2. Try SHAP, fall back to feature_importances_ if unavailable
        try:
            if self.shap_explainer:
                shap_values = self.shap_explainer.shap_values(X)
                # Handle binary vs multiclass
                if isinstance(shap_values, list):
                    shap_array = np.abs(np.stack([np.asarray(v) for v in shap_values], axis=0))
                else:
                    shap_array = np.abs(np.asarray(shap_values))
                # Mean absolute SHAP per feature
                mean_shap = np.mean(shap_array.reshape(-1, shap_array.shape[-1]), axis=0)
                return dict(zip(self.feature_names, mean_shap))
        except Exception:
            pass
        
        # Fallback to feature importance
        if self.model:
            importances = self.model.get_feature_importance()
            return dict(zip(self.feature_names, importances))
        
        return {}
    
    def save_model(self, model_path: Path) -> None:
        """Save model to .cbm file."""
        pass
    
    def load_model(self, model_path: Path) -> None:
        """Load model from .cbm file."""
        pass
```

**Model parameters (default):**
```python
{
    "iterations": 100,
    "depth": 4,
    "learning_rate": 0.1,
    "loss_function": "Logloss",
    "verbose": False,
    "random_seed": 42,
    "task_type": "GPU"  # Use GPU if available
}
```

**Reference:** See `case_law_data/scripts/run_catboost_shap_party.py` for SHAP integration pattern

---

### 4. Implement Training Data Loading

**Label file format:**

Create `case_law_data/facts_labels_sealing.csv`:
```csv
fact_id,label_sealing_critical,notes
F001,1,"Plaintiff in PRC during publications"
F002,1,"Xi slide content + PRC risk"
F003,0,"Generic court description"
...
```

**Function:**

```python
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional

def load_training_data(
    facts_csv_path: Path,
    labels_csv_path: Path
) -> Tuple[List[Fact], List[int]]:
    """
    Load facts and labels for training.
    
    Args:
        facts_csv_path: Path to facts_truth_table_v2.csv
        labels_csv_path: Path to facts_labels_sealing.csv
        
    Returns:
        Tuple of (facts, labels) where labels are 1 (sealing_critical) or 0 (not)
    """
    # Load facts
    facts = load_truth_facts(facts_csv_path)
    
    # Load labels
    labels_df = pd.read_csv(labels_csv_path)
    label_dict = dict(zip(labels_df['fact_id'], labels_df['label_sealing_critical']))
    
    # Join on fact_id
    train_facts = []
    train_labels = []
    for fact in facts:
        if fact.fact_id in label_dict:
            train_facts.append(fact)
            train_labels.append(int(label_dict[fact.fact_id]))
    
    return train_facts, train_labels
```

**Note:** If `facts_labels_sealing.csv` doesn't exist yet, fall back to heuristics:
- Label=1 if: `safety_risk in ["High", "Extreme"]` OR `event_type in ["Risk", "Harm"]`
- Label=0 if: `safety_risk == "none"` OR `event_type == "Unknown"`

---

### 5. Add to Package Exports

**Update `fact_engine/__init__.py`:**
```python
from .ml_importance import FactImportanceModel, extract_features, prepare_training_data
```

---

## ✅ Success Criteria

1. Can create `FactImportanceModel()` and train on facts
2. Can call `explain_facts()` and get SHAP values per fact
3. Can call `get_global_importance()` and see top features
4. Model saves/loads correctly (.cbm format)
5. Features include TF-IDF, binary flags, metadata, and graph features (if KG available)

---

## 🧪 Testing

Create test script:
```python
from fact_engine import load_truth_facts, FactImportanceModel, extract_features
from pathlib import Path

# Load facts
facts = load_truth_facts(Path("case_law_data/facts_truth_table.csv"))

# Prepare training data (heuristic labels for now)
from fact_engine.ml_importance import prepare_training_data
train_facts, labels = prepare_training_data(facts)

# Train model
model = FactImportanceModel()
metrics = model.train(train_facts, labels)
print(f"Accuracy: {metrics['accuracy']:.3f}")

# Explain facts
importances = model.explain_facts(facts[:10])
for imp in importances:
    print(f"Fact {imp.fact_id}: score={imp.importance_score:.3f}")

# Global importance
global_imp = model.get_global_importance(facts)
print("\nTop 10 features:")
for feat, val in list(global_imp.items())[:10]:
    print(f"  {feat}: {val:.4f}")

# Save model
model.save_model(Path("case_law_data/models/fact_importance_model.cbm"))
```

---

## 📝 Notes

- Use GPU if available (CatBoost `task_type="GPU"`)
- Handle missing KG gracefully (graph features = 0 if KG not available)
- TF-IDF should be fit on training set only, transform on test
- SHAP can be slow on large datasets - consider sampling for global importance
- Store feature names with model for consistent feature extraction on new facts

