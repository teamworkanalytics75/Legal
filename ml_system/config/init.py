"""Configuration files for ML system."""

import json
from pathlib import Path
from typing import Dict, Any

def load_ml_config(config_path: str = "ml_system/config/ml_config.json") -> Dict[str, Any]:
    """Load ML configuration from JSON file.

    Falls back to case-variant filenames if the default is missing
    (e.g., `MlConfig.json`).
    """
    candidates = [
        Path(config_path),
        Path("ml_system/config/MlConfig.json"),
        Path("ml_system/config/MLConfig.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, 'r') as f:
                return json.load(f)

    # Return default config if no file exists
    return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default ML configuration."""
    return {
        "data": {
            "legal_docs_db": "lawsuit_docs",
            "agent_history_db": "jobs.db",
            "train_test_split": 0.8,
            "validation_split": 0.1
        },
        "models": {
            "supervised": {
                "outcome_predictor": {
                    "type": "gradient_boost",
                    "n_estimators": 100,
                    "max_depth": 5
                },
                "document_classifier": {
                    "type": "random_forest",
                    "n_estimators": 200
                }
            },
            "deep_learning": {
                "legal_lstm": {
                    "framework": "pytorch",
                    "hidden_dim": 256,
                    "num_layers": 2
                },
                "legal_bert": {
                    "framework": "pytorch",
                    "model_name": "nlpaueb/legal-bert-base-uncased"
                }
            }
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "early_stopping_patience": 3
        }
    }

__all__ = ["load_ml_config", "get_default_config"]
