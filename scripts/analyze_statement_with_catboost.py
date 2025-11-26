#!/usr/bin/env python3
"""
Analyze Statement of Claim using ACTUAL CatBoost model and feature extraction.

This uses the real ML system:
- Section 1782 CatBoost model (trained on actual case data)
- Feature extraction from analyze_ma_motion_doc.py
- Comparison to historical successful cases
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analysis"))
sys.path.insert(0, str(project_root / "case_law_data" / "scripts"))

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("ERROR: CatBoost not available. Install with: pip install catboost")
    sys.exit(1)

def load_catboost_model(model_path: Path):
    """Load the Section 1782 CatBoost model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading CatBoost model from: {model_path}")
    model = cb.CatBoostClassifier()
    model.load_model(str(model_path))
    print(f"Model loaded successfully")
    return model

def extract_features_from_statement(statement_text: str):
    """Extract features from statement using the actual feature extraction system."""
    try:
        from analyze_ma_motion_doc import compute_draft_features
        print("Using feature extraction from analyze_ma_motion_doc.py")
        features = compute_draft_features(statement_text, base_year=2025, court="US Federal")
        return features
    except ImportError:
        print("WARNING: Could not import compute_draft_features, using fallback")
        return _simple_feature_extraction(statement_text)

def _simple_feature_extraction(text: str) -> Dict[str, float]:
    """Fallback simple feature extraction."""
    text_lower = text.lower()
    return {
        "word_count": float(len(text.split())),
        "text_length": float(len(text)),
        "mentions_privacy": float(text_lower.count("privacy")),
        "mentions_harassment": float(text_lower.count("harass")),
        "mentions_safety": float(text_lower.count("safety")),
        "mentions_retaliation": float(text_lower.count("retaliat")),
        "citation_count": float(len([w for w in text.split() if any(c in w for c in ["v.", "f.", "U.S."])])),
        "public_interest_mentions": float(text_lower.count("public interest")),
        "transparency_mentions": float(text_lower.count("transparency")),
        "privacy_harm_count": float(text_lower.count("harm")),
        "paragraph_count": float(len(text.split('\n\n'))),
        "section_count": float(text.count('PART')),
    }

def compare_to_historical_cases(features: Dict[str, float], features_csv_path: Path):
    """Compare statement features to historical successful Section 1782 cases."""
    if not features_csv_path.exists():
        print(f"WARNING: Features CSV not found: {features_csv_path}")
        return {}

    try:
        df = pd.read_csv(features_csv_path)

        # Filter for Section 1782 successful cases
        # Assuming there's a column indicating success/outcome
        success_cols = [c for c in df.columns if 'success' in c.lower() or 'outcome' in c.lower() or 'granted' in c.lower()]

        if not success_cols:
            print("WARNING: Could not find success/outcome column in CSV")
            return {}

        success_col = success_cols[0]
        successful_cases = df[df[success_col] == 1] if len(df[success_col].unique()) <= 2 else df

        # Compare to successful case averages
        comparison = {}
        for feature_name, value in features.items():
            if feature_name in successful_cases.columns:
                avg_value = successful_cases[feature_name].mean()
                std_value = successful_cases[feature_name].std()
                if not pd.isna(avg_value):
                    comparison[feature_name] = {
                        "statement_value": value,
                        "successful_avg": float(avg_value),
                        "successful_std": float(std_value) if not pd.isna(std_value) else 0.0,
                        "difference": float(value - avg_value),
                        "z_score": float((value - avg_value) / std_value) if std_value > 0 else 0.0
                    }

        return comparison
    except Exception as e:
        print(f"WARNING: Could not compare to historical cases: {e}")
        return {}

def analyze_with_catboost(statement_text: str, model_path: Path, features_csv_path: Path = None):
    """Full analysis using CatBoost model."""

    print("="*70)
    print("CATBOOST-BASED NATIONAL SECURITY ANALYSIS")
    print("="*70)
    print(f"\nStatement length: {len(statement_text)} characters")

    # Load model
    model = load_catboost_model(model_path)

    # Extract features
    print("\nExtracting features from statement...")
    features = extract_features_from_statement(statement_text)
    print(f"Extracted {len(features)} features")

    # Prepare feature dataframe
    feature_df = pd.DataFrame([features])

    # Ensure all model features are present
    if hasattr(model, 'feature_names_'):
        model_features = model.feature_names_
        print(f"\nModel expects {len(model_features)} features")

        # Add missing features as zeros
        for col in model_features:
            if col not in feature_df.columns:
                feature_df[col] = 0.0

        # Reorder to match model
        feature_df = feature_df[model_features]
    else:
        print("WARNING: Model does not have feature_names_, using all extracted features")

    # Predict
    print("\nRunning CatBoost prediction...")
    proba = model.predict_proba(feature_df)[0]
    prediction = model.predict(feature_df)[0]
    confidence = float(np.max(proba))
    predicted_class = int(prediction)

    # Get class labels if available
    try:
        if hasattr(model, 'classes_'):
            class_labels = model.classes_
            probabilities = {f"class_{int(cls)}": float(proba[int(cls)]) for cls in class_labels}
        else:
            probabilities = {f"class_{i}": float(p) for i, p in enumerate(proba)}
    except:
        probabilities = {"class_0": float(proba[0]), "class_1": float(proba[1]) if len(proba) > 1 else 0.0}

    print(f"Prediction complete")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.1%}")

    # Compare to historical cases
    historical_comparison = {}
    if features_csv_path:
        print("\nComparing to historical successful cases...")
        historical_comparison = compare_to_historical_cases(features, features_csv_path)
        if historical_comparison:
            print(f"Compared {len(historical_comparison)} features to historical averages")

    # Top features (if we can get SHAP values)
    top_features = {}
    try:
        from catboost import Pool
        pool = Pool(feature_df, feature_names=feature_df.columns.tolist())
        feature_importance = model.get_feature_importance(pool, type="FeatureImportance")

        # Get top 10 most important features for this prediction
        feature_names = feature_df.columns.tolist()
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = {
            name: {
                "importance": float(imp),
                "value": float(features.get(name, 0))
            }
            for name, imp in importance_pairs[:10]
        }
        print(f"Extracted top {len(top_features)} features by importance")
    except Exception as e:
        print(f"WARNING: Could not extract feature importance: {e}")

    return {
        "prediction": int(predicted_class),
        "confidence": confidence,
        "probabilities": probabilities,
        "features": features,
        "top_features": top_features,
        "historical_comparison": historical_comparison,
        "model_path": str(model_path),
        "feature_count": len(features)
    }

def main():
    """Main execution."""
    # Paths
    statement_path = project_root / "case_law_data" / "tmp_corpus" / "Exhibit 2 — Certified Statement of Claim (Hong Kong, 2 Jun 2025).txt"
    model_path = project_root / "case_law_data" / "models" / "section_1782_discovery_model.cbm"
    features_csv_path = project_root / "case_law_data" / "features" / "unified_features.csv"

    # Load statement
    if not statement_path.exists():
        print(f"ERROR: Statement not found: {statement_path}")
        return

    with open(statement_path, 'r', encoding='utf-8') as f:
        statement_text = f.read()

    # Run analysis
    results = analyze_with_catboost(statement_text, model_path, features_csv_path)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nPrediction: Class {results['prediction']}")
    print(f"Confidence: {results['confidence']:.1%}")
    print(f"\nClass Probabilities:")
    for cls, prob in results['probabilities'].items():
        print(f"  {cls}: {prob:.1%}")

    if results['top_features']:
        print(f"\nTop Features by Importance:")
        for name, data in list(results['top_features'].items())[:5]:
            print(f"  {name}: importance={data['importance']:.4f}, value={data['value']:.2f}")

    # Save results
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "catboost_national_security_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFull results saved to: {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()

