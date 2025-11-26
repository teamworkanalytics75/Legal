#!/usr/bin/env python3
"""
Petition Model Trainer

Trains a predictive model to predict §1782 petition outcomes based on petition features.
Uses proper validation and multiple ML algorithms.

Usage: python scripts/train_petition_model.py
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PetitionModelTrainer:
    def __init__(self):
        self.unified_dataset_file = "data/case_law/unified_dataset.csv"
        self.output_dir = "data/case_law/petition_model"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Model parameters
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5

    def load_unified_dataset(self) -> pd.DataFrame:
        """Load the unified dataset."""
        logger.info("Loading unified dataset...")

        if not os.path.exists(self.unified_dataset_file):
            logger.error(f"Unified dataset file not found: {self.unified_dataset_file}")
            return None

        df = pd.read_csv(self.unified_dataset_file)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")

        return df

    def prepare_features_and_targets(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets for training."""
        logger.info("Preparing features and targets...")

        # Exclude non-feature columns
        exclude_columns = [
            'petition_id', 'case_name', 'docket_id', 'docket_number',
            'court', 'date_filed', 'outcome', 'petition_type'
        ]

        feature_columns = [col for col in df.columns if col not in exclude_columns]

        # Prepare features
        X = df[feature_columns].copy()

        # Handle missing values
        X = X.fillna(0)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        # Convert categorical variables to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        # Prepare targets
        y = df['is_success'].copy()

        logger.info(f"Prepared {len(feature_columns)} features for {len(X)} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y, feature_columns

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models."""
        logger.info("Training ML models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state)
        }

        results = {}

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Use scaled features for Logistic Regression
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.cv_folds)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds)

            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Calculate AUC
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc_score = 0.0

            # Generate classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_labels': y_test,
                'classification_report': class_report
            }

            logger.info(f"✓ {name}: {accuracy:.3f} accuracy, {cv_mean:.3f} ± {cv_std:.3f} CV, {auc_score:.3f} AUC")

        # Save scaler
        scaler_path = os.path.join(self.output_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return results, X_train, X_test, y_train, y_test

    def analyze_feature_importance(self, results: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        logger.info("Analyzing feature importance...")

        feature_importance = {}

        for model_name, result in results.items():
            model = result['model']

            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_importance[model_name] = dict(zip(feature_columns, importances))
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_[0])
                feature_importance[model_name] = dict(zip(feature_columns, coefficients))
            else:
                # No feature importance available
                feature_importance[model_name] = {}

        return feature_importance

    def save_models_and_results(self, results: Dict[str, Any], feature_importance: Dict[str, Any],
                               feature_columns: List[str], X_train: pd.DataFrame, y_train: pd.Series):
        """Save models and results."""
        logger.info("Saving models and results...")

        # Save models
        for name, result in results.items():
            model_path = os.path.join(self.output_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)

        # Save training data
        X_train.to_csv(os.path.join(self.output_dir, "training_features.csv"), index=False)
        y_train.to_csv(os.path.join(self.output_dir, "training_targets.csv"), index=False)

        # Save feature importance
        importance_file = os.path.join(self.output_dir, "feature_importance.json")
        with open(importance_file, 'w', encoding='utf-8') as f:
            json.dump(feature_importance, f, indent=2, ensure_ascii=False)

        # Save results summary
        results_summary = {
            'training_date': datetime.now().isoformat(),
            'total_samples': len(X_train),
            'total_features': len(feature_columns),
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'model_performance': {
                name: {
                    'accuracy': result['accuracy'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'auc_score': result['auc_score']
                }
                for name, result in results.items()
            },
            'best_model': max(results.keys(), key=lambda k: results[k]['cv_mean'])
        }

        results_file = os.path.join(self.output_dir, "model_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Models and results saved to: {self.output_dir}")

    def generate_model_report(self, results: Dict[str, Any], feature_importance: Dict[str, Any],
                            feature_columns: List[str]):
        """Generate a comprehensive model report."""
        logger.info("Generating model report...")

        report_file = os.path.join(self.output_dir, "model_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📊 Petition Model Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Features**: {len(feature_columns)}\n")
            f.write(f"**Cross-Validation Folds**: {self.cv_folds}\n\n")

            f.write("## 🎯 Model Performance\n\n")
            f.write("| Model | Accuracy | CV Mean | CV Std | AUC Score |\n")
            f.write("|-------|----------|---------|--------|----------|\n")

            for name, result in results.items():
                f.write(f"| {name} | {result['accuracy']:.3f} | {result['cv_mean']:.3f} | {result['cv_std']:.3f} | {result['auc_score']:.3f} |\n")

            # Find best model
            best_model = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            f.write(f"\n**Best Model**: {best_model}\n\n")

            f.write("## 🔍 Feature Importance\n\n")

            # Show top features for each model
            for model_name, importance in feature_importance.items():
                if importance:
                    f.write(f"### {model_name}\n\n")
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

                    f.write("| Feature | Importance |\n")
                    f.write("|---------|----------|\n")

                    for feature, imp in sorted_features[:15]:  # Top 15 features
                        f.write(f"| {feature} | {imp:.4f} |\n")
                    f.write("\n")

            f.write("## 📈 Model Insights\n\n")

            # Analyze top features across all models
            all_features = {}
            for importance in feature_importance.values():
                for feature, imp in importance.items():
                    all_features[feature] = all_features.get(feature, 0) + imp

            if all_features:
                f.write("### Top Features Across All Models\n\n")
                sorted_all = sorted(all_features.items(), key=lambda x: x[1], reverse=True)

                f.write("| Feature | Combined Importance |\n")
                f.write("|---------|-------------------|\n")

                for feature, imp in sorted_all[:20]:  # Top 20 features
                    f.write(f"| {feature} | {imp:.4f} |\n")

            f.write("\n## 🚀 Next Steps\n\n")
            f.write("1. **Feature Engineering**: Refine features based on importance analysis\n")
            f.write("2. **Model Tuning**: Optimize hyperparameters for best model\n")
            f.write("3. **Validation**: Test on additional petition data\n")
            f.write("4. **Deployment**: Apply model to new petition analysis\n")

        logger.info(f"✓ Model report saved to: {report_file}")

    def run_training(self):
        """Run the complete model training process."""
        logger.info("🚀 Starting Petition Model Training")
        logger.info("="*80)

        # Load dataset
        df = self.load_unified_dataset()
        if df is None:
            return

        # Prepare features and targets
        X, y, feature_columns = self.prepare_features_and_targets(df)

        # Train models
        results, X_train, X_test, y_train, y_test = self.train_models(X, y)

        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(results, feature_columns)

        # Save models and results
        self.save_models_and_results(results, feature_importance, feature_columns, X_train, y_train)

        # Generate report
        self.generate_model_report(results, feature_importance, feature_columns)

        logger.info("🎉 Petition model training complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total samples: {len(X_train)}")
        print(f"  Total features: {len(feature_columns)}")
        print(f"  Best model: {max(results.keys(), key=lambda k: results[k]['cv_mean'])}")

        for name, result in results.items():
            print(f"  {name}: {result['accuracy']:.3f} accuracy, {result['cv_mean']:.3f} ± {result['cv_std']:.3f} CV")

        return results, feature_importance

def main():
    """Main function."""
    print("🤖 Petition Model Trainer")
    print("="*80)

    trainer = PetitionModelTrainer()
    results, feature_importance = trainer.run_training()

    print("\n✅ Petition model training complete!")
    print("Check data/case_law/petition_model/ for results.")

if __name__ == "__main__":
    main()
