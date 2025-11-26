"""
SHAP Insight Plugin

Provides explainable AI insights based on SHAP values for CatBoost models.
Converts predictions into actionable recommendations.
"""

from .shap_insight_plugin import SHAPInsightPlugin

__all__ = ['SHAPInsightPlugin']

