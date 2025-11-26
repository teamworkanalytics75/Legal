"""Machine Learning System for The Matrix Legal AI.

This module provides comprehensive ML capabilities including:
- Supervised learning for legal case prediction
- Deep learning models (PyTorch & TensorFlow)
- Advanced ML pipelines for training and inference
- ML-focused atomic agents
- Integration with existing agent architecture

Version: 1.0.0
Author: The Matrix Legal AI Team
"""

from .data import LegalDataLoader, FeatureEngineer
from .models import ModelRegistry
from .pipelines import InferencePipeline, AutomatedTrainingPipeline
from .agents import OutcomePredictorAgent, DocumentClassifierAgent, PatternRecognizerAgent

__version__ = "1.0.0"
__all__ = [
    "LegalDataLoader",
    "FeatureEngineer",
    "ModelRegistry",
    "InferencePipeline",
    "AutomatedTrainingPipeline",
    "OutcomePredictorAgent",
    "DocumentClassifierAgent",
    "PatternRecognizerAgent"
]
