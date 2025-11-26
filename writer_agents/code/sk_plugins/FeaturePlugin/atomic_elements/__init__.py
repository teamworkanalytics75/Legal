#!/usr/bin/env python3
"""
Atomic Elements Plugin Category

This category contains plugins that monitor atomic elements of legal documents:
- Words (individual word usage)
- Sentences (sentence structure and content)
- Paragraphs (paragraph structure and content)
- Sections (section organization)
- Formatting (formatting rules)
- Rules (legal rule compliance)

Each atomic plugin watches over its own domain and can interact with other
atomic plugins to recalibrate the document as a whole.
"""

from .word_monitors import (
    CatBoostWordMonitorPlugin,
    OrderWordMonitorPlugin,
    HarmWordMonitorPlugin,
    SafetyWordMonitorPlugin,
    ImmediateWordMonitorPlugin,
    PseudonymWordMonitorPlugin,
    RiskWordMonitorPlugin,
    SecurityWordMonitorPlugin,
    SeriousWordMonitorPlugin,
    SealedWordMonitorPlugin,
    MotionWordMonitorPlugin,
    CitizenWordMonitorPlugin,
    CompleteWordMonitorPlugin,
    BodilyWordMonitorPlugin,
    ThreatWordMonitorPlugin,
    ImpoundWordMonitorPlugin,
    ProtectiveWordMonitorPlugin,
    NationalWordMonitorPlugin,
)

__all__ = [
    # Base word monitor
    "CatBoostWordMonitorPlugin",
    # Word monitor plugins
    "OrderWordMonitorPlugin",
    "HarmWordMonitorPlugin",
    "SafetyWordMonitorPlugin",
    "ImmediateWordMonitorPlugin",
    "PseudonymWordMonitorPlugin",
    "RiskWordMonitorPlugin",
    "SecurityWordMonitorPlugin",
    "SeriousWordMonitorPlugin",
    "SealedWordMonitorPlugin",
    "MotionWordMonitorPlugin",
    "CitizenWordMonitorPlugin",
    "CompleteWordMonitorPlugin",
    "BodilyWordMonitorPlugin",
    "ThreatWordMonitorPlugin",
    "ImpoundWordMonitorPlugin",
    "ProtectiveWordMonitorPlugin",
    "NationalWordMonitorPlugin",
]

