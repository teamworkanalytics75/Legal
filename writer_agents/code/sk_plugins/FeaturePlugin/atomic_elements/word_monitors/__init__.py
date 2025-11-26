#!/usr/bin/env python3
"""
Word Monitor Plugins

Plugins that monitor individual word usage in legal documents.
Each plugin tracks a specific CatBoost success signal word.
"""

from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin
from .order_word_monitor_plugin import OrderWordMonitorPlugin
from .harm_word_monitor_plugin import HarmWordMonitorPlugin
from .safety_word_monitor_plugin import SafetyWordMonitorPlugin
from .immediate_word_monitor_plugin import ImmediateWordMonitorPlugin
from .pseudonym_word_monitor_plugin import PseudonymWordMonitorPlugin
from .risk_word_monitor_plugin import RiskWordMonitorPlugin
from .security_word_monitor_plugin import SecurityWordMonitorPlugin
from .serious_word_monitor_plugin import SeriousWordMonitorPlugin
from .sealed_word_monitor_plugin import SealedWordMonitorPlugin
from .motion_word_monitor_plugin import MotionWordMonitorPlugin
from .citizen_word_monitor_plugin import CitizenWordMonitorPlugin
from .complete_word_monitor_plugin import CompleteWordMonitorPlugin
from .bodily_word_monitor_plugin import BodilyWordMonitorPlugin
from .threat_word_monitor_plugin import ThreatWordMonitorPlugin
from .impound_word_monitor_plugin import ImpoundWordMonitorPlugin
from .protective_word_monitor_plugin import ProtectiveWordMonitorPlugin
from .national_word_monitor_plugin import NationalWordMonitorPlugin

__all__ = [
    "CatBoostWordMonitorPlugin",
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

