"""Background agent implementations."""

from .document_monitor import DocumentMonitorAgent
from .legal_research import LegalResearchAgent
from .citation_network import CitationNetworkAgent
from .pattern_detection import PatternDetectionAgent
from .settlement_optimizer import SettlementOptimizerAgent
from .project_organizer import ProjectOrganizerAgent

__all__ = [
    "DocumentMonitorAgent",
    "LegalResearchAgent",
    "CitationNetworkAgent",
    "PatternDetectionAgent",
    "SettlementOptimizerAgent",
    "ProjectOrganizerAgent",
]
