"""Static project context for all 49 The Matrix agents.

This file contains role descriptions, team positions, and dependencies
for each agent type. Updated manually as agents are added/changed.
"""

from typing import Dict, Any

AGENT_CONTEXTS: Dict[str, Dict[str, Any]] = {
    # ===== ATOMIC AGENTS (25) =====

    # Citation Pipeline (5 agents)
    "CitationFinderAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Find raw citation strings in legal text using 6 regex patterns",
        "phase": "Citation",
        "supervisor": "CitationSupervisor",
        "team_size": 5,
        "team_position": "1/5 - First in citation pipeline",
        "upstream": "DraftingSupervisor (receives draft text)",
        "downstream": "CitationNormalizerAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "L - Detective (Tier 1)", # From ChatGPT mapping
        "cluster": "Gatekeeper / Verification Engine"
    },

    "CitationNormalizerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Normalize citations to Bluebook format using templates",
        "phase": "Citation",
        "supervisor": "CitationSupervisor",
        "team_size": 5,
        "team_position": "2/5 - Second in citation pipeline",
        "upstream": "CitationFinderAgent",
        "downstream": "CitationVerifierAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    "CitationVerifierAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Verify citations against case law database",
        "phase": "Citation",
        "supervisor": "CitationSupervisor",
        "team_size": 5,
        "team_position": "3/5 - Third in citation pipeline",
        "upstream": "CitationNormalizerAgent",
        "downstream": "CitationLocatorAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    "CitationLocatorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Map citation tokens to file paths or URLs in database",
        "phase": "Citation",
        "supervisor": "CitationSupervisor",
        "team_size": 5,
        "team_position": "4/5 - Fourth in citation pipeline",
        "upstream": "CitationVerifierAgent",
        "downstream": "CitationInserterAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    "CitationInserterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Insert formatted citations into document text",
        "phase": "Citation",
        "supervisor": "CitationSupervisor",
        "team_size": 5,
        "team_position": "5/5 - Final in citation pipeline",
        "upstream": "CitationLocatorAgent",
        "downstream": "QASupervisor (sends cited text)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    # Research Pipeline (6 agents)
    "FactExtractorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Extract discrete facts from case documents",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "1/6 - Parallel executor",
        "upstream": "MasterSupervisor (receives CaseInsights + summary)",
        "downstream": "DraftingSupervisor (facts used in drafting)",
        "deterministic": False,
        "cost": "~$0.005 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "PrecedentFinderAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Find relevant precedent cases using LLM suggestions",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "2/6 - Parallel executor",
        "upstream": "MasterSupervisor (receives CaseInsights + summary)",
        "downstream": "PrecedentRankerAgent",
        "deterministic": False,
        "cost": "~$0.003 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "PrecedentRankerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Rank precedent cases by relevance using LLM scoring",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "3/6 - Parallel executor",
        "upstream": "PrecedentFinderAgent",
        "downstream": "PrecedentSummarizerAgent",
        "deterministic": False,
        "cost": "~$0.004 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "PrecedentSummarizerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Summarize precedent cases in 1-2 sentences using LLM",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "4/6 - Parallel executor",
        "upstream": "PrecedentRankerAgent",
        "downstream": "DraftingSupervisor (summaries used in drafting)",
        "deterministic": False,
        "cost": "~$0.003 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "StatuteLocatorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Find statute text from database using SQL queries",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "5/6 - Parallel executor",
        "upstream": "MasterSupervisor (receives CaseInsights + summary)",
        "downstream": "DraftingSupervisor (statutes used in drafting)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "ExhibitFetcherAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Fetch exhibit files from storage using file system/DB",
        "phase": "Research",
        "supervisor": "ResearchSupervisor",
        "team_size": 6,
        "team_position": "6/6 - Parallel executor",
        "upstream": "MasterSupervisor (receives CaseInsights + summary)",
        "downstream": "DraftingSupervisor (exhibits used in drafting)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Artemis - Hunter Gatherer (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    # Drafting Pipeline (4 agents)
    "OutlineBuilderAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Build structured section outline for legal document using LLM planning",
        "phase": "Drafting",
        "supervisor": "DraftingSupervisor",
        "team_size": 4,
        "team_position": "1/4 - First in drafting pipeline",
        "upstream": "ResearchSupervisor (receives research findings)",
        "downstream": "SectionWriterAgent",
        "deterministic": False,
        "cost": "~$0.003 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "SectionWriterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Write one section of legal document using LLM generation",
        "phase": "Drafting",
        "supervisor": "DraftingSupervisor",
        "team_size": 4,
        "team_position": "2/4 - Parallel executor (spawns N copies)",
        "upstream": "OutlineBuilderAgent",
        "downstream": "TransitionAgent",
        "deterministic": False,
        "cost": "~$0.010 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "ParagraphWriterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Write one paragraph within a section using LLM generation",
        "phase": "Drafting",
        "supervisor": "DraftingSupervisor",
        "team_size": 4,
        "team_position": "3/4 - Parallel executor",
        "upstream": "SectionWriterAgent",
        "downstream": "TransitionAgent",
        "deterministic": False,
        "cost": "~$0.005 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "TransitionAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Create smooth transitions between sections using LLM",
        "phase": "Drafting",
        "supervisor": "DraftingSupervisor",
        "team_size": 4,
        "team_position": "4/4 - Final in drafting pipeline",
        "upstream": "SectionWriterAgent, ParagraphWriterAgent",
        "downstream": "CitationSupervisor (sends draft text)",
        "deterministic": False,
        "cost": "~$0.002 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    # QA/Review Pipeline (7 agents)
    "GrammarFixerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Fix grammar and typos in text using LLM correction",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "1/7 - First in QA pipeline",
        "upstream": "CitationSupervisor (receives cited text)",
        "downstream": "StyleCheckerAgent",
        "deterministic": False,
        "cost": "~$0.005 per execution",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "StyleCheckerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Check document against style guide rules using rule-based checks",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "2/7 - Second in QA pipeline",
        "upstream": "GrammarFixerAgent",
        "downstream": "LogicCheckerAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "LogicCheckerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Check argument logic and coherence using LLM analysis",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "3/7 - Third in QA pipeline",
        "upstream": "StyleCheckerAgent",
        "downstream": "ConsistencyCheckerAgent",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "Apollo - Logic Master (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "ConsistencyCheckerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Check term and name consistency throughout document using dictionary tracking",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "4/7 - Fourth in QA pipeline",
        "upstream": "LogicCheckerAgent",
        "downstream": "RedactionAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "RedactionAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Apply redaction rules to protect sensitive information using regex patterns",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "5/7 - Fifth in QA pipeline",
        "upstream": "ConsistencyCheckerAgent",
        "downstream": "ComplianceAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Hades - Shadow Censor (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "ComplianceAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Verify document format compliance using rule validation",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "6/7 - Sixth in QA pipeline",
        "upstream": "RedactionAgent",
        "downstream": "ExpertQAAgent",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "ExpertQAAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Expert-level final QA review using LLM (gpt-4o)",
        "phase": "QA",
        "supervisor": "QASupervisor",
        "team_size": 7,
        "team_position": "7/7 - Final in QA pipeline",
        "upstream": "ComplianceAgent",
        "downstream": "OutputSupervisor (sends final text)",
        "deterministic": False,
        "cost": "~$0.020 per execution (conditional)",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    # Output Pipeline (3 agents)
    "MarkdownExporterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Export document to Markdown format using template rendering",
        "phase": "Output",
        "supervisor": "OutputSupervisor",
        "team_size": 3,
        "team_position": "1/3 - Parallel executor",
        "upstream": "QASupervisor (receives final text)",
        "downstream": "User (final markdown file)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "DocxExporterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Export document to DOCX format using python-docx",
        "phase": "Output",
        "supervisor": "OutputSupervisor",
        "team_size": 3,
        "team_position": "2/3 - Parallel executor",
        "upstream": "QASupervisor (receives final text)",
        "downstream": "User (final docx file)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "MetadataTaggerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Add metadata tags to document using JSON serialization",
        "phase": "Output",
        "supervisor": "OutputSupervisor",
        "team_size": 3,
        "team_position": "3/3 - Parallel executor",
        "upstream": "QASupervisor (receives final text)",
        "downstream": "User (metadata file)",
        "deterministic": True,
        "cost": "$0.00",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    # ===== ADVANCED AGENTS (10) =====

    "StrategicPlannerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "High-level strategic planning for legal document creation",
        "phase": "Strategic Planning",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "1/10 - Strategic level",
        "upstream": "MasterSupervisor (receives case context)",
        "downstream": "TacticalPlannerAgent",
        "deterministic": False,
        "cost": "~$0.010 per execution",
        "codename": "Light Yagami - MasterSupervisor (Tier 1)",
        "cluster": "MasterSupervisor (Strategic Orchestration)"
    },

    "TacticalPlannerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Mid-level tactical planning and coordination",
        "phase": "Strategic Planning",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "2/10 - Tactical level",
        "upstream": "StrategicPlannerAgent",
        "downstream": "OperationalPlannerAgent",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "Neo - System Optimizer (Tier 1)",
        "cluster": "System Optimizer / Self-updating Supervisor"
    },

    "OperationalPlannerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Low-level operational planning for paragraph and sentence-level tasks",
        "phase": "Strategic Planning",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "3/10 - Operational level",
        "upstream": "TacticalPlannerAgent",
        "downstream": "ResearchAgent",
        "deterministic": False,
        "cost": "~$0.006 per execution",
        "codename": "Neo - System Optimizer (Tier 1)",
        "cluster": "System Optimizer / Self-updating Supervisor"
    },

    "ResearchAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Dedicated research and fact-checking for legal documents",
        "phase": "Advanced Research",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "4/10 - Research specialist",
        "upstream": "OperationalPlannerAgent",
        "downstream": "PrimaryWriterAgent",
        "deterministic": False,
        "cost": "~$0.012 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "PrimaryWriterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Primary content generation for legal documents",
        "phase": "Advanced Writing",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "5/10 - Primary writer",
        "upstream": "ResearchAgent",
        "downstream": "ContentReviewerAgent",
        "deterministic": False,
        "cost": "~$0.015 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "ContentReviewerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Multi-level content-focused review (basic, intermediate, advanced)",
        "phase": "Advanced Review",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "6/10 - Content reviewer",
        "upstream": "PrimaryWriterAgent",
        "downstream": "TechnicalReviewerAgent",
        "deterministic": False,
        "cost": "~$0.010 per execution",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "TechnicalReviewerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Technical accuracy and citation review",
        "phase": "Advanced Review",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "7/10 - Technical reviewer",
        "upstream": "ContentReviewerAgent",
        "downstream": "StyleReviewerAgent",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    "StyleReviewerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Style and tone review for legal documents",
        "phase": "Advanced Review",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "8/10 - Style reviewer",
        "upstream": "TechnicalReviewerAgent",
        "downstream": "QualityAssuranceAgent",
        "deterministic": False,
        "cost": "~$0.006 per execution",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "QualityAssuranceAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Final quality assurance and gatekeeper",
        "phase": "Advanced Review",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "9/10 - QA gatekeeper",
        "upstream": "StyleReviewerAgent",
        "downstream": "AdaptiveOrchestratorAgent",
        "deterministic": False,
        "cost": "~$0.012 per execution",
        "codename": "Aragorn - Gatekeeper (Tier 1)",
        "cluster": "Gatekeeper / Trusted Authority"
    },

    "AdaptiveOrchestratorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Adaptive workflow orchestration and optimization",
        "phase": "Orchestration",
        "supervisor": "AdvancedWriterOrchestrator",
        "team_size": 10,
        "team_position": "10/10 - Orchestrator",
        "upstream": "QualityAssuranceAgent",
        "downstream": "Output (final deliverable)",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "Neo - System Optimizer (Tier 1)",
        "cluster": "System Optimizer / Self-updating Supervisor"
    },

    # ===== BASE AGENTS (5) =====

    "PlannerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Produce section plans for legal documents",
        "phase": "Legacy Planning",
        "supervisor": "WriterOrchestrator",
        "team_size": 5,
        "team_position": "1/5 - Legacy planner",
        "upstream": "MasterSupervisor (receives case context)",
        "downstream": "WriterAgent",
        "deterministic": False,
        "cost": "~$0.005 per execution",
        "codename": "Athena - Strategic Oracle (Tier 1)",
        "cluster": "MasterSupervisor (Strategic Orchestration)"
    },

    "WriterAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Draft prose for legal documents",
        "phase": "Legacy Writing",
        "supervisor": "WriterOrchestrator",
        "team_size": 5,
        "team_position": "2/5 - Legacy writer",
        "upstream": "PlannerAgent",
        "downstream": "EditorAgent",
        "deterministic": False,
        "cost": "~$0.010 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    },

    "EditorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Holistic review of legal documents",
        "phase": "Legacy Review",
        "supervisor": "WriterOrchestrator",
        "team_size": 5,
        "team_position": "3/5 - Legacy editor",
        "upstream": "WriterAgent",
        "downstream": "DoubleCheckerAgent",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    "DoubleCheckerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Factual validation and double-checking",
        "phase": "Legacy Review",
        "supervisor": "WriterOrchestrator",
        "team_size": 5,
        "team_position": "4/5 - Legacy checker",
        "upstream": "EditorAgent",
        "downstream": "StylistAgent",
        "deterministic": False,
        "cost": "~$0.006 per execution",
        "codename": "L - Detective (Tier 1)",
        "cluster": "Gatekeeper / Verification Engine"
    },

    "StylistAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Suggest refinements before final editing",
        "phase": "Legacy Review",
        "supervisor": "WriterOrchestrator",
        "team_size": 5,
        "team_position": "5/5 - Legacy stylist",
        "upstream": "DoubleCheckerAgent",
        "downstream": "Output (final deliverable)",
        "deterministic": False,
        "cost": "~$0.004 per execution",
        "codename": "Tyrion Lannister - QualityChecker (Tier 1)",
        "cluster": "QualityChecker / Policy & Style"
    },

    # ===== STRATEGIC AGENTS (4) =====

    "SettlementOptimizerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Run Monte Carlo simulation (10k iterations) to find optimal settlement range",
        "phase": "Strategic Analysis",
        "supervisor": "StrategicIntegrationEngine",
        "team_size": 4,
        "team_position": "1/4 - Settlement analysis",
        "upstream": "BayesianNetwork (receives posteriors)",
        "downstream": "Report generation",
        "deterministic": True,
        "cost": "$0.00 (pure computation)",
        "codename": "Oracle - Predictive Evaluator (Tier 1)",
        "cluster": "Predictive Evaluator / Risk Forecaster"
    },

    "BATNAAnalyzerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Analyze Best Alternative To Negotiated Agreement",
        "phase": "Strategic Analysis",
        "supervisor": "StrategicIntegrationEngine",
        "team_size": 4,
        "team_position": "2/4 - BATNA analysis",
        "upstream": "BayesianNetwork (receives posteriors)",
        "downstream": "NashEquilibriumCalculatorAgent",
        "deterministic": True,
        "cost": "$0.00 (pure computation)",
        "codename": "Oracle - Predictive Evaluator (Tier 1)",
        "cluster": "Predictive Evaluator / Risk Forecaster"
    },

    "NashEquilibriumCalculatorAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Calculate Nash equilibrium for negotiation strategies",
        "phase": "Strategic Analysis",
        "supervisor": "StrategicIntegrationEngine",
        "team_size": 4,
        "team_position": "3/4 - Game theory analysis",
        "upstream": "BATNAAnalyzerAgent",
        "downstream": "ReputationRiskScorerAgent",
        "deterministic": True,
        "cost": "$0.00 (pure computation)",
        "codename": "Oracle - Predictive Evaluator (Tier 1)",
        "cluster": "Predictive Evaluator / Risk Forecaster"
    },

    "ReputationRiskScorerAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Score reputation risk and institutional damage",
        "phase": "Strategic Analysis",
        "supervisor": "StrategicIntegrationEngine",
        "team_size": 4,
        "team_position": "4/4 - Risk assessment",
        "upstream": "NashEquilibriumCalculatorAgent",
        "downstream": "Report generation",
        "deterministic": True,
        "cost": "$0.00 (pure computation)",
        "codename": "Oracle - Predictive Evaluator (Tier 1)",
        "cluster": "Predictive Evaluator / Risk Forecaster"
    },

    # ===== SPECIALIZED AGENTS (5) =====

    "TimelineAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Build temporal analysis and event timelines",
        "phase": "Specialized Analysis",
        "supervisor": "AutogenIntegrationEngine",
        "team_size": 5,
        "team_position": "1/5 - Timeline specialist",
        "upstream": "MasterSupervisor (receives case context)",
        "downstream": "CausalAgent",
        "deterministic": False,
        "cost": "~$0.007 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "LinguistAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Advanced linguistic analysis and pattern recognition",
        "phase": "Specialized Analysis",
        "supervisor": "AutogenIntegrationEngine",
        "team_size": 5,
        "team_position": "2/5 - Linguistics specialist",
        "upstream": "MasterSupervisor (receives case context)",
        "downstream": "LegalAgent",
        "deterministic": False,
        "cost": "~$0.009 per execution",
        "codename": "Hermes - Language Messenger (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "CausalAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Causal analysis and relationship mapping",
        "phase": "Specialized Analysis",
        "supervisor": "AutogenIntegrationEngine",
        "team_size": 5,
        "team_position": "3/5 - Causal specialist",
        "upstream": "TimelineAgent",
        "downstream": "WriterAgent (autogen)",
        "deterministic": False,
        "cost": "~$0.008 per execution",
        "codename": "Varys - The Spider (Tier 1)",
        "cluster": "Intelligence Broker / Evidence & Adversary Modeling"
    },

    "LegalAgent": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Legal domain expertise and precedent analysis",
        "phase": "Specialized Analysis",
        "supervisor": "AutogenIntegrationEngine",
        "team_size": 5,
        "team_position": "4/5 - Legal specialist",
        "upstream": "LinguistAgent",
        "downstream": "WriterAgent (autogen)",
        "deterministic": False,
        "cost": "~$0.011 per execution",
        "codename": "Oracle - Predictive Evaluator (Tier 1)",
        "cluster": "Predictive Evaluator / Risk Forecaster"
    },

    "WriterAgent (autogen)": {
        "project": "The Matrix - Bayesian Legal AI System",
        "version": "2.1.1",
        "role": "Autogen-integrated writing and synthesis",
        "phase": "Specialized Writing",
        "supervisor": "AutogenIntegrationEngine",
        "team_size": 5,
        "team_position": "5/5 - Autogen writer",
        "upstream": "CausalAgent, LegalAgent",
        "downstream": "Output (final deliverable)",
        "deterministic": False,
        "cost": "~$0.013 per execution",
        "codename": "Gandalf - Narrative Mentor (Tier 1)",
        "cluster": "Narrative Mentor / Writer-Orchestrator"
    }
}

# System-wide context (all agents get this)
SYSTEM_CONTEXT = """
**The Matrix 2.1.1** - Advanced Bayesian Legal AI System

**Mission:** Generate professional legal analysis at $0.07-0.12 per case vs. $60K-130K traditional cost.

**Core Capabilities:**
- Bayesian probabilistic reasoning (pgmpy/PyMC)
- LangChain-powered evidence retrieval
- Strategic settlement optimization (Monte Carlo + Game Theory)
- Distributed cognition (49 specialized agents)
- Complete audit trail (SQLite job tracking)

**Your Team:** You are part of a 49-agent system working together to produce legal memorandums.
Each agent has ONE singular duty. Stay focused on your task.

**Quality Standards:**
- Accuracy > speed
- Citations must be verifiable
- Clear, professional prose
- Respect upstream/downstream contracts
"""

def get_agent_context(agent_type: str) -> Dict[str, Any]:
    """Get full context for an agent type.

    Args:
        agent_type: Name of agent class

    Returns:
        Dictionary with complete context including system context
    """
    context = AGENT_CONTEXTS.get(agent_type, {})
    context['system_context'] = SYSTEM_CONTEXT
    return context
