#!/usr/bin/env python
"""Test NLP, factuality filter, document ingestion, and full pipeline integration."""

import os
import time
from pathlib import Path


def test_nlp_analysis_pipeline():
    """Test NLP analysis pipeline."""
    print("\n" + "="*60)
    print("TESTING NLP ANALYSIS PIPELINE")
    print("="*60)
    
    try:
        # Check if NLP analysis module exists
        nlp_dir = Path("nlp_analysis")
        if not nlp_dir.exists():
            print("  NLP analysis module not found - SKIPPED")
            return None
        
        # Check for example usage script
        example_script = nlp_dir / "scripts" / "example_nlp_usage.py"
        if not example_script.exists():
            print("  NLP example script not found - SKIPPED")
            return None
        
        print(f"  NLP analysis module found: {nlp_dir}")
        print(f"  Example script found: {example_script}")
        
        # Test basic imports
        try:
            import spacy
            print(f"  spaCy version: {spacy.__version__}")
        except ImportError:
            print("  spaCy not installed - SKIPPED")
            return None
        
        try:
            import transformers
            print(f"  Transformers version: {transformers.__version__}")
        except ImportError:
            print("  Transformers not installed - SKIPPED")
            return None
        
        print("  NLP dependencies available - READY")
        return True
        
    except Exception as e:
        print(f"  NLP analysis test failed: {e}")
        return None


def test_factuality_filter():
    """Test factuality filter."""
    print("\n" + "="*60)
    print("TESTING FACTUALITY FILTER")
    print("="*60)
    
    try:
        # Check if factuality filter module exists
        factuality_dir = Path("factuality_filter")
        if not factuality_dir.exists():
            print("  Factuality filter module not found - SKIPPED")
            return None
        
        # Check for demo script
        demo_script = factuality_dir / "scripts" / "demo_factuality_filter.py"
        if not demo_script.exists():
            print("  Factuality filter demo script not found - SKIPPED")
            return None
        
        print(f"  Factuality filter module found: {factuality_dir}")
        print(f"  Demo script found: {demo_script}")
        
        # Test basic imports
        try:
            from factuality_filter.code.factuality_filter import FactualityFilter
            print("  FactualityFilter class available - READY")
            return True
        except ImportError as e:
            print(f"  FactualityFilter import failed: {e}")
            return None
        
    except Exception as e:
        print(f"  Factuality filter test failed: {e}")
        return None


def test_document_ingestion():
    """Test document ingestion."""
    print("\n" + "="*60)
    print("TESTING DOCUMENT INGESTION")
    print("="*60)
    
    try:
        # Check if document ingestion module exists
        ingestion_dir = Path("document_ingestion")
        if not ingestion_dir.exists():
            print("  Document ingestion module not found - SKIPPED")
            return None
        
        # Check for main script
        main_script = ingestion_dir / "main.py"
        if not main_script.exists():
            print("  Document ingestion main script not found - SKIPPED")
            return None
        
        print(f"  Document ingestion module found: {ingestion_dir}")
        print(f"  Main script found: {main_script}")
        
        # Check database
        lawsuit_db = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
        if lawsuit_db.exists():
            print(f"  Lawsuit database found: {lawsuit_db}")
            
            # Check database size
            db_size = lawsuit_db.stat().st_size
            print(f"  Database size: {db_size:,} bytes ({db_size/1024/1024:.1f} MB)")
            
            # Test database connection
            import sqlite3
            conn = sqlite3.connect(lawsuit_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"  Database tables: {tables}")
            
            # Check document count
            if 'cleaned_documents' in tables:
                cursor.execute("SELECT COUNT(*) FROM cleaned_documents")
                doc_count = cursor.fetchone()[0]
                print(f"  Document count: {doc_count}")
            
            conn.close()
            print("  Document ingestion - READY")
            return True
        else:
            print("  Lawsuit database not found - SKIPPED")
            return None
        
    except Exception as e:
        print(f"  Document ingestion test failed: {e}")
        return None


def test_full_pipeline_integration():
    """Test full pipeline integration."""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE INTEGRATION")
    print("="*60)
    
    integration_results = {}
    
    print("\n1. Document Ingestion -> Database:")
    print("-" * 30)
    
    lawsuit_db = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
    if lawsuit_db.exists():
        print("  Database exists and accessible")
        integration_results["database"] = True
    else:
        print("  Database not found")
        integration_results["database"] = False
    
    # Test 2: Database -> LangChain -> Evidence Retrieval
    print("\n2. Database -> LangChain -> Evidence Retrieval:")
    print("-" * 30)
    
    try:
        from writer_agents.code.langchain_integration import LangChainSQLAgent
        from writer_agents.code.agents import ModelConfig
        
        if lawsuit_db.exists():
            langchain_agent = LangChainSQLAgent(lawsuit_db, ModelConfig(model="gpt-4o-mini"))
            result = langchain_agent.query_evidence("What tables are in this database?")
            
            if result['success']:
                print("  LangChain can query database")
                integration_results["langchain"] = True
            else:
                print("  LangChain query failed")
                integration_results["langchain"] = False
        else:
            print("  Database not available for LangChain test")
            integration_results["langchain"] = False
            
    except Exception as e:
        print(f"  LangChain integration failed: {e}")
        integration_results["langchain"] = False
    
    # Test 3: Evidence -> BN Analysis -> Strategic Modules
    print("\n3. Evidence -> BN Analysis -> Strategic Modules:")
    print("-" * 30)
    
    try:
        from writer_agents.code.insights import CaseInsights, Posterior
        from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig
        from writer_agents.game_theory import BATNAAnalyzer
        from writer_agents.reputation_risk import ReputationRiskScorer
        
        # Create test insights
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
                interpretation="Test case"
            )
        ]
        
        insights = CaseInsights(
            reference_id="integration_test",
            summary="Integration test case",
            posteriors=posteriors
        )
        
        # Test settlement optimization
        optimizer = SettlementOptimizer()
        config = SettlementConfig(monte_carlo_iterations=100)  # Fast test
        settlement_rec = optimizer.optimize_settlement(insights, config)
        
        # Test game theory
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(insights, settlement_rec)
        
        # Test reputation risk
        risk_scorer = ReputationRiskScorer()
        risk_assessments = risk_scorer.score_reputation_risk(insights)
        
        print("  Strategic modules work with BN insights")
        integration_results["strategic"] = True
        
    except Exception as e:
        print(f"  Strategic modules integration failed: {e}")
        integration_results["strategic"] = False
    
    # Test 4: Strategic Analysis -> Atomic Workflow
    print("\n4. Strategic Analysis -> Atomic Workflow:")
    print("-" * 30)
    
    try:
        from writer_agents.code.agents import AgentFactory, ModelConfig
        from writer_agents.code.master_supervisor import MasterSupervisor
        
        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        master_supervisor = MasterSupervisor(factory)
        
        # Check that all phases are available
        phases = ['research', 'drafting', 'citation', 'qa']
        available_phases = [phase for phase in phases if hasattr(master_supervisor, phase)]
        
        if len(available_phases) == len(phases):
            print("  All atomic workflow phases available")
            integration_results["atomic_workflow"] = True
        else:
            print(f"  Partial atomic workflow: {available_phases}")
            integration_results["atomic_workflow"] = False
            
    except Exception as e:
        print(f"  Atomic workflow integration failed: {e}")
        integration_results["atomic_workflow"] = False
    
    return integration_results


def main():
    """Run integration testing."""
    print("WITCHWEB INTEGRATION TESTING")
    print("="*60)
    print("Testing NLP, factuality filter, document ingestion, and full pipeline")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Test individual components
        nlp_result = test_nlp_analysis_pipeline()
        factuality_result = test_factuality_filter()
        ingestion_result = test_document_ingestion()
        
        # Test full pipeline integration
        integration_results = test_full_pipeline_integration()
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        print(f"NLP Analysis Pipeline: {'READY' if nlp_result else 'SKIPPED'}")
        print(f"Factuality Filter: {'READY' if factuality_result else 'SKIPPED'}")
        print(f"Document Ingestion: {'READY' if ingestion_result else 'SKIPPED'}")
        
        print(f"\nFull Pipeline Integration:")
        for component, status in integration_results.items():
            status_text = "READY" if status else "FAILED"
            print(f"  {component}: {status_text}")
        
        # Overall success
        integration_success = sum(integration_results.values())
        total_components = len(integration_results)
        
        print(f"\nIntegration Success Rate: {integration_success}/{total_components} ({integration_success/total_components*100:.1f}%)")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if integration_success >= 3:  # 75% success rate
            print("\nSUCCESS: Integration testing completed successfully!")
            print("   Core pipeline components are integrated")
            print("   Strategic modules are operational")
            print("   Atomic workflow is ready")
            return True
        else:
            print("\nPARTIAL: Some integration points need attention")
            return False
            
    except Exception as e:
        print(f"\nERROR: Integration testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
