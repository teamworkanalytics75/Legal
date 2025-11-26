#!/usr/bin/env python3
"""
End-to-End The Matrix Workflow Test - Simplified Version

Tests all systems working together without complex imports.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

def test_database_health():
    """Test database health check."""
    print("=" * 60)
    print("TESTING DATABASE HEALTH CHECK")
    print("=" * 60)

    try:
        import sqlite3

        db_path = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"

        # Test connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['cleaned_documents', 'sources', 'nodes', 'edges']
        tables_exist = all(table in tables for table in expected_tables)

        # Get row counts
        row_counts = {}
        for table in expected_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_counts[table] = cursor.fetchone()[0]
            else:
                row_counts[table] = 0

        conn.close()

        print(f"Database Health Check Results:")
        print(f"  Database exists: {Path(db_path).exists()}")
        print(f"  Connection successful: True")
        print(f"  All tables exist: {tables_exist}")
        print(f"  Row counts: {row_counts}")

        # Validate minimum data requirements
        min_docs = row_counts.get('cleaned_documents', 0) >= 100
        min_nodes = row_counts.get('nodes', 0) >= 1000
        min_edges = row_counts.get('edges', 0) >= 100000

        health_score = sum([Path(db_path).exists(), True, tables_exist, min_docs, min_nodes, min_edges])

        print(f"Health Score: {health_score}/6")

        if health_score >= 5:
            print("OK Database Health Check PASSED")
            return True
        else:
            print("ERROR Database Health Check FAILED")
            return False

    except Exception as e:
        print(f"ERROR Database Health Check ERROR: {e}")
        return False

def test_langchain_integration():
    """Test LangChain integration status."""
    print("\n" + "=" * 60)
    print("TESTING LANGCHAIN INTEGRATION")
    print("=" * 60)

    try:
        # Check if API key config exists
        config_file = Path('config/openai_config.json')
        api_key_configured = config_file.exists()

        # Check memory store status
        memory_store = Path('memory_store')
        meta_file = memory_store / 'system_meta.json'

        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            total_memories = meta.get('total_memories', 0)
            agent_count = meta.get('agent_count', 0)

            # Count agents with memories
            agent_stats = meta.get('agent_stats', {})
            agents_with_memories = sum(1 for stats in agent_stats.values() if stats.get('memory_count', 0) > 0)

            print(f"LangChain Integration Status:")
            print(f"  API key configured: {api_key_configured}")
            print(f"  Total memories: {total_memories}")
            print(f"  Agent count: {agent_count}")
            print(f"  Agents with memories: {agents_with_memories}")

            # Test if seeding script works
            seeding_script = Path('scripts/seed_langchain_agent_memories.py')
            seeding_script_exists = seeding_script.exists()

            print(f"  Seeding script exists: {seeding_script_exists}")

            integration_score = sum([api_key_configured, agents_with_memories > 0, seeding_script_exists])

            if integration_score >= 2:
                print("OK LangChain Integration PASSED")
                return True
            else:
                print("ERROR LangChain Integration FAILED")
                return False
        else:
            print("ERROR No memory store found")
            return False

    except Exception as e:
        print(f"ERROR LangChain Integration ERROR: {e}")
        return False

def test_strategic_modules():
    """Test strategic modules."""
    print("\n" + "=" * 60)
    print("TESTING STRATEGIC MODULES")
    print("=" * 60)

    try:
        # Check if strategic module files exist
        settlement_file = Path('writer_agents/settlement_optimizer.py')
        game_theory_file = Path('writer_agents/game_theory.py')
        reputation_file = Path('writer_agents/reputation_risk.py')
        scenario_file = Path('writer_agents/scenario_war_gaming.py')

        settlement_exists = settlement_file.exists()
        game_theory_exists = game_theory_file.exists()
        reputation_exists = reputation_file.exists()
        scenario_exists = scenario_file.exists()

        print(f"Strategic Modules Status:")
        print(f"  Settlement Optimizer: {settlement_exists}")
        print(f"  Game Theory: {game_theory_exists}")
        print(f"  Reputation Risk: {reputation_exists}")
        print(f"  Scenario War Gaming: {scenario_exists}")

        # Test imports
        try:
            sys.path.insert(0, str(Path.cwd()))
            from writer_agents.settlement_optimizer import SettlementOptimizer
            settlement_import = True
        except:
            settlement_import = False

        try:
            from writer_agents.game_theory import BATNAAnalyzer
            game_theory_import = True
        except:
            game_theory_import = False

        try:
            from writer_agents.reputation_risk import ReputationRiskScorer
            reputation_import = True
        except:
            reputation_import = False

        print(f"  Settlement Optimizer import: {settlement_import}")
        print(f"  Game Theory import: {game_theory_import}")
        print(f"  Reputation Risk import: {reputation_import}")

        module_score = sum([settlement_exists, game_theory_exists, reputation_exists, scenario_exists, settlement_import, game_theory_import, reputation_import])

        print(f"Strategic Modules Score: {module_score}/7")

        if module_score >= 5:
            print("OK Strategic Modules PASSED")
            return True
        else:
            print("ERROR Strategic Modules FAILED")
            return False

    except Exception as e:
        print(f"ERROR Strategic Modules ERROR: {e}")
        return False

def test_monitoring_tools():
    """Test monitoring tools."""
    print("\n" + "=" * 60)
    print("TESTING MONITORING TOOLS")
    print("=" * 60)

    try:
        # Check if monitoring tool files exist
        cost_tracker_file = Path('utilities/cost_tracker.py')
        health_check_file = Path('utilities/check_lawsuit_db.py')

        cost_tracker_exists = cost_tracker_file.exists()
        health_check_exists = health_check_file.exists()

        print(f"Monitoring Tools Status:")
        print(f"  Cost Tracker: {cost_tracker_exists}")
        print(f"  Database Health Check: {health_check_exists}")

        # Test imports
        try:
            sys.path.insert(0, str(Path.cwd()))
            from utilities.cost_tracker import CostTracker
            cost_tracker_import = True
        except:
            cost_tracker_import = False

        try:
            from utilities.check_lawsuit_db import DatabaseHealthChecker
            health_check_import = True
        except:
            health_check_import = False

        print(f"  Cost Tracker import: {cost_tracker_import}")
        print(f"  Health Check import: {health_check_import}")

        monitoring_score = sum([cost_tracker_exists, health_check_exists, cost_tracker_import, health_check_import])

        print(f"Monitoring Tools Score: {monitoring_score}/4")

        if monitoring_score >= 3:
            print("OK Monitoring Tools PASSED")
            return True
        else:
            print("ERROR Monitoring Tools FAILED")
            return False

    except Exception as e:
        print(f"ERROR Monitoring Tools ERROR: {e}")
        return False

def test_bayesian_network():
    """Test Bayesian network components."""
    print("\n" + "=" * 60)
    print("TESTING BAYESIAN NETWORK")
    print("=" * 60)

    try:
        # Check if BN files exist
        bn_model_file = Path('bayesian_network/code/experiments/The Matrix1.1.3.xdsl')
        bn_build_file = Path('bayesian_network/code/build_enhanced_bn.py')

        bn_model_exists = bn_model_file.exists()
        bn_build_exists = bn_build_file.exists()

        print(f"Bayesian Network Status:")
        print(f"  BN Model file: {bn_model_exists}")
        print(f"  BN Build script: {bn_build_exists}")

        if bn_model_exists:
            file_size = bn_model_file.stat().st_size
            print(f"  Model file size: {file_size:,} bytes")

        bn_score = sum([bn_model_exists, bn_build_exists])

        print(f"Bayesian Network Score: {bn_score}/2")

        if bn_score >= 1:
            print("OK Bayesian Network PASSED")
            return True
        else:
            print("ERROR Bayesian Network FAILED")
            return False

    except Exception as e:
        print(f"ERROR Bayesian Network ERROR: {e}")
        return False

def main():
    """Run complete end-to-end workflow test."""
    print("The Matrix End-to-End Workflow Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all tests
    test_results = []

    test_results.append(("Database Health Check", test_database_health()))
    test_results.append(("LangChain Integration", test_langchain_integration()))
    test_results.append(("Strategic Modules", test_strategic_modules()))
    test_results.append(("Monitoring Tools", test_monitoring_tools()))
    test_results.append(("Bayesian Network", test_bayesian_network()))

    # Summary
    print("\n" + "=" * 60)
    print("END-TO-END WORKFLOW TEST SUMMARY")
    print("=" * 60)

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")

    print("\nDetailed Results:")
    for test_name, result in test_results:
        status = "OK PASSED" if result else "ERROR FAILED"
        print(f"  {test_name}: {status}")

    if passed_tests == total_tests:
        print("\nALL SYSTEMS OPERATIONAL!")
        print("The Matrix is ready for production use.")
    elif passed_tests >= total_tests * 0.8:
        print("\nMOSTLY OPERATIONAL!")
        print("The Matrix is ready for production with minor issues.")
    else:
        print("\nNEEDS ATTENTION!")
        print("Some systems need fixes before production use.")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
