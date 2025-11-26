"""Comprehensive test script for the advanced multi-agent writing system."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def _load_parse_evidence() -> callable | None:
    """Load parse_evidence_input from the experiments directory if available."""
    experiments_dir = Path(__file__).parent.parent / "experiments"
    candidate = experiments_dir / "WizardWeb1.1.4_STABLE.py"
    if not candidate.exists():
        return None

    import importlib.util

    spec = importlib.util.spec_from_file_location("wizardweb_stable", candidate)
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "parse_evidence_input", None)


parse_evidence_input = _load_parse_evidence()

if parse_evidence_input is None:
    def parse_evidence_input(_):
        raise RuntimeError(
            "WizardWeb1.1.4_STABLE.py not found; please place the file in writer_agents/experiments."
        )
from writer_agents.bn_integration import BNIntegrationConfig, BNWritingIntegrator
from writer_agents.advanced_agents import AdvancedAgentConfig, AdvancedWriterOrchestrator
from writer_agents.enhanced_orchestrator import EnhancedOrchestratorConfig, EnhancedWriterOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedSystemTester:
    """Comprehensive tester for the advanced multi-agent writing system."""
    
    def __init__(self) -> None:
        self._test_results: List[Dict] = []
        self._performance_metrics: Dict[str, List[float]] = {}
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        print("=" * 80)
        print("ADVANCED MULTI-AGENT WRITING SYSTEM - COMPREHENSIVE TESTING")
        print("=" * 80)
        
        tests = [
            ("Basic Advanced Workflow", self._test_basic_advanced_workflow),
            ("Enhanced Orchestrator", self._test_enhanced_orchestrator),
            ("BN Integration", self._test_bn_integration),
            ("Workflow Comparison", self._test_workflow_comparison),
            ("Error Handling", self._test_error_handling),
            ("Performance Analysis", self._test_performance_analysis),
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"RUNNING TEST: {test_name}")
            print(f"{'='*60}")
            
            try:
                result = await test_func()
                self._test_results.append({
                    "test_name": test_name,
                    "status": "PASSED" if result else "FAILED",
                    "details": result if isinstance(result, dict) else {}
                })
                
                if result:
                    print(f"[ok] {test_name}: PASSED")
                else:
                    print(f"x {test_name}: FAILED")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self._test_results.append({
                    "test_name": test_name,
                    "status": "ERROR",
                    "error": str(e)
                })
                print(f" {test_name}: ERROR - {e}")
                all_passed = False
        
        # Generate test report
        self._generate_test_report()
        
        return all_passed
    
    async def _test_basic_advanced_workflow(self) -> bool:
        """Test the basic advanced workflow functionality."""
        try:
            # Create test case insights
            evidence = parse_evidence_input("OGC_Email_Apr18_2025=Sent,PRC_Awareness=Direct")
            
            # Configure advanced system
            config = AdvancedAgentConfig(
                max_review_rounds=1, # Reduced for testing
                enable_research_agents=True,
                enable_quality_gates=True,
                enable_adaptive_workflow=True,
            )
            
            orchestrator = AdvancedWriterOrchestrator(config)
            
            try:
                # Create mock insights
                from writer_agents.insights import CaseInsights, EvidenceItem, Posterior
                
                insights = CaseInsights(
                    reference_id="TEST-001",
                    summary="Test case for advanced workflow validation",
                    posteriors=[
                        Posterior(
                            node_id="LegalSuccess_US",
                            probabilities={"High": 0.3, "Moderate": 0.5, "Low": 0.2}
                        )
                    ],
                    evidence=[
                        EvidenceItem(node_id="OGC_Email_Apr18_2025", state="Sent"),
                        EvidenceItem(node_id="PRC_Awareness", state="Direct")
                    ],
                    jurisdiction="US",
                    case_style="Memorandum"
                )
                
                # Run workflow
                result = await orchestrator.run_advanced_workflow(insights)
                
                # Validate results
                assert result.plan is not None, "Plan should not be None"
                assert len(result.sections) > 0, "Should have generated sections"
                assert len(result.edited_document) > 0, "Should have generated document"
                assert "workflow_type" in result.metadata, "Should have workflow metadata"
                
                print(f"[ok] Generated {len(result.sections)} sections")
                print(f"[ok] Document length: {len(result.edited_document)} characters")
                print(f"[ok] Workflow type: {result.metadata.get('workflow_type')}")
                
                return True
                
            finally:
                await orchestrator.close()
                
        except Exception as e:
            logger.error(f"Basic advanced workflow test failed: {e}")
            return False
    
    async def _test_enhanced_orchestrator(self) -> bool:
        """Test the enhanced orchestrator with intelligent workflow selection."""
        try:
            # Configure enhanced orchestrator
            config = EnhancedOrchestratorConfig(
                use_advanced_workflow=True,
                complexity_threshold=0.6,
                enable_hybrid_mode=True,
                enable_performance_monitoring=True,
            )
            
            orchestrator = EnhancedWriterOrchestrator(config)
            
            try:
                # Test with different complexity cases
                test_cases = [
                    {
                        "name": "Simple Case",
                        "evidence": "OGC_Email_Apr18_2025=Sent",
                        "summary": "Basic test case",
                        "expected_workflow": "traditional"
                    },
                    {
                        "name": "Complex Case",
                        "evidence": "OGC_Email_Apr18_2025=Sent,PRC_Awareness=Direct,FinancialImpact=Severe",
                        "summary": "Complex multi-faceted legal matter requiring comprehensive analysis and strategic recommendations",
                        "expected_workflow": "advanced"
                    }
                ]
                
                for test_case in test_cases:
                    evidence = parse_evidence_input(test_case["evidence"])
                    
                    # Create insights
                    from writer_agents.insights import CaseInsights, EvidenceItem, Posterior
                    
                    insights = CaseInsights(
                        reference_id=f"TEST-{test_case['name'].replace(' ', '-')}",
                        summary=test_case["summary"],
                        posteriors=[
                            Posterior(
                                node_id="TestNode",
                                probabilities={"High": 0.4, "Medium": 0.4, "Low": 0.2}
                            )
                        ],
                        evidence=[EvidenceItem(node_id=k, state=v) for k, v in evidence.items()],
                        jurisdiction="US",
                        case_style="Memorandum"
                    )
                    
                    # Get recommendations
                    recommendations = orchestrator.get_workflow_recommendations(insights)
                    
                    # Run workflow
                    result = await orchestrator.run_intelligent_workflow(insights)
                    
                    # Validate
                    assert result.metadata.get("workflow_type") is not None, "Should have workflow type"
                    assert len(result.edited_document) > 0, "Should generate document"
                    
                    print(f"[ok] {test_case['name']}: {recommendations['recommended_workflow']} workflow")
                    print(f" Complexity: {recommendations['complexity_score']:.2f}")
                    print(f" Actual: {result.metadata.get('workflow_type')}")
                
                return True
                
            finally:
                await orchestrator.close()
                
        except Exception as e:
            logger.error(f"Enhanced orchestrator test failed: {e}")
            return False
    
    async def _test_bn_integration(self) -> bool:
        """Test BN integration with writing agents."""
        try:
            # Configure BN integration
            config = BNIntegrationConfig(
                model_path=None, # Will use mock data
                enable_pysmile=False, # Disable for testing
                fallback_to_mock=True,
                use_advanced_workflow=True,
                enable_hybrid_mode=True,
            )
            
            integrator = BNWritingIntegrator(config)
            
            try:
                # Test evidence validation
                evidence = {"OGC_Email_Apr18_2025": "Sent", "PRC_Awareness": "Direct"}
                is_valid, errors = await integrator._evidence_validator.validate_evidence(evidence)
                
                assert is_valid, f"Evidence should be valid: {errors}"
                print("[ok] Evidence validation passed")
                
                # Test workflow recommendations
                recommendations = integrator.get_workflow_recommendations(
                    evidence, "Test case for BN integration"
                )
                
                assert "recommended_workflow" in recommendations, "Should have workflow recommendation"
                print(f"[ok] Workflow recommendation: {recommendations['recommended_workflow']}")
                
                # Test full integration workflow
                result = await integrator.run_bn_writing_workflow(
                    evidence=evidence,
                    summary="Test case for BN integration validation",
                    reference_id="BN-TEST-001",
                    jurisdiction="US",
                    case_style="Memorandum"
                )
                
                assert result.case_insights is not None, "Should have case insights"
                assert result.writing_result is not None, "Should have writing result"
                assert len(result.integration_metrics) > 0, "Should have integration metrics"
                
                print(f"[ok] BN integration completed")
                print(f" Evidence items: {result.integration_metrics.get('evidence_count', 0)}")
                print(f" Posterior nodes: {result.integration_metrics.get('posterior_count', 0)}")
                print(f" Document length: {result.integration_metrics.get('document_length', 0)}")
                
                return True
                
            finally:
                await integrator.close()
                
        except Exception as e:
            logger.error(f"BN integration test failed: {e}")
            return False
    
    async def _test_workflow_comparison(self) -> bool:
        """Test comparison between different workflows."""
        try:
            # Create test insights
            from writer_agents.insights import CaseInsights, EvidenceItem, Posterior
            
            insights = CaseInsights(
                reference_id="COMPARISON-TEST",
                summary="Test case for workflow comparison",
                posteriors=[
                    Posterior(
                        node_id="TestNode",
                        probabilities={"High": 0.5, "Medium": 0.3, "Low": 0.2}
                    )
                ],
                evidence=[
                    EvidenceItem(node_id="TestEvidence", state="Present")
                ],
                jurisdiction="US",
                case_style="Memorandum"
            )
            
            # Test traditional workflow
            from writer_agents.orchestrator import WriterOrchestrator, WriterOrchestratorConfig
            
            traditional_config = WriterOrchestratorConfig()
            traditional_orchestrator = WriterOrchestrator(traditional_config)
            
            try:
                traditional_result = await traditional_orchestrator.run(insights)
                traditional_metrics = {
                    "sections": len(traditional_result.sections),
                    "document_length": len(traditional_result.edited_document),
                    "reviews": len(traditional_result.reviews),
                }
            finally:
                await traditional_orchestrator.close()
            
            # Test advanced workflow
            advanced_config = AdvancedAgentConfig(max_review_rounds=1)
            advanced_orchestrator = AdvancedWriterOrchestrator(advanced_config)
            
            try:
                advanced_result = await advanced_orchestrator.run_advanced_workflow(insights)
                advanced_metrics = {
                    "sections": len(advanced_result.sections),
                    "document_length": len(advanced_result.edited_document),
                    "reviews": len(advanced_result.reviews),
                    "workflow_type": advanced_result.metadata.get("workflow_type"),
                }
            finally:
                await advanced_orchestrator.close()
            
            # Compare results
            print("[ok] Workflow Comparison Results:")
            print(f" Traditional: {traditional_metrics['sections']} sections, {traditional_metrics['document_length']} chars")
            print(f" Advanced: {advanced_metrics['sections']} sections, {advanced_metrics['document_length']} chars")
            print(f" Advanced workflow type: {advanced_metrics['workflow_type']}")
            
            # Validate that both workflows produce results
            assert traditional_metrics['document_length'] > 0, "Traditional workflow should produce document"
            assert advanced_metrics['document_length'] > 0, "Advanced workflow should produce document"
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow comparison test failed: {e}")
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test error handling and recovery mechanisms."""
        try:
            # Test with invalid evidence
            config = BNIntegrationConfig(
                enable_pysmile=False,
                fallback_to_mock=True,
                use_advanced_workflow=False, # Use simpler workflow for error testing
            )
            
            integrator = BNWritingIntegrator(config)
            
            try:
                # Test with empty evidence
                result = await integrator.run_bn_writing_workflow(
                    evidence={},
                    summary="Test with empty evidence",
                    reference_id="ERROR-TEST-001"
                )
                
                # Should still work with fallback
                assert result.case_insights is not None, "Should handle empty evidence gracefully"
                print("[ok] Empty evidence handled gracefully")
                
                # Test with invalid evidence format
                invalid_evidence = {"": "", "invalid": ""}
                is_valid, errors = await integrator._evidence_validator.validate_evidence(invalid_evidence)
                
                assert not is_valid, "Should detect invalid evidence"
                assert len(errors) > 0, "Should provide error messages"
                print(f"[ok] Invalid evidence detected: {len(errors)} errors")
                
                return True
                
            finally:
                await integrator.close()
                
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    async def _test_performance_analysis(self) -> bool:
        """Test performance analysis and monitoring."""
        try:
            import time
            
            # Test performance monitoring
            config = EnhancedOrchestratorConfig(
                enable_performance_monitoring=True,
                use_advanced_workflow=True,
            )
            
            orchestrator = EnhancedWriterOrchestrator(config)
            
            try:
                # Create test insights
                from writer_agents.insights import CaseInsights, EvidenceItem, Posterior
                
                insights = CaseInsights(
                    reference_id="PERF-TEST",
                    summary="Performance test case",
                    posteriors=[
                        Posterior(
                            node_id="PerfNode",
                            probabilities={"High": 0.6, "Low": 0.4}
                        )
                    ],
                    evidence=[
                        EvidenceItem(node_id="PerfEvidence", state="Test")
                    ],
                    jurisdiction="US",
                    case_style="Memorandum"
                )
                
                # Measure execution time
                start_time = time.time()
                result = await orchestrator.run_intelligent_workflow(insights)
                execution_time = time.time() - start_time
                
                # Validate performance metrics
                assert execution_time > 0, "Should have measurable execution time"
                assert "performance_metrics" in result.metadata, "Should have performance metrics"
                
                metrics = result.metadata.get("performance_metrics", {})
                print(f"[ok] Performance Analysis:")
                print(f" Execution time: {execution_time:.2f} seconds")
                print(f" Workflow type: {result.metadata.get('workflow_type')}")
                print(f" Complexity score: {result.metadata.get('complexity_score', 0):.2f}")
                
                # Store performance data
                self._performance_metrics.setdefault("execution_time", []).append(execution_time)
                self._performance_metrics.setdefault("document_length", []).append(len(result.edited_document))
                
                return True
                
            finally:
                await orchestrator.close()
                
        except Exception as e:
            logger.error(f"Performance analysis test failed: {e}")
            return False
    
    def _generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("TEST REPORT SUMMARY")
        print("=" * 80)
        
        # Test results summary
        passed = sum(1 for result in self._test_results if result["status"] == "PASSED")
        failed = sum(1 for result in self._test_results if result["status"] == "FAILED")
        errors = sum(1 for result in self._test_results if result["status"] == "ERROR")
        total = len(self._test_results)
        
        print(f"\nTest Results:")
        print(f" [ok] Passed: {passed}/{total}")
        print(f" x Failed: {failed}/{total}")
        print(f" Errors: {errors}/{total}")
        print(f" Success Rate: {(passed/total)*100:.1f}%")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for result in self._test_results:
            status_icon = {
                "PASSED": "[ok]",
                "FAILED": "x", 
                "ERROR": ""
            }.get(result["status"], "?")
            
            print(f" {status_icon} {result['test_name']}: {result['status']}")
            if result["status"] == "ERROR" and "error" in result:
                print(f" Error: {result['error']}")
        
        # Performance summary
        if self._performance_metrics:
            print(f"\nPerformance Summary:")
            for metric, values in self._performance_metrics.items():
                if values:
                    avg_value = sum(values) / len(values)
                    print(f" {metric}: {avg_value:.2f} (avg), {min(values):.2f}-{max(values):.2f} (range)")
        
        # Save detailed report
        report_data = {
            "test_results": self._test_results,
            "performance_metrics": self._performance_metrics,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": (passed/total)*100 if total > 0 else 0
            }
        }
        
        report_path = Path("test_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n Detailed report saved to: {report_path}")


async def main():
    """Main test execution function."""
    tester = AdvancedSystemTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            print("\n ALL TESTS PASSED! The advanced multi-agent writing system is working correctly.")
            return 0
        else:
            print("\nWARNING SOME TESTS FAILED! Please review the test report for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n TEST EXECUTION FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
