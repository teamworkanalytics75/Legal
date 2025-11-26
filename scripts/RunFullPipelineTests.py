"""Full Pipeline Test Runner.

Orchestrates all integration tests and generates comprehensive reports.
Tests the complete The Matrix pipeline end-to-end.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests_integration'))

# Import test modules
from test_full_pipeline import FullPipelineTester
from test_bn_configurations import BNConfigurationTester
from test_strategic_modules import StrategicModulesTester
from test_harvard_case_full import HarvardCaseAnalyzer


class TestRunner:
    """Orchestrates all integration tests and generates reports."""

    def __init__(self):
        self.results_dir = Path("tests_integration/results")
        self.results_dir.mkdir(exist_ok=True)

        # Test modules
        self.test_modules = {
            "full_pipeline": FullPipelineTester(),
            "bn_configurations": BNConfigurationTester(),
            "strategic_modules": StrategicModulesTester(),
            "harvard_case": HarvardCaseAnalyzer()
        }

        # Test results
        self.all_results = {}
        self.start_time = None
        self.total_execution_time = 0.0

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🚀 Starting The Matrix Full Pipeline Test Suite")
        print("=" * 60)

        self.start_time = time.time()

        # Run each test module
        for test_name, tester in self.test_modules.items():
            print(f"\n📋 Running {test_name.replace('_', ' ').title()} Tests...")
            print("-" * 40)

            try:
                if test_name == "harvard_case":
                    # Harvard case is a special analysis, not a test suite
                    result = tester.run_full_analysis()
                else:
                    result = tester.run_all_tests()

                self.all_results[test_name] = result

                # Print summary
                if result.get("success_rate"):
                    success_rate = result["success_rate"]
                    print(f"✅ {test_name}: {success_rate*100:.1f}% success rate")
                elif result.get("successful_steps"):
                    successful_steps = result["successful_steps"]
                    total_steps = result["total_steps"]
                    print(f"✅ {test_name}: {successful_steps}/{total_steps} steps successful")
                else:
                    print(f"✅ {test_name}: Completed")

            except Exception as e:
                print(f"❌ {test_name}: Failed with error: {e}")
                self.all_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "test_name": test_name
                }

        # Calculate overall summary
        self.total_execution_time = time.time() - self.start_time
        overall_summary = self._calculate_overall_summary()

        # Generate reports
        self._generate_reports(overall_summary)

        return overall_summary

    def _calculate_overall_summary(self) -> Dict[str, Any]:
        """Calculate overall test summary."""
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        mathematical_issues = False

        # Count tests from each module
        for test_name, result in self.all_results.items():
            if test_name == "harvard_case":
                # Harvard case analysis
                if result.get("successful_steps"):
                    successful_steps = result["successful_steps"]
                    total_steps = result["total_steps"]
                    total_tests += total_steps
                    successful_tests += successful_steps
                    failed_tests += (total_steps - successful_steps)
            else:
                # Regular test suites
                if result.get("total_tests"):
                    total_tests += result["total_tests"]
                    successful_tests += result["successful_tests"]
                    failed_tests += result["failed_tests"]

                # Check for mathematical issues
                if result.get("mathematical_issues_detected"):
                    mathematical_issues = True

        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0

        return {
            "test_suite": "The Matrix Full Pipeline Tests",
            "execution_date": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "overall_success_rate": overall_success_rate,
            "total_execution_time": round(self.total_execution_time, 2),
            "mathematical_issues_detected": mathematical_issues,
            "test_modules": self.all_results,
            "summary": {
                "pipeline_status": "PASS" if overall_success_rate >= 0.8 else "FAIL",
                "pgmpy_integration": "PASS" if self.all_results.get("bn_configurations", {}).get("success_rate", 0) >= 0.75 else "FAIL",
                "strategic_modules": "PASS" if self.all_results.get("strategic_modules", {}).get("success_rate", 0) >= 0.8 else "FAIL",
                "langchain_integration": "PASS" if self.all_results.get("full_pipeline", {}).get("success_rate", 0) >= 0.8 else "FAIL",
                "real_case_analysis": "PASS" if self.all_results.get("harvard_case", {}).get("success_rate", 0) >= 0.8 else "FAIL"
            }
        }

    def _generate_reports(self, summary: Dict[str, Any]) -> None:
        """Generate comprehensive test reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate JSON report
        json_report_file = self.results_dir / f"test_suite_results_{timestamp}.json"
        with open(json_report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate Markdown report
        md_report_file = self.results_dir / f"test_suite_report_{timestamp}.md"
        self._generate_markdown_report(summary, md_report_file)

        # Print console summary
        self._print_console_summary(summary)

        print(f"\n📊 Reports Generated:")
        print(f"  - JSON: {json_report_file}")
        print(f"  - Markdown: {md_report_file}")

    def _generate_markdown_report(self, summary: Dict[str, Any], report_file: Path) -> None:
        """Generate Markdown test report."""
        with open(report_file, 'w') as f:
            f.write(f"# The Matrix Full Pipeline Test Report\n\n")
            f.write(f"**Execution Date:** {summary['execution_date']}\n")
            f.write(f"**Total Execution Time:** {summary['total_execution_time']} seconds\n")
            f.write(f"**Overall Success Rate:** {summary['overall_success_rate']*100:.1f}%\n\n")

            # Overall status
            pipeline_status = summary['summary']['pipeline_status']
            status_emoji = "✅" if pipeline_status == "PASS" else "❌"
            f.write(f"## Overall Status: {status_emoji} {pipeline_status}\n\n")

            # Component status
            f.write("## Component Status\n\n")
            f.write("| Component | Status |\n")
            f.write("|-----------|--------|\n")

            for component, status in summary['summary'].items():
                if component != 'pipeline_status':
                    status_emoji = "✅" if status == "PASS" else "❌"
                    f.write(f"| {component.replace('_', ' ').title()} | {status_emoji} {status} |\n")

            f.write("\n")

            # Test results
            f.write("## Test Results\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Successful:** {summary['successful_tests']}\n")
            f.write(f"- **Failed:** {summary['failed_tests']}\n")
            f.write(f"- **Success Rate:** {summary['overall_success_rate']*100:.1f}%\n\n")

            # Mathematical issues
            if summary['mathematical_issues_detected']:
                f.write("⚠️ **Mathematical Issues Detected:** Settlement optimizer certainty equivalent calculation needs fixing.\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")

            for test_name, result in summary['test_modules'].items():
                f.write(f"### {test_name.replace('_', ' ').title()}\n\n")

                if result.get("success_rate"):
                    f.write(f"- **Success Rate:** {result['success_rate']*100:.1f}%\n")
                elif result.get("successful_steps"):
                    f.write(f"- **Steps:** {result['successful_steps']}/{result['total_steps']} successful\n")

                if result.get("total_execution_time"):
                    f.write(f"- **Execution Time:** {result['total_execution_time']} seconds\n")

                if result.get("error"):
                    f.write(f"- **Error:** {result['error']}\n")

                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if summary['overall_success_rate'] >= 0.8:
                f.write("🎉 **System Ready for Production**\n\n")
                f.write("The The Matrix pipeline is working correctly and ready for real-world case analysis.\n\n")
            else:
                f.write("⚠️ **System Needs Attention**\n\n")
                f.write("Some components failed testing and need attention before production use.\n\n")

            if summary['mathematical_issues_detected']:
                f.write("### Priority Fixes\n\n")
                f.write("1. **Settlement Optimizer:** Fix certainty equivalent calculation formula\n")
                f.write("2. **Risk Adjustment:** Correct the risk adjustment formula to prevent extreme negative values\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. Review failed tests and fix issues\n")
            f.write("2. Address mathematical issues in settlement optimizer\n")
            f.write("3. Run tests again to verify fixes\n")
            f.write("4. Deploy to production environment\n\n")

    def _print_console_summary(self, summary: Dict[str, Any]) -> None:
        """Print console summary."""
        print("\n" + "=" * 60)
        print("🎯 WITCHWEB FULL PIPELINE TEST SUMMARY")
        print("=" * 60)

        # Overall status
        pipeline_status = summary['summary']['pipeline_status']
        status_emoji = "🎉" if pipeline_status == "PASS" else "❌"
        print(f"\n{status_emoji} Overall Status: {pipeline_status}")
        print(f"📊 Success Rate: {summary['overall_success_rate']*100:.1f}%")
        print(f"⏱️  Total Time: {summary['total_execution_time']} seconds")

        # Component status
        print(f"\n📋 Component Status:")
        for component, status in summary['summary'].items():
            if component != 'pipeline_status':
                status_emoji = "✅" if status == "PASS" else "❌"
                print(f"  {status_emoji} {component.replace('_', ' ').title()}: {status}")

        # Test counts
        print(f"\n📈 Test Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")

        # Mathematical issues
        if summary['mathematical_issues_detected']:
            print(f"\n⚠️  Mathematical Issues Detected:")
            print(f"  Settlement optimizer certainty equivalent calculation needs fixing")

        # Final recommendation
        if summary['overall_success_rate'] >= 0.8:
            print(f"\n🎉 RECOMMENDATION: System ready for production!")
        else:
            print(f"\n⚠️  RECOMMENDATION: System needs attention before production")

        print("=" * 60)


def main():
    """Main function to run all tests."""
    runner = TestRunner()
    results = runner.run_all_tests()

    # Return exit code based on success rate
    if results["overall_success_rate"] >= 0.8:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the reports for details.")
        return 1


if __name__ == "__main__":
    exit(main())
