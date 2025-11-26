"""
AI Asset-to-Revit Pipeline - Pipeline Orchestrator

Main pipeline controller that orchestrates all phases and integrates with WitchWeb.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import pipeline components
from asset_analyzer.geometry_parser import parse_asset
from asset_analyzer.constraint_detector import detect_constraints
from asset_analyzer.revit_mapper import map_constraints_to_revit
from dynamo_generator.graph_builder import build_dynamo_graph
from design_explorer.variation_generator import explore_design_variations

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    asset_path: str
    output_dir: str
    num_variations: int
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cultural_style: Optional[str] = None
    optimization_method: str = 'witchweb'
    enable_ai_analysis: bool = True
    enable_dynamo_generation: bool = True
    enable_design_exploration: bool = True

@dataclass
class PipelineResult:
    """Complete pipeline result"""
    asset_name: str
    geometry_data: Any
    constraint_map: Any
    revit_mapping: Any
    dynamo_graph_path: Optional[str]
    optimization_result: Any
    execution_time: float
    success: bool
    errors: List[str]
    metadata: Dict[str, Any]

class PipelineOrchestrator:
    """Main pipeline orchestrator"""

    def __init__(self):
        # Simple initialization - no complex integrations needed
        pass

    def run_pipeline(self, config: PipelineConfig) -> PipelineResult:
        """Run complete pipeline"""

        logger.info(f"Starting pipeline for: {config.asset_path}")
        start_time = time.time()

        errors = []
        geometry_data = None
        constraint_map = None
        revit_mapping = None
        dynamo_graph_path = None
        optimization_result = None

        try:
            # Phase 1: Asset Analysis
            logger.info("Phase 1: Asset Analysis")
            geometry_data = self._run_phase1(config)

            # Phase 2: Constraint Detection
            logger.info("Phase 2: Constraint Detection")
            constraint_map = self._run_phase2(geometry_data, config)

            # Phase 3: Revit Mapping
            logger.info("Phase 3: Revit Mapping")
            revit_mapping = self._run_phase3(constraint_map)

            # Phase 4: Dynamo Generation
            if config.enable_dynamo_generation:
                logger.info("Phase 4: Dynamo Generation")
                dynamo_graph_path = self._run_phase4(revit_mapping, config)

            # Phase 5: Design Exploration
            if config.enable_design_exploration:
                logger.info("Phase 5: Design Exploration")
                optimization_result = self._run_phase5(revit_mapping, config)

            success = True

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            errors.append(str(e))
            success = False

        execution_time = time.time() - start_time

        return PipelineResult(
            asset_name=Path(config.asset_path).stem,
            geometry_data=geometry_data,
            constraint_map=constraint_map,
            revit_mapping=revit_mapping,
            dynamo_graph_path=dynamo_graph_path,
            optimization_result=optimization_result,
            execution_time=execution_time,
            success=success,
            errors=errors,
            metadata={
                'config': asdict(config),
                'pipeline_version': '1.0.0',
                'timestamp': time.time()
            }
        )

    def _run_phase1(self, config: PipelineConfig):
        """Run Phase 1: Asset Analysis"""

        try:
            geometry_data = parse_asset(config.asset_path)
            logger.info(f"Phase 1 complete: Parsed {geometry_data.asset_name}")
            return geometry_data
        except Exception as e:
            logger.error(f"Phase 1 failed: {str(e)}")
            raise

    def _run_phase2(self, geometry_data, config: PipelineConfig):
        """Run Phase 2: Constraint Detection"""

        try:
            if config.enable_ai_analysis:
                constraint_map = detect_constraints(
                    geometry_data,
                    config.openai_api_key,
                    config.anthropic_api_key
                )
            else:
                # Use rule-based analysis
                from asset_analyzer.constraint_detector import ConstraintDetector
                detector = ConstraintDetector()
                constraint_map = detector._rule_based_analysis(geometry_data)

            logger.info(f"Phase 2 complete: Detected {len(constraint_map.variable_parameters)} parameters")
            return constraint_map
        except Exception as e:
            logger.error(f"Phase 2 failed: {str(e)}")
            raise

    def _run_phase3(self, constraint_map):
        """Run Phase 3: Revit Mapping"""

        try:
            revit_mapping = map_constraints_to_revit(constraint_map)
            logger.info(f"Phase 3 complete: Mapped to {revit_mapping.family_type.name}")
            return revit_mapping
        except Exception as e:
            logger.error(f"Phase 3 failed: {str(e)}")
            raise

    def _run_phase4(self, revit_mapping, config: PipelineConfig):
        """Run Phase 4: Dynamo Generation"""

        try:
            output_path = Path(config.output_dir) / f"{revit_mapping.asset_name}_graph.dyn"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            dynamo_graph_path = build_dynamo_graph(revit_mapping, str(output_path))
            logger.info(f"Phase 4 complete: Generated {dynamo_graph_path}")
            return dynamo_graph_path
        except Exception as e:
            logger.error(f"Phase 4 failed: {str(e)}")
            raise

    def _run_phase5(self, revit_mapping, config: PipelineConfig):
        """Run Phase 5: Design Exploration"""

        try:
            optimization_result = explore_design_variations(revit_mapping, config.num_variations)
            logger.info(f"Phase 5 complete: Explored {len(optimization_result.all_variations)} variations")
            return optimization_result
        except Exception as e:
            logger.error(f"Phase 5 failed: {str(e)}")
            raise

    def run_batch_pipeline(self, configs: List[PipelineConfig]) -> List[PipelineResult]:
        """Run pipeline for multiple assets"""

        logger.info(f"Running batch pipeline for {len(configs)} assets")

        results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all pipeline runs
            future_to_config = {
                executor.submit(self.run_pipeline, config): config
                for config in configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed pipeline for: {result.asset_name}")
                except Exception as e:
                    logger.error(f"Pipeline failed for {config.asset_path}: {str(e)}")
                    # Create error result
                    error_result = PipelineResult(
                        asset_name=Path(config.asset_path).stem,
                        geometry_data=None,
                        constraint_map=None,
                        revit_mapping=None,
                        dynamo_graph_path=None,
                        optimization_result=None,
                        execution_time=0,
                        success=False,
                        errors=[str(e)],
                        metadata={'config': asdict(config)}
                    )
                    results.append(error_result)

        return results

    def export_results(self, result: PipelineResult, output_dir: str):
        """Export pipeline results"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export constraint map
        if result.constraint_map:
            constraint_file = output_path / f"{result.asset_name}_constraints.json"
            with open(constraint_file, 'w') as f:
                json.dump(asdict(result.constraint_map), f, indent=2)

        # Export Revit mapping
        if result.revit_mapping:
            mapping_file = output_path / f"{result.asset_name}_revit_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(asdict(result.revit_mapping), f, indent=2)

        # Export optimization results
        if result.optimization_result:
            optimization_file = output_path / f"{result.asset_name}_optimization.json"
            with open(optimization_file, 'w') as f:
                json.dump(asdict(result.optimization_result), f, indent=2)

        # Export complete result
        result_file = output_path / f"{result.asset_name}_pipeline_result.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(f"Results exported to: {output_path}")

# Removed WitchWeb and LangChain integrations - keeping it simple

def run_pipeline(asset_path: str,
                output_dir: str = "outputs",
                num_variations: int = 20,
                openai_api_key: Optional[str] = None,
                anthropic_api_key: Optional[str] = None,
                cultural_style: Optional[str] = None) -> PipelineResult:
    """Main function to run the complete pipeline"""

    config = PipelineConfig(
        asset_path=asset_path,
        output_dir=output_dir,
        num_variations=num_variations,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        cultural_style=cultural_style
    )

    orchestrator = PipelineOrchestrator()
    result = orchestrator.run_pipeline(config)

    # Export results
    orchestrator.export_results(result, output_dir)

    return result

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        asset_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
        num_variations = int(sys.argv[3]) if len(sys.argv) > 3 else 20

        try:
            result = run_pipeline(asset_path, output_dir, num_variations)

            if result.success:
                print(f"Pipeline completed successfully!")
                print(f"Asset: {result.asset_name}")
                print(f"Execution time: {result.execution_time:.2f} seconds")
                print(f"Dynamo graph: {result.dynamo_graph_path}")
                print(f"Variations explored: {len(result.optimization_result.all_variations) if result.optimization_result else 0}")
                print(f"Best score: {result.optimization_result.best_variation.overall_score:.2f}" if result.optimization_result else "N/A")
            else:
                print(f"Pipeline failed: {result.errors}")

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
    else:
        print("Usage: python pipeline_orchestrator.py <asset_file> [output_dir] [num_variations]")
