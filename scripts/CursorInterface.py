"""
AI Asset-to-Revit Pipeline - Cursor IDE Integration

Easy-to-use interface for running the pipeline from Cursor.
Enhanced with steampunk house integration and WitchWeb optimization.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import pipeline components
try:
    from pipeline_orchestrator import run_pipeline, PipelineConfig, PipelineOrchestrator
    from ai_asset_pipeline.asset_analyzer.geometry_parser import parse_asset
    from ai_asset_pipeline.asset_analyzer.constraint_detector import detect_constraints
    from ai_asset_pipeline.asset_analyzer.revit_mapper import map_constraints_to_revit
    from ai_asset_pipeline.dynamo_generator.graph_builder import build_dynamo_graph
    from ai_asset_pipeline.design_explorer.variation_generator import explore_design_variations
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Pipeline components not available: {e}")
    PIPELINE_AVAILABLE = False

# Removed WitchWeb integration - keeping it simple
WITCHWEB_AVAILABLE = False

logger = logging.getLogger(__name__)

class CursorPipelineInterface:
    """Cursor-friendly interface for the AI Asset-to-Revit pipeline"""

    def __init__(self):
        self.orchestrator = PipelineOrchestrator()
        self.last_result = None

    def analyze_asset(self, asset_path: str,
                     cultural_style: Optional[str] = None,
                     openai_api_key: Optional[str] = None,
                     anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Analyze 3D asset and extract constraints"""

        print(f"🔍 Analyzing asset: {asset_path}")

        try:
            # Run pipeline with analysis only
            config = PipelineConfig(
                asset_path=asset_path,
                output_dir="outputs",
                num_variations=0,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                cultural_style=cultural_style,
                enable_dynamo_generation=False,
                enable_design_exploration=False
            )

            result = self.orchestrator.run_pipeline(config)

            if result.success:
                print("✅ Asset analysis complete!")
                print(f"📊 Found {len(result.constraint_map.variable_parameters)} variable parameters")
                print(f"🏗️ Detected {len(result.constraint_map.fixed_constraints)} fixed constraints")
                print(f"🎨 Cultural style: {result.constraint_map.cultural_style}")

                self.last_result = result
                return {
                    'success': True,
                    'asset_name': result.asset_name,
                    'cultural_style': result.constraint_map.cultural_style,
                    'variable_parameters': len(result.constraint_map.variable_parameters),
                    'fixed_constraints': len(result.constraint_map.fixed_constraints),
                    'constraint_map': result.constraint_map
                }
            else:
                print(f"❌ Analysis failed: {result.errors}")
                return {'success': False, 'errors': result.errors}

        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def generate_dynamo_graph(self, asset_path: str,
                            cultural_style: Optional[str] = None,
                            openai_api_key: Optional[str] = None,
                            anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Generate Dynamo graph for asset"""

        print(f"🔧 Generating Dynamo graph for: {asset_path}")

        try:
            # Run pipeline with Dynamo generation
            config = PipelineConfig(
                asset_path=asset_path,
                output_dir="outputs",
                num_variations=0,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                cultural_style=cultural_style,
                enable_dynamo_generation=True,
                enable_design_exploration=False
            )

            result = self.orchestrator.run_pipeline(config)

            if result.success:
                print("✅ Dynamo graph generated!")
                print(f"📁 Graph saved to: {result.dynamo_graph_path}")

                self.last_result = result
                return {
                    'success': True,
                    'asset_name': result.asset_name,
                    'dynamo_graph_path': result.dynamo_graph_path,
                    'revit_mapping': result.revit_mapping
                }
            else:
                print(f"❌ Dynamo generation failed: {result.errors}")
                return {'success': False, 'errors': result.errors}

        except Exception as e:
            print(f"❌ Dynamo generation error: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def explore_design_variations(self, asset_path: str,
                                num_variations: int = 20,
                                cultural_style: Optional[str] = None,
                                openai_api_key: Optional[str] = None,
                                anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Explore design variations and optimization"""

        print(f"🎨 Exploring {num_variations} design variations for: {asset_path}")

        try:
            # Run complete pipeline
            config = PipelineConfig(
                asset_path=asset_path,
                output_dir="outputs",
                num_variations=num_variations,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                cultural_style=cultural_style,
                enable_dynamo_generation=True,
                enable_design_exploration=True
            )

            result = self.orchestrator.run_pipeline(config)

            if result.success:
                print("✅ Design exploration complete!")
                print(f"🏆 Best variation: {result.optimization_result.best_variation.id}")
                print(f"📊 Overall score: {result.optimization_result.best_variation.overall_score:.2f}")
                print(f"🎯 Aesthetic: {result.optimization_result.best_variation.aesthetic_score:.2f}")
                print(f"🔧 Functionality: {result.optimization_result.best_variation.functionality_score:.2f}")
                print(f"🏯 Cultural: {result.optimization_result.best_variation.cultural_score:.2f}")
                print(f"🏗️ Structural: {result.optimization_result.best_variation.structural_score:.2f}")

                self.last_result = result
                return {
                    'success': True,
                    'asset_name': result.asset_name,
                    'best_variation': result.optimization_result.best_variation,
                    'total_variations': len(result.optimization_result.all_variations),
                    'dynamo_graph_path': result.dynamo_graph_path,
                    'optimization_result': result.optimization_result
                }
            else:
                print(f"❌ Design exploration failed: {result.errors}")
                return {'success': False, 'errors': result.errors}

        except Exception as e:
            print(f"❌ Design exploration error: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def run_complete_pipeline(self, asset_path: str,
                            num_variations: int = 20,
                            cultural_style: Optional[str] = None,
                            openai_api_key: Optional[str] = None,
                            anthropic_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Run complete pipeline with all phases"""

        print(f"🚀 Running complete pipeline for: {asset_path}")
        print(f"📊 Variations: {num_variations}")
        print(f"🎨 Cultural style: {cultural_style or 'auto-detect'}")

        try:
            result = run_pipeline(
                asset_path=asset_path,
                output_dir="outputs",
                num_variations=num_variations,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                cultural_style=cultural_style
            )

            if result.success:
                print("🎉 Complete pipeline successful!")
                print(f"⏱️ Execution time: {result.execution_time:.2f} seconds")
                print(f"📁 Output directory: outputs/")
                print(f"🔧 Dynamo graph: {result.dynamo_graph_path}")
                print(f"🏆 Best variation: {result.optimization_result.best_variation.id}")
                print(f"📊 Best score: {result.optimization_result.best_variation.overall_score:.2f}")

                self.last_result = result
                return {
                    'success': True,
                    'asset_name': result.asset_name,
                    'execution_time': result.execution_time,
                    'dynamo_graph_path': result.dynamo_graph_path,
                    'best_variation': result.optimization_result.best_variation,
                    'total_variations': len(result.optimization_result.all_variations),
                    'result': result
                }
            else:
                print(f"❌ Pipeline failed: {result.errors}")
                return {'success': False, 'errors': result.errors}

        except Exception as e:
            print(f"❌ Pipeline error: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def show_last_result(self):
        """Show details of last pipeline result"""

        if not self.last_result:
            print("❌ No previous result available")
            return

        result = self.last_result

        print("📊 Last Pipeline Result:")
        print(f"Asset: {result.asset_name}")
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.2f} seconds")

        if result.constraint_map:
            print(f"Cultural style: {result.constraint_map.cultural_style}")
            print(f"Variable parameters: {len(result.constraint_map.variable_parameters)}")
            print(f"Fixed constraints: {len(result.constraint_map.fixed_constraints)}")

        if result.dynamo_graph_path:
            print(f"Dynamo graph: {result.dynamo_graph_path}")

        if result.optimization_result:
            print(f"Total variations: {len(result.optimization_result.all_variations)}")
            print(f"Best variation: {result.optimization_result.best_variation.id}")
            print(f"Best score: {result.optimization_result.best_variation.overall_score:.2f}")

    def list_available_styles(self):
        """List available cultural styles"""

        styles = {
            'japanese': 'Traditional Japanese architecture (pagodas, temples)',
            'steampunk': 'Victorian industrial steampunk style',
            'modern': 'Contemporary modern architecture',
            'hybrid': 'Mixed traditional and modern elements',
            'generic': 'Generic architectural style'
        }

        print("🎨 Available Cultural Styles:")
        for style, description in styles.items():
            print(f"  {style}: {description}")

    def run_steampunk_house_pipeline(self, num_variations: int = 20) -> Dict[str, Any]:
        """Run complete pipeline on the steampunk house"""

        steampunk_house_path = r"C:\Users\Owner\Desktop\SteamPunk House\source\tutorial_steampunk_house_no_LM.fbx"

        if not Path(steampunk_house_path).exists():
            print(f"❌ Steampunk house not found at: {steampunk_house_path}")
            return {'success': False, 'errors': ['Steampunk house file not found']}

        print(f"🏠 Running steampunk house pipeline...")
        print(f"📁 Asset: {steampunk_house_path}")
        print(f"📊 Variations: {num_variations}")

        try:
            result = self.run_complete_pipeline(
                asset_path=steampunk_house_path,
                num_variations=num_variations,
                cultural_style='steampunk'
            )

            if result['success']:
                print("🎉 Steampunk house pipeline complete!")
                print("🏯 Your pagodas should now appear in Revit!")

            return result

        except Exception as e:
            print(f"❌ Steampunk house pipeline failed: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def create_pagoda_variations(self, num_pagodas: int = 5) -> Dict[str, Any]:
        """Create multiple pagoda variations in Revit"""

        print(f"🏯 Creating {num_pagodas} pagoda variations...")

        try:
            # Run steampunk house pipeline
            result = self.run_steampunk_house_pipeline(num_pagodas * 4)  # Generate more variations

            if not result['success']:
                return result

            # Select best pagodas
            all_variations = result['result'].optimization_result.all_variations
            sorted_variations = sorted(all_variations, key=lambda v: v.overall_score, reverse=True)
            best_pagodas = sorted_variations[:num_pagodas]

            print(f"🏆 Selected {len(best_pagodas)} best pagoda variations:")
            for i, pagoda in enumerate(best_pagodas):
                print(f"  {i+1}. {pagoda.name} - Score: {pagoda.overall_score:.2f}")

            # Create pagoda configurations for Revit
            pagoda_configs = []
            for i, pagoda in enumerate(best_pagodas):
                config = {
                    'id': f"pagoda_{i+1:02d}",
                    'name': pagoda.name,
                    'parameters': pagoda.parameters,
                    'scores': {
                        'aesthetic': pagoda.aesthetic_score,
                        'functionality': pagoda.functionality_score,
                        'cultural': pagoda.cultural_score,
                        'structural': pagoda.structural_score,
                        'overall': pagoda.overall_score
                    },
                    'position': {
                        'x': i * 50,  # Space pagodas 50 feet apart
                        'y': 0,
                        'z': 0
                    }
                }
                pagoda_configs.append(config)

            # Export pagoda configurations
            output_path = Path("outputs") / "pagoda_variations.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(pagoda_configs, f, indent=2)

            print(f"💾 Pagoda configurations exported to: {output_path}")
            print("🎯 Ready to import into Revit!")

            return {
                'success': True,
                'pagoda_configs': pagoda_configs,
                'output_path': str(output_path),
                'total_pagodas': len(pagoda_configs)
            }

        except Exception as e:
            print(f"❌ Pagoda creation failed: {str(e)}")
            return {'success': False, 'errors': [str(e)]}

    def get_help(self):
        """Show help information"""

        print("🤖 AI Asset-to-Revit Pipeline - Cursor Interface")
        print("=" * 50)
        print()
        print("Available functions:")
        print()
        print("📊 Analysis:")
        print("  analyze_asset(asset_path, cultural_style=None)")
        print("  - Analyze 3D asset and extract constraints")
        print()
        print("🔧 Dynamo Generation:")
        print("  generate_dynamo_graph(asset_path, cultural_style=None)")
        print("  - Generate Dynamo parametric graph")
        print()
        print("🎨 Design Exploration:")
        print("  explore_design_variations(asset_path, num_variations=20)")
        print("  - Generate and optimize design variations")
        print()
        print("🚀 Complete Pipeline:")
        print("  run_complete_pipeline(asset_path, num_variations=20)")
        print("  - Run all phases: analysis → Dynamo → optimization")
        print()
        print("🏠 Steampunk House Integration:")
        print("  run_steampunk_house_pipeline(num_variations=20)")
        print("  - Run pipeline on your steampunk house")
        print()
        print("🏯 Pagoda Creation:")
        print("  create_pagoda_variations(num_pagodas=5)")
        print("  - Create multiple pagoda variations for Revit")
        print()
        print("📋 Utilities:")
        print("  show_last_result() - Show last pipeline result")
        print("  list_available_styles() - List cultural styles")
        print("  get_help() - Show this help")
        print()
        print("Example usage:")
        print("  result = run_steampunk_house_pipeline(20)")
        print("  pagodas = create_pagoda_variations(5)")

# Global instance for easy use
pipeline = CursorPipelineInterface()

# Convenience functions
def analyze_asset(asset_path: str, cultural_style: Optional[str] = None):
    """Analyze 3D asset and extract constraints"""
    return pipeline.analyze_asset(asset_path, cultural_style)

def generate_dynamo_graph(asset_path: str, cultural_style: Optional[str] = None):
    """Generate Dynamo graph for asset"""
    return pipeline.generate_dynamo_graph(asset_path, cultural_style)

def explore_design_variations(asset_path: str, num_variations: int = 20, cultural_style: Optional[str] = None):
    """Explore design variations and optimization"""
    return pipeline.explore_design_variations(asset_path, num_variations, cultural_style)

def run_complete_pipeline(asset_path: str, num_variations: int = 20, cultural_style: Optional[str] = None):
    """Run complete pipeline with all phases"""
    return pipeline.run_complete_pipeline(asset_path, num_variations, cultural_style)

def show_last_result():
    """Show details of last pipeline result"""
    return pipeline.show_last_result()

def list_available_styles():
    """List available cultural styles"""
    return pipeline.list_available_styles()

def get_help():
    """Show help information"""
    return pipeline.get_help()

def run_steampunk_house_pipeline(num_variations: int = 20):
    """Run complete pipeline on the steampunk house"""
    return pipeline.run_steampunk_house_pipeline(num_variations)

def create_pagoda_variations(num_pagodas: int = 5):
    """Create multiple pagoda variations for Revit"""
    return pipeline.create_pagoda_variations(num_pagodas)

# Demo function
def demo_pipeline():
    """Demo the pipeline with the steampunk house"""

    print("🎬 AI Asset-to-Revit Pipeline Demo")
    print("=" * 40)

    # Use the steampunk house
    steampunk_path = r"C:\Users\Owner\Desktop\SteamPunk House\source\tutorial_steampunk_house_no_LM.fbx"

    if not Path(steampunk_path).exists():
        print(f"⚠️  Steampunk house not found: {steampunk_path}")
        print("Please check the file path")
        return

    print(f"📁 Using steampunk house: {steampunk_path}")

    # Run steampunk house pipeline
    result = run_steampunk_house_pipeline(10)

    if result['success']:
        print("\n🎉 Demo completed successfully!")
        print("🏯 Your pagodas should now appear in Revit!")
        print("Check the 'outputs/' directory for results")

        # Create pagoda variations
        pagodas = create_pagoda_variations(3)
        if pagodas['success']:
            print(f"🏯 Created {pagodas['total_pagodas']} pagoda variations!")
    else:
        print(f"\n❌ Demo failed: {result['errors']}")

def demo_steampunk_pagodas():
    """Demo creating steampunk pagodas"""

    print("🎬 Steampunk Pagoda Demo")
    print("=" * 40)

    # Create pagoda variations
    result = create_pagoda_variations(5)

    if result['success']:
        print(f"🎉 Created {result['total_pagodas']} steampunk pagodas!")
        print(f"📁 Configurations saved to: {result['output_path']}")
        print("🏯 Ready to import into Revit!")
    else:
        print(f"❌ Demo failed: {result['errors']}")

if __name__ == "__main__":
    # Show help by default
    get_help()

    # If arguments provided, run demo
    if len(sys.argv) > 1:
        demo_pipeline()
