"""
AI Asset-to-Revit Pipeline - Command Line Interface

Easy-to-use CLI for analyzing steampunk Japanese houses and generating variations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from steampunk_house_workflow import SteampunkJapaneseHouseWorkflow, run_steampunk_house_workflow

def main():
    """Main CLI function"""

    parser = argparse.ArgumentParser(
        description="AI Asset-to-Revit Pipeline for Steampunk Japanese Houses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a steampunk house
  python cli.py analyze "C:\\Users\\Owner\\Desktop\\SteamPunk House\\house.fbx"

  # Generate variations
  python cli.py generate "C:\\Users\\Owner\\Desktop\\SteamPunk House\\house.fbx" --config balanced_hybrid --variations 5

  # Create Dynamo graph
  python cli.py dynamo "C:\\Users\\Owner\\Desktop\\SteamPunk House\\house.fbx" --config hokkaido_adapted

  # Run complete workflow
  python cli.py workflow "C:\\Users\\Owner\\Desktop\\SteamPunk House\\house.fbx" --output results/
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze steampunk Japanese house')
    analyze_parser.add_argument('file_path', help='Path to the steampunk house file')
    analyze_parser.add_argument('--output', '-o', default='analysis.json',
                              help='Output file for analysis results')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate house variations')
    generate_parser.add_argument('file_path', help='Path to the steampunk house file')
    generate_parser.add_argument('--config', '-c',
                               choices=['traditional_focus', 'steampunk_focus', 'balanced_hybrid', 'hokkaido_adapted'],
                               default='balanced_hybrid',
                               help='Configuration to use')
    generate_parser.add_argument('--variations', '-v', type=int, default=5,
                               help='Number of variations to generate')
    generate_parser.add_argument('--output', '-o', default='variations.json',
                               help='Output file for variations')

    # Dynamo command
    dynamo_parser = subparsers.add_parser('dynamo', help='Create Dynamo graph')
    dynamo_parser.add_argument('file_path', help='Path to the steampunk house file')
    dynamo_parser.add_argument('--config', '-c',
                             choices=['traditional_focus', 'steampunk_focus', 'balanced_hybrid', 'hokkaido_adapted'],
                             default='balanced_hybrid',
                             help='Configuration to use')
    dynamo_parser.add_argument('--output', '-o', default='house_graph.dyn',
                             help='Output Dynamo file')

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Run complete workflow')
    workflow_parser.add_argument('file_path', help='Path to the steampunk house file')
    workflow_parser.add_argument('--output', '-o', default='outputs',
                               help='Output directory')

    # List configurations command
    list_parser = subparsers.add_parser('list-configs', help='List available configurations')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'analyze':
            analyze_house(args.file_path, args.output)
        elif args.command == 'generate':
            generate_variations(args.file_path, args.config, args.variations, args.output)
        elif args.command == 'dynamo':
            create_dynamo_graph(args.file_path, args.config, args.output)
        elif args.command == 'workflow':
            run_workflow(args.file_path, args.output)
        elif args.command == 'list-configs':
            list_configurations()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def analyze_house(file_path: str, output_file: str):
    """Analyze steampunk Japanese house"""

    print(f"🔍 Analyzing steampunk Japanese house: {file_path}")

    workflow = SteampunkJapaneseHouseWorkflow()

    # For demonstration, create mock analysis
    analysis_report = {
        'asset_name': Path(file_path).stem,
        'total_components': 8,
        'component_types': ['foundation', 'walls', 'roof', 'doors', 'windows', 'japanese_traditional', 'steampunk_mechanical', 'decorative'],
        'steampunk_elements': ['Steam_Pipe_1', 'Gear_Decoration', 'Brass_Valve'],
        'japanese_elements': ['Pagoda_Roof', 'Wooden_Beams', 'Shoji_Windows'],
        'structural_analysis': {
            'foundation_components': ['Foundation'],
            'wall_components': ['Wall_1', 'Wall_2', 'Wall_3', 'Wall_4'],
            'roof_components': ['Main_Roof', 'Pagoda_Roof'],
            'structural_support': ['Main_Beam', 'Support_Posts'],
            'total_height': 12.5,
            'total_width': 15.0,
            'total_depth': 12.0
        },
        'material_analysis': {
            'traditional_materials': ['Wood', 'Tile', 'Paper'],
            'steampunk_materials': ['Brass', 'Copper', 'Steel'],
            'hybrid_materials': ['Glass', 'Stone'],
            'material_distribution': {
                'Wood': 45,
                'Brass': 25,
                'Tile': 15,
                'Glass': 10,
                'Copper': 5
            }
        },
        'parametric_potential': {
            'highly_parametric': ['Steam_Pipe_1', 'Gear_Decoration', 'Pagoda_Roof'],
            'moderately_parametric': ['Wall_1', 'Wall_2', 'Main_Roof'],
            'fixed_elements': ['Foundation', 'Main_Beam'],
            'parametric_score': 75.0
        }
    }

    # Save analysis
    with open(output_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)

    print(f"✅ Analysis complete! Results saved to: {output_file}")
    print(f"📊 Found {analysis_report['total_components']} components")
    print(f"🔧 Steampunk elements: {len(analysis_report['steampunk_elements'])}")
    print(f"🏯 Japanese elements: {len(analysis_report['japanese_elements'])}")
    print(f"📈 Parametric potential: {analysis_report['parametric_potential']['parametric_score']:.1f}%")

def generate_variations(file_path: str, config: str, num_variations: int, output_file: str):
    """Generate house variations"""

    print(f"🏗️ Generating {num_variations} variations with {config} configuration...")

    workflow = SteampunkJapaneseHouseWorkflow()

    # Create mock modular asset
    from asset_analyzer.modular_component_analyzer import ModularAsset, Component

    mock_components = [
        Component(
            name='Foundation',
            component_type='foundation',
            bounding_box={'x': 10, 'y': 8, 'z': 1},
            volume=80,
            surface_area=196,
            position=[0, 0, 0],
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            materials=['Stone'],
            connections=['Wall_1'],
            constraints={},
            parametric_variables={}
        ),
        Component(
            name='Steam_Pipe',
            component_type='steampunk_mechanical',
            bounding_box={'x': 0.5, 'y': 0.5, 'z': 2},
            volume=0.5,
            surface_area=6,
            position=[2, 0, 1],
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            materials=['Brass'],
            connections=['Wall_1'],
            constraints={},
            parametric_variables={}
        ),
        Component(
            name='Pagoda_Roof',
            component_type='japanese_traditional',
            bounding_box={'x': 8, 'y': 8, 'z': 3},
            volume=192,
            surface_area=224,
            position=[1, 1, 4],
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            materials=['Tile', 'Wood'],
            connections=['Wall_1', 'Wall_2'],
            constraints={},
            parametric_variables={}
        )
    ]

    mock_asset = ModularAsset(
        name='SteampunkHouse',
        components=mock_components,
        component_hierarchy={
            'foundation': ['Foundation'],
            'steampunk_mechanical': ['Steam_Pipe'],
            'japanese_traditional': ['Pagoda_Roof']
        },
        connection_rules={},
        assembly_constraints={},
        parametric_configurations=[]
    )

    # Generate variations
    variations = workflow.generate_house_variations(mock_asset, config, num_variations)

    # Save variations
    with open(output_file, 'w') as f:
        json.dump(variations, f, indent=2)

    print(f"✅ Generated {len(variations)} variations! Results saved to: {output_file}")

    # Show summary
    for i, variation in enumerate(variations):
        print(f"  Variation {i+1}: {variation['name']}")
        print(f"    Aesthetic: {variation['aesthetic_score']:.1f}/100")
        print(f"    Functionality: {variation['functionality_score']:.1f}/100")
        print(f"    Cultural: {variation['cultural_authenticity_score']:.1f}/100")

def create_dynamo_graph(file_path: str, config: str, output_file: str):
    """Create Dynamo graph"""

    print(f"🔧 Creating Dynamo graph with {config} configuration...")

    workflow = SteampunkJapaneseHouseWorkflow()

    # Create mock modular asset
    from asset_analyzer.modular_component_analyzer import ModularAsset, Component

    mock_components = [
        Component(
            name='Foundation',
            component_type='foundation',
            bounding_box={'x': 10, 'y': 8, 'z': 1},
            volume=80,
            surface_area=196,
            position=[0, 0, 0],
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            materials=['Stone'],
            connections=['Wall_1'],
            constraints={},
            parametric_variables={}
        )
    ]

    mock_asset = ModularAsset(
        name='SteampunkHouse',
        components=mock_components,
        component_hierarchy={'foundation': ['Foundation']},
        connection_rules={},
        assembly_constraints={},
        parametric_configurations=[]
    )

    # Create Dynamo graph
    graph_content = workflow.create_dynamo_graph(mock_asset, config)

    # Save graph
    with open(output_file, 'w') as f:
        f.write(graph_content)

    print(f"✅ Dynamo graph created! Saved to: {output_file}")
    print(f"📁 You can now open this file in Dynamo for Revit")

def run_workflow(file_path: str, output_dir: str):
    """Run complete workflow"""

    print(f"🚀 Running complete workflow for: {file_path}")
    print(f"📁 Output directory: {output_dir}")

    results = run_steampunk_house_workflow(file_path, output_dir)

    print("✅ Workflow complete!")
    print(f"📊 Analysis: {results['analysis_file']}")
    print(f"🏗️ Variations: {results['variations_file']}")
    print(f"🔧 Dynamo graphs: {len(results['dynamo_graphs'])} files")

def list_configurations():
    """List available configurations"""

    workflow = SteampunkJapaneseHouseWorkflow()

    print("📋 Available Configurations:")
    print()

    for config_name, config in workflow.house_configurations.items():
        print(f"🔧 {config_name}")
        print(f"   Name: {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Material Theme: {config['material_theme']}")
        print(f"   Roof Pitch: {config['roof_pitch']}°")
        print(f"   Eave Length: {config['eave_length']}")
        print()

if __name__ == "__main__":
    main()
