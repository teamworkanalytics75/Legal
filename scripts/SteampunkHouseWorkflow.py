"""
AI Asset-to-Revit Pipeline - Steampunk Japanese House Workflow

Specialized workflow for analyzing and generating variations of steampunk Japanese houses.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from asset_analyzer.modular_component_analyzer import ModularComponentAnalyzer, ModularAsset
from dynamo_generator.graph_builder import DynamoGraphBuilder

logger = logging.getLogger(__name__)

class SteampunkJapaneseHouseWorkflow:
    """Complete workflow for steampunk Japanese house analysis and generation"""

    def __init__(self):
        self.component_analyzer = ModularComponentAnalyzer()
        self.graph_builder = DynamoGraphBuilder()

        # Define steampunk Japanese house specific configurations
        self.house_configurations = {
            'traditional_focus': {
                'name': 'Traditional Japanese House',
                'description': 'Minimal steampunk elements, maximum traditional Japanese aesthetics',
                'component_scales': {
                    'foundation': 1.0,
                    'walls': 1.0,
                    'roof': 1.2,
                    'doors': 1.0,
                    'windows': 1.0,
                    'japanese_traditional': 1.5,
                    'steampunk_mechanical': 0.3,
                    'decorative': 1.0
                },
                'material_theme': 'traditional',
                'roof_pitch': 35,
                'eave_length': 0.6,
                'beam_spacing': 2.0
            },
            'steampunk_focus': {
                'name': 'Steampunk Industrial House',
                'description': 'Maximum steampunk elements with Japanese base structure',
                'component_scales': {
                    'foundation': 1.0,
                    'walls': 1.0,
                    'roof': 0.8,
                    'doors': 1.2,
                    'windows': 1.0,
                    'japanese_traditional': 0.5,
                    'steampunk_mechanical': 2.0,
                    'decorative': 1.5
                },
                'material_theme': 'steampunk',
                'roof_pitch': 25,
                'eave_length': 0.3,
                'beam_spacing': 1.5
            },
            'balanced_hybrid': {
                'name': 'Steampunk Japanese Hybrid',
                'description': 'Balanced mix of traditional and steampunk elements',
                'component_scales': {
                    'foundation': 1.0,
                    'walls': 1.0,
                    'roof': 1.0,
                    'doors': 1.0,
                    'windows': 1.0,
                    'japanese_traditional': 1.0,
                    'steampunk_mechanical': 1.0,
                    'decorative': 1.0
                },
                'material_theme': 'hybrid',
                'roof_pitch': 30,
                'eave_length': 0.4,
                'beam_spacing': 1.8
            },
            'hokkaido_adapted': {
                'name': 'Hokkaido Steampunk House',
                'description': 'Adapted for Hokkaido climate with steampunk heating systems',
                'component_scales': {
                    'foundation': 1.2,  # Elevated for snow
                    'walls': 1.1,      # Thicker insulation
                    'roof': 1.3,       # Steeper for snow shedding
                    'doors': 1.0,
                    'windows': 0.8,    # Smaller for heat retention
                    'japanese_traditional': 0.8,
                    'steampunk_mechanical': 1.5,  # More heating systems
                    'decorative': 0.7
                },
                'material_theme': 'hokkaido',
                'roof_pitch': 40,      # Steeper for snow
                'eave_length': 0.2,    # Shorter eaves
                'beam_spacing': 2.2,   # Wider spacing for insulation
                'insulation_factor': 1.5,
                'heating_systems': True
            }
        }

    def analyze_steampunk_house(self, file_path: str) -> Dict[str, Any]:
        """Analyze steampunk Japanese house and extract components"""

        logger.info(f"Analyzing steampunk Japanese house: {file_path}")

        # Analyze modular components
        modular_asset = self.component_analyzer.analyze_modular_asset(file_path)

        # Create analysis report
        analysis_report = {
            'asset_name': modular_asset.name,
            'total_components': len(modular_asset.components),
            'component_types': list(modular_asset.component_hierarchy.keys()),
            'steampunk_elements': self._identify_steampunk_elements(modular_asset),
            'japanese_elements': self._identify_japanese_elements(modular_asset),
            'structural_analysis': self._analyze_structure(modular_asset),
            'material_analysis': self._analyze_materials(modular_asset),
            'parametric_potential': self._assess_parametric_potential(modular_asset)
        }

        return analysis_report

    def generate_house_variations(self, modular_asset: ModularAsset,
                                configuration: str = 'balanced_hybrid',
                                num_variations: int = 5) -> List[Dict[str, Any]]:
        """Generate parametric variations of the house"""

        logger.info(f"Generating {num_variations} variations with {configuration} configuration")

        if configuration not in self.house_configurations:
            configuration = 'balanced_hybrid'

        config = self.house_configurations[configuration]
        variations = []

        for i in range(num_variations):
            variation = self._create_variation(modular_asset, config, i)
            variations.append(variation)

        return variations

    def create_dynamo_graph(self, modular_asset: ModularAsset,
                          configuration: str = 'balanced_hybrid') -> str:
        """Create Dynamo graph for parametric house generation"""

        logger.info(f"Creating Dynamo graph for {configuration} configuration")

        # Create specialized graph for steampunk Japanese house
        graph_content = self._create_steampunk_house_graph(modular_asset, configuration)

        return graph_content

    def _identify_steampunk_elements(self, modular_asset: ModularAsset) -> List[str]:
        """Identify steampunk elements in the house"""

        steampunk_elements = []

        for component in modular_asset.components:
            if component.component_type == 'steampunk_mechanical':
                steampunk_elements.append(component.name)

        return steampunk_elements

    def _identify_japanese_elements(self, modular_asset: ModularAsset) -> List[str]:
        """Identify Japanese elements in the house"""

        japanese_elements = []

        for component in modular_asset.components:
            if component.component_type == 'japanese_traditional':
                japanese_elements.append(component.name)

        return japanese_elements

    def _analyze_structure(self, modular_asset: ModularAsset) -> Dict[str, Any]:
        """Analyze structural characteristics"""

        structure_analysis = {
            'foundation_components': [],
            'wall_components': [],
            'roof_components': [],
            'structural_support': [],
            'total_height': 0,
            'total_width': 0,
            'total_depth': 0
        }

        for component in modular_asset.components:
            if component.component_type == 'foundation':
                structure_analysis['foundation_components'].append(component.name)
            elif component.component_type == 'walls':
                structure_analysis['wall_components'].append(component.name)
            elif component.component_type == 'roof':
                structure_analysis['roof_components'].append(component.name)
            elif component.component_type == 'structural':
                structure_analysis['structural_support'].append(component.name)

            # Calculate overall dimensions
            structure_analysis['total_height'] = max(
                structure_analysis['total_height'],
                component.position[2] + component.bounding_box['z']
            )
            structure_analysis['total_width'] = max(
                structure_analysis['total_width'],
                component.position[0] + component.bounding_box['x']
            )
            structure_analysis['total_depth'] = max(
                structure_analysis['total_depth'],
                component.position[1] + component.bounding_box['y']
            )

        return structure_analysis

    def _analyze_materials(self, modular_asset: ModularAsset) -> Dict[str, Any]:
        """Analyze material usage"""

        material_analysis = {
            'traditional_materials': [],
            'steampunk_materials': [],
            'hybrid_materials': [],
            'material_distribution': {}
        }

        for component in modular_asset.components:
            for material in component.materials:
                if material in ['Wood', 'Tile', 'Paper', 'Bamboo']:
                    if material not in material_analysis['traditional_materials']:
                        material_analysis['traditional_materials'].append(material)
                elif material in ['Brass', 'Copper', 'Steel', 'Iron']:
                    if material not in material_analysis['steampunk_materials']:
                        material_analysis['steampunk_materials'].append(material)
                else:
                    if material not in material_analysis['hybrid_materials']:
                        material_analysis['hybrid_materials'].append(material)

                # Count material usage
                material_analysis['material_distribution'][material] = \
                    material_analysis['material_distribution'].get(material, 0) + 1

        return material_analysis

    def _assess_parametric_potential(self, modular_asset: ModularAsset) -> Dict[str, Any]:
        """Assess parametric design potential"""

        potential_analysis = {
            'highly_parametric': [],
            'moderately_parametric': [],
            'fixed_elements': [],
            'parametric_score': 0
        }

        total_components = len(modular_asset.components)
        parametric_count = 0

        for component in modular_asset.components:
            if len(component.parametric_variables) > 3:
                potential_analysis['highly_parametric'].append(component.name)
                parametric_count += 1
            elif len(component.parametric_variables) > 1:
                potential_analysis['moderately_parametric'].append(component.name)
                parametric_count += 0.5
            else:
                potential_analysis['fixed_elements'].append(component.name)

        potential_analysis['parametric_score'] = (parametric_count / total_components) * 100

        return potential_analysis

    def _create_variation(self, modular_asset: ModularAsset,
                        config: Dict[str, Any], variation_index: int) -> Dict[str, Any]:
        """Create a specific variation of the house"""

        variation = {
            'variation_id': variation_index,
            'name': f"{config['name']} Variation {variation_index + 1}",
            'description': config['description'],
            'component_modifications': {},
            'material_overrides': {},
            'structural_changes': {},
            'aesthetic_score': 0,
            'functionality_score': 0,
            'cultural_authenticity_score': 0
        }

        # Apply component modifications
        for comp_type, scale in config['component_scales'].items():
            variation['component_modifications'][comp_type] = {
                'scale_factor': scale,
                'position_offset': [0, 0, 0],
                'rotation_offset': [0, 0, 0],
                'material_override': None
            }

        # Apply material overrides
        material_theme = config['material_theme']
        if material_theme == 'traditional':
            variation['material_overrides'] = {
                'primary': 'Wood',
                'secondary': 'Tile',
                'accent': 'Bamboo'
            }
        elif material_theme == 'steampunk':
            variation['material_overrides'] = {
                'primary': 'Brass',
                'secondary': 'Steel',
                'accent': 'Copper'
            }
        elif material_theme == 'hybrid':
            variation['material_overrides'] = {
                'primary': 'Wood',
                'secondary': 'Brass',
                'accent': 'Copper'
            }
        elif material_theme == 'hokkaido':
            variation['material_overrides'] = {
                'primary': 'Insulated Wood',
                'secondary': 'Steel',
                'accent': 'Copper',
                'insulation': 'High'
            }

        # Apply structural changes
        variation['structural_changes'] = {
            'roof_pitch': config.get('roof_pitch', 30),
            'eave_length': config.get('eave_length', 0.4),
            'beam_spacing': config.get('beam_spacing', 2.0),
            'insulation_factor': config.get('insulation_factor', 1.0),
            'heating_systems': config.get('heating_systems', False)
        }

        # Calculate scores
        variation['aesthetic_score'] = self._calculate_aesthetic_score(variation, config)
        variation['functionality_score'] = self._calculate_functionality_score(variation, config)
        variation['cultural_authenticity_score'] = self._calculate_cultural_score(variation, config)

        return variation

    def _calculate_aesthetic_score(self, variation: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Calculate aesthetic score for variation"""

        score = 50  # Base score

        # Material harmony bonus
        material_theme = config['material_theme']
        if material_theme == 'hybrid':
            score += 20  # Hybrid is most aesthetically interesting
        elif material_theme == 'traditional':
            score += 15
        elif material_theme == 'steampunk':
            score += 10

        # Component balance bonus
        scales = config['component_scales']
        balance_score = 100 - abs(scales.get('steampunk_mechanical', 1.0) - scales.get('japanese_traditional', 1.0)) * 20
        score += balance_score * 0.3

        return min(100, max(0, score))

    def _calculate_functionality_score(self, variation: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Calculate functionality score for variation"""

        score = 60  # Base score

        # Structural soundness
        roof_pitch = variation['structural_changes']['roof_pitch']
        if 25 <= roof_pitch <= 45:
            score += 20  # Good roof pitch range

        # Insulation factor
        insulation = variation['structural_changes'].get('insulation_factor', 1.0)
        score += insulation * 10

        # Heating systems
        if variation['structural_changes'].get('heating_systems', False):
            score += 15

        return min(100, max(0, score))

    def _calculate_cultural_score(self, variation: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Calculate cultural authenticity score"""

        score = 40  # Base score

        # Traditional elements bonus
        japanese_scale = config['component_scales'].get('japanese_traditional', 1.0)
        score += japanese_scale * 20

        # Steampunk elements bonus
        steampunk_scale = config['component_scales'].get('steampunk_mechanical', 1.0)
        score += steampunk_scale * 15

        # Material authenticity
        material_theme = config['material_theme']
        if material_theme == 'traditional':
            score += 25
        elif material_theme == 'hybrid':
            score += 15

        return min(100, max(0, score))

    def _create_steampunk_house_graph(self, modular_asset: ModularAsset,
                                    configuration: str) -> str:
        """Create specialized Dynamo graph for steampunk Japanese house"""

        # This would create a more sophisticated graph
        # For now, return a basic graph structure

        graph_content = f"""
# Steampunk Japanese House - {configuration} Configuration
# Generated Dynamo Graph

# Input Parameters
Configuration = "{configuration}"
MaterialTheme = "{self.house_configurations[configuration]['material_theme']}"
RoofPitch = {self.house_configurations[configuration]['roof_pitch']}
EaveLength = {self.house_configurations[configuration]['eave_length']}

# Component Scales
ComponentScales = {json.dumps(self.house_configurations[configuration]['component_scales'])}

# Assembly Logic
# This graph will generate parametric variations of the steampunk Japanese house
# with the specified configuration parameters.
"""

        return graph_content

def run_steampunk_house_workflow(file_path: str, output_dir: str = "outputs"):
    """Run complete workflow for steampunk Japanese house"""

    workflow = SteampunkJapaneseHouseWorkflow()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Step 1: Analyze the house
    print("Step 1: Analyzing steampunk Japanese house...")
    analysis_report = workflow.analyze_steampunk_house(file_path)

    # Save analysis report
    analysis_file = output_path / f"{Path(file_path).stem}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")

    # Step 2: Generate variations
    print("Step 2: Generating house variations...")

    # Create mock modular asset for demonstration
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
        )
    ]

    mock_asset = ModularAsset(
        name='SteampunkHouse',
        components=mock_components,
        component_hierarchy={'foundation': ['Foundation'], 'steampunk_mechanical': ['Steam_Pipe']},
        connection_rules={},
        assembly_constraints={},
        parametric_configurations=[]
    )

    # Generate variations for each configuration
    all_variations = []
    for config_name in workflow.house_configurations.keys():
        variations = workflow.generate_house_variations(mock_asset, config_name, 3)
        all_variations.extend(variations)

    # Save variations
    variations_file = output_path / f"{Path(file_path).stem}_variations.json"
    with open(variations_file, 'w') as f:
        json.dump(all_variations, f, indent=2)
    print(f"Variations saved to: {variations_file}")

    # Step 3: Create Dynamo graphs
    print("Step 3: Creating Dynamo graphs...")

    for config_name in workflow.house_configurations.keys():
        graph_content = workflow.create_dynamo_graph(mock_asset, config_name)
        graph_file = output_path / f"{Path(file_path).stem}_{config_name}.dyn"
        with open(graph_file, 'w') as f:
            f.write(graph_content)
        print(f"Dynamo graph saved to: {graph_file}")

    print("Workflow complete!")
    return {
        'analysis_file': analysis_file,
        'variations_file': variations_file,
        'dynamo_graphs': list(output_path.glob("*.dyn"))
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

        try:
            results = run_steampunk_house_workflow(file_path, output_dir)
            print(f"Workflow completed successfully!")
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error running workflow: {str(e)}")
    else:
        print("Usage: python steampunk_workflow.py <steampunk_house_file> [output_dir]")
        print("\nExample:")
        print("python steampunk_workflow.py 'C:\\Users\\Owner\\Desktop\\SteamPunk House\\house.fbx'")
