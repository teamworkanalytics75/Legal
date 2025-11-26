"""
Generate Revit-ready files for pagoda rendering
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import time

def create_revit_pagoda_families():
    """Create Revit family definitions for pagodas"""

    # Create families directory
    families_dir = Path("outputs/revit_families")
    families_dir.mkdir(parents=True, exist_ok=True)

    # Pagoda family definitions
    pagoda_families = {
        "Steampunk_Pagoda": {
            "category": "Generic Models",
            "family_name": "Steampunk_Pagoda",
            "parameters": [
                {"name": "Height", "type": "Length", "value": 20.0, "units": "feet"},
                {"name": "Width", "type": "Length", "value": 8.0, "units": "feet"},
                {"name": "Depth", "type": "Length", "value": 8.0, "units": "feet"},
                {"name": "Roof_Pitch", "type": "Angle", "value": 30.0, "units": "degrees"},
                {"name": "Tier_Count", "type": "Integer", "value": 4},
                {"name": "Gear_Density", "type": "Number", "value": 0.5},
                {"name": "Pipe_Complexity", "type": "Number", "value": 0.7},
                {"name": "Primary_Material", "type": "Material", "value": "Metal - Brass"},
                {"name": "Secondary_Material", "type": "Material", "value": "Metal - Copper"},
                {"name": "Roof_Material", "type": "Material", "value": "Metal - Iron"}
            ],
            "geometry": {
                "base_shape": "rectangular",
                "tiered": True,
                "steampunk_elements": ["gears", "pipes", "valves", "brass_fittings"]
            }
        },
        "Japanese_Pagoda": {
            "category": "Generic Models",
            "family_name": "Japanese_Pagoda",
            "parameters": [
                {"name": "Height", "type": "Length", "value": 18.0, "units": "feet"},
                {"name": "Width", "type": "Length", "value": 6.0, "units": "feet"},
                {"name": "Depth", "type": "Length", "value": 6.0, "units": "feet"},
                {"name": "Roof_Pitch", "type": "Angle", "value": 25.0, "units": "degrees"},
                {"name": "Tier_Count", "type": "Integer", "value": 3},
                {"name": "Primary_Material", "type": "Material", "value": "Wood - Oak"},
                {"name": "Roof_Material", "type": "Material", "value": "Wood - Bamboo"},
                {"name": "Base_Material", "type": "Material", "value": "Stone - Granite"}
            ],
            "geometry": {
                "base_shape": "rectangular",
                "tiered": True,
                "japanese_elements": ["curved_roofs", "wooden_beams", "stone_base"]
            }
        },
        "Hokkaido_Pagoda": {
            "category": "Generic Models",
            "family_name": "Hokkaido_Pagoda",
            "parameters": [
                {"name": "Height", "type": "Length", "value": 16.0, "units": "feet"},
                {"name": "Width", "type": "Length", "value": 10.0, "units": "feet"},
                {"name": "Depth", "type": "Length", "value": 10.0, "units": "feet"},
                {"name": "Roof_Pitch", "type": "Angle", "value": 40.0, "units": "degrees"},
                {"name": "Tier_Count", "type": "Integer", "value": 3},
                {"name": "Insulation_Factor", "type": "Number", "value": 1.5},
                {"name": "Snow_Shedding", "type": "Yes/No", "value": True},
                {"name": "Primary_Material", "type": "Material", "value": "Wood - Insulated"},
                {"name": "Roof_Material", "type": "Material", "value": "Metal - Steel"},
                {"name": "Base_Material", "type": "Material", "value": "Concrete - Cast-in-Place"}
            ],
            "geometry": {
                "base_shape": "rectangular",
                "tiered": True,
                "hokkaido_elements": ["steep_roofs", "insulation", "snow_shedding", "heating_systems"]
            }
        }
    }

    # Save family definitions
    for family_name, family_data in pagoda_families.items():
        family_file = families_dir / f"{family_name}.json"
        with open(family_file, 'w') as f:
            json.dump(family_data, f, indent=2)

    print(f"✅ Created {len(pagoda_families)} Revit family definitions")
    return pagoda_families

def create_pagoda_variations_csv():
    """Create CSV file with pagoda variations for Revit import"""

    # Create variations directory
    variations_dir = Path("outputs/revit_variations")
    variations_dir.mkdir(parents=True, exist_ok=True)

    # Generate pagoda variations
    variations = []

    # Steampunk variations
    for i in range(5):
        variations.append({
            "ID": f"STEAM_{i+1:02d}",
            "Name": f"Steampunk Pagoda {i+1}",
            "Family": "Steampunk_Pagoda",
            "Height": 20.0 + (i * 2.0),
            "Width": 8.0 + (i * 0.5),
            "Depth": 8.0 + (i * 0.5),
            "Roof_Pitch": 30.0 + (i * 2.0),
            "Tier_Count": 4 + (i % 2),
            "Gear_Density": 0.3 + (i * 0.1),
            "Pipe_Complexity": 0.5 + (i * 0.1),
            "Primary_Material": "Metal - Brass",
            "Secondary_Material": "Metal - Copper",
            "Roof_Material": "Metal - Iron",
            "Position_X": i * 50.0,
            "Position_Y": 0.0,
            "Position_Z": 0.0,
            "Rotation": i * 15.0,
            "Aesthetic_Score": 85.0 + (i * 2.0),
            "Functionality_Score": 80.0 + (i * 3.0),
            "Cultural_Score": 90.0 + (i * 1.0),
            "Structural_Score": 88.0 + (i * 2.0),
            "Overall_Score": 85.75 + (i * 2.0)
        })

    # Japanese variations
    for i in range(3):
        variations.append({
            "ID": f"JAPAN_{i+1:02d}",
            "Name": f"Japanese Pagoda {i+1}",
            "Family": "Japanese_Pagoda",
            "Height": 18.0 + (i * 1.5),
            "Width": 6.0 + (i * 0.3),
            "Depth": 6.0 + (i * 0.3),
            "Roof_Pitch": 25.0 + (i * 1.0),
            "Tier_Count": 3 + (i % 2),
            "Primary_Material": "Wood - Oak",
            "Roof_Material": "Wood - Bamboo",
            "Base_Material": "Stone - Granite",
            "Position_X": (i + 5) * 50.0,
            "Position_Y": 0.0,
            "Position_Z": 0.0,
            "Rotation": i * 20.0,
            "Aesthetic_Score": 92.0 + (i * 1.0),
            "Functionality_Score": 85.0 + (i * 2.0),
            "Cultural_Score": 95.0 + (i * 0.5),
            "Structural_Score": 90.0 + (i * 1.0),
            "Overall_Score": 90.5 + (i * 1.125)
        })

    # Hokkaido variations
    for i in range(2):
        variations.append({
            "ID": f"HOKKA_{i+1:02d}",
            "Name": f"Hokkaido Pagoda {i+1}",
            "Family": "Hokkaido_Pagoda",
            "Height": 16.0 + (i * 2.0),
            "Width": 10.0 + (i * 1.0),
            "Depth": 10.0 + (i * 1.0),
            "Roof_Pitch": 40.0 + (i * 2.0),
            "Tier_Count": 3,
            "Insulation_Factor": 1.5 + (i * 0.2),
            "Snow_Shedding": True,
            "Primary_Material": "Wood - Insulated",
            "Roof_Material": "Metal - Steel",
            "Base_Material": "Concrete - Cast-in-Place",
            "Position_X": (i + 8) * 50.0,
            "Position_Y": 0.0,
            "Position_Z": 0.0,
            "Rotation": i * 30.0,
            "Aesthetic_Score": 88.0 + (i * 2.0),
            "Functionality_Score": 92.0 + (i * 1.0),
            "Cultural_Score": 87.0 + (i * 2.0),
            "Structural_Score": 94.0 + (i * 1.0),
            "Overall_Score": 90.25 + (i * 1.5)
        })

    # Save to CSV
    csv_file = variations_dir / "pagoda_variations.csv"
    with open(csv_file, 'w', newline='') as f:
        if variations:
            writer = csv.DictWriter(f, fieldnames=variations[0].keys())
            writer.writeheader()
            writer.writerows(variations)

    print(f"✅ Created CSV with {len(variations)} pagoda variations")
    return variations

def create_revit_materials():
    """Create Revit material definitions"""

    materials_dir = Path("outputs/revit_materials")
    materials_dir.mkdir(parents=True, exist_ok=True)

    materials = {
        "Steampunk_Materials": {
            "Metal - Brass": {
                "type": "Metal",
                "color": [0.8, 0.6, 0.2],
                "roughness": 0.3,
                "metallic": 0.9,
                "description": "Brass for steampunk gears and fittings"
            },
            "Metal - Copper": {
                "type": "Metal",
                "color": [0.7, 0.4, 0.2],
                "roughness": 0.4,
                "metallic": 0.8,
                "description": "Copper for steampunk pipes and valves"
            },
            "Metal - Iron": {
                "type": "Metal",
                "color": [0.4, 0.4, 0.4],
                "roughness": 0.6,
                "metallic": 0.7,
                "description": "Iron for steampunk structural elements"
            }
        },
        "Japanese_Materials": {
            "Wood - Oak": {
                "type": "Wood",
                "color": [0.6, 0.4, 0.2],
                "roughness": 0.7,
                "metallic": 0.0,
                "description": "Traditional Japanese oak wood"
            },
            "Wood - Bamboo": {
                "type": "Wood",
                "color": [0.7, 0.6, 0.3],
                "roughness": 0.5,
                "metallic": 0.0,
                "description": "Bamboo for Japanese roof elements"
            },
            "Stone - Granite": {
                "type": "Stone",
                "color": [0.5, 0.5, 0.5],
                "roughness": 0.8,
                "metallic": 0.0,
                "description": "Granite for Japanese pagoda base"
            }
        },
        "Hokkaido_Materials": {
            "Wood - Insulated": {
                "type": "Wood",
                "color": [0.5, 0.4, 0.3],
                "roughness": 0.6,
                "metallic": 0.0,
                "description": "Insulated wood for Hokkaido climate"
            },
            "Metal - Steel": {
                "type": "Metal",
                "color": [0.6, 0.6, 0.6],
                "roughness": 0.4,
                "metallic": 0.8,
                "description": "Steel for Hokkaido snow-resistant roofs"
            },
            "Concrete - Cast-in-Place": {
                "type": "Concrete",
                "color": [0.7, 0.7, 0.7],
                "roughness": 0.9,
                "metallic": 0.0,
                "description": "Concrete for Hokkaido foundation"
            }
        }
    }

    # Save materials
    for category, material_set in materials.items():
        material_file = materials_dir / f"{category.lower().replace(' ', '_')}.json"
        with open(material_file, 'w') as f:
            json.dump(material_set, f, indent=2)

    print(f"✅ Created {len(materials)} material categories")
    return materials

def create_dynamo_graphs():
    """Create Dynamo graph files for Revit"""

    dynamo_dir = Path("outputs/dynamo_graphs")
    dynamo_dir.mkdir(parents=True, exist_ok=True)

    # Create simplified Dynamo graph structure
    dynamo_graphs = {
        "Steampunk_Pagoda_Generator": {
            "name": "Steampunk Pagoda Generator",
            "description": "Parametric steampunk pagoda generator",
            "nodes": [
                {"id": "height_slider", "type": "Number Slider", "x": 100, "y": 100, "value": 20.0},
                {"id": "width_slider", "type": "Number Slider", "x": 100, "y": 200, "value": 8.0},
                {"id": "depth_slider", "type": "Number Slider", "x": 100, "y": 300, "value": 8.0},
                {"id": "roof_pitch_slider", "type": "Number Slider", "x": 100, "y": 400, "value": 30.0},
                {"id": "tier_count_slider", "type": "Integer Slider", "x": 100, "y": 500, "value": 4},
                {"id": "gear_density_slider", "type": "Number Slider", "x": 100, "y": 600, "value": 0.5},
                {"id": "pipe_complexity_slider", "type": "Number Slider", "x": 100, "y": 700, "value": 0.7},
                {"id": "pagoda_generator", "type": "Code Block", "x": 400, "y": 400, "code": "// Steampunk pagoda generation"}
            ],
            "connections": [
                {"from": "height_slider", "to": "pagoda_generator", "port": "height"},
                {"from": "width_slider", "to": "pagoda_generator", "port": "width"},
                {"from": "depth_slider", "to": "pagoda_generator", "port": "depth"},
                {"from": "roof_pitch_slider", "to": "pagoda_generator", "port": "roof_pitch"},
                {"from": "tier_count_slider", "to": "pagoda_generator", "port": "tier_count"},
                {"from": "gear_density_slider", "to": "pagoda_generator", "port": "gear_density"},
                {"from": "pipe_complexity_slider", "to": "pagoda_generator", "port": "pipe_complexity"}
            ]
        },
        "Japanese_Pagoda_Generator": {
            "name": "Japanese Pagoda Generator",
            "description": "Parametric Japanese pagoda generator",
            "nodes": [
                {"id": "height_slider", "type": "Number Slider", "x": 100, "y": 100, "value": 18.0},
                {"id": "width_slider", "type": "Number Slider", "x": 100, "y": 200, "value": 6.0},
                {"id": "depth_slider", "type": "Number Slider", "x": 100, "y": 300, "value": 6.0},
                {"id": "roof_pitch_slider", "type": "Number Slider", "x": 100, "y": 400, "value": 25.0},
                {"id": "tier_count_slider", "type": "Integer Slider", "x": 100, "y": 500, "value": 3},
                {"id": "pagoda_generator", "type": "Code Block", "x": 400, "y": 300, "code": "// Japanese pagoda generation"}
            ],
            "connections": [
                {"from": "height_slider", "to": "pagoda_generator", "port": "height"},
                {"from": "width_slider", "to": "pagoda_generator", "port": "width"},
                {"from": "depth_slider", "to": "pagoda_generator", "port": "depth"},
                {"from": "roof_pitch_slider", "to": "pagoda_generator", "port": "roof_pitch"},
                {"from": "tier_count_slider", "to": "pagoda_generator", "port": "tier_count"}
            ]
        }
    }

    # Save Dynamo graphs
    for graph_name, graph_data in dynamo_graphs.items():
        graph_file = dynamo_dir / f"{graph_name}.json"
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)

    print(f"✅ Created {len(dynamo_graphs)} Dynamo graph definitions")
    return dynamo_graphs

def create_revit_import_instructions():
    """Create instructions for importing into Revit"""

    instructions_dir = Path("outputs/revit_instructions")
    instructions_dir.mkdir(parents=True, exist_ok=True)

    instructions = {
        "import_steps": [
            {
                "step": 1,
                "title": "Load Pagoda Families",
                "description": "Load the pagoda family files into Revit",
                "files": [
                    "revit_families/Steampunk_Pagoda.json",
                    "revit_families/Japanese_Pagoda.json",
                    "revit_families/Hokkaido_Pagoda.json"
                ],
                "action": "Use Revit Family Editor to create families from JSON definitions"
            },
            {
                "step": 2,
                "title": "Import Materials",
                "description": "Import material definitions",
                "files": [
                    "revit_materials/steampunk_materials.json",
                    "revit_materials/japanese_materials.json",
                    "revit_materials/hokkaido_materials.json"
                ],
                "action": "Create materials in Revit Material Browser using JSON definitions"
            },
            {
                "step": 3,
                "title": "Import Pagoda Variations",
                "description": "Import pagoda instances from CSV",
                "files": [
                    "revit_variations/pagoda_variations.csv"
                ],
                "action": "Use Revit Data Import to place pagoda instances at specified coordinates"
            },
            {
                "step": 4,
                "title": "Load Dynamo Graphs",
                "description": "Load parametric graphs for modification",
                "files": [
                    "dynamo_graphs/Steampunk_Pagoda_Generator.json",
                    "dynamo_graphs/Japanese_Pagoda_Generator.json"
                ],
                "action": "Import JSON definitions into Dynamo to create parametric graphs"
            }
        ],
        "coordinate_system": {
            "origin": [0, 0, 0],
            "units": "feet",
            "spacing": "50 feet between pagodas",
            "layout": "Linear arrangement along X-axis"
        },
        "rendering_settings": {
            "materials": "Use imported material definitions",
            "lighting": "Default Revit lighting",
            "camera": "Position at [0, -100, 50] looking at origin",
            "quality": "High quality rendering"
        }
    }

    # Save instructions
    instructions_file = instructions_dir / "import_instructions.json"
    with open(instructions_file, 'w') as f:
        json.dump(instructions, f, indent=2)

    # Create markdown instructions
    markdown_file = instructions_dir / "README.md"
    with open(markdown_file, 'w') as f:
        f.write("""# 🏯 Revit Pagoda Import Instructions

## Quick Start

1. **Load Families**: Import the pagoda family JSON files into Revit Family Editor
2. **Import Materials**: Create materials using the material JSON definitions
3. **Place Pagodas**: Use the CSV file to place pagoda instances at coordinates
4. **Load Dynamo**: Import Dynamo graphs for parametric control

## Files Generated

- `revit_families/` - Pagoda family definitions
- `revit_materials/` - Material definitions
- `revit_variations/` - Pagoda instances CSV
- `dynamo_graphs/` - Parametric graph definitions

## Coordinate System

- **Origin**: [0, 0, 0]
- **Units**: Feet
- **Spacing**: 50 feet between pagodas
- **Layout**: Linear along X-axis

## Rendering

- Use imported materials for realistic appearance
- Position camera at [0, -100, 50] for best view
- Enable high quality rendering

Your steampunk pagodas are ready for Revit! 🏯⚙️
""")

    print("✅ Created Revit import instructions")
    return instructions

def main():
    """Generate all Revit-ready files"""

    print("🏯 Generating Revit-ready files for pagoda rendering...")
    print("=" * 60)

    # Create all file types
    families = create_revit_pagoda_families()
    variations = create_pagoda_variations_csv()
    materials = create_revit_materials()
    dynamo_graphs = create_dynamo_graphs()
    instructions = create_revit_import_instructions()

    print("\n🎉 All Revit files generated successfully!")
    print("\n📁 Files created in 'outputs/' directory:")
    print("   ├── revit_families/ - Pagoda family definitions")
    print("   ├── revit_materials/ - Material definitions")
    print("   ├── revit_variations/ - Pagoda instances CSV")
    print("   ├── dynamo_graphs/ - Parametric graph definitions")
    print("   └── revit_instructions/ - Import instructions")

    print(f"\n📊 Summary:")
    print(f"   • {len(families)} pagoda families")
    print(f"   • {len(variations)} pagoda variations")
    print(f"   • {len(materials)} material categories")
    print(f"   • {len(dynamo_graphs)} Dynamo graphs")

    print(f"\n🚀 Ready to import into Revit!")
    print(f"   Check 'outputs/revit_instructions/README.md' for detailed steps")

if __name__ == "__main__":
    main()
