"""
Simple test script to verify the AI Asset-to-Revit Pipeline components
"""

import sys
from pathlib import Path

# Test imports
print("🧪 Testing AI Asset-to-Revit Pipeline Components")
print("=" * 50)

# Test 1: Geometry Parser
print("\n1. Testing Geometry Parser...")
try:
    from ai_asset_pipeline.asset_analyzer.geometry_parser import parse_asset, GeometryParser
    print("✅ Geometry parser imported successfully")

    # Test with mock data
    parser = GeometryParser()
    print(f"✅ Parser initialized with {len(parser.supported_formats)} supported formats")

except Exception as e:
    print(f"❌ Geometry parser failed: {e}")

# Test 2: Constraint Detector
print("\n2. Testing Constraint Detector...")
try:
    from ai_asset_pipeline.asset_analyzer.constraint_detector import detect_constraints, ConstraintDetector
    print("✅ Constraint detector imported successfully")

    detector = ConstraintDetector()
    print(f"✅ Detector initialized with {len(detector.cultural_styles)} cultural styles")

except Exception as e:
    print(f"❌ Constraint detector failed: {e}")

# Test 3: Revit Mapper
print("\n3. Testing Revit Mapper...")
try:
    from ai_asset_pipeline.asset_analyzer.revit_mapper import map_constraints_to_revit, RevitMapper
    print("✅ Revit mapper imported successfully")

    mapper = RevitMapper()
    print(f"✅ Mapper initialized with {len(mapper.family_templates)} family templates")

except Exception as e:
    print(f"❌ Revit mapper failed: {e}")

# Test 4: Dynamo Graph Builder
print("\n4. Testing Dynamo Graph Builder...")
try:
    from ai_asset_pipeline.dynamo_generator.graph_builder import build_dynamo_graph, DynamoGraphBuilder
    print("✅ Dynamo graph builder imported successfully")

    builder = DynamoGraphBuilder()
    print(f"✅ Builder initialized with {len(builder.node_templates)} node templates")

except Exception as e:
    print(f"❌ Dynamo graph builder failed: {e}")

# Test 5: Variation Generator
print("\n5. Testing Variation Generator...")
try:
    from ai_asset_pipeline.design_explorer.variation_generator import explore_design_variations, VariationGenerator
    print("✅ Variation generator imported successfully")

    generator = VariationGenerator()
    print(f"✅ Generator initialized with {len(generator.cultural_scoring)} cultural scoring configs")

except Exception as e:
    print(f"❌ Variation generator failed: {e}")

# Test 6: Cursor Interface
print("\n6. Testing Cursor Interface...")
try:
    from cursor_interface import CursorPipelineInterface, run_steampunk_house_pipeline
    print("✅ Cursor interface imported successfully")

    interface = CursorPipelineInterface()
    print("✅ Interface initialized successfully")

except Exception as e:
    print(f"❌ Cursor interface failed: {e}")

# Test 7: Steampunk House Path
print("\n7. Testing Steampunk House Path...")
steampunk_path = Path(r"C:\Users\Owner\Desktop\SteamPunk House\source\tutorial_steampunk_house_no_LM.fbx")
if steampunk_path.exists():
    print(f"✅ Steampunk house found: {steampunk_path}")
    print(f"📊 File size: {steampunk_path.stat().st_size:,} bytes")
else:
    print(f"❌ Steampunk house not found: {steampunk_path}")

print("\n🎉 Component testing complete!")
print("\nTo run the full pipeline:")
print("1. Open Cursor IDE")
print("2. Run: run_steampunk_house_pipeline(20)")
print("3. Or run: create_pagoda_variations(5)")

if __name__ == "__main__":
    pass
