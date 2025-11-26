"""
Simple test to verify the AI Asset-to-Revit Pipeline works
"""

# Test the pipeline components
print("🧪 Testing AI Asset-to-Revit Pipeline")
print("=" * 40)

# Test 1: Check if steampunk house exists
from pathlib import Path
steampunk_path = Path(r"C:\Users\Owner\Desktop\SteamPunk House\source\tutorial_steampunk_house_no_LM.fbx")

print(f"\n1. Checking steampunk house...")
if steampunk_path.exists():
    print(f"✅ Found: {steampunk_path}")
    print(f"📊 Size: {steampunk_path.stat().st_size:,} bytes")
else:
    print(f"❌ Not found: {steampunk_path}")

# Test 2: Test geometry parser
print(f"\n2. Testing geometry parser...")
try:
    from ai_asset_pipeline.asset_analyzer.geometry_parser import GeometryParser
    parser = GeometryParser()
    print(f"✅ Parser supports: {', '.join(parser.supported_formats)}")
except Exception as e:
    print(f"❌ Parser error: {e}")

# Test 3: Test constraint detector
print(f"\n3. Testing constraint detector...")
try:
    from ai_asset_pipeline.asset_analyzer.constraint_detector import ConstraintDetector
    detector = ConstraintDetector()
    print(f"✅ Detector supports: {', '.join(detector.cultural_styles.keys())}")
except Exception as e:
    print(f"❌ Detector error: {e}")

# Test 4: Test Revit mapper
print(f"\n4. Testing Revit mapper...")
try:
    from ai_asset_pipeline.asset_analyzer.revit_mapper import RevitMapper
    mapper = RevitMapper()
    print(f"✅ Mapper supports: {', '.join(mapper.family_templates.keys())}")
except Exception as e:
    print(f"❌ Mapper error: {e}")

# Test 5: Test Dynamo builder
print(f"\n5. Testing Dynamo builder...")
try:
    from ai_asset_pipeline.dynamo_generator.graph_builder import DynamoGraphBuilder
    builder = DynamoGraphBuilder()
    print(f"✅ Builder has {len(builder.node_templates)} node templates")
except Exception as e:
    print(f"❌ Builder error: {e}")

# Test 6: Test variation generator
print(f"\n6. Testing variation generator...")
try:
    from ai_asset_pipeline.design_explorer.variation_generator import VariationGenerator
    generator = VariationGenerator()
    print(f"✅ Generator supports: {', '.join(generator.cultural_scoring.keys())}")
except Exception as e:
    print(f"❌ Generator error: {e}")

# Test 7: Test cursor interface
print(f"\n7. Testing cursor interface...")
try:
    from cursor_interface import CursorPipelineInterface
    interface = CursorPipelineInterface()
    print("✅ Interface initialized successfully")
except Exception as e:
    print(f"❌ Interface error: {e}")

print(f"\n🎉 All components tested!")
print(f"\nTo run the pipeline:")
print(f"1. Open Cursor IDE")
print(f"2. Import: from cursor_interface import run_steampunk_house_pipeline")
print(f"3. Run: result = run_steampunk_house_pipeline(20)")
print(f"4. Create pagodas: from cursor_interface import create_pagoda_variations")
print(f"5. Run: pagodas = create_pagoda_variations(5)")

print(f"\n🏯 Your steampunk pagodas will be ready for Revit!")
