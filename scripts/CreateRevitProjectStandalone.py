"""
Create Revit Project - Standalone Version
==========================================

This version creates a Revit project using the external automation service.
You can run this from your normal Python environment (outside Revit).

It will:
1. Connect to Revit (or launch it)
2. Create a new project
3. Add walls, floor, and roof
4. Create a 3D view
5. Save the project

Usage:
    python CreateRevitProjectStandalone.py

Requirements:
    - Revit installed
    - Revit Automation Service running (or it will try to start Revit)
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "revit_agent" / "code"))

try:
    from RevitController import RevitController
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    print("⚠️  RevitController not available")
    print("   Make sure revit_agent/code/RevitController.py exists")


def create_building_via_controller():
    """Create building using external Revit controller."""
    
    if not CONTROLLER_AVAILABLE:
        print("❌ Cannot create building - RevitController not available")
        return False
    
    print("=" * 60)
    print("🏗️  Create Your First Revit Project (Standalone)")
    print("=" * 60)
    print()
    
    # Define building constraints
    building_constraints = {
        "design_name": "My First Building",
        "elements": [
            {
                "type": "walls",
                "count": 4,
                "width": 20.0,  # feet
                "depth": 15.0,  # feet
                "height": 10.0  # feet
            },
            {
                "type": "floor",
                "width": 20.0,
                "depth": 15.0
            },
            {
                "type": "roof",
                "width": 20.0,
                "depth": 15.0,
                "height": 10.0
            },
            {
                "type": "view_3d",
                "name": "My First Building"
            }
        ]
    }
    
    # Save constraints to file
    import json
    constraints_file = Path("data") / "building_constraints.json"
    constraints_file.parent.mkdir(exist_ok=True)
    
    with open(constraints_file, 'w') as f:
        json.dump(building_constraints, f, indent=2)
    
    print(f"📝 Created constraints file: {constraints_file}")
    print()
    
    # Connect to Revit
    print("🔌 Connecting to Revit...")
    print("   (If Revit is not running, this may take a moment)")
    print()
    
    try:
        controller = RevitController()
        
        if not controller.connect(timeout=60):
            print("❌ Failed to connect to Revit")
            print()
            print("💡 To use external automation:")
            print("   1. Open Revit")
            print("   2. Start the automation service:")
            print("      - Via pyRevit: Run RevitAutomationService.py")
            print("      - Or use the internal script method")
            print("   3. Run this script again")
            return False
        
        print("✅ Connected to Revit!")
        print()
        
        # Apply constraints (create building)
        print("🏗️  Creating building...")
        result = controller.apply_constraints(str(constraints_file))
        
        if result.get("success"):
            print("✅ Building created successfully!")
            print()
            
            # Get status
            status = controller.get_status()
            print("📊 Project Status:")
            print(f"   • Current file: {status.get('current_file', 'Unknown')}")
            print(f"   • Total elements: {status.get('element_count', 0)}")
            print(f"   • Walls: {status.get('wall_count', 0)}")
            print()
            
            # Export elements for verification
            print("📤 Exporting elements...")
            elements = controller.export_elements("data/my_first_building.json")
            print(f"✅ Exported {len(elements.get('walls', []))} walls")
            print()
            
            print("🎉 Success! Your building is ready!")
            print()
            print("💡 Next steps:")
            print("   1. Check Revit - you should see your building")
            print("   2. Go to 3D Views → 'My First Building'")
            print("   3. Use navigation cube to explore")
            print("   4. Try different visual styles")
            print()
            
            controller.disconnect()
            return True
        else:
            print(f"❌ Failed to create building: {result.get('error', 'Unknown error')}")
            controller.disconnect()
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    success = create_building_via_controller()
    
    if not success:
        print()
        print("💡 Alternative: Use CreateFirstRevitProject.py")
        print("   (runs inside Revit via pyRevit or Dynamo)")
    
    return success


if __name__ == "__main__":
    main()

