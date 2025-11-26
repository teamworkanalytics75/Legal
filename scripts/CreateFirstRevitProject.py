"""
Create Your First Revit Project
================================

This script creates a simple Revit project file (.rvt) with basic building elements
that you can open in Revit and view. Perfect for getting started!

What it creates:
- A simple building with 4 walls
- A floor
- A roof
- A 3D view to see everything

Usage:
    python CreateFirstRevitProject.py

Requirements:
    - Revit must be installed
    - Run this script from within Revit (via pyRevit or Dynamo)
    - OR use the external automation service
"""

import sys
import os
from pathlib import Path

# Try to import Revit API
try:
    import clr
    clr.AddReference('RevitAPI')
    clr.AddReference('RevitServices')
    from Autodesk.Revit.DB import *
    from Autodesk.Revit.UI import *
    from RevitServices.Persistence import DocumentManager
    from RevitServices.Transactions import TransactionManager
    REVIT_AVAILABLE = True
except ImportError:
    REVIT_AVAILABLE = False
    print("⚠️  Revit API not available. This script needs to run inside Revit.")
    print("   Options:")
    print("   1. Run via pyRevit in Revit")
    print("   2. Run via Dynamo Python node")
    print("   3. Use external automation service")


def create_simple_building(doc, output_path=None):
    """
    Create a simple building with walls, floor, and roof.
    
    Args:
        doc: Revit Document
        output_path: Optional path to save the project
    """
    
    if not REVIT_AVAILABLE:
        print("❌ Cannot create building - Revit API not available")
        return False
    
    print("🏗️  Creating simple building...")
    
    try:
        # Start transaction
        TransactionManager.Instance.EnsureInTransaction(doc)
        
        # Get or create a level
        level = get_or_create_level(doc, 0.0, "Level 1")
        print(f"   ✅ Using level: {level.Name}")
        
        # Get wall type
        wall_type = get_wall_type(doc)
        if not wall_type:
            print("   ❌ No wall type found")
            return False
        print(f"   ✅ Using wall type: {wall_type.Name}")
        
        # Create walls (simple rectangle: 20' x 15')
        walls = create_walls(doc, level, wall_type, 
                            width=20.0,  # 20 feet
                            depth=15.0,  # 15 feet
                            height=10.0) # 10 feet tall
        print(f"   ✅ Created {len(walls)} walls")
        
        # Create floor
        floor = create_floor(doc, level, width=20.0, depth=15.0)
        if floor:
            print(f"   ✅ Created floor")
        
        # Create roof
        roof = create_roof(doc, level, width=20.0, depth=15.0, wall_height=10.0)
        if roof:
            print(f"   ✅ Created roof")
        
        # Create a 3D view
        view_3d = create_3d_view(doc, "My First Building")
        if view_3d:
            print(f"   ✅ Created 3D view: {view_3d.Name}")
        
        # Commit transaction
        TransactionManager.Instance.TransactionTaskDone()
        
        # Save if path provided
        if output_path:
            save_options = SaveAsOptions()
            doc.SaveAs(output_path, save_options)
            print(f"   ✅ Saved project to: {output_path}")
        
        print("\n🎉 Building created successfully!")
        print("\n📋 What was created:")
        print("   • 4 walls forming a 20' x 15' rectangle")
        print("   • 1 floor")
        print("   • 1 roof")
        print("   • 1 3D view")
        print("\n💡 In Revit:")
        print("   • Go to 3D Views → 'My First Building' to see your building")
        print("   • Use the navigation cube to rotate and zoom")
        print("   • Try different visual styles (Shaded, Realistic, etc.)")
        
        return True
        
    except Exception as e:
        TransactionManager.Instance.TransactionTaskDone()
        print(f"   ❌ Error creating building: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_or_create_level(doc, elevation, name):
    """Get existing level or create a new one."""
    
    # Try to find existing level
    collector = FilteredElementCollector(doc)
    levels = collector.OfClass(Level).ToElements()
    
    for level in levels:
        if abs(level.Elevation - elevation) < 0.01:
            return level
    
    # Create new level
    level = Level.Create(doc, elevation)
    level.Name = name
    return level


def get_wall_type(doc):
    """Get the first available wall type."""
    
    collector = FilteredElementCollector(doc)
    wall_types = collector.OfClass(WallType).ToElements()
    
    if wall_types:
        return wall_types[0]
    
    return None


def create_walls(doc, level, wall_type, width, depth, height):
    """Create 4 walls forming a rectangle."""
    
    walls = []
    
    # Define corners (in feet, converted to internal units)
    # Revit uses feet internally
    corners = [
        XYZ(0, 0, 0),
        XYZ(width, 0, 0),
        XYZ(width, depth, 0),
        XYZ(0, depth, 0)
    ]
    
    # Create walls between corners
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        
        line = Line.CreateBound(start, end)
        wall = Wall.Create(doc, line, wall_type.Id, level.Id, height, 0, False, False)
        walls.append(wall)
    
    return walls


def create_floor(doc, level, width, depth):
    """Create a floor from boundary curves."""
    
    try:
        # Get floor type
        collector = FilteredElementCollector(doc)
        floor_types = collector.OfClass(FloorType).ToElements()
        
        if not floor_types:
            print("   ⚠️  No floor type found, skipping floor")
            return None
        
        floor_type = floor_types[0]
        
        # Create boundary curves
        curves = CurveArray()
        curves.Append(Line.CreateBound(XYZ(0, 0, 0), XYZ(width, 0, 0)))
        curves.Append(Line.CreateBound(XYZ(width, 0, 0), XYZ(width, depth, 0)))
        curves.Append(Line.CreateBound(XYZ(width, depth, 0), XYZ(0, depth, 0)))
        curves.Append(Line.CreateBound(XYZ(0, depth, 0), XYZ(0, 0, 0)))
        
        # Create floor
        floor = Floor.Create(doc, curves, floor_type.Id, level.Id)
        return floor
        
    except Exception as e:
        print(f"   ⚠️  Could not create floor: {str(e)}")
        return None


def create_roof(doc, level, width, depth, wall_height):
    """Create a simple flat roof."""
    
    try:
        # Get roof type
        collector = FilteredElementCollector(doc)
        roof_types = collector.OfClass(RoofType).ToElements()
        
        if not roof_types:
            print("   ⚠️  No roof type found, skipping roof")
            return None
        
        roof_type = roof_types[0]
        
        # Create boundary curves at roof level
        z = wall_height
        curves = CurveArray()
        curves.Append(Line.CreateBound(XYZ(0, 0, z), XYZ(width, 0, z)))
        curves.Append(Line.CreateBound(XYZ(width, 0, z), XYZ(width, depth, z)))
        curves.Append(Line.CreateBound(XYZ(width, depth, z), XYZ(0, depth, z)))
        curves.Append(Line.CreateBound(XYZ(0, depth, z), XYZ(0, 0, z)))
        
        # Create roof
        roof = Roof.Create(doc, curves, roof_type.Id, level.Id)
        return roof
        
    except Exception as e:
        print(f"   ⚠️  Could not create roof: {str(e)}")
        return None


def create_3d_view(doc, view_name):
    """Create a 3D view to see the building."""
    
    try:
        # Get 3D view family type
        collector = FilteredElementCollector(doc)
        view_family_types = collector.OfClass(ViewFamilyType).ToElements()
        
        view_family_type_3d = None
        for vft in view_family_types:
            if vft.ViewFamily == ViewFamily.ThreeDimensional:
                view_family_type_3d = vft
                break
        
        if not view_family_type_3d:
            print("   ⚠️  Could not find 3D view family type")
            return None
        
        # Create 3D view
        view_3d = View3D.CreateIsometric(doc, view_family_type_3d.Id)
        view_3d.Name = view_name
        
        # Set up camera position (looking at the building)
        eye = XYZ(30, -30, 15)  # Camera position
        target = XYZ(10, 7.5, 5)  # Looking at center of building
        up = XYZ.BasisZ  # Up direction
        
        orientation = ViewOrientation3D(eye, target, up)
        view_3d.SetOrientation(orientation)
        
        return view_3d
        
    except Exception as e:
        print(f"   ⚠️  Could not create 3D view: {str(e)}")
        return None


def main():
    """Main function to create the building."""
    
    print("=" * 60)
    print("🏗️  Create Your First Revit Project")
    print("=" * 60)
    print()
    
    if not REVIT_AVAILABLE:
        print("❌ This script needs to run inside Revit.")
        print()
        print("📖 How to run:")
        print("   1. Open Revit")
        print("   2. Create a new project (Architectural Template)")
        print("   3. Use pyRevit or Dynamo to run this script")
        print("   4. OR use the external automation service")
        print()
        print("💡 Alternative: Use CreateRevitProjectStandalone.py")
        print("   (creates project via external automation)")
        return False
    
    # Get current document
    try:
        doc = DocumentManager.Instance.CurrentDBDocument
        if not doc:
            print("❌ No active Revit document found")
            print("   Please open or create a project in Revit first")
            return False
        
        print(f"📄 Active document: {doc.Title}")
        print()
        
        # Create building
        success = create_simple_building(doc)
        
        if success:
            print()
            print("✅ Success! Your building is ready to view in Revit!")
            return True
        else:
            print()
            print("❌ Failed to create building")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

