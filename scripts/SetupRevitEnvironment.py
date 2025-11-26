"""
Revit Setup Script

Easy setup script that handles all Revit connection requirements.
Run this first to set up your Revit connection.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from revit_connection_manager import setup_revit_connection
from cursor_revit_interface import CursorRevitInterface

def check_revit_requirements():
    """Check if Revit requirements are met"""

    print("🔍 Checking Revit requirements...")
    print("=" * 50)

    requirements_met = True

    # Check 1: Revit Installation
    print("1. Checking Revit installation...")
    connection_manager = RevitConnectionManager()
    revit_info = connection_manager.detect_revit_installation()

    if revit_info['installed']:
        print(f"   ✅ Revit {revit_info['version']} found at {revit_info['path']}")

        if revit_info['api_available']:
            print("   ✅ Revit API available")
        else:
            print("   ⚠️  Revit API not available")
            requirements_met = False
    else:
        print("   ❌ Revit not installed")
        requirements_met = False

    # Check 2: Python Environment
    print("\n2. Checking Python environment...")
    try:
        import clr
        print("   ✅ Python.NET (clr) available")
    except ImportError:
        print("   ❌ Python.NET (clr) not available")
        print("   Install with: pip install pythonnet")
        requirements_met = False

    # Check 3: Revit API Access
    print("\n3. Checking Revit API access...")
    try:
        clr.AddReference('RevitAPI')
        print("   ✅ Revit API accessible")
    except Exception as e:
        print(f"   ❌ Revit API not accessible: {str(e)}")
        requirements_met = False

    return requirements_met, revit_info

def setup_revit_environment():
    """Set up Revit environment for pagoda visualization"""

    print("\n🔧 Setting up Revit environment...")
    print("=" * 50)

    # Check requirements
    requirements_met, revit_info = check_revit_requirements()

    if not requirements_met:
        print("\n❌ Requirements not met. Please fix the issues above.")
        return False

    # Set up connection
    print("\n🔌 Setting up Revit connection...")
    connection_manager = setup_revit_connection()

    if not connection_manager.is_connected:
        print("\n❌ Failed to connect to Revit")
        print("Please make sure Revit is open with an active document")
        return False

    # Set up pagoda project
    print("\n🏯 Setting up pagoda project...")
    if connection_manager.setup_pagoda_project():
        print("✅ Pagoda project created successfully!")
    else:
        print("⚠️  Failed to create pagoda project, but connection is working")

    return True

def test_pagoda_system():
    """Test the pagoda visualization system"""

    print("\n🧪 Testing pagoda visualization system...")
    print("=" * 50)

    try:
        # Create interface
        interface = CursorRevitInterface()

        # Set up pagodas
        if interface.setup_pagodas(3):
            print("✅ Pagoda system setup successful!")

            # Test parameter adjustments
            print("\n🎮 Testing parameter adjustments...")

            # Adjust height
            if interface.adjust_height(0, 20.0):
                print("   ✅ Height adjustment working")

            # Change style
            if interface.change_style(1, 'steampunk'):
                print("   ✅ Style change working")

            # Adjust roof pitch
            if interface.adjust_roof_pitch(2, 40.0):
                print("   ✅ Roof pitch adjustment working")

            # Show configurations
            print("\n📊 Current pagoda configurations:")
            interface.show_all_pagodas()

            print("\n🎉 All tests passed! Your pagoda system is ready!")
            return True
        else:
            print("❌ Pagoda system setup failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main setup function"""

    print("🏯 Revit Pagoda Visualization Setup")
    print("=" * 50)
    print("This script will set up your Revit environment for pagoda visualization")
    print()

    # Step 1: Check requirements
    requirements_met, revit_info = check_revit_requirements()

    if not requirements_met:
        print("\n❌ Setup failed - requirements not met")
        print("\nPlease fix the issues above and try again")
        return False

    # Step 2: Set up environment
    if not setup_revit_environment():
        print("\n❌ Setup failed - environment setup failed")
        return False

    # Step 3: Test system
    if not test_pagoda_system():
        print("\n❌ Setup failed - system test failed")
        return False

    print("\n🎉 Setup complete!")
    print("=" * 50)
    print("Your Revit pagoda visualization system is ready!")
    print()
    print("You can now use these functions in Cursor:")
    print("  setup_pagodas(5)           # Set up 5 pagodas")
    print("  adjust_height(0, 25.0)     # Make pagoda 0 taller")
    print("  change_style(1, 'steampunk') # Change pagoda 1 to steampunk")
    print("  show_all_pagodas()         # Show all configurations")
    print()
    print("Make sure Revit stays open while using the system!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup successful! You're ready to visualize pagodas!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        print("Please check your Revit installation and try again")
