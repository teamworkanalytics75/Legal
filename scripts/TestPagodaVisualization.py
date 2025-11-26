"""
Test Script for Live Revit Visualization

Run this to test the pagoda visualization system.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from cursor_revit_interface import *

def test_pagoda_system():
    """Test the pagoda visualization system"""

    print("🧪 Testing Pagoda Visualization System")
    print("=" * 50)

    # Test 1: Setup
    print("Test 1: Setting up pagodas...")
    if setup_pagodas(3):
        print("✅ Setup successful!")
    else:
        print("❌ Setup failed - make sure Revit is open!")
        return False

    # Test 2: Show configurations
    print("\nTest 2: Showing configurations...")
    show_all_pagodas()

    # Test 3: Adjust parameters
    print("\nTest 3: Adjusting parameters...")

    # Adjust height
    if adjust_height(0, 20.0):
        print("✅ Height adjustment successful!")

    # Adjust roof pitch
    if adjust_roof_pitch(1, 40.0):
        print("✅ Roof pitch adjustment successful!")

    # Change style
    if change_style(2, 'steampunk'):
        print("✅ Style change successful!")

    # Test 4: Move pagoda
    print("\nTest 4: Moving pagoda...")
    if move_pagoda(0, 10.0, 5.0):
        print("✅ Move successful!")

    # Test 5: Create render view
    print("\nTest 5: Creating render view...")
    if create_render_view("Test_Pagodas"):
        print("✅ Render view created!")

    # Test 6: Refresh view
    print("\nTest 6: Refreshing view...")
    refresh_view()
    print("✅ View refreshed!")

    print("\n🎉 All tests completed!")
    print("Check Revit to see your pagoda variations!")

    return True

def quick_pagoda_demo():
    """Quick demo for testing"""

    print("⚡ Quick Pagoda Demo")
    print("=" * 30)

    # Setup
    setup_pagodas(2)

    # Show initial state
    show_all_pagodas()

    # Make changes
    adjust_height(0, 18.0)
    change_style(1, 'hokkaido')

    # Show final state
    show_all_pagodas()

    print("✅ Quick demo complete!")

if __name__ == "__main__":
    # Run tests
    test_pagoda_system()
