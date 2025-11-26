"""
Cursor Integration for Live Revit Visualization

Easy-to-use functions that you can call from Cursor to update Revit in real-time.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from revit_visualizer import LiveRevitVisualizer, PagodaVariationGenerator, run_live_pagoda_visualization
from revit_connection_manager import RevitConnectionManager, setup_revit_connection

class CursorRevitInterface:
    """Cursor-friendly interface for Revit visualization"""

    def __init__(self):
        self.visualizer = None
        self.pagoda_configs = []
        self.is_setup = False
        self.connection_manager = None

    def setup_pagodas(self, num_pagodas: int = 5) -> bool:
        """Set up pagodas in Revit - call this first!"""

        print(f"🏯 Setting up {num_pagodas} pagodas in Revit...")

        try:
            # First, set up Revit connection
            print("🔌 Setting up Revit connection...")
            self.connection_manager = setup_revit_connection()

            if not self.connection_manager.is_connected:
                print("❌ Failed to connect to Revit!")
                print("Please make sure Revit is open with an active document")
                return False

            # Now set up pagoda visualization
            self.visualizer, self.pagoda_configs = run_live_pagoda_visualization(num_pagodas)

            if self.visualizer:
                self.is_setup = True
                print("✅ Setup complete! You can now see pagodas in Revit")
                print("🎮 Use the functions below to adjust parameters:")
                print("   - adjust_height(pagoda_id, new_height)")
                print("   - adjust_roof_pitch(pagoda_id, new_pitch)")
                print("   - adjust_width(pagoda_id, new_width)")
                print("   - change_style(pagoda_id, 'steampunk')")
                return True
            else:
                print("❌ Setup failed - make sure Revit is open!")
                return False

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False

    def adjust_height(self, pagoda_id: int, new_height: float) -> bool:
        """Adjust pagoda height - see changes instantly in Revit!"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        if pagoda_id < 0 or pagoda_id >= len(self.pagoda_configs):
            print(f"❌ Invalid pagoda ID. Use 0-{len(self.pagoda_configs)-1}")
            return False

        print(f"📏 Adjusting pagoda {pagoda_id} height to {new_height}m...")

        # Update configuration
        self.pagoda_configs[pagoda_id]['height'] = new_height

        # Update in Revit
        success = self.visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", self.pagoda_configs[pagoda_id])

        if success:
            print("✅ Height updated in Revit!")
            self.visualizer.refresh_view()
        else:
            print("❌ Failed to update height")

        return success

    def adjust_roof_pitch(self, pagoda_id: int, new_pitch: float) -> bool:
        """Adjust roof pitch - see changes instantly in Revit!"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        if pagoda_id < 0 or pagoda_id >= len(self.pagoda_configs):
            print(f"❌ Invalid pagoda ID. Use 0-{len(self.pagoda_configs)-1}")
            return False

        print(f"🏠 Adjusting pagoda {pagoda_id} roof pitch to {new_pitch}°...")

        # Update configuration
        self.pagoda_configs[pagoda_id]['roof_pitch'] = new_pitch

        # Update in Revit
        success = self.visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", self.pagoda_configs[pagoda_id])

        if success:
            print("✅ Roof pitch updated in Revit!")
            self.visualizer.refresh_view()
        else:
            print("❌ Failed to update roof pitch")

        return success

    def adjust_width(self, pagoda_id: int, new_width: float) -> bool:
        """Adjust pagoda width - see changes instantly in Revit!"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        if pagoda_id < 0 or pagoda_id >= len(self.pagoda_configs):
            print(f"❌ Invalid pagoda ID. Use 0-{len(self.pagoda_configs)-1}")
            return False

        print(f"📐 Adjusting pagoda {pagoda_id} width to {new_width}m...")

        # Update configuration
        self.pagoda_configs[pagoda_id]['width'] = new_width

        # Update in Revit
        success = self.visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", self.pagoda_configs[pagoda_id])

        if success:
            print("✅ Width updated in Revit!")
            self.visualizer.refresh_view()
        else:
            print("❌ Failed to update width")

        return success

    def change_style(self, pagoda_id: int, new_style: str) -> bool:
        """Change pagoda style - see changes instantly in Revit!"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        if pagoda_id < 0 or pagoda_id >= len(self.pagoda_configs):
            print(f"❌ Invalid pagoda ID. Use 0-{len(self.pagoda_configs)-1}")
            return False

        valid_styles = ['traditional', 'steampunk', 'hybrid', 'hokkaido', 'modern']
        if new_style not in valid_styles:
            print(f"❌ Invalid style. Use one of: {valid_styles}")
            return False

        print(f"🎨 Changing pagoda {pagoda_id} to {new_style} style...")

        # Get style configuration
        generator = PagodaVariationGenerator()
        style_config = generator.base_configs[new_style].copy()

        # Update configuration
        self.pagoda_configs[pagoda_id].update(style_config)

        # Update in Revit
        success = self.visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", self.pagoda_configs[pagoda_id])

        if success:
            print(f"✅ Style changed to {new_style} in Revit!")
            self.visualizer.refresh_view()
        else:
            print("❌ Failed to change style")

        return success

    def move_pagoda(self, pagoda_id: int, x: float, y: float) -> bool:
        """Move pagoda to new position - see changes instantly in Revit!"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        if pagoda_id < 0 or pagoda_id >= len(self.pagoda_configs):
            print(f"❌ Invalid pagoda ID. Use 0-{len(self.pagoda_configs)-1}")
            return False

        print(f"📍 Moving pagoda {pagoda_id} to position ({x}, {y})...")

        # Update configuration
        self.pagoda_configs[pagoda_id]['x'] = x
        self.pagoda_configs[pagoda_id]['y'] = y

        # Update in Revit
        success = self.visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", self.pagoda_configs[pagoda_id])

        if success:
            print("✅ Position updated in Revit!")
            self.visualizer.refresh_view()
        else:
            print("❌ Failed to update position")

        return success

    def show_all_pagodas(self):
        """Show all pagoda configurations"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return

        print("🏯 All Pagoda Configurations:")
        print("=" * 50)

        for i, config in enumerate(self.pagoda_configs):
            print(f"Pagoda {i}: {config['name']}")
            print(f"  Height: {config['height']:.1f}m")
            print(f"  Width: {config['width']:.1f}m")
            print(f"  Roof Pitch: {config['roof_pitch']}°")
            print(f"  Tiers: {config['tier_count']}")
            print(f"  Style: {config['style']}")
            print(f"  Position: ({config['x']:.1f}, {config['y']:.1f})")
            print()

    def create_render_view(self, view_name: str = "Pagoda_Render") -> bool:
        """Create a 3D render view for pagodas"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return False

        print(f"📸 Creating render view: {view_name}...")

        render_view = self.visualizer.create_render_view(view_name)

        if render_view:
            print("✅ Render view created!")
            return True
        else:
            print("❌ Failed to create render view")
            return False

    def refresh_view(self):
        """Refresh the Revit view"""

        if not self.is_setup:
            print("❌ Run setup_pagodas() first!")
            return

        self.visualizer.refresh_view()
        print("🔄 View refreshed!")

# Global instance for easy use
revit = CursorRevitInterface()

# Convenience functions for direct use in Cursor
def setup_pagodas(num_pagodas: int = 5):
    """Set up pagodas in Revit - call this first!"""
    return revit.setup_pagodas(num_pagodas)

def adjust_height(pagoda_id: int, new_height: float):
    """Adjust pagoda height - see changes instantly in Revit!"""
    return revit.adjust_height(pagoda_id, new_height)

def adjust_roof_pitch(pagoda_id: int, new_pitch: float):
    """Adjust roof pitch - see changes instantly in Revit!"""
    return revit.adjust_roof_pitch(pagoda_id, new_pitch)

def adjust_width(pagoda_id: int, new_width: float):
    """Adjust pagoda width - see changes instantly in Revit!"""
    return revit.adjust_width(pagoda_id, new_width)

def change_style(pagoda_id: int, new_style: str):
    """Change pagoda style - see changes instantly in Revit!"""
    return revit.change_style(pagoda_id, new_style)

def move_pagoda(pagoda_id: int, x: float, y: float):
    """Move pagoda to new position - see changes instantly in Revit!"""
    return revit.move_pagoda(pagoda_id, x, y)

def show_all_pagodas():
    """Show all pagoda configurations"""
    return revit.show_all_pagodas()

def create_render_view(view_name: str = "Pagoda_Render"):
    """Create a 3D render view for pagodas"""
    return revit.create_render_view(view_name)

def refresh_view():
    """Refresh the Revit view"""
    return revit.refresh_view()

# Example usage function
def demo_pagoda_variations():
    """Demo function showing how to create pagoda variations"""

    print("🎬 Starting Pagoda Variation Demo...")
    print("=" * 50)

    # Step 1: Set up pagodas
    print("Step 1: Setting up 5 pagodas...")
    if not setup_pagodas(5):
        print("❌ Demo failed - make sure Revit is open!")
        return

    # Step 2: Show initial configurations
    print("\nStep 2: Initial configurations:")
    show_all_pagodas()

    # Step 3: Create variations
    print("Step 3: Creating variations...")

    # Make pagoda 0 taller
    adjust_height(0, 25.0)

    # Make pagoda 1 steampunk style
    change_style(1, 'steampunk')

    # Make pagoda 2 wider
    adjust_width(2, 15.0)

    # Make pagoda 3 steeper roof
    adjust_roof_pitch(3, 45.0)

    # Move pagoda 4
    move_pagoda(4, 30.0, 20.0)

    # Step 4: Show final configurations
    print("\nStep 4: Final configurations:")
    show_all_pagodas()

    # Step 5: Create render view
    print("\nStep 5: Creating render view...")
    create_render_view("Demo_Pagodas")

    print("\n✅ Demo complete! Check Revit to see your pagoda variations!")

if __name__ == "__main__":
    # Run demo
    demo_pagoda_variations()
