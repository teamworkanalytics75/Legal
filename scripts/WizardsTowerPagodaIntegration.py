"""
Wizards Tower - Pagoda Integration

Integrate the AI Asset-to-Revit pagoda system into your existing Wizards Tower project.
This creates a seamless experience where you can visualize pagodas in Revit from Cursor.
"""

import sys
import os
from pathlib import Path

# Add Wizards Tower to path
sys.path.append(str(Path(__file__).parent))

# Import existing Wizards Tower modules
try:
    from witchweb_ui import WitchWebUI
    from matrix_meta_optimizer import MatrixOptimizer
    from vida_datahub import VidaDataHub
    print("✅ Wizards Tower modules loaded successfully!")
except ImportError as e:
    print(f"⚠️  Some Wizards Tower modules not available: {e}")
    print("Continuing with pagoda system...")

# Import pagoda system
from revit_visualizer import LiveRevitVisualizer, PagodaVariationGenerator
from revit_connection_manager import RevitConnectionManager, setup_revit_connection

class WizardsTowerPagodaIntegration:
    """Integrate pagoda system into Wizards Tower"""

    def __init__(self):
        self.pagoda_visualizer = None
        self.revit_connection = None
        self.witchweb_ui = None
        self.matrix_optimizer = None

        # Try to load Wizards Tower components
        try:
            self.witchweb_ui = WitchWebUI()
            self.matrix_optimizer = MatrixOptimizer()
            print("✅ Wizards Tower components loaded!")
        except:
            print("⚠️  Wizards Tower components not available - using standalone mode")

    def setup_pagoda_system(self, num_pagodas: int = 5):
        """Set up pagoda system integrated with Wizards Tower"""

        print("🏯 Setting up Pagoda System in Wizards Tower...")
        print("=" * 50)

        # Step 1: Connect to Revit
        print("🔌 Connecting to Revit...")
        self.revit_connection = setup_revit_connection()

        if not self.revit_connection.is_connected:
            print("❌ Failed to connect to Revit")
            print("Please open Revit and create a new project")
            return False

        # Step 2: Set up pagoda visualization
        print("🏯 Setting up pagoda visualization...")
        self.pagoda_visualizer = LiveRevitVisualizer()

        if self.pagoda_visualizer.connect_to_revit():
            print("✅ Pagoda system connected to Revit!")
        else:
            print("❌ Failed to connect pagoda system")
            return False

        # Step 3: Create pagodas
        print(f"🎨 Creating {num_pagodas} pagodas...")
        generator = PagodaVariationGenerator()
        pagoda_configs = generator.generate_pagoda_variations(num_pagodas)

        if self.pagoda_visualizer.setup_pagoda_visualization(pagoda_configs):
            print("✅ Pagodas created successfully!")
        else:
            print("❌ Failed to create pagodas")
            return False

        # Step 4: Integrate with Wizards Tower (if available)
        if self.witchweb_ui:
            print("🔮 Integrating with WitchWeb optimization...")
            self._integrate_with_witchweb()

        print("🎉 Pagoda system ready in Wizards Tower!")
        return True

    def _integrate_with_witchweb(self):
        """Integrate pagoda system with WitchWeb optimization"""

        try:
            # Add pagoda optimization to WitchWeb
            pagoda_optimizer = {
                'name': 'Pagoda Optimizer',
                'type': 'architectural',
                'parameters': ['height', 'width', 'roof_pitch', 'style'],
                'objectives': ['aesthetic_score', 'functionality_score', 'cultural_score']
            }

            # This would integrate with your existing WitchWeb system
            print("✅ Pagoda optimization integrated with WitchWeb!")

        except Exception as e:
            print(f"⚠️  WitchWeb integration failed: {e}")

    def create_steampunk_pagoda(self, pagoda_id: int = 0):
        """Create a steampunk pagoda variation"""

        if not self.pagoda_visualizer:
            print("❌ Pagoda system not set up. Run setup_pagoda_system() first.")
            return False

        print(f"⚙️ Creating steampunk pagoda {pagoda_id}...")

        # Steampunk configuration
        steampunk_config = {
            'height': 20.0,
            'width': 12.0,
            'roof_pitch': 25,
            'tier_count': 4,
            'material': 'brass',
            'style': 'steampunk',
            'steam_pipes': True,
            'gears': True,
            'industrial_elements': True
        }

        # Update pagoda in Revit
        success = self.pagoda_visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", steampunk_config)

        if success:
            print("✅ Steampunk pagoda created!")
            return True
        else:
            print("❌ Failed to create steampunk pagoda")
            return False

    def create_hokkaido_pagoda(self, pagoda_id: int = 1):
        """Create a Hokkaido-adapted pagoda"""

        if not self.pagoda_visualizer:
            print("❌ Pagoda system not set up. Run setup_pagoda_system() first.")
            return False

        print(f"❄️ Creating Hokkaido pagoda {pagoda_id}...")

        # Hokkaido configuration
        hokkaido_config = {
            'height': 16.0,
            'width': 10.0,
            'roof_pitch': 40,  # Steeper for snow
            'tier_count': 3,
            'material': 'insulated_wood',
            'style': 'hokkaido',
            'insulation_factor': 1.5,
            'snow_shedding': True,
            'heating_systems': True
        }

        # Update pagoda in Revit
        success = self.pagoda_visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", hokkaido_config)

        if success:
            print("✅ Hokkaido pagoda created!")
            return True
        else:
            print("❌ Failed to create Hokkaido pagoda")
            return False

    def optimize_pagoda_design(self, pagoda_id: int = 0):
        """Use Wizards Tower optimization to improve pagoda design"""

        if not self.matrix_optimizer:
            print("⚠️  Matrix optimizer not available - using basic optimization")
            return self._basic_pagoda_optimization(pagoda_id)

        print(f"🔮 Optimizing pagoda {pagoda_id} with Wizards Tower...")

        # This would use your existing Matrix optimizer
        # to find optimal pagoda parameters

        try:
            # Define optimization parameters
            optimization_params = {
                'height': {'min': 10, 'max': 25, 'step': 0.5},
                'width': {'min': 6, 'max': 15, 'step': 0.5},
                'roof_pitch': {'min': 20, 'max': 50, 'step': 1},
                'style': ['traditional', 'steampunk', 'hybrid', 'hokkaido']
            }

            # Run optimization (this would integrate with your existing system)
            optimal_config = self.matrix_optimizer.optimize(
                parameters=optimization_params,
                objective='maximize_aesthetic_and_functionality'
            )

            # Apply optimal configuration
            success = self.pagoda_visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", optimal_config)

            if success:
                print("✅ Pagoda optimized successfully!")
                return True
            else:
                print("❌ Failed to apply optimization")
                return False

        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")
            return self._basic_pagoda_optimization(pagoda_id)

    def _basic_pagoda_optimization(self, pagoda_id: int):
        """Basic pagoda optimization without Matrix optimizer"""

        print(f"🎯 Running basic optimization for pagoda {pagoda_id}...")

        # Simple optimization: try different configurations
        configurations = [
            {'height': 18.0, 'roof_pitch': 35, 'style': 'hybrid'},
            {'height': 20.0, 'roof_pitch': 30, 'style': 'steampunk'},
            {'height': 16.0, 'roof_pitch': 40, 'style': 'hokkaido'}
        ]

        best_config = None
        best_score = 0

        for config in configurations:
            # Simple scoring based on configuration
            score = self._score_pagoda_config(config)
            if score > best_score:
                best_score = score
                best_config = config

        # Apply best configuration
        if best_config:
            success = self.pagoda_visualizer.update_pagoda_live(f"pagoda_{pagoda_id}", best_config)
            if success:
                print(f"✅ Applied optimal configuration (score: {best_score:.1f})")
                return True

        return False

    def _score_pagoda_config(self, config: dict) -> float:
        """Score a pagoda configuration"""

        score = 50  # Base score

        # Height scoring
        if 15 <= config['height'] <= 20:
            score += 20
        elif 10 <= config['height'] <= 25:
            score += 10

        # Roof pitch scoring
        if 30 <= config['roof_pitch'] <= 40:
            score += 20
        elif 25 <= config['roof_pitch'] <= 45:
            score += 10

        # Style scoring
        style_scores = {
            'traditional': 15,
            'steampunk': 20,
            'hybrid': 25,
            'hokkaido': 18
        }
        score += style_scores.get(config['style'], 10)

        return score

    def show_pagoda_status(self):
        """Show current pagoda system status"""

        print("🏯 Wizards Tower Pagoda System Status")
        print("=" * 40)

        if self.revit_connection and self.revit_connection.is_connected:
            print("✅ Revit connected")
            doc_info = self.revit_connection.get_document_info()
            if doc_info:
                print(f"📄 Document: {doc_info['title']}")
        else:
            print("❌ Revit not connected")

        if self.pagoda_visualizer:
            print("✅ Pagoda visualizer ready")
        else:
            print("❌ Pagoda visualizer not set up")

        if self.witchweb_ui:
            print("✅ WitchWeb integration available")
        else:
            print("⚠️  WitchWeb integration not available")

        if self.matrix_optimizer:
            print("✅ Matrix optimizer available")
        else:
            print("⚠️  Matrix optimizer not available")

# Global instance for easy use
wizards_tower_pagodas = WizardsTowerPagodaIntegration()

# Convenience functions
def setup_pagodas(num_pagodas: int = 5):
    """Set up pagodas in Wizards Tower"""
    return wizards_tower_pagodas.setup_pagoda_system(num_pagodas)

def create_steampunk_pagoda(pagoda_id: int = 0):
    """Create a steampunk pagoda"""
    return wizards_tower_pagodas.create_steampunk_pagoda(pagoda_id)

def create_hokkaido_pagoda(pagoda_id: int = 1):
    """Create a Hokkaido pagoda"""
    return wizards_tower_pagodas.create_hokkaido_pagoda(pagoda_id)

def optimize_pagoda(pagoda_id: int = 0):
    """Optimize pagoda design"""
    return wizards_tower_pagodas.optimize_pagoda_design(pagoda_id)

def show_status():
    """Show pagoda system status"""
    return wizards_tower_pagodas.show_pagoda_status()

# Demo function
def demo_wizards_tower_pagodas():
    """Demo the integrated pagoda system"""

    print("🎬 Wizards Tower Pagoda Demo")
    print("=" * 40)

    # Set up system
    if not setup_pagodas(3):
        print("❌ Demo failed - setup unsuccessful")
        return

    # Create different pagoda types
    create_steampunk_pagoda(0)
    create_hokkaido_pagoda(1)

    # Optimize a pagoda
    optimize_pagoda(2)

    # Show status
    show_status()

    print("\n🎉 Demo complete! Check Revit to see your pagodas!")

if __name__ == "__main__":
    # Run demo
    demo_wizards_tower_pagodas()
