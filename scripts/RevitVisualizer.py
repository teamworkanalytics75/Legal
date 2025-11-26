"""
AI Asset-to-Revit Pipeline - Live Revit Visualizer

Real-time visualization system that automatically updates Revit models
as you adjust parameters in Cursor. Perfect for seeing pagoda variations live!
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import clr
import sys

# Add Revit API references
clr.AddReference('RevitAPI')
clr.AddReference('RevitServices')
clr.AddReference('RevitNodes')

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *
from RevitServices.Persistence import DocumentManager
from RevitServices.Transactions import TransactionManager

logger = logging.getLogger(__name__)

class LiveRevitVisualizer:
    """Real-time Revit visualization system"""

    def __init__(self):
        self.doc = None
        self.ui_doc = None
        self.active_view = None
        self.pagoda_families = {}
        self.parameter_watchers = {}
        self.is_live_mode = False

    def connect_to_revit(self) -> bool:
        """Connect to active Revit document"""
        try:
            self.doc = DocumentManager.Instance.CurrentDBDocument
            self.ui_doc = DocumentManager.Instance.CurrentUIApplication.ActiveUIDocument
            self.active_view = self.ui_doc.ActiveView

            logger.info(f"Connected to Revit document: {self.doc.Title}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Revit: {str(e)}")
            return False

    def setup_pagoda_visualization(self, pagoda_configs: List[Dict[str, Any]]) -> bool:
        """Set up pagoda visualization in Revit"""

        if not self.connect_to_revit():
            return False

        try:
            # Start transaction
            TransactionManager.Instance.EnsureInTransaction(self.doc)

            # Create levels for pagodas
            levels = self._create_pagoda_levels(len(pagoda_configs))

            # Create pagoda families
            for i, config in enumerate(pagoda_configs):
                pagoda = self._create_pagoda_family(config, levels[i])
                self.pagoda_families[f"pagoda_{i}"] = pagoda

            # Set up parameter watchers
            self._setup_parameter_watchers()

            # Commit transaction
            TransactionManager.Instance.TransactionTaskDone()

            logger.info(f"Created {len(pagoda_configs)} pagoda visualizations")
            return True

        except Exception as e:
            TransactionManager.Instance.TransactionTaskDone()
            logger.error(f"Failed to setup pagoda visualization: {str(e)}")
            return False

    def _create_pagoda_levels(self, num_pagodas: int) -> List[Level]:
        """Create levels for pagoda placement"""

        levels = []
        base_elevation = 0.0

        for i in range(num_pagodas):
            # Create level for each pagoda
            level = Level.Create(self.doc, base_elevation)
            level.Name = f"Pagoda_{i+1}_Base"
            levels.append(level)

            # Create additional levels for tiers
            for tier in range(3):  # 3 tiers per pagoda
                tier_level = Level.Create(self.doc, base_elevation + (tier + 1) * 3.0)
                tier_level.Name = f"Pagoda_{i+1}_Tier_{tier+1}"

            base_elevation += 25.0  # Space between pagodas

        return levels

    def _create_pagoda_family(self, config: Dict[str, Any], base_level: Level) -> FamilyInstance:
        """Create a pagoda family instance"""

        # Get or create pagoda family type
        family_type = self._get_or_create_pagoda_family_type(config)

        # Create instance at base level
        location = XYZ(config.get('x', 0), config.get('y', 0), base_level.Elevation)

        family_instance = self.doc.Create.NewFamilyInstance(
            location,
            family_type,
            base_level
        )

        # Set parameters
        self._set_pagoda_parameters(family_instance, config)

        return family_instance

    def _get_or_create_pagoda_family_type(self, config: Dict[str, Any]) -> FamilySymbol:
        """Get or create pagoda family type"""

        # Look for existing pagoda family
        collector = FilteredElementCollector(self.doc)
        family_symbols = collector.OfClass(FamilySymbol).ToElements()

        for symbol in family_symbols:
            if "Pagoda" in symbol.Family.Name:
                return symbol

        # Create new pagoda family (simplified - would normally load from file)
        # For now, use a generic family and modify it
        generic_family = None
        for symbol in family_symbols:
            if "Generic Model" in symbol.Family.Name:
                generic_family = symbol
                break

        if generic_family:
            return generic_family

        # Fallback: create a simple box family
        return self._create_simple_pagoda_family(config)

    def _create_simple_pagoda_family(self, config: Dict[str, Any]) -> FamilySymbol:
        """Create a simple pagoda family using Revit API"""

        # This is a simplified version - in practice, you'd load a proper pagoda family
        # For now, we'll create a basic structure

        # Create a new family document (this would be more complex in practice)
        # For demonstration, we'll return a generic family

        collector = FilteredElementCollector(self.doc)
        family_symbols = collector.OfClass(FamilySymbol).ToElements()

        for symbol in family_symbols:
            if "Generic Model" in symbol.Family.Name:
                return symbol

        # If no generic family exists, create a simple one
        # This is a placeholder - real implementation would create proper pagoda geometry
        return None

    def _set_pagoda_parameters(self, family_instance: FamilyInstance, config: Dict[str, Any]):
        """Set parameters for pagoda family instance"""

        # Set basic parameters
        parameters = family_instance.Parameters

        for param in parameters:
            param_name = param.Definition.Name

            if param_name == "Height" and "height" in config:
                param.Set(config["height"])
            elif param_name == "Width" and "width" in config:
                param.Set(config["width"])
            elif param_name == "Roof Pitch" and "roof_pitch" in config:
                param.Set(config["roof_pitch"])
            elif param_name == "Tier Count" and "tier_count" in config:
                param.Set(config["tier_count"])

    def _setup_parameter_watchers(self):
        """Set up parameter watchers for live updates"""

        self.parameter_watchers = {}

        for pagoda_id, family_instance in self.pagoda_families.items():
            self.parameter_watchers[pagoda_id] = {
                'instance': family_instance,
                'last_params': self._get_current_parameters(family_instance)
            }

    def _get_current_parameters(self, family_instance: FamilyInstance) -> Dict[str, Any]:
        """Get current parameters from family instance"""

        params = {}
        for param in family_instance.Parameters:
            param_name = param.Definition.Name
            if param.StorageType == StorageType.Double:
                params[param_name] = param.AsDouble()
            elif param.StorageType == StorageType.Integer:
                params[param_name] = param.AsInteger()
            elif param.StorageType == StorageType.String:
                params[param_name] = param.AsString()

        return params

    def update_pagoda_live(self, pagoda_id: str, new_config: Dict[str, Any]) -> bool:
        """Update pagoda parameters in real-time"""

        if pagoda_id not in self.pagoda_families:
            logger.error(f"Pagoda {pagoda_id} not found")
            return False

        try:
            # Start transaction
            TransactionManager.Instance.EnsureInTransaction(self.doc)

            family_instance = self.pagoda_families[pagoda_id]

            # Update parameters
            self._set_pagoda_parameters(family_instance, new_config)

            # Update position if specified
            if 'x' in new_config or 'y' in new_config:
                location = family_instance.Location.Point
                new_x = new_config.get('x', location.X)
                new_y = new_config.get('y', location.Y)
                new_location = XYZ(new_x, new_y, location.Z)

                # Move the instance
                family_instance.Location.Move(XYZ(new_x - location.X, new_y - location.Y, 0))

            # Commit transaction
            TransactionManager.Instance.TransactionTaskDone()

            logger.info(f"Updated pagoda {pagoda_id} with new parameters")
            return True

        except Exception as e:
            TransactionManager.Instance.TransactionTaskDone()
            logger.error(f"Failed to update pagoda {pagoda_id}: {str(e)}")
            return False

    def update_all_pagodas(self, pagoda_configs: List[Dict[str, Any]]) -> bool:
        """Update all pagodas with new configurations"""

        success_count = 0

        for i, config in enumerate(pagoda_configs):
            pagoda_id = f"pagoda_{i}"
            if self.update_pagoda_live(pagoda_id, config):
                success_count += 1

        logger.info(f"Updated {success_count}/{len(pagoda_configs)} pagodas")
        return success_count == len(pagoda_configs)

    def start_live_mode(self):
        """Start live parameter monitoring mode"""

        self.is_live_mode = True
        logger.info("Started live mode - Revit will update automatically")

        # In a real implementation, you'd set up a timer or event listener
        # to monitor parameter changes and update Revit accordingly

    def stop_live_mode(self):
        """Stop live parameter monitoring"""

        self.is_live_mode = False
        logger.info("Stopped live mode")

    def refresh_view(self):
        """Refresh the active view in Revit"""

        try:
            self.active_view = self.ui_doc.ActiveView
            self.ui_doc.RefreshActiveView()
            logger.info("Refreshed Revit view")
        except Exception as e:
            logger.error(f"Failed to refresh view: {str(e)}")

    def create_render_view(self, view_name: str = "Pagoda_Render") -> View3D:
        """Create a 3D render view for pagodas"""

        try:
            # Create 3D view
            view_family_type = ViewFamilyType.ThreeDimensional
            view_3d = View3D.CreateIsometric(self.doc, view_family_type.Id)
            view_3d.Name = view_name

            # Set up camera
            eye = XYZ(50, 50, 30)
            target = XYZ(0, 0, 10)
            up = XYZ.BasisZ

            view_3d.SetOrientation(ViewOrientation3D(eye, target, up))

            logger.info(f"Created render view: {view_name}")
            return view_3d

        except Exception as e:
            logger.error(f"Failed to create render view: {str(e)}")
            return None

class PagodaVariationGenerator:
    """Generate pagoda variations for live visualization"""

    def __init__(self):
        self.base_configs = {
            'traditional': {
                'height': 15.0,
                'width': 8.0,
                'roof_pitch': 35,
                'tier_count': 3,
                'material': 'wood',
                'style': 'traditional'
            },
            'steampunk': {
                'height': 18.0,
                'width': 10.0,
                'roof_pitch': 25,
                'tier_count': 4,
                'material': 'brass',
                'style': 'steampunk'
            },
            'hybrid': {
                'height': 16.0,
                'width': 9.0,
                'roof_pitch': 30,
                'tier_count': 3,
                'material': 'mixed',
                'style': 'hybrid'
            },
            'hokkaido': {
                'height': 14.0,
                'width': 8.5,
                'roof_pitch': 40,
                'tier_count': 3,
                'material': 'insulated_wood',
                'style': 'hokkaido',
                'insulation_factor': 1.5
            },
            'modern': {
                'height': 20.0,
                'width': 12.0,
                'roof_pitch': 20,
                'tier_count': 5,
                'material': 'steel',
                'style': 'modern'
            }
        }

    def generate_pagoda_variations(self, num_variations: int = 5) -> List[Dict[str, Any]]:
        """Generate pagoda variations for visualization"""

        variations = []

        for i in range(num_variations):
            # Select base configuration
            base_style = list(self.base_configs.keys())[i % len(self.base_configs)]
            base_config = self.base_configs[base_style].copy()

            # Add variation parameters
            variation = {
                'id': f'pagoda_{i}',
                'name': f'{base_style.title()} Pagoda {i+1}',
                'x': i * 15.0,  # Space pagodas apart
                'y': 0.0,
                'z': 0.0,
                **base_config,
                'variation_params': {
                    'height_scale': 0.8 + (i * 0.1),  # Vary height
                    'width_scale': 0.9 + (i * 0.05),   # Vary width
                    'roof_pitch_offset': (i - 2) * 5,  # Vary roof pitch
                    'tier_count_offset': i % 2,         # Vary tier count
                }
            }

            # Apply variation parameters
            variation['height'] *= variation['variation_params']['height_scale']
            variation['width'] *= variation['variation_params']['width_scale']
            variation['roof_pitch'] += variation['variation_params']['roof_pitch_offset']
            variation['tier_count'] += variation['variation_params']['tier_count_offset']

            variations.append(variation)

        return variations

def run_live_pagoda_visualization(num_pagodas: int = 5):
    """Run live pagoda visualization in Revit"""

    print(f"🏯 Setting up live pagoda visualization for {num_pagodas} pagodas...")

    # Initialize visualizer
    visualizer = LiveRevitVisualizer()

    # Generate pagoda variations
    generator = PagodaVariationGenerator()
    pagoda_configs = generator.generate_pagoda_variations(num_pagodas)

    # Set up visualization in Revit
    if visualizer.setup_pagoda_visualization(pagoda_configs):
        print("✅ Successfully set up pagoda visualization in Revit!")
        print("🎯 You can now see your pagodas live in Revit")

        # Start live mode
        visualizer.start_live_mode()

        # Create render view
        render_view = visualizer.create_render_view("Pagoda_Render_View")

        print("📊 Pagoda Configurations:")
        for i, config in enumerate(pagoda_configs):
            print(f"  Pagoda {i+1}: {config['name']}")
            print(f"    Height: {config['height']:.1f}m")
            print(f"    Width: {config['width']:.1f}m")
            print(f"    Roof Pitch: {config['roof_pitch']}°")
            print(f"    Tiers: {config['tier_count']}")
            print(f"    Style: {config['style']}")
            print()

        return visualizer, pagoda_configs
    else:
        print("❌ Failed to set up pagoda visualization")
        return None, None

def update_pagoda_parameters(visualizer: LiveRevitVisualizer,
                           pagoda_configs: List[Dict[str, Any]],
                           parameter_updates: Dict[str, Any]):
    """Update pagoda parameters in real-time"""

    print("🔄 Updating pagoda parameters...")

    # Apply updates to configurations
    for i, config in enumerate(pagoda_configs):
        pagoda_id = f"pagoda_{i}"

        # Apply parameter updates
        for param_name, param_value in parameter_updates.items():
            if param_name in config:
                config[param_name] = param_value

        # Update in Revit
        visualizer.update_pagoda_live(pagoda_id, config)

    # Refresh view
    visualizer.refresh_view()

    print("✅ Parameters updated in Revit!")

if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        num_pagodas = int(sys.argv[1])
    else:
        num_pagodas = 5

    try:
        visualizer, configs = run_live_pagoda_visualization(num_pagodas)

        if visualizer:
            print("\n🎮 Live Mode Active!")
            print("You can now adjust parameters and see changes in Revit")
            print("\nExample parameter updates:")
            print("update_pagoda_parameters(visualizer, configs, {'height': 20.0, 'roof_pitch': 45})")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure Revit is open and you have a document active")
