"""
Plugin Calibrator - Recalibrates Plugins Based on Perfect Outline

Uses the OutlineManager to:
1. Set plugin targets based on section requirements
2. Adjust plugin priorities based on section importance
3. Enable/disable plugins based on section context
4. Calibrate enumeration requirements
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .outline_manager import OutlineManager, load_outline_manager

logger = logging.getLogger(__name__)


class PluginCalibrator:
    """
    Calibrates plugins based on perfect outline structure.

    Ensures plugins:
    - Know their target values based on section requirements
    - Are prioritized correctly based on section importance
    - Follow enumeration requirements
    - Are organized by section
    """

    def __init__(self, outline_manager: Optional[OutlineManager] = None):
        """
        Initialize Plugin Calibrator.

        Args:
            outline_manager: Optional OutlineManager instance
        """
        self.outline_manager = outline_manager or load_outline_manager()
        self.calibrations: Dict[str, Dict[str, Any]] = {}
        self._generate_calibrations()

    def _generate_calibrations(self):
        """Generate calibrations for all plugins based on outline."""
        logger.info("Generating plugin calibrations from perfect outline...")

        # Get all plugins organized by section
        plugins_by_section = self.outline_manager.get_all_plugins_by_section()

        for section_name, plugins in plugins_by_section.items():
            # Calibrate required plugins
            for plugin_name in plugins["required"]:
                calibration = self.outline_manager.calibrate_plugin_targets(
                    section_name, plugin_name
                )
                self.calibrations[plugin_name] = {
                    **calibration,
                    "priority": "required",
                    "section": section_name,
                }

            # Calibrate optional plugins
            for plugin_name in plugins["optional"]:
                calibration = self.outline_manager.calibrate_plugin_targets(
                    section_name, plugin_name
                )
                self.calibrations[plugin_name] = {
                    **calibration,
                    "priority": "optional",
                    "section": section_name,
                }

        logger.info(f"Generated calibrations for {len(self.calibrations)} plugins")

    def get_calibration(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get calibration data for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Calibration data dictionary
        """
        return self.calibrations.get(plugin_name, {})

    def get_target_value(self, plugin_name: str, section_name: Optional[str] = None) -> Optional[float]:
        """
        Get target value for a plugin based on section requirements.

        Args:
            plugin_name: Name of the plugin
            section_name: Optional section name (if known)

        Returns:
            Target value or None
        """
        calibration = self.get_calibration(plugin_name)

        # Get from calibration if available
        if "min_citations" in calibration:
            return float(calibration["min_citations"])
        elif "min_privacy_mentions" in calibration:
            return float(calibration["min_privacy_mentions"])
        elif "min_harm_types" in calibration:
            return float(calibration["min_harm_types"])
        elif "min_safety_mentions" in calibration:
            return float(calibration["min_safety_mentions"])

        # Default: check section requirements
        if section_name:
            section = self.outline_manager.get_section(section_name)
            if section:
                # Use section-specific defaults
                if "citation" in plugin_name.lower():
                    return 3.0  # Default citations
                elif "privacy" in plugin_name.lower():
                    return 5.0  # Default privacy mentions
                elif "enumeration" in plugin_name.lower():
                    return float(section.enumeration_min_count) if section.enumeration_min_count else 0.0

        return None

    def get_plugins_for_section(self, section_name: str) -> Dict[str, List[str]]:
        """
        Get plugins organized by priority for a section.

        Args:
            section_name: Name of the section

        Returns:
            Dictionary with 'required' and 'optional' plugin lists
        """
        required, optional = self.outline_manager.get_plugins_for_section(section_name)
        return {
            "required": required,
            "optional": optional
        }

    def get_plugin_priority(self, plugin_name: str, section_name: Optional[str] = None) -> str:
        """
        Get priority level for a plugin.

        Args:
            plugin_name: Name of the plugin
            section_name: Optional section name

        Returns:
            Priority level: "required", "optional", or "disabled"
        """
        calibration = self.get_calibration(plugin_name)
        if calibration:
            return calibration.get("priority", "optional")

        # Check if plugin is in any section
        plugins_by_section = self.outline_manager.get_all_plugins_by_section()
        for section, plugins in plugins_by_section.items():
            if plugin_name in plugins["required"]:
                return "required"
            elif plugin_name in plugins["optional"]:
                return "optional"

        return "optional"  # Default

    def should_enable_plugin(self, plugin_name: str, section_name: Optional[str] = None) -> bool:
        """
        Determine if a plugin should be enabled for a section.

        Args:
            plugin_name: Name of the plugin
            section_name: Optional section name

        Returns:
            True if plugin should be enabled
        """
        priority = self.get_plugin_priority(plugin_name, section_name)
        return priority in ["required", "optional"]

    def get_enumeration_requirements(self, section_name: str) -> Dict[str, Any]:
        """
        Get enumeration requirements for a section.

        Args:
            section_name: Name of the section

        Returns:
            Enumeration requirements dictionary
        """
        section = self.outline_manager.get_section(section_name)
        if not section:
            return {}

        return {
            "required": section.enumeration_required,
            "min_count": section.enumeration_min_count,
            "style": section.metadata.get("enumeration_style", "bullet_points"),
            "overall_min_count": self.outline_manager.ENUMERATION_REQUIREMENTS["overall_min_count"]
        }

    def validate_plugin_configuration(self, section_name: str, enabled_plugins: List[str]) -> Dict[str, Any]:
        """
        Validate plugin configuration for a section.

        Args:
            section_name: Name of the section
            enabled_plugins: List of enabled plugin names

        Returns:
            Validation result
        """
        plugins_for_section = self.get_plugins_for_section(section_name)
        required_plugins = plugins_for_section["required"]

        missing_required = [p for p in required_plugins if p not in enabled_plugins]
        extra_plugins = [p for p in enabled_plugins if p not in required_plugins and p not in plugins_for_section["optional"]]

        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "extra_plugins": extra_plugins,
            "recommended": required_plugins + plugins_for_section["optional"],
            "section": section_name
        }

    def get_all_calibrations(self) -> Dict[str, Dict[str, Any]]:
        """Get all plugin calibrations."""
        return self.calibrations.copy()

    def update_plugin_targets(self, plugin_name: str, current_target: float, section_name: Optional[str] = None) -> float:
        """
        Update plugin target based on outline requirements.

        Args:
            plugin_name: Name of the plugin
            current_target: Current target value
            section_name: Optional section name

        Returns:
            Updated target value
        """
        calibration = self.get_calibration(plugin_name)
        if not calibration:
            return current_target

        # Get target from calibration
        target = self.get_target_value(plugin_name, section_name)
        if target is not None:
            # Use calibration target, but ensure it's reasonable
            if target > current_target * 1.5:
                logger.info(f"Updating {plugin_name} target from {current_target} to {target} (based on outline)")
            return max(target, current_target)  # Don't lower targets

        return current_target

