"""
Revit Connection Manager

Handles Revit API connections, authentication, and document management.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import winreg

logger = logging.getLogger(__name__)

class RevitConnectionManager:
    """Manages Revit API connections and document access"""

    def __init__(self):
        self.revit_app = None
        self.revit_doc = None
        self.ui_doc = None
        self.is_connected = False
        self.revit_version = None
        self.revit_path = None

    def detect_revit_installation(self) -> Dict[str, Any]:
        """Detect Revit installation on system"""

        logger.info("Detecting Revit installation...")

        revit_info = {
            'installed': False,
            'version': None,
            'path': None,
            'api_available': False
        }

        try:
            # Check registry for Revit installation
            revit_versions = self._check_revit_registry()

            if revit_versions:
                # Get latest version
                latest_version = max(revit_versions.keys())
                revit_info['installed'] = True
                revit_info['version'] = latest_version
                revit_info['path'] = revit_versions[latest_version]['path']
                revit_info['api_available'] = revit_versions[latest_version]['api_available']

                logger.info(f"Found Revit {latest_version} at {revit_info['path']}")
            else:
                logger.warning("No Revit installation detected")

        except Exception as e:
            logger.error(f"Error detecting Revit: {str(e)}")

        return revit_info

    def _check_revit_registry(self) -> Dict[str, Dict[str, Any]]:
        """Check Windows registry for Revit installations"""

        revit_versions = {}

        try:
            # Check common Revit registry locations
            registry_paths = [
                r"SOFTWARE\Autodesk\Revit",
                r"SOFTWARE\WOW6432Node\Autodesk\Revit"
            ]

            for reg_path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        # Enumerate subkeys (versions)
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                if subkey_name.isdigit():  # Version number
                                    version = subkey_name

                                    # Get installation path
                                    with winreg.OpenKey(key, subkey_name) as version_key:
                                        try:
                                            path, _ = winreg.QueryValueEx(version_key, "InstallPath")
                                            revit_versions[version] = {
                                                'path': path,
                                                'api_available': self._check_api_availability(path)
                                            }
                                        except FileNotFoundError:
                                            pass

                                i += 1
                            except OSError:
                                break

                except FileNotFoundError:
                    continue

        except Exception as e:
            logger.error(f"Error checking registry: {str(e)}")

        return revit_versions

    def _check_api_availability(self, revit_path: str) -> bool:
        """Check if Revit API is available"""

        try:
            # Check for RevitAPI.dll
            api_dll = Path(revit_path) / "RevitAPI.dll"
            return api_dll.exists()
        except Exception:
            return False

    def launch_revit(self, version: Optional[str] = None) -> bool:
        """Launch Revit application"""

        try:
            revit_info = self.detect_revit_installation()

            if not revit_info['installed']:
                logger.error("Revit not installed")
                return False

            # Use specified version or latest
            target_version = version or revit_info['version']
            revit_path = revit_info['path']

            # Find Revit executable
            revit_exe = Path(revit_path) / "Revit.exe"
            if not revit_exe.exists():
                logger.error(f"Revit.exe not found at {revit_exe}")
                return False

            # Launch Revit
            logger.info(f"Launching Revit {target_version}...")
            # Launch without shell to avoid command injection and ensure consistent behavior
            subprocess.Popen([str(revit_exe)])

            # Wait for Revit to start
            time.sleep(10)  # Give Revit time to start

            return True

        except Exception as e:
            logger.error(f"Failed to launch Revit: {str(e)}")
            return False

    def connect_to_revit(self, timeout: int = 30) -> bool:
        """Connect to Revit API"""

        logger.info("Connecting to Revit API...")

        try:
            # Import Revit API
            import clr
            clr.AddReference('RevitAPI')
            clr.AddReference('RevitServices')
            clr.AddReference('RevitNodes')

            from Autodesk.Revit.ApplicationServices import Application
            from Autodesk.Revit.DB import Document
            from Autodesk.Revit.UI import UIApplication
            from RevitServices.Persistence import DocumentManager

            # Get Revit application
            self.revit_app = DocumentManager.Instance.CurrentUIApplication
            if not self.revit_app:
                logger.error("No Revit application found")
                return False

            # Get active document
            self.revit_doc = DocumentManager.Instance.CurrentDBDocument
            if not self.revit_doc:
                logger.error("No active Revit document found")
                return False

            self.ui_doc = DocumentManager.Instance.CurrentUIApplication.ActiveUIDocument
            if not self.ui_doc:
                logger.error("No active UI document found")
                return False

            self.is_connected = True
            logger.info(f"Connected to Revit document: {self.revit_doc.Title}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Revit: {str(e)}")
            return False

    def create_new_document(self, template_path: Optional[str] = None) -> bool:
        """Create a new Revit document"""

        if not self.is_connected:
            logger.error("Not connected to Revit")
            return False

        try:
            # Start transaction
            from RevitServices.Transactions import TransactionManager
            TransactionManager.Instance.EnsureInTransaction(self.revit_doc)

            # Create new document
            if template_path and Path(template_path).exists():
                # Use template
                new_doc = self.revit_app.Application.NewDocument(template_path)
            else:
                # Use default template
                new_doc = self.revit_app.Application.NewDocument()

            # Switch to new document
            self.revit_doc = new_doc
            self.ui_doc = self.revit_app.ActiveUIDocument

            TransactionManager.Instance.TransactionTaskDone()

            logger.info("Created new Revit document")
            return True

        except Exception as e:
            logger.error(f"Failed to create new document: {str(e)}")
            return False

    def open_document(self, file_path: str) -> bool:
        """Open existing Revit document"""

        if not self.is_connected:
            logger.error("Not connected to Revit")
            return False

        try:
            # Check if file exists
            if not Path(file_path).exists():
                logger.error(f"File not found: {file_path}")
                return False

            # Open document
            self.revit_doc = self.revit_app.Application.OpenDocumentFile(file_path)
            self.ui_doc = self.revit_app.ActiveUIDocument

            logger.info(f"Opened document: {self.revit_doc.Title}")
            return True

        except Exception as e:
            logger.error(f"Failed to open document: {str(e)}")
            return False

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about current document"""

        if not self.is_connected:
            return {}

        try:
            return {
                'title': self.revit_doc.Title,
                'path': self.revit_doc.PathName,
                'is_family': self.revit_doc.IsFamilyDocument,
                'is_workshared': self.revit_doc.IsWorkshared,
                'units': str(self.revit_doc.DisplayUnitSystem),
                'version': self.revit_doc.VersionNumber
            }
        except Exception as e:
            logger.error(f"Failed to get document info: {str(e)}")
            return {}

    def setup_pagoda_project(self) -> bool:
        """Set up a new project specifically for pagoda visualization"""

        if not self.is_connected:
            logger.error("Not connected to Revit")
            return False

        try:
            # Create new document for pagodas
            if not self.create_new_document():
                return False

            # Set up project settings
            self._setup_project_settings()

            # Create levels for pagodas
            self._create_pagoda_levels()

            # Set up views
            self._create_pagoda_views()

            logger.info("Pagoda project setup complete")
            return True

        except Exception as e:
            logger.error(f"Failed to setup pagoda project: {str(e)}")
            return False

    def _setup_project_settings(self):
        """Set up project settings for pagoda visualization"""

        try:
            # Set units to metric
            from Autodesk.Revit.DB import UnitSystem, DisplayUnitType

            # This would set project units - simplified for now
            pass

        except Exception as e:
            logger.error(f"Failed to setup project settings: {str(e)}")

    def _create_pagoda_levels(self):
        """Create levels for pagoda placement"""

        try:
            from Autodesk.Revit.DB import Level, Transaction
            from RevitServices.Transactions import TransactionManager

            TransactionManager.Instance.EnsureInTransaction(self.revit_doc)

            # Create base level
            base_level = Level.Create(self.revit_doc, 0.0)
            base_level.Name = "Pagoda Base"

            # Create tier levels
            for i in range(5):  # 5 tiers max
                tier_level = Level.Create(self.revit_doc, (i + 1) * 3.0)
                tier_level.Name = f"Pagoda Tier {i + 1}"

            TransactionManager.Instance.TransactionTaskDone()

        except Exception as e:
            logger.error(f"Failed to create pagoda levels: {str(e)}")

    def _create_pagoda_views(self):
        """Create views for pagoda visualization"""

        try:
            from Autodesk.Revit.DB import View3D, ViewFamilyType, ViewOrientation3D, XYZ

            # Create 3D view
            view_family_type = ViewFamilyType.ThreeDimensional
            view_3d = View3D.CreateIsometric(self.revit_doc, view_family_type.Id)
            view_3d.Name = "Pagoda Visualization"

            # Set camera position
            eye = XYZ(50, 50, 30)
            target = XYZ(0, 0, 10)
            up = XYZ.BasisZ

            view_3d.SetOrientation(ViewOrientation3D(eye, target, up))

        except Exception as e:
            logger.error(f"Failed to create pagoda views: {str(e)}")

    def disconnect(self):
        """Disconnect from Revit"""

        if self.is_connected:
            self.revit_app = None
            self.revit_doc = None
            self.ui_doc = None
            self.is_connected = False
            logger.info("Disconnected from Revit")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""

        return {
            'connected': self.is_connected,
            'version': self.revit_version,
            'path': self.revit_path,
            'document': self.get_document_info() if self.is_connected else None
        }

def setup_revit_connection() -> RevitConnectionManager:
    """Set up Revit connection - main function to call"""

    print("🔌 Setting up Revit connection...")

    # Create connection manager
    connection_manager = RevitConnectionManager()

    # Detect Revit installation
    revit_info = connection_manager.detect_revit_installation()

    if not revit_info['installed']:
        print("❌ Revit not installed or not detected")
        print("Please install Autodesk Revit and try again")
        return connection_manager

    print(f"✅ Found Revit {revit_info['version']} at {revit_info['path']}")

    # Try to connect
    if connection_manager.connect_to_revit():
        print("✅ Connected to Revit successfully!")

        # Get document info
        doc_info = connection_manager.get_document_info()
        if doc_info:
            print(f"📄 Active document: {doc_info['title']}")
        else:
            print("⚠️  No active document - creating new one...")
            if connection_manager.setup_pagoda_project():
                print("✅ Created new pagoda project")
            else:
                print("❌ Failed to create new project")
    else:
        print("❌ Failed to connect to Revit")
        print("Make sure Revit is open with an active document")

    return connection_manager

if __name__ == "__main__":
    # Test connection
    connection = setup_revit_connection()

    if connection.is_connected:
        print("🎉 Revit connection successful!")
        print("You can now use the pagoda visualization system")
    else:
        print("❌ Revit connection failed")
        print("Please check your Revit installation and try again")
