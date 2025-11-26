"""
Simple Revit API Test
Run this to test if Revit API is working
"""

def test_revit_api():
    """Test Revit API connection"""

    print("🔍 Testing Revit API connection...")

    try:
        # Import Revit API
        import clr
        clr.AddReference('RevitAPI')
        clr.AddReference('RevitServices')

        from Autodesk.Revit.DB import *
        from RevitServices.Persistence import DocumentManager

        # Get current document
        doc = DocumentManager.Instance.CurrentDBDocument

        if doc:
            print(f"✅ Connected to Revit!")
            print(f"📄 Document: {doc.Title}")
            print(f"📁 Path: {doc.PathName}")
            return True
        else:
            print("❌ No active Revit document found")
            print("Please open Revit and create/open a project")
            return False

    except ImportError as e:
        print("❌ Revit API not available")
        print(f"Error: {str(e)}")
        print("Make sure Revit is installed and Python.NET is available")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_revit_api()
