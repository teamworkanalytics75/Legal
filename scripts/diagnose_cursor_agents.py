#!/usr/bin/env python3
"""
Diagnostic script for Cursor multi-agent execution issues.

Tests:
- Agent execution environment (paths, env vars)
- API connectivity from agent context
- File system access
- WSL integration status
- Cursor agent execution simulation
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


def check_environment() -> Dict[str, Any]:
    """Check environment variables and system info."""
    print_header("Environment Check")

    results = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "is_wsl": False,
        "workspace_path": None,
        "current_dir": os.getcwd(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "env_vars": {},
        "path_entries": [],
    }

    # Check if running in WSL
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    results["is_wsl"] = True
                    print_success("Running in WSL (Windows Subsystem for Linux)")
        except Exception:
            pass

    if not results["is_wsl"]:
        print_warning("Not running in WSL - agents may have path issues")

    # Check workspace path
    workspace_path = os.getenv("WORKSPACE_PATH") or os.getcwd()
    results["workspace_path"] = workspace_path
    print_info(f"Workspace path: {workspace_path}")
    print_info(f"Current directory: {results['current_dir']}")

    # Check if workspace is in WSL home
    if results["is_wsl"]:
        if workspace_path.startswith("/home/"):
            print_success("Workspace is in WSL home directory (good for agent execution)")
        elif workspace_path.startswith("/mnt/"):
            print_warning("Workspace is on Windows mount (/mnt/) - may be slower")
        else:
            print_warning(f"Workspace path format: {workspace_path}")

    # Check Python version
    print_info(f"Python version: {sys.version.split()[0]}")
    print_info(f"Python executable: {sys.executable}")

    # Check critical environment variables
    critical_vars = [
        "OPENAI_API_KEY",
        "PATH",
        "PYTHONPATH",
        "HOME",
        "USER",
        "SHELL",
        "WORKSPACE_PATH",
    ]

    for var in critical_vars:
        value = os.getenv(var)
        if value:
            if var == "OPENAI_API_KEY":
                # Mask API key
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                results["env_vars"][var] = masked
                print_success(f"{var}: {masked}")
            elif var == "PATH":
                results["env_vars"][var] = value
                path_entries = value.split(os.pathsep)
                results["path_entries"] = path_entries[:10]  # First 10 entries
                print_success(f"{var}: {len(path_entries)} entries")
                print_info(f"  First entries: {', '.join(path_entries[:3])}")
            else:
                results["env_vars"][var] = value
                print_success(f"{var}: {value}")
        else:
            if var == "OPENAI_API_KEY":
                print_error(f"{var}: NOT SET (required for agent execution)")
            elif var in ["HOME", "USER"]:
                print_warning(f"{var}: NOT SET (may cause issues)")
            else:
                print_info(f"{var}: not set")

    return results


def check_api_connectivity() -> Dict[str, Any]:
    """Check API connectivity for agent execution."""
    print_header("API Connectivity Check")

    results = {
        "openai_accessible": False,
        "openai_error": None,
        "ollama_accessible": False,
        "ollama_error": None,
    }

    # Check OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not set - cannot test OpenAI connectivity")
        results["openai_error"] = "API key not set"
    else:
        try:
            import requests
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            if response.status_code == 200:
                print_success("OpenAI API: Accessible")
                results["openai_accessible"] = True
            else:
                print_error(f"OpenAI API: HTTP {response.status_code}")
                results["openai_error"] = f"HTTP {response.status_code}"
        except ImportError:
            print_warning("requests library not installed - cannot test API connectivity")
            results["openai_error"] = "requests library missing"
        except Exception as e:
            print_error(f"OpenAI API: {str(e)}")
            results["openai_error"] = str(e)

    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print_success("Ollama: Accessible")
            results["ollama_accessible"] = True
        else:
            print_warning(f"Ollama: HTTP {response.status_code}")
    except ImportError:
        print_warning("requests library not installed - cannot test Ollama")
    except Exception as e:
        print_info(f"Ollama: Not running or not accessible ({str(e)})")
        results["ollama_error"] = str(e)

    return results


def check_file_system_access() -> Dict[str, Any]:
    """Check file system access patterns agents would use."""
    print_header("File System Access Check")

    results = {
        "workspace_readable": False,
        "workspace_writable": False,
        "can_create_files": False,
        "can_read_files": False,
        "path_issues": [],
    }

    workspace = Path(os.getcwd())

    # Check workspace access
    if workspace.exists():
        print_success(f"Workspace exists: {workspace}")
        results["workspace_readable"] = os.access(workspace, os.R_OK)
        results["workspace_writable"] = os.access(workspace, os.W_OK)

        if results["workspace_readable"]:
            print_success("Workspace is readable")
        else:
            print_error("Workspace is NOT readable")
            results["path_issues"].append("Workspace not readable")

        if results["workspace_writable"]:
            print_success("Workspace is writable")
        else:
            print_error("Workspace is NOT writable")
            results["path_issues"].append("Workspace not writable")
    else:
        print_error(f"Workspace does not exist: {workspace}")
        results["path_issues"].append("Workspace does not exist")

    # Test file creation
    test_file = workspace / ".cursor_agent_test"
    try:
        test_file.write_text("test")
        print_success("Can create files in workspace")
        results["can_create_files"] = True
        test_file.unlink()
    except Exception as e:
        print_error(f"Cannot create files: {e}")
        results["path_issues"].append(f"Cannot create files: {e}")

    # Test file reading
    test_read_file = workspace / "README.md"
    if test_read_file.exists():
        try:
            test_read_file.read_text()
            print_success("Can read files in workspace")
            results["can_read_files"] = True
        except Exception as e:
            print_error(f"Cannot read files: {e}")
            results["path_issues"].append(f"Cannot read files: {e}")
    else:
        print_info("No README.md found to test reading")

    # Check for path issues
    if workspace.as_posix().startswith("/mnt/"):
        print_warning("Workspace on Windows mount - may have permission issues")
        results["path_issues"].append("Workspace on Windows mount")

    return results


def check_wsl_integration() -> Dict[str, Any]:
    """Check WSL integration status."""
    print_header("WSL Integration Check")

    results = {
        "is_wsl": False,
        "wsl_version": None,
        "cursor_wsl_mode": False,
        "terminal_type": None,
    }

    # Check if WSL
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read()
                if "microsoft" in version_info.lower() or "wsl" in version_info.lower():
                    results["is_wsl"] = True
                    print_success("Running in WSL")

                    # Try to detect WSL version
                    if "WSL2" in version_info or "microsoft-standard-WSL2" in version_info:
                        results["wsl_version"] = "WSL2"
                        print_success("WSL2 detected")
                    else:
                        results["wsl_version"] = "WSL1"
                        print_info("WSL1 detected")
        except Exception:
            pass

    if not results["is_wsl"]:
        print_warning("Not running in WSL")
        return results

    # Check terminal type
    term = os.getenv("TERM") or "unknown"
    results["terminal_type"] = term
    print_info(f"Terminal type: {term}")

    # Check if Cursor might be in WSL mode
    # This is heuristic - check for Cursor-specific env vars or paths
    cursor_indicators = [
        "CURSOR" in os.getenv("VSCODE_IPC_HOOK_CLI", ""),
        os.getenv("VSCODE_INJECTION", ""),
    ]

    if any(cursor_indicators):
        print_info("Cursor environment detected")
        results["cursor_wsl_mode"] = True

    # Check workspace path format
    workspace = os.getcwd()
    if workspace.startswith("/home/"):
        print_success("Workspace in WSL home (optimal for agents)")
    elif workspace.startswith("/mnt/"):
        print_warning("Workspace on Windows mount - agents may have issues")
        results["path_issues"] = ["Workspace on Windows mount"]

    return results


def simulate_agent_execution() -> Dict[str, Any]:
    """Simulate what Cursor agents would do."""
    print_header("Agent Execution Simulation")

    results = {
        "can_import_openai": False,
        "can_import_requests": False,
        "can_execute_simple_task": False,
        "execution_error": None,
    }

    # Test imports agents would use
    try:
        import openai
        print_success("Can import openai library")
        results["can_import_openai"] = True
    except ImportError:
        print_error("Cannot import openai library")
        results["execution_error"] = "openai library not installed"

    try:
        import requests
        print_success("Can import requests library")
        results["can_import_requests"] = True
    except ImportError:
        print_error("Cannot import requests library")
        if not results["execution_error"]:
            results["execution_error"] = "requests library not installed"

    # Simulate a simple agent task
    if results["can_import_openai"] and os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Just test if we can create a client, don't make actual API call
            print_success("Can create OpenAI client")
            results["can_execute_simple_task"] = True
        except Exception as e:
            print_error(f"Cannot create OpenAI client: {e}")
            results["execution_error"] = str(e)

    return results


def check_cursor_config() -> Dict[str, Any]:
    """Check Cursor/VS Code configuration."""
    print_header("Cursor Configuration Check")

    results = {
        "settings_file_exists": False,
        "settings_path": None,
        "wsl_settings": {},
    }

    workspace = Path(os.getcwd())
    settings_file = workspace / ".vscode" / "settings.json"

    if settings_file.exists():
        print_success(f"Settings file exists: {settings_file}")
        results["settings_file_exists"] = True
        results["settings_path"] = str(settings_file)

        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)

            # Check for WSL-related settings
            wsl_keys = [
                "remote.WSL.fileWatcher.polling",
                "remote.WSL.useShellEnvironment",
                "terminal.integrated.defaultProfile.linux",
                "python.defaultInterpreterPath",
            ]

            for key in wsl_keys:
                if key in settings:
                    results["wsl_settings"][key] = settings[key]
                    print_info(f"  {key}: {settings[key]}")

            if not results["wsl_settings"]:
                print_warning("No WSL-specific settings found")
        except Exception as e:
            print_error(f"Cannot read settings file: {e}")
    else:
        print_warning(f"Settings file not found: {settings_file}")

    return results


def generate_report(all_results: Dict[str, Any]) -> str:
    """Generate a diagnostic report."""
    print_header("Diagnostic Report Summary")

    issues = []
    warnings = []

    # Check environment
    env = all_results.get("environment", {})
    if not env.get("is_wsl") and platform.system() == "Linux":
        warnings.append("Not running in WSL - may cause path issues")
    if not env.get("env_vars", {}).get("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set - agents cannot execute")

    # Check API connectivity
    api = all_results.get("api_connectivity", {})
    if not api.get("openai_accessible"):
        issues.append(f"OpenAI API not accessible: {api.get('openai_error', 'Unknown error')}")

    # Check file system
    fs = all_results.get("file_system", {})
    if fs.get("path_issues"):
        issues.extend(fs["path_issues"])
    if not fs.get("can_create_files"):
        issues.append("Cannot create files in workspace")

    # Check agent execution
    agent = all_results.get("agent_execution", {})
    if not agent.get("can_execute_simple_task"):
        issues.append(f"Agent execution simulation failed: {agent.get('execution_error', 'Unknown')}")

    # Print summary
    if issues:
        print_error(f"Found {len(issues)} critical issue(s):")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print_success("No critical issues found!")

    if warnings:
        print_warning(f"Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  • {warning}")

    # Recommendations
    print_header("Recommendations")

    if not env.get("is_wsl"):
        print("1. Ensure Cursor is connected to WSL (Ctrl+Shift+P → 'WSL: Connect to WSL')")

    if not env.get("env_vars", {}).get("OPENAI_API_KEY"):
        print("2. Set OPENAI_API_KEY environment variable in WSL")
        print("   Run: export OPENAI_API_KEY='your-key-here'")
        print("   Add to ~/.bashrc for persistence")

    if fs.get("path_issues") and any("Windows mount" in str(issue) for issue in fs["path_issues"]):
        print("3. Consider moving workspace to WSL home directory for better performance")
        print("   Current: /mnt/c/...")
        print("   Recommended: /home/serteamwork/projects/TheMatrix")

    if not agent.get("can_import_openai"):
        print("4. Install required Python packages:")
        print("   pip install openai requests")

    return json.dumps(all_results, indent=2)


def main():
    """Run all diagnostic checks."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print("Cursor Multi-Agent Diagnostic Tool")
    print("="*70)
    print(f"{Colors.RESET}\n")

    all_results = {}

    # Run all checks
    all_results["environment"] = check_environment()
    all_results["api_connectivity"] = check_api_connectivity()
    all_results["file_system"] = check_file_system_access()
    all_results["wsl_integration"] = check_wsl_integration()
    all_results["agent_execution"] = simulate_agent_execution()
    all_results["cursor_config"] = check_cursor_config()

    # Generate report
    report = generate_report(all_results)

    # Save report
    report_file = Path(os.getcwd()) / ".cursor_agent_diagnostic.json"
    try:
        report_file.write_text(report)
        print_success(f"\nFull diagnostic report saved to: {report_file}")
    except Exception as e:
        print_error(f"Cannot save report: {e}")

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

