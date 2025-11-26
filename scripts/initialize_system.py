#!/usr/bin/env python3
"""
System Initialization Script

Initializes all core components:
1. Motion to Seal and Pseudonym functionality
2. CatBoost/SHAP analysis setup
3. Semantic Kernel with local LLM configuration
4. AutoGen with local LLM support
5. Refinement loop system

Usage:
    python scripts/initialize_system.py [--verify-only] [--skip-catboost] [--skip-llm-check]
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))


class SystemInitializer:
    """Unified system initialization for all components."""
    
    def __init__(self, verify_only: bool = False, skip_catboost: bool = False, skip_llm_check: bool = False):
        self.verify_only = verify_only
        self.skip_catboost = skip_catboost
        self.skip_llm_check = skip_llm_check
        self.project_root = project_root
        self.results: Dict[str, Any] = {
            "motion_to_seal": {},
            "catboost_shap": {},
            "semantic_kernel": {},
            "autogen": {},
            "refinement_loop": {}
        }
    
    def initialize_all(self) -> Dict[str, Any]:
        """Initialize all components."""
        logger.info("=" * 80)
        logger.info("SYSTEM INITIALIZATION")
        logger.info("=" * 80)
        
        try:
            # 1. Motion to Seal and Pseudonym
            logger.info("\n📋 Initializing Motion to Seal/Pseudonym functionality...")
            self.results["motion_to_seal"] = self._init_motion_to_seal()
            
            # 2. CatBoost/SHAP
            if not self.skip_catboost:
                logger.info("\n📊 Initializing CatBoost/SHAP analysis...")
                self.results["catboost_shap"] = self._init_catboost_shap()
            else:
                logger.info("\n⏭️  Skipping CatBoost/SHAP initialization")
                self.results["catboost_shap"] = {"status": "skipped"}
            
            # 3. Semantic Kernel
            logger.info("\n🧠 Initializing Semantic Kernel...")
            self.results["semantic_kernel"] = self._init_semantic_kernel()
            
            # 4. AutoGen
            logger.info("\n🤖 Initializing AutoGen with local LLMs...")
            self.results["autogen"] = self._init_autogen()
            
            # 5. Refinement Loop
            logger.info("\n🔄 Initializing Refinement Loop...")
            self.results["refinement_loop"] = self._init_refinement_loop()
            
            # Summary
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.results["error"] = str(e)
            return self.results
    
    def _init_motion_to_seal(self) -> Dict[str, Any]:
        """Initialize motion to seal/pseudonym functionality."""
        result = {"status": "success", "components": []}
        
        try:
            # Check for motion generation scripts
            motion_scripts = [
                "create_motion_for_seal.py",
                "create_motion_local.py",
                "writer_agents/generate_full_motion_to_seal.py"
            ]
            
            found_scripts = []
            for script in motion_scripts:
                script_path = self.project_root / script
                if script_path.exists():
                    found_scripts.append(str(script_path))
                    result["components"].append(f"Found: {script}")
            
            if found_scripts:
                logger.info(f"✅ Found {len(found_scripts)} motion generation scripts")
            else:
                logger.warning("⚠️  No motion generation scripts found")
                result["status"] = "warning"
                result["message"] = "Motion scripts not found"
            
            # Check for motion templates/content
            motion_content_paths = [
                "complete_motion_to_seal.txt",
                "motion_for_seal_pseudonym_draft_*.txt"
            ]
            
            # Verify motion classification functionality
            try:
                from case_law_data.scripts.analyze_only_seal_pseudonym_motions import is_seal_pseudonym_motion
                result["components"].append("Motion classification function available")
                logger.info("✅ Motion classification function available")
            except ImportError as e:
                logger.warning(f"⚠️  Motion classification not available: {e}")
                result["status"] = "warning"
            
            return result
            
        except Exception as e:
            logger.error(f"Motion to seal initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _init_catboost_shap(self) -> Dict[str, Any]:
        """Initialize CatBoost/SHAP analysis setup."""
        result = {"status": "success", "components": []}
        
        try:
            # Check CatBoost installation
            try:
                import catboost
                result["components"].append(f"CatBoost {catboost.__version__} installed")
                logger.info(f"✅ CatBoost {catboost.__version__} available")
            except ImportError:
                logger.error("❌ CatBoost not installed. Install with: pip install catboost")
                result["status"] = "error"
                result["error"] = "CatBoost not installed"
                return result
            
            # Check SHAP installation
            try:
                import shap
                result["components"].append(f"SHAP {shap.__version__} installed")
                logger.info(f"✅ SHAP {shap.__version__} available")
            except ImportError:
                logger.error("❌ SHAP not installed. Install with: pip install shap")
                result["status"] = "error"
                result["error"] = "SHAP not installed"
                return result
            
            # Check for CatBoost models
            model_paths = [
                "models/",
                "Agents_1782_ML_Dataset/ml_system/models/",
                "case_law_data/models/"
            ]
            
            found_models = []
            for model_path in model_paths:
                full_path = self.project_root / model_path
                if full_path.exists():
                    # Look for .cbm files (CatBoost model format)
                    cbm_files = list(full_path.glob("*.cbm"))
                    if cbm_files:
                        found_models.extend([str(f) for f in cbm_files])
            
            if found_models:
                result["components"].append(f"Found {len(found_models)} CatBoost model(s)")
                logger.info(f"✅ Found {len(found_models)} CatBoost model(s)")
            else:
                logger.warning("⚠️  No CatBoost models found (models will need to be trained)")
                result["components"].append("No models found (training required)")
            
            # Check for SHAP analysis scripts
            shap_scripts = [
                "case_law_data/scripts/run_catboost_shap_all_corpora.py",
                "case_law_data/analysis_tools/permutation_importance_analyzer.py"
            ]
            
            found_shap_scripts = []
            for script in shap_scripts:
                script_path = self.project_root / script
                if script_path.exists():
                    found_shap_scripts.append(str(script_path))
            
            if found_shap_scripts:
                result["components"].append(f"Found {len(found_shap_scripts)} SHAP analysis script(s)")
                logger.info(f"✅ Found {len(found_shap_scripts)} SHAP analysis script(s)")
            
            # Verify CatBoostPredictor
            try:
                from writer_agents.code.sk_plugins.FeaturePlugin.catboost_predictor import CatBoostPredictor
                result["components"].append("CatBoostPredictor class available")
                logger.info("✅ CatBoostPredictor class available")
            except ImportError as e:
                logger.warning(f"⚠️  CatBoostPredictor not available: {e}")
            
            # Verify SHAPAnalyzer
            try:
                from writer_agents.code.sk_plugins.FeaturePlugin.shap_analyzer import SHAPAnalyzer
                result["components"].append("SHAPAnalyzer class available")
                logger.info("✅ SHAPAnalyzer class available")
            except ImportError as e:
                logger.warning(f"⚠️  SHAPAnalyzer not available: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"CatBoost/SHAP initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _init_semantic_kernel(self) -> Dict[str, Any]:
        """Initialize Semantic Kernel with local LLM configuration."""
        result = {"status": "success", "components": []}
        
        try:
            # Check Semantic Kernel installation
            try:
                import semantic_kernel
                result["components"].append(f"Semantic Kernel installed")
                logger.info("✅ Semantic Kernel available")
            except ImportError:
                logger.error("❌ Semantic Kernel not installed")
                result["status"] = "error"
                result["error"] = "Semantic Kernel not installed"
                return result
            
            # Check Ollama connector
            try:
                from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
                result["components"].append("Ollama connector available")
                logger.info("✅ Ollama connector available")
            except ImportError:
                logger.warning("⚠️  Ollama connector not available (will use OpenAI fallback)")
                result["components"].append("Ollama connector not available")
            
            # Initialize SK config
            try:
                from writer_agents.code.sk_config import (
                    SKConfig, create_sk_kernel, get_default_sk_config
                )
                result["components"].append("SK configuration module available")
                logger.info("✅ SK configuration module available")
                
                # Test kernel creation (if not verify-only)
                if not self.verify_only:
                    try:
                        config = get_default_sk_config(use_local=True)
                        result["components"].append(f"Default config: local_model={config.local_model}")
                        logger.info(f"✅ Default SK config: local_model={config.local_model}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not create default config: {e}")
                
            except ImportError as e:
                logger.error(f"❌ SK configuration not available: {e}")
                result["status"] = "error"
                result["error"] = str(e)
                return result
            
            # Check local LLM availability (if not skipped)
            if not self.skip_llm_check:
                ollama_available = self._check_ollama_server()
                result["ollama_available"] = ollama_available
                if ollama_available:
                    result["components"].append("Ollama server is running")
                    logger.info("✅ Ollama server is running")
                else:
                    logger.warning("⚠️  Ollama server not running (start with: ollama serve)")
                    result["components"].append("Ollama server not running")
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic Kernel initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _init_autogen(self) -> Dict[str, Any]:
        """Initialize AutoGen with local LLM support."""
        result = {"status": "success", "components": []}
        
        try:
            # Check AutoGen installation
            try:
                from autogen_agentchat.agents import AssistantAgent
                result["components"].append("AutoGen agentchat available")
                logger.info("✅ AutoGen agentchat available")
            except ImportError:
                try:
                    from autogen import AssistantAgent
                    result["components"].append("AutoGen (legacy) available")
                    logger.info("✅ AutoGen (legacy) available")
                except ImportError:
                    logger.error("❌ AutoGen not installed")
                    result["status"] = "error"
                    result["error"] = "AutoGen not installed"
                    return result
            
            # Check Ollama client
            try:
                from ollama_client import OllamaConfig, OllamaChatClient, check_ollama_server
                result["components"].append("Ollama client available")
                logger.info("✅ Ollama client available")
            except ImportError:
                logger.warning("⚠️  Ollama client not available (will use OpenAI fallback)")
                result["components"].append("Ollama client not available")
            
            # Check AgentFactory
            try:
                from writer_agents.code.agents import AgentFactory, ModelConfig
                result["components"].append("AgentFactory available")
                logger.info("✅ AgentFactory available")
                
                # Test factory creation (if not verify-only)
                if not self.verify_only:
                    try:
                        config = ModelConfig(use_local=True, local_model="qwen2.5:14b")
                        factory = AgentFactory(config)
                        result["components"].append(f"AgentFactory initialized: local_model={config.local_model}")
                        logger.info(f"✅ AgentFactory initialized: local_model={config.local_model}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not create AgentFactory: {e}")
                
            except ImportError as e:
                logger.warning(f"⚠️  AgentFactory not available: {e}")
                result["components"].append("AgentFactory not available")
            
            # Check local LLM availability (if not skipped)
            if not self.skip_llm_check:
                ollama_available = self._check_ollama_server()
                result["ollama_available"] = ollama_available
                if ollama_available:
                    result["components"].append("Ollama server is running")
                    logger.info("✅ Ollama server is running")
                else:
                    logger.warning("⚠️  Ollama server not running")
                    result["components"].append("Ollama server not running")
            
            return result
            
        except Exception as e:
            logger.error(f"AutoGen initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _init_refinement_loop(self) -> Dict[str, Any]:
        """Initialize refinement loop system."""
        result = {"status": "success", "components": []}
        
        try:
            # Check RefinementLoop class
            try:
                from writer_agents.code.sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop
                result["components"].append("RefinementLoop class available")
                logger.info("✅ RefinementLoop class available")
            except ImportError as e:
                logger.error(f"❌ RefinementLoop not available: {e}")
                result["status"] = "error"
                result["error"] = str(e)
                return result
            
            # Check FeatureExtractor
            try:
                from writer_agents.code.sk_plugins.FeaturePlugin.feature_extractor import FeatureExtractor
                result["components"].append("FeatureExtractor available")
                logger.info("✅ FeatureExtractor available")
            except ImportError as e:
                logger.warning(f"⚠️  FeatureExtractor not available: {e}")
            
            # Check CatBoostPredictor (already checked in catboost_shap, but verify here too)
            try:
                from writer_agents.code.sk_plugins.FeaturePlugin.catboost_predictor import CatBoostPredictor
                result["components"].append("CatBoostPredictor available for refinement")
                logger.info("✅ CatBoostPredictor available for refinement")
            except ImportError as e:
                logger.warning(f"⚠️  CatBoostPredictor not available: {e}")
            
            # Check Conductor (main orchestrator that uses RefinementLoop)
            try:
                from writer_agents.code.WorkflowOrchestrator import Conductor
                result["components"].append("Conductor (main orchestrator) available")
                logger.info("✅ Conductor (main orchestrator) available")
            except ImportError as e:
                logger.warning(f"⚠️  Conductor not available: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Refinement loop initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_ollama_server(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _print_summary(self):
        """Print initialization summary."""
        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZATION SUMMARY")
        logger.info("=" * 80)
        
        for component, result in self.results.items():
            if component == "error":
                continue
            
            status = result.get("status", "unknown")
            status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
            
            logger.info(f"\n{status_icon} {component.upper().replace('_', ' ')}: {status}")
            
            if "components" in result:
                for comp in result["components"]:
                    logger.info(f"   • {comp}")
            
            if "error" in result:
                logger.error(f"   Error: {result['error']}")
        
        logger.info("\n" + "=" * 80)
        
        # Check for critical errors
        has_errors = any(
            r.get("status") == "error" 
            for r in self.results.values() 
            if isinstance(r, dict)
        )
        
        if has_errors:
            logger.warning("\n⚠️  Some components failed to initialize. Check errors above.")
        else:
            logger.info("\n✅ All components initialized successfully!")
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save initialization results to JSON file."""
        if output_path is None:
            output_path = self.project_root / "reports" / "system_initialization.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize system components")
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify components, don't initialize")
    parser.add_argument("--skip-catboost", action="store_true",
                       help="Skip CatBoost/SHAP initialization")
    parser.add_argument("--skip-llm-check", action="store_true",
                       help="Skip LLM server availability check")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    initializer = SystemInitializer(
        verify_only=args.verify_only,
        skip_catboost=args.skip_catboost,
        skip_llm_check=args.skip_llm_check
    )
    
    results = initializer.initialize_all()
    
    if args.save_results:
        initializer.save_results()
    
    # Exit with error code if any component failed
    has_errors = any(
        r.get("status") == "error" 
        for r in results.values() 
        if isinstance(r, dict)
    )
    
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()

