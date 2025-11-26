#!/usr/bin/env python3
"""
Unified CLI to run training and explainability scripts with optional MLflow logging.

Usage:
  python scripts/ml_train_explain_cli.py --task comprehensive    # 08 ComprehensiveSuccessAnalysis
  python scripts/ml_train_explain_cli.py --task ma_motion        # 09 MaMotionAnalysis
  python scripts/ml_train_explain_cli.py --task ma_federal       # 10 MaFederalMotionAnalysis
  python scripts/ml_train_explain_cli.py --task sota             # 07 TrainSotaModels (auto)

Options:
  --enable-mlflow / --disable-mlflow

Notes:
  - This CLI delegates to existing scripts in Agents_1782_ML_Dataset/scripts/.
  - It sets ENABLE_MLFLOW environment variable to control logging.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "Agents_1782_ML_Dataset" / "scripts"


def run_script(script_name: str, args: list[str] | None = None) -> int:
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        return 1
    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ML training/explain scripts")
    parser.add_argument(
        "--task",
        choices=["comprehensive", "ma_motion", "ma_federal", "sota"],
        required=True,
        help="Which workflow to run",
    )
    parser.add_argument("--enable-mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--sota-mode", choices=["auto", "tiny", "small", "medium"], default="auto")

    args = parser.parse_args()

    if args.enable_mlflow and args.disable_mlflow:
        print("Cannot set both --enable-mlflow and --disable-mlflow")
        return 2

    # Control MLflow via env var
    if args.enable_mlflow:
        os.environ["ENABLE_MLFLOW"] = "1"
    if args.disable_mlflow:
        os.environ["ENABLE_MLFLOW"] = "0"

    if args.task == "comprehensive":
        return run_script("08ComprehensiveSuccessAnalysis.py")
    if args.task == "ma_motion":
        return run_script("09MaMotionAnalysis.py")
    if args.task == "ma_federal":
        return run_script("10MaFederalMotionAnalysis.py")
    if args.task == "sota":
        return run_script("07TrainSotaModels.py", ["--mode", args.sota_mode])

    print("Unknown task")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

