#!/usr/bin/env python3
"""Generate model/artifact inventory with experiment metadata."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

MODEL_EXTENSIONS: Sequence[str] = (".cbm", ".joblib")
MODEL_ROOTS: Sequence[Path] = (Path("case_law_data/models"), Path("ml_system/models/trained"))
CALIBRATION_DIR = Path("Agents_1782_ML_Dataset/ml_system/outputs/calibration")
HPO_DIR = Path("Agents_1782_ML_Dataset/ml_system/outputs/hpo")
EXPERIMENT_DIR = Path("Agents_1782_ML_Dataset/ml_system/outputs/experiments")
SHAP_PATH = Path("case_law_data/analysis/motion_seal_shap_latest.json")

INVENTORY_FIELDS = [
    "artifact_name",
    "artifact_path",
    "size_bytes",
    "modified_iso",
    "artifact_type",
    "calibration_method",
    "ece",
    "brier",
    "notes",
    "priority_rank",
    "experiment_id",
    "hpo_best_params",
    "hpo_best_f1",
    "model_accuracy",
    "model_f1",
]


@dataclass
class ArtifactRecord:
    artifact_name: str
    artifact_path: str
    size_bytes: int
    modified_iso: str
    artifact_type: str
    calibration_method: str = ""
    ece: str = ""
    brier: str = ""
    notes: str = ""
    priority_rank: str = ""
    experiment_id: str = ""
    hpo_best_params: str = ""
    hpo_best_f1: str = ""
    model_accuracy: str = ""
    model_f1: str = ""

    def apply_metadata(self, meta: Dict[str, str]) -> None:
        if not meta:
            return
        for field in (
            "experiment_id",
            "hpo_best_params",
            "hpo_best_f1",
            "model_accuracy",
            "model_f1",
            "calibration_method",
            "ece",
            "brier",
        ):
            value = meta.get(field)
            if value:
                setattr(self, field, value)


def _priority_for(filename: str) -> str:
    name = filename.lower()
    if name == "catboost_outline_unified.cbm":
        return "1"
    if name.startswith("catboost_outline_both"):
        return "2"
    if name == "catboost_motion_seal_pseudonym.cbm":
        return "3"
    if name == "catboost_motion.cbm":
        return "4"
    if name == "section_1782_discovery_model.cbm":
        return "5"
    if name == "logistic_model.joblib":
        return "6"
    return ""


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_record(path: Path, artifact_type: str, notes: str = "", priority: str = "") -> ArtifactRecord:
    stat = path.stat()
    return ArtifactRecord(
        artifact_name=path.name,
        artifact_path=str(path),
        size_bytes=stat.st_size,
        modified_iso=_utc_iso(stat.st_mtime),
        artifact_type=artifact_type,
        notes=notes,
        priority_rank=priority,
    )


def collect_model_records() -> List[ArtifactRecord]:
    records: List[ArtifactRecord] = []
    for root in MODEL_ROOTS:
        if not root.exists():
            continue
        for ext in MODEL_EXTENSIONS:
            for file in root.rglob(f"*{ext}"):
                records.append(
                    _make_record(
                        file,
                        artifact_type="model",
                        priority=_priority_for(file.name),
                    )
                )
    return records


def collect_calibration_records() -> List[ArtifactRecord]:
    if not CALIBRATION_DIR.exists():
        return []
    records: List[ArtifactRecord] = []
    for file in sorted(CALIBRATION_DIR.glob("calibration_summary_*.json")):
        try:
            data = json.loads(file.read_text())
        except Exception:
            data = {}
        logistic_metrics = (
            data.get("models", {}).get("logistic", {}) if isinstance(data, dict) else {}
        )
        record = _make_record(
            file,
            artifact_type="calibration_metadata",
            notes="Calibration summary",
        )
        record.calibration_method = data.get("method", "")
        record.ece = str(logistic_metrics.get("ece", ""))
        record.brier = str(logistic_metrics.get("brier", ""))
        records.append(record)
    return records


def collect_hpo_records() -> List[ArtifactRecord]:
    if not HPO_DIR.exists():
        return []
    records: List[ArtifactRecord] = []
    for file in sorted(HPO_DIR.glob("hpo_summary_*.json")):
        try:
            data = json.loads(file.read_text())
        except Exception:
            data = {}
        record = _make_record(
            file,
            artifact_type="hpo_summary",
            notes="Optuna HPO summary",
        )
        best_params = data.get("best_params")
        if best_params:
            record.hpo_best_params = json.dumps(best_params)
        best_value = data.get("best_value")
        if best_value is not None:
            record.hpo_best_f1 = str(best_value)
        records.append(record)
    return records


def collect_shap_record() -> List[ArtifactRecord]:
    if not SHAP_PATH.exists():
        return []
    record = _make_record(
        SHAP_PATH,
        artifact_type="shap_export",
        notes="CatBoost SHAP export",
    )
    return [record]


def load_experiment_metadata() -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not EXPERIMENT_DIR.exists():
        return mapping
    for log_file in sorted(EXPERIMENT_DIR.glob("experiment_*.json")):
        try:
            data = json.loads(log_file.read_text())
        except Exception:
            continue
        exp_id = data.get("experiment_id", "")
        metrics = data.get("metrics", {})
        calibration = metrics.get("calibration") or {}
        logistic_cal = (
            calibration.get("models", {}).get("logistic", {}) if isinstance(calibration, dict) else {}
        )
        hpo = metrics.get("hyperparameter_tuning") or {}
        perf = metrics.get("model_performance") or {}
        logistic_perf = perf.get("logistic") or {}

        meta_common = {
            "experiment_id": exp_id,
            "calibration_method": calibration.get("method", "") if isinstance(calibration, dict) else "",
            "ece": str(logistic_cal.get("ece", "")),
            "brier": str(logistic_cal.get("brier", "")),
            "hpo_best_params": json.dumps(hpo.get("best_params")) if hpo.get("best_params") else "",
            "hpo_best_f1": str(hpo.get("best_value", "")) if hpo.get("best_value") is not None else "",
            "model_accuracy": str(logistic_perf.get("accuracy", "")),
            "model_f1": str(logistic_perf.get("f1_score", "")),
        }

        artifacts = data.get("artifacts", {}) or {}
        for path in artifacts.get("models", []) or []:
            mapping[str(Path(path))] = meta_common
        for special_key in ("calibration_summary", "hpo_summary", "shap_export"):
            special_path = artifacts.get(special_key)
            if special_path:
                mapping[str(Path(special_path))] = meta_common
    return mapping


def write_inventory_csv(records: List[ArtifactRecord], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "artifact_name": record.artifact_name,
                "artifact_path": record.artifact_path,
                "size_bytes": record.size_bytes,
                "modified_iso": record.modified_iso,
                "artifact_type": record.artifact_type,
                "calibration_method": record.calibration_method,
                "ece": record.ece,
                "brier": record.brier,
                "notes": record.notes,
                "priority_rank": record.priority_rank,
                "experiment_id": record.experiment_id,
                "hpo_best_params": record.hpo_best_params,
                "hpo_best_f1": record.hpo_best_f1,
                "model_accuracy": record.model_accuracy,
                "model_f1": record.model_f1,
            })


def write_long_listing(records: List[ArtifactRecord], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                f"{r.artifact_path}\n"
                f"  size: {r.size_bytes}\n"
                f"  modified_utc: {r.modified_iso}\n"
                f"  type: {r.artifact_type}\n"
                f"  experiment: {r.experiment_id}\n\n"
            )


def flag_stale_models(records: List[ArtifactRecord], max_days: int = 30) -> None:
    threshold = datetime.now(timezone.utc).timestamp() - max_days * 86400
    for r in records:
        if r.artifact_type != "model":
            continue
        ts = datetime.strptime(r.modified_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if ts < threshold:
            print(f"[DRIFT] {r.artifact_name} is older than {max_days} days ({r.modified_iso})")


def apply_metadata(records: List[ArtifactRecord], metadata: Dict[str, Dict[str, str]]) -> None:
    for record in records:
        meta = metadata.get(record.artifact_path)
        if not meta:
            continue
        record.apply_metadata(meta)


def main() -> None:
    outdir = Path("reports/analysis_outputs")
    inventory_csv = outdir / "model_artifacts_inventory.csv"
    long_listing = outdir / "model_artifacts_long_listing.txt"

    metadata_index = load_experiment_metadata()
    records = []
    records.extend(collect_model_records())
    records.extend(collect_calibration_records())
    records.extend(collect_hpo_records())
    records.extend(collect_shap_record())

    apply_metadata(records, metadata_index)
    records.sort(key=lambda r: (r.artifact_type, r.artifact_name))

    write_inventory_csv(records, inventory_csv)
    write_long_listing(records, long_listing)
    flag_stale_models(records)

    print(f"Wrote inventory: {inventory_csv}")
    print(f"Wrote long listing: {long_listing}")


if __name__ == "__main__":
    main()
