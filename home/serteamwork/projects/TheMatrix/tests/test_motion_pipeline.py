import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

CALIBRATION_DIR = Path('Agents_1782_ML_Dataset/ml_system/outputs/calibration')
SHAP_EXPORT = Path('case_law_data/analysis/motion_seal_shap_latest.json')
LOGISTIC_MODEL = Path('ml_system/models/trained/outcome_predictor/logistic_model.joblib')


def _latest_calibration_summary() -> Path:
    summaries = sorted(CALIBRATION_DIR.glob('calibration_summary_*.json'))
    if not summaries:
        raise FileNotFoundError('No calibration summaries available')
    return summaries[-1]


@pytest.mark.unit
def test_calibrated_model_and_shap_export_are_healthy(tmp_path):
    assert LOGISTIC_MODEL.exists(), 'Calibrated logistic model missing'
    model = joblib.load(LOGISTIC_MODEL)
    assert model is not None, 'Unable to load calibrated logistic model'

    summary_path = _latest_calibration_summary()
    summary = json.loads(summary_path.read_text())
    logistic_metrics = summary.get('models', {}).get('logistic', {})
    ece = float(logistic_metrics.get('ece', 1.0))
    brier = float(logistic_metrics.get('brier', 1.0))
    assert ece < 0.05, f'ECE too high: {ece}'
    assert brier < 0.05, f'Brier too high: {brier}'

    assert SHAP_EXPORT.exists(), 'SHAP export missing'
    shap_payload = json.loads(SHAP_EXPORT.read_text())
    top_features = shap_payload.get('top_features_shap', [])
    assert top_features, 'SHAP export lacks feature importance'

    env = os.environ.copy()
    env['MOTION_TEST_MODE'] = '1'
    cmd = [
        sys.executable,
        'writer_agents/generate_full_motion_to_seal.py',
        '--dry-run',
        '--local-llm',
    ]
    subprocess.run(cmd, check=True, env=env)
