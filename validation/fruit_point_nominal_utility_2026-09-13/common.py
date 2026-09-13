"""Frozen dependencies for the bounded nominal-estimator utility comparison."""
from pathlib import Path
import importlib.util
import json
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent/'fruit_point_operational_gate_trial_2026-09-12'
REPAIR = HERE.parent/'fruit_point_bounded_repair_2026-09-12'
SENSITIVITY = HERE.parent/'fruit_point_feedback_sensitivity_2026-09-13'
PREVIOUS = HERE.parent/'fruit_point_coherent_feedback_2026-09-11'
OLD = Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1')
SAVED = Path('/private/tmp/sci-fruit-point-operational-gates-20260912-r0.1')
OUT = Path('/private/tmp/sci-fruit-point-nominal-utility-20260913-r0.1')
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
b = module('utility_frozen_base', PREVIOUS/'experiment_r02.py')
candidate = module('utility_frozen_starlet', PRIOR/'candidate.py')
sys.modules['candidate'] = candidate
def read(path):
    return json.loads(Path(path).read_text())
def verify_freeze():
    frozen = read(HERE/'FREEZE.json')
    for row in frozen['files']:
        assert b.digest(row['path']) == row['sha256'], row['path']
    return frozen
