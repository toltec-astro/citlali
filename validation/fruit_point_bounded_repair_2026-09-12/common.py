"""Read-only bindings for the saved-map stage of one authorized repair."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE.parent / 'fruit_point_operational_gate_trial_2026-09-12'
BASE = HERE.parent / 'fruit_point_coherent_feedback_2026-09-11'
SAVED = Path('/private/tmp/sci-fruit-point-operational-gates-20260912-r0.1')
NULL = Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1/null20260911/R/pass00_maps.npz')
OUT = Path('/private/tmp/sci-fruit-point-bounded-repair-20260912-r0.1')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

b = module('repair_frozen_base', BASE/'experiment_r02.py')
candidate = module('repair_frozen_starlet', PRIOR/'candidate.py')
sys.modules['candidate'] = candidate

def read(path):
    return json.loads(Path(path).read_text())

def setup():
    with np.load(SAVED/'geometry.npz') as z:
        data = SimpleNamespace(x=z['x'], ygrid=z['y'], shape=tuple(z['shape']),
                               S=z['S'], D=z['D'], O=z['O'], Q=z['Q'])
    with np.load(NULL) as z:
        null = z['total']
    estimators = []
    for a in range(3):
        e = candidate.Estimator(data.x, data.ygrid, data.shape, data.S[a], data.D[a], data.O[a], data.Q[a])
        assert e.calibrate(null[a])['available']
        e.D = e.D & (np.hypot(data.x, data.ygrid) <= 60)
        estimators.append(e)
    return data, estimators

def verify_freeze():
    frozen = read(HERE/'FREEZE.json')
    for row in frozen['files']:
        assert b.digest(row['path']) == row['sha256'], row['path']
    return frozen
