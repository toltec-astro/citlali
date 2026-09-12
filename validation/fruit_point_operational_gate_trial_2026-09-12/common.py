"""Frozen local experimental dependencies; no production imports."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'fruit_point_coherent_feedback_2026-09-11'
OLD = Path('/private/tmp/sci-fruit-point-rbf-feedback-20260911-r0.1')
CENTRAL = HERE.parent / 'fruit_point_starlet_central_domain_2026-09-12'
OUT = Path('/private/tmp/sci-fruit-point-operational-gates-20260912-r0.1')

spec = importlib.util.spec_from_file_location('point_frozen_base', PREVIOUS/'experiment_r02.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def read(path):
    return json.loads(Path(path).read_text())

def verify_freeze():
    frozen = read(HERE/'FREEZE.json')
    for row in frozen['files']:
        if b.digest(row['path']) != row['sha256']:
            raise ValueError('changed frozen source/input: '+row['path'])
    return frozen
