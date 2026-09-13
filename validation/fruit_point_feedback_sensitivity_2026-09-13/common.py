"""Exact saved models and existing diagnostic feedback path; no new estimator."""
from pathlib import Path
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPAIR = HERE.parent/'fruit_point_bounded_repair_2026-09-12'
PRIOR = HERE.parent/'fruit_point_operational_gate_trial_2026-09-12'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

old = module('sensitivity_saved_common', REPAIR/'common.py')
b, candidate = old.b, old.candidate
SAVED, SOLUTIONS, OLD = old.SAVED, old.OUT, old.NULL.parents[2]
OUT = Path('/private/tmp/sci-fruit-point-feedback-sensitivity-20260913-r0.1')
evaluation = module('sensitivity_repaired_evaluator', REPAIR/'reevaluate.py')

def forbidden(*_, **__):
    raise RuntimeError('new feedback optimization or inference is outside this experiment')

candidate.minimize = forbidden
b.inference = forbidden

def read(path):
    return json.loads(Path(path).read_text())

def initialize():
    data = b.Data(read(old.BASE/'INPUT_PREFLIGHT.json'))
    geometry, estimators = old.setup()
    for a,e in enumerate(estimators):
        e.array=a
    for key in ['x','ygrid','S','D','O','Q']:
        np.testing.assert_array_equal(getattr(data,key),getattr(geometry,key))
    with np.load(SAVED/'geometry.npz') as z:
        for key, value in [('uid',data.uid),('valid',data.valid),('edges',data.edges)]:
            np.testing.assert_array_equal(value,z[key])
    return data, estimators

def state_rows():
    return [r for r in read(SAVED/'RECEIPTS.json') if r['arm']=='C']

def models(data, row, pairs):
    n, t = np.zeros((3,data.npix)), np.zeros((3,data.npix))
    present = []
    for a, array in enumerate(b.ARRAYS):
        p = pairs[row['case'],row['pass_index'],array]
        with np.load(SOLUTIONS/'stopping'/(p['name']+'.npz')) as z:
            ok = 'declared' in z and 'tight' in z
            if ok:
                n[a],t[a] = z['declared'],z['tight']
        present.append(ok)
    return n,t,present

def source_core(data, e, original_measurement):
    fit = original_measurement.get('fit')
    if fit is None or fit['peak'] <= 0 or not np.isfinite(fit['parameters']).all():
        return np.zeros(data.npix,bool)
    G = b.gaussian(fit['parameters'],data.x,data.ygrid)
    return e.D & (G >= .1*fit['peak'])

def verify_freeze():
    for row in read(HERE/'FREEZE.json')['files']:
        assert b.digest(row['path'])==row['sha256'],row['path']

def array_hash(v):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
