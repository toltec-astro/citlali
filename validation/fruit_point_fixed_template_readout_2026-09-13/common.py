"""Local saved-product dependencies; imports no reduction or fitting implementation."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DIAG=HERE.parent/'fruit_point_peak_failure_diagnosis_2026-09-13'
PRIOR=HERE.parent/'fruit_point_nominal_utility_2026-09-13'
OUT=Path('/private/tmp/sci-fruit-point-nominal-utility-20260913-r0.1')
spec=importlib.util.spec_from_file_location('readout_saved_diagnosis',DIAG/'diagnose.py')
saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)
read,write,digest=saved.read,saved.write,saved.digest
ARRAYS=saved.ARRAYS
SEEDS=saved.SEEDS

def preserve():
    result=saved.preserve()
    m=DIAG/'RESULT_MANIFEST.json';rows=read(m)['files']
    for r in rows:assert digest(m.parent/r['path'])==r['sha256'],r['path']
    result['packets'].append(dict(path=str(m),sha256=digest(m),payloads=len(rows)))
    return result

def verify_freeze():
    f=read(HERE/'FREEZE.json')
    for r in f['files']:assert digest(r['path'])==r['sha256'],r['path']
    return f
