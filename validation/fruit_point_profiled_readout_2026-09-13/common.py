"""Saved-map-only dependencies. No reduction or feedback implementation imports."""
from pathlib import Path
import ast
import importlib.util
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d, minimum_filter1d
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIXED=HERE.parent/'fruit_point_fixed_template_readout_2026-09-13'
PRIOR=HERE.parent/'fruit_point_nominal_utility_2026-09-13'
OUT=Path('/private/tmp/sci-fruit-point-nominal-utility-20260913-r0.1')
BASE=HERE.parent/'fruit_point_coherent_feedback_2026-09-11/experiment_r02.py'
EVAL=HERE.parent/'fruit_point_operational_gate_trial_2026-09-12/evaluation.py'
REPAIR=HERE.parent/'fruit_point_bounded_repair_2026-09-12/reevaluate.py'
CANDIDATE=HERE.parent/'fruit_point_operational_gate_trial_2026-09-12/candidate.py'
spec=importlib.util.spec_from_file_location('previous_saved_helpers',FIXED/'common.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
read,write,digest=previous.read,previous.write,previous.digest
ARRAYS=previous.ARRAYS
SEEDS=previous.SEEDS
FWHM=2*np.sqrt(2*np.log(2))
def selected_functions(path,names,env):
    tree=ast.parse(path.read_text())
    nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names]
    assert {v.name for v in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env
base=selected_functions(BASE,['gaussian','model_jac','coherent_start'],dict(np=np,signal=signal,FWHM=FWHM))
gaussian,model_jac,coherent_start=[base[k] for k in ['gaussian','model_jac','coherent_start']]
filters=[]
for level in range(4):
    h=np.zeros(4*2**level+1);h[::2**level]=np.array([1,4,6,4,1])/16;filters.append(h)
wave=selected_functions(CANDIDATE,['smooth','analysis','eligible','mad'],dict(np=np,FILTERS=filters,convolve1d=convolve1d,minimum_filter1d=minimum_filter1d))
analysis,eligible,mad=[wave[k] for k in ['analysis','eligible','mad']]
judgments=selected_functions(REPAIR,['judgments'],dict(np=np))['judgments']
def preserve():
    value=previous.preserve()
    manifest=FIXED/'RESULT_MANIFEST.json'
    rows=read(manifest)['files']
    for r in rows:assert digest(FIXED/r['path'])==r['sha256'],r['path']
    value['packets'].append(dict(path=str(manifest),sha256=digest(manifest),payloads=len(rows)))
    return value
def verify_freeze():
    f=read(HERE/'FREEZE.json')
    for row in f['files']:assert digest(row['path'])==row['sha256'],row['path']
    return f
