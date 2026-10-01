"""Read saved products and exercise existing LMTOOF reader; never reduce or fit."""
import os,sys,json,subprocess
from pathlib import Path
import numpy as np,netCDF4
from astropy.io import fits
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from oof_saved_interface import attach_historical,require_lm_sequence
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from inspect_forward_map import require
L=Path.home()/'work_toltec/local_data';LM=Path.home()/'GitHub/LMTOOF';D=Path(__file__).parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
sources=[LM/'C++'/n for n in ('BaseGrid.cpp','DataGrid.cpp','DishParams.cpp','rng.cpp','besj1.cpp','TransferFunction.cpp','Subref.cpp')]
binary=out/'lmtoof-reader-fixture'
command=['c++','-O1','-I'+str(LM/'C++'),'-I/opt/homebrew/include','-L/opt/homebrew/lib',str(D/'lmtoof_reader_fixture.cpp'),*[str(p) for p in sources],'-lcfitsio','-lfftw3','-lgsl','-lgslcblas','-o',str(binary)]
with (out/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
results=[];tel=[]
for obs,z in zip((152385,152386,152387),(-3.566596717269,-1.566596717269,.433403282731)):
    src=L/f'citlali-validation/v2/oof/refactor/1146+399/reduced/redu00/{obs}/raw/toltec_commissioning_a1100_oof_{obs}_citlali.fits'
    target=out/f'{obs}-interface-test.fits';r=attach_historical(src,target,requested_focus_mm=z,pupil_stop_m=22.)
    with fits.open(target) as h:a=h[0].data.copy()
    dump=out/f'{obs}-reader.bin';p=subprocess.run([str(binary),str(target),str(len(a)),str(dump)],capture_output=True,text=True,check=True)
    (out/f'{obs}-reader.log').write_text(p.stdout+p.stderr)
    require('ERROR' not in p.stdout and 'ERROR' not in p.stderr,'LMTOOF reader error')
    b=np.fromfile(dump,dtype=np.float64).reshape(a.shape);require(np.array_equal(a,b,equal_nan=True),'C++ reader changed pixels/axes/missingness')
    fields=next(x for x in p.stdout.splitlines() if x.startswith('CHECK ')).split()[1:];n,count=map(int,fields[:2]);step,zero,one,lam,focus,stop,f0,f20=map(float,fields[2:])
    require(n==len(a) and count==np.isfinite(a).sum() and abs(step-1)<1e-12 and zero==0 and abs(one-1)<1e-12 and lam==.0011 and abs(focus-1000*z)<1e-8 and stop==22 and f0==-2 and f20>f0,'native reader/focus convention mismatch')
    r['native_reader']=dict(exact_pixel_roundtrip=True,supported_pixels=count,focus_path_derivative_center=f0,focus_path_derivative_at20m=f20,log=bound(out/f'{obs}-reader.log'));results.append(r)
    original=next((L/'citlali-validation/v1/oof/data').glob(f'tel*{obs}*nc'));derived=L/f'citlali-validation/v2/oof/refactor/1146+399/reduced/tel_toltec_2026-02-19_{obs}_00_0001_recomputed.nc'
    changed=[]
    with netCDF4.Dataset(original) as x,netCDF4.Dataset(derived) as y:
        require(set(x.variables)==set(y.variables),'recomputed telescope variable inventory differs')
        for name in x.variables:
            a=np.asarray(x[name][...]);b=np.asarray(y[name][...])
            if a.dtype.kind not in 'fiu':continue
            if not np.array_equal(a,b,equal_nan=True):changed.append(dict(field=name,max_absolute_difference=float(np.nanmax(abs(a-b)))))
        pa=np.asarray(y['Data.TelescopeBackend.ActParAng'][...]);parange=np.rad2deg([np.nanmin(pa),np.nanmax(pa)]).tolist()
    tel.append(dict(observation=obs,original=bound(original),selected_historical_recomputed=bound(derived),changed_numeric_fields=changed,selected_parallactic_angle_deg=parange,derivation_provenance='not recovered; exact historical bytes selected for matched control, not raw stream identity'))
require(len({tuple(r['shape']) for r in results})==1,'three-map solver grid mismatch')
try:
    require_lm_sequence(results)
except ValueError as exc:
    require('unequal valid-pixel' in str(exc),'unexpected sequence rejection')
    joint=dict(status='blocked',reason=str(exc),support_changed=False)
else:raise ValueError('known unequal support was incorrectly admitted')
identities=[bound(p) for p in sources]+[bound(p) for p in sorted((LM/'C++').glob('*.h'))]+[*[bound(LM/'C++'/n) for n in ('OOF.cpp','OOFLM.cpp','OOFModelLM.cpp','glm_fit_data_files.cpp','FFTGrid.cpp','bandpass.cpp')],bound(LM/'C++/OOFModel.cpp'),bound(LM/'C++/LMTGrid.cpp'),bound(LM/'python/TolTECBeamMap3.py'),bound(LM/'python/OOFBeamMap.py'),bound(LM/'python/OOFDriverFrontend.py')]
save_json(out/'SAVED_INTERFACE.json',dict(status='pass-interface-only',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),maps=results,joint_lm_sequence=joint,telescope=tel,lmtoof_sources=identities,harness=bound(D/'lmtoof_reader_fixture.cpp'),command=command,inversion_runs=0,observation_starts=0))
print('three saved maps passed exact C++ pixel/support/axis/focus roundtrip; no inversion')
