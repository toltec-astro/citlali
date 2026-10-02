"""Saved operational preprocessing only; no optical fit, inference, or telescope writes.
The existing Gaussian crop locator is executed unchanged, never used as admission.
"""
import os
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/oof2-mpl',XDG_CACHE_HOME='/private/tmp/oof2-cache',PYTHONDONTWRITEBYTECODE='1')
import sys,ast,json,subprocess,contextlib
from pathlib import Path
from unittest.mock import patch
import numpy as np,yaml
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import griddata
W=Path(__file__).resolve().parents[2];D=Path(__file__).parent
LM=Path.home()/'GitHub/LMTOOF';ML=Path.home()/'GitHub/oof_ml_project'
L=Path.home()/'work_toltec/local_data';R=L/'citlali-validation/v2/oof/refactor/1146+399/reduced/redu00'
sys.path[:0]=[str(W/'tools/timestream_successor'),str(LM/'python')]
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from inspect_forward_map import require
from oof_saved_interface import require_lm_sequence
from TolTECBeamMap3 import TolTECBeamMap
from ToltecSignalFits import ToltecSignalFits
OBS=(152385,152386,152387)

def stats(a,mask=None):
    ok=np.isfinite(a) if mask is None else np.isfinite(a)&mask
    v=a[ok];return dict(shape=list(a.shape),finite_pixels=int(np.isfinite(a).sum()),supported_pixels=int(ok.sum()),unsupported_pixels=int(a.size-ok.sum()),minimum=float(v.min()),maximum=float(v.max()),median=float(np.median(v)),negative_pixels=int((v<0).sum()))

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    source=ML/'oof_ml/inference/test_toltec_pipeline.py'
    # Compile precisely the existing loader class, excluding TF/model imports and inference.
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MultiBandDataLoader')
    ns=dict(np=np,yaml=yaml,Path=Path,TolTECBeamMap=TolTECBeamMap)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    cfg=out/'ml';cfg.mkdir();(cfg/'config.yaml').write_text(yaml.safe_dump(dict(path=str(R),obsnums=list(OBS))))
    captured=[];centers=[]
    original=ToltecSignalFits.plotCutout;fit=ToltecSignalFits.fitGaussian
    def cut(self,*a,**kw):
        c=original(self,*a,**kw);captured.append(c);return c
    def center(self,*a,**kw):
        r=fit(self,*a,**kw);centers.append([float(x) for x in r['center_pixel']]);return r
    sources=[LM/'C++'/n for n in ('BaseGrid.cpp','DataGrid.cpp','DishParams.cpp','rng.cpp','besj1.cpp','TransferFunction.cpp','Subref.cpp')]
    binary=out/'reader';command=['c++','-O1','-I'+str(LM/'C++'),'-I/opt/homebrew/include','-L/opt/homebrew/lib',str(D/'consumer_reader.cpp'),*[str(p) for p in sources],'-lcfitsio','-lfftw3','-lgsl','-lgslcblas','-o',str(binary)]
    with (out/'compile.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    rows=[]
    with (out/'preprocessing.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
      with patch.object(ToltecSignalFits,'plotCutout',cut),patch.object(ToltecSignalFits,'fitGaussian',center):
        loader=ns['MultiBandDataLoader'](cfg,crop_size=(32,32),bands=['a2000'])
        mlcuts=[captured[i] for i in (1,3,5)]
        for i,obs in enumerate(OBS):
            src=next((R/str(obs)/'raw').glob('toltec_*a2000*.fits'))
            require(len(list(src.parent.glob('toltec_*a2000*.fits')))==1,'ambiguous source map')
            before=bound(src)
            with fits.open(src) as h:
                ph=dict(h[0].header);hh=h['signal_I'].header;z=np.squeeze(h['signal_I'].data).copy();weight=np.squeeze(h['weight_I'].data).copy();wcs=WCS(hh).sub(2)
                header={k:hh.get(k) for k in ('CTYPE1','CTYPE2','CUNIT1','CUNIT2','CDELT1','CDELT2','BUNIT')}
            require(ph['OOF_ID']==2000 and ph['OOF_W']==.002,'wavelength identity')
            tbm=TolTECBeamMap(str(src),obs,2,str(src.parent),0,0);c=captured[-1]
            threshold=tbm.tsf.findWeightThreshold();valid=np.isfinite(z)&np.isfinite(weight)&(weight>0)&(weight>=threshold)
            cropmask=valid[c.slices_original];require(cropmask.shape==tbm.signal.shape,'truncated crop not supported by diagnostic')
            tbm.grid(256,4,0);tbm.write_FITS(str(out/'lm'))
            # Separate support interpolation; never overwrite consumer values or zero-fill data.
            coords=np.column_stack((tbm.xmap,tbm.ymap));axis=(np.arange(256)-127.5)*4;xx,yy=np.meshgrid(axis,axis)
            support_grid=griddata(coords,cropmask.astype(float).ravel(),(xx,yy),method='linear').T
            window=tbm.grid_array[116:141,116:141];support=(support_grid[116:141,116:141]>=1-1e-12)&np.isfinite(window)
            target=Path(tbm.fits_file_name);dump=out/f'{obs}-reader.bin'
            p=subprocess.run([str(binary),str(target),str(dump)],capture_output=True,text=True,check=True)
            (out/f'{obs}-reader.log').write_text(p.stdout+p.stderr)
            fields=next(x.split()[1:] for x in p.stdout.splitlines() if x.startswith('CHECK '));n,count=map(int,fields[:2]);step,lam,focus=map(float,fields[2:])
            actual=np.fromfile(dump,dtype=np.float64).reshape((256,256));require(np.array_equal(actual,tbm.grid_array,equal_nan=True),'C++ data orientation differs')
            require(n==256 and count==np.isfinite(window).sum() and abs(step-4)<1e-10 and lam==.002 and abs(focus-ph['OOF_M2Z'])<1e-8,'reader boundary differs')
            mt=loader.fp['a2000'][obs]['tbm'];mc=mlcuts[i];mraw=mt.signal[34:66,34:66];ms=valid[mc.slices_original][34:66,34:66];norm=loader.images['a2000'][0,:,:,i]
            require(np.array_equal(norm,mraw/mraw.max(),equal_nan=True),'ML normalization differs')
            np.savez_compressed(out/f'{obs}-windows.npz',lm_window=window,lm_support=support,ml_raw=mraw,ml_normalized=norm,ml_support=ms,lm_crop=tbm.signal,lm_crop_support=cropmask)
            def crop_record(c):
                slices=[[s.start,s.stop] for s in c.slices_original];y,x=c.slices_original;bounds=wcs.all_pix2world([[x.start,y.start],[x.stop-1,y.stop-1]],0).tolist()
                return dict(parent_yx_slices=slices,parent_corner_world=bounds,parent_world_units=[str(u) for u in wcs.wcs.cunit])
            rows.append(dict(observation=obs,input=before,parent=stats(z,np.isfinite(weight)&(weight>0)),header=header,requested_focus_microns=ph['OOF_M2Z'],weight_threshold=threshold,
                lm=dict(crop=crop_record(c),locator_center_parent_pixel=centers[-1],crop_stats=stats(tbm.signal,cropmask),window=stats(window,support),reader_valid_samples=count,reader=bound(out/f'{obs}-reader.log'),fits=bound(target),units='numeric input mJy/beam preserved; wrapper label arbs',normalization='none',background_subtraction='none',sample_coordinate_centers_arcsec=[-46,50],reader_coordinate_centers_arcsec=[-48,48],finite_but_unsupported=int((np.isfinite(window)&~support).sum())),
                ml=dict(crop=crop_record(mc),window=stats(mraw,ms),normalized=stats(norm,ms),normalization_divisor=float(mraw.max()),x_centers_arcsec=mt.axis1[34:66].tolist(),y_centers_arcsec=mt.axis2[34:66].tolist()),windows=bound(out/f'{obs}-windows.npz')))
            require(bound(src)==before,'historical input mutated')
    counts=[r['lm']['reader_valid_samples'] for r in rows]
    if len(set(counts))==1:require_lm_sequence([dict(shape=[256,256],angular_pixel_arcsec=4,wavelength_m=.002,supported_pixels=n) for n in counts])
    bindings=[bound(p) for p in sources]+[bound(LM/'python'/p) for p in ('TolTECBeamMap3.py','ToltecSignalFits.py','OOFBeamMap.py','OOFDriverFrontend.py')]+[bound(source),bound(ML/'scripts/04_test_toltec.py'),bound(LM/'etc/default_config.yaml'),bound(LM/'etc/default_config_2000.yaml')]
    save_json(out/'CONSUMERS.json',dict(maps=rows,sources=bindings,loader_isolation='unchanged actual MultiBandDataLoader class AST, model/inference imports excluded',configuration=dict(lm=dict(pointing_option=2,nfft=256,pix_size_arcsec=4,grid_type=0,map_size_x=12,map_size_y=12,weight_cut=.5),ml=dict(bands=['a2000'],crop_size=[32,32],pointing_option=0)),configuration_authority='installed defaults plus saved operational a2000 examples; past invocation on 152385 not recovered',lm_fixed_count_warning=dict(counts=counts,status='not encountered after operational preprocessing' if len(set(counts))==1 else 'blocks optical fitting only',general_defect_repaired=False),inversion_runs=0,model_inferences=0,observation_starts=0))
    print(json.dumps(dict(lm_counts=counts,ml_support=[r['ml']['window']['supported_pixels'] for r in rows])))
if __name__=='__main__':main(Path(sys.argv[1]))
