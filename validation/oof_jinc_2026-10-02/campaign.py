"""Bounded saved-OOF adapter. Shared C++ owns all JINC science arithmetic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/oof-jinc-mpl',XDG_CACHE_HOME='/private/tmp/oof-jinc-cache')
import argparse,json,sys,subprocess,shutil,time,resource
from pathlib import Path
import numpy as np,yaml
from astropy.io import fits
from astropy.nddata import Cutout2D
W=Path(__file__).resolve().parents[2];D=Path(__file__).parent
sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_fruit_trajectory import bound,verified
from point_additive_diagnostics import save_json
from point_jinc_interval import enclose_cache
from inspect_forward_map import require,digest
from point_storage import tree_bytes
sys.path.insert(0,str(W/'validation/oof_a2000_m0_2026-10-01'))
from run_campaign import ROOT as PARENT,PRIOR,CAP
OBS=(152385,152386,152387)
ROOT=PARENT.with_name('successor-oof-a2000-jinc-20261002')
HELPER=W/'build/bin/citlali_jinc_diagnostic'
PROFILE='OOF-A2000-JINC-NUMERICAL-ADEQUACY-v1'
RESERVE=1_000_000_000
MANDATORY_GATES=('configure','build','ctest','baseline','point-tools','oof-tools','config','native-point-contract','saved-preparation')

def read(p):return json.loads(Path(p).read_text())
def yread(p):return yaml.load(Path(p).read_text(),Loader=yaml.CSafeLoader)
def learning(p):
    lines=[]
    with Path(p).open() as f:
        for line in f:
            if line.startswith("spectra:"):break
            lines.append(line)
    return yaml.load("".join(lines),Loader=yaml.CSafeLoader)
def revision():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'clean source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
def capacity():
    inventory=[dict(path=str(p),bytes=tree_bytes(p)) for p in [*PRIOR,PARENT]]
    prior=sum(p['bytes'] for p in inventory);actual=tree_bytes(ROOT) if ROOT.exists() else 0
    require(actual<=RESERVE and prior+RESERVE<CAP,'shared disk reservation exceeded')
    require(shutil.disk_usage(ROOT.parent).free>max(0,RESERVE-actual)+2_000_000_000,'2GB free reserve unavailable')
    return dict(prior=inventory,new_reservation_bytes=RESERVE,new_actual_bytes=actual,shared_bound_bytes=prior+RESERVE,ceiling_bytes=CAP)
def invoke(request,out):
    with out.with_suffix('.log').open('w') as log:
        p=subprocess.run([str(HELPER),str(request),str(out)],stdout=log,stderr=subprocess.STDOUT)
    require(p.returncode==0,'native JINC failed; inspect '+str(out.with_suffix('.log')))
    require(out.with_suffix('.log').stat().st_size==0,'unexpected native output')
    return yread(out/'receipt.yaml')
class SavedInputs:
    def __init__(self,role):
        record=role/'COMPLETED-SCRATCH-REMOVALS.json'
        self.expected={b['path']:b['sha256'] for b in read(record)['preserved']}
        self.used={str(record):bound(record)}
    def get(self,p,sha=None):
        p=Path(p);h=self.expected.get(str(p));require(h is not None,'unpreserved source '+str(p))
        require(sha is None or sha==h,'receipt source hash differs')
        ref=dict(path=str(p),sha256=h);verified(ref);self.used[str(p)]=ref;return p
    def yaml(self,p,sha=None):return yread(self.get(p,sha))

def selected_values(values,causes,coordinates,bits):
    retained=causes==0;count=int(retained.sum())
    require(coordinates.shape==(count,2),'retained AST shape differs')
    eligible=np.unpackbits(bits,bitorder='little')[:count].astype(bool)
    require(len(bits)==(count+7)//8 and eligible.all(),'bounded OOF requires original all-retained admission')
    z=values[retained];require(np.isfinite(z).all() and np.isfinite(coordinates).all(),'nonfinite admitted sample')
    return z

def replay(z,xy,h,original,normalization,pixels,numerator=None,support=None):
    px=xy[:,0]/h['CDELT1']+h['CRPIX1'];py=xy[:,1]/h['CDELT2']+h['CRPIX2']
    nx=h['NAXIS1'];ny=h['NAXIS2'];ix=np.floor(px-.5).astype('i8');iy=np.floor(py-.5).astype('i8')
    require(np.all((ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)),'original AST outside destination')
    actual=iy*nx+ix;require(np.array_equal(actual,pixels),'same-occurrence AST membership replay differs')
    n=np.bincount(actual,weights=z,minlength=nx*ny).reshape(ny,nx)
    c=np.bincount(actual,minlength=nx*ny).reshape(ny,nx)
    require(np.array_equal(c,normalization),'original uniform normalization differs')
    if numerator is not None:require(np.allclose(n,numerator,rtol=1e-11,atol=1e-7),'original full numerator replay failed')
    ok=np.isfinite(original) if support is None else support
    require(ok.any() and np.array_equal(np.isfinite(original),ok) and np.all(c[ok]>0),'original signal support/missingness differs')
    delta=float(np.max(np.abs(n[ok]/c[ok]-original[ok])))
    require(np.allclose(n[ok]/c[ok],original[ok],rtol=1e-11,atol=1e-9),'original signal replay failed')
    return dict(samples=len(z),membership_exact=True,normalization_exact=True,max_signal_difference=delta,max_numerator_difference=float(np.max(np.abs(n-numerator))) if numerator is not None else None,published_signal_pixels=int(ok.sum()),positive_normalization_pixels=int((c>0).sum()))

def prepare():
    rev=revision();capacity();out=ROOT/'prepared'/rev;out.mkdir(parents=True,exist_ok=False)
    measurements=read(PARENT/'report/MEASUREMENTS.json');states=[];bindings=[bound(PARENT/'FREEZE.json'),bound(PARENT/'CLOSEOUT.json'),bound(PARENT/'report/MEASUREMENTS.json'),bound(D/'WORK_ORDER.md')]
    for obs in OBS:
        role=PARENT/f'S-{obs}';auth=SavedInputs(role);base=role/'M0/products/a2000'
        bundle=auth.yaml(base/'bundle.yaml');plan=auth.yaml(base/'plan.yaml',bundle['plan_sha256'])
        require(plan['observation']==bundle['observation']==f'obs:{obs}_0_2','unrecognized predecessor identity; reassess')
        require(bundle['array']=='a2000' and bundle['AST_invalid_coordinates']==0 and bundle['MAP_route_admitted']==bundle['PTC_retained'],'parent population/admission differs')
        with fits.open(auth.get(base/'map.fits',bundle['files']['map.fits'])) as f:
            h=f[0].header.copy();original=f[0].data.copy();norm=f['NORMALIZATION'].data.copy();numerator=f['NUMERATOR'].data.copy();support=f['SCI_SUPPORT'].data.astype(bool)
        require(h['RADESYS']=='FK5' and h['CDELT1']==-2/3600 and h['CDELT2']==2/3600,'original grid differs')
        zs=[];xys=[];frequencies=[];segments=[];seen=set();cursor=0
        for e in plan['ordered_column_domain']:
            pub=auth.yaml(base/e['publication_file'],e['publication_sha256']);s=pub['source_segment'];nw=pub['network'];require(nw in (11,12),'foreign network')
            coef=auth.yaml(base/e['coefficient_file'],e['coefficient_sha256']);ast=auth.yaml(base/e['AST_binding_file'],e['AST_binding_sha256'])
            require(coef['scalar']==1 and str(coef['unit'])=='1' and coef['PTC_publication_sha256']==e['publication_sha256'] and coef['stored_support_reference']==pub['support_reference']==e['support_reference'],'uniform coefficient association differs')
            require(coef['family']=='SCI-PTC:uniform_constant@draft-0.1' and coef['profile']=='SCI-PTC:uniform_coefficient_handoff@draft-0.1','coefficient family differs')
            cfg=read(auth.get(role/f'inputs/network{nw}-input.json'))
            require((cfg['observation'],cfg['subobservation'],cfg['scan'],cfg['mode'],cfg['network'],cfg['development_reference'])==(obs,0,1,'OOF',nw,'oof-a2000-m0-v1'),'actual acquisition differs')
            cal=auth.yaml(Path(pub['source_CAL_root'])/'receipt.yaml',pub['source_CAL_receipt_sha256'])
            require(ast['CAL_parent']==pub['CAL_parent']=='sha256:'+pub['source_CAL_receipt_sha256'] and ast['RTC_grid']==pub['RTC_grid'] and ast['PTC_publication']==pub['identity']==e['PTC_publication'],'AST/PTC/CAL join differs')
            require(ast['AST_telescope']==cal['AST_telescope'] and ast['AST_pointing_offset']==cal['AST_pointing_offset'] and f'inputs:{obs}_0_1:astrometry' in ast['AST_pointing_offset'],'AST acquisition differs')
            ptcroot=Path(pub['source_PTC_root']);ptc=auth.yaml(ptcroot/'receipt.yaml',pub['source_PTC_receipt_sha256'])
            stem=next(k.removesuffix('-cleaned.f64') for k in s['files'] if k.endswith('-cleaned.f64'))
            require(ptc['segments'][int(stem.split('-')[-1])]==s and pub['rank']==ptc['rank']==3,'PTC segment/rank differs')
            nt,nd=s['scheduled_times'],s['output_detectors'];cols=s['output_detector_indices']
            learn=learning(auth.get(ptcroot.parent.parent/'learning-receipt.yaml'))
            expected=[x['occurrence']+':association='+x['array_association'] for x in learn['detectors']]
            require(pub['detector_occurrences']==[expected[i] for i in cols],'detector occurrence order differs')
            def array(suffix,dtype):return np.fromfile(auth.get(ptcroot/(stem+suffix),s['files'][stem+suffix]),dtype=dtype)
            values=array('-cleaned.f64','<f8').reshape(nt,nd);causes=array('-causes.u8','u1').reshape(nt,nd);slots=array('-slots.i64','<i8')
            require(len(slots)==nt and len(set(slots))==nt and not any((nw,int(t)) in seen for t in slots),'duplicate temporal support')
            seen.update((nw,int(t)) for t in slots)
            xy=np.fromfile(auth.get(base/e['AST_coordinate_file'],e['AST_coordinate_sha256']),dtype='<f8').reshape(-1,2)
            bits=np.fromfile(auth.get(base/e['MAP_eligibility_file'],e['MAP_eligibility_sha256']),dtype='u1')
            z=selected_values(values,causes,xy,bits)
            require(len(z)==pub['retained']==e['input_rows'] and cursor==e['first_input_row'],'column occurrence domain differs');cursor+=len(z)
            frequency=1/(learn['cadence_interval_seconds']*pub['RTC_factor'])
            zs.append(z);xys.append(xy);frequencies.append(np.full(len(z),frequency))
            segments.append(dict(network=nw,publication=bound(base/e['publication_file']),input=bound(role/f'inputs/network{nw}-input.json'),slots=bound(ptcroot/(stem+'-slots.i64')),frequency_hz=frequency,samples=len(z)))
        z=np.concatenate(zs);xy=np.concatenate(xys);fs=np.concatenate(frequencies)
        pixels=np.fromfile(auth.get(base/'operator-pixel.i32',bundle['files']['operator-pixel.i32']),dtype='<i4')
        checks=replay(z,xy,h,original,norm,pixels,numerator,support)
        # Preserve reference center and full center-to-center extent; halve cell size.
        nh=h.copy();nh['CDELT1']=-1/3600;nh['CDELT2']=1/3600
        for i in (1,2):nh['NAXIS'+str(i)]=2*h['NAXIS'+str(i)]-1;nh['CRPIX'+str(i)]=2*h['CRPIX'+str(i)]-1
        nh['OBSNUM']=obs;nh['SCAN']=1;nh['SUBOBS']=0;nh['MAPMAKER']='JINC';nh['BUNIT']='mJy/beam'
        geometry=np.column_stack([xy[:,0]/nh['CDELT1']+nh['CRPIX1'],xy[:,1]/nh['CDELT2']+nh['CRPIX2'],np.ones(len(z)),fs])
        require(np.all((geometry[:,0]>=.5)&(geometry[:,0]<nh['NAXIS1']+.5)&(geometry[:,1]>=.5)&(geometry[:,1]<nh['NAXIS2']+.5)),'new grid excludes centers')
        dest=out/str(obs);dest.mkdir();geometry.astype('<f8').tofile(dest/'geometry.f64');z.astype('<f8').tofile(dest/'values.f64')
        header={k:v for k,v in nh.items() if k not in ('COMMENT','HISTORY','')};save_json(dest/'header.json',header)
        spec=dict(rows=nh['NAXIS2'],columns=nh['NAXIS1'],phases=1,pixel_matrix_rad=[-np.pi/180/3600,0.,0.,np.pi/180/3600],scale_rad=(2.0/1000)/45,a=1.1,b=3.17,c=2.,rmax=1.5)
        request=dict(schema='oof-a2000-jinc-diagnostic-v1',observation=obs,subobservation=0,scan=1,array='a2000',frame='FK5',mode='cache',spec=spec)
        save_json(dest/'cache-request.json',request);receipt=invoke(dest/'cache-request.json',dest/'cache')
        require(receipt['half_width']==14,'OOF square footprint differs')
        cache=np.fromfile(dest/'cache/cache.f64',dtype='<f8');enclosure=enclose_cache(cache,14,[-1/3600,1/3600],'2.0','3.17',phases=1,rmax='1.5')
        enclosure['errors'].astype('<f8').tofile(dest/'coefficient-errors.f64')
        peak=next(x['S_native_32arcsec']['peak'] for x in measurements['observations'] if x['observation']==obs)
        require(peak==float(np.nanmax(Cutout2D(original,(original.shape[1]/2,original.shape[0]/2),(16,16)).data)), 'frozen old central peak differs')
        request.update(mode='accumulate',profile=PROFILE,cache_sha256=digest(dest/'cache/cache.f64'),coefficient_errors=bound(dest/'coefficient-errors.f64'),samples=len(z),geometry_payload=bound(dest/'geometry.f64'),value_payload=bound(dest/'values.f64'),workers=1,frozen_peak=peak)
        save_json(dest/'request.json',request)
        state=dict(observation=obs,identity=f'obs:{obs}_0_1',predecessor_identity=plan['observation'],identity_repair='scan1 authenticated from actual OOF input and CAL/AST; original erroneous scan2 strings preserved',request=bound(dest/'request.json'),header=bound(dest/'header.json'),ordinary_replay=checks,segments=segments,peak_scale=peak,peak_scale_hex=float(peak).hex(),certificate=dict(max_coefficient_error=float(enclosure['errors'].max()),unique_radii=enclosure['unique_radii'],precision=enclosure['precision'],cache=bound(dest/'cache/cache.f64'),errors=bound(dest/'coefficient-errors.f64')))
        states.append(state);bindings.extend(auth.used.values());print(str(obs)+' saved replay/cache passed',flush=True)
    refs={r['path']:r for r in bindings}
    for p in out.rglob('*'):
        if p.is_file():refs[str(p)]=bound(p)
    for b in refs.values():verified(b)
    save_json(ROOT/'PREPARED.json',dict(source_revision=rev,helper=bound(HELPER),states=states,bindings=list(refs.values()),storage=capacity(),observational_accumulations=0,RTC_CAL_PTC_executions=0))

def execution_identity(prepared):
    require(revision()==prepared['source_revision'],'execution source changed')
    require(Path(prepared['helper']['path'])==HELPER,'selected helper path differs')
    verified(prepared['helper'])
def validate_gates(gates,rev):
    require(gates['source_revision']==rev and gates.get('complete') is True,'gates incomplete or wrong source')
    require(tuple(g['name'] for g in gates['gates'])==MANDATORY_GATES,'mandatory gate inventory differs')
    require(all(g['exit_code']==0 for g in gates['gates']),'gate failure')
    for g in gates['gates']:verified(g['log'])
def check(frozen=True):
    rev=revision();p=read(ROOT/'PREPARED.json');execution_identity(p)
    for b in p['bindings']:verified(b)
    if frozen:
        f=read(ROOT/'FREEZE.json');require(f['source_revision']==rev,'freeze source differs')
        for b in f['bindings']:verified(b)
        require(not (ROOT/'LAUNCHED.json').exists(),'single map-only launch already consumed')
    return p,capacity()
def freeze():
    p,storage=check(False);review=read(ROOT/'review.json');gates=read(ROOT/'GATES.json')
    require(review['source_revision']==gates['source_revision']==p['source_revision'] and review['verdict']=='pass','exact source review/gates missing')
    validate_gates(gates,p['source_revision'])
    require(not (ROOT/'FREEZE.json').exists(),'preserve existing freeze')
    refs=[bound(ROOT/n) for n in ('PREPARED.json','GATES.json','review.json')]+[g['log'] for g in gates['gates']]
    save_json(ROOT/'FREEZE.json',dict(source_revision=p['source_revision'],bindings=refs,storage=storage,roles=list(OBS),new_map_only_starts=3,new_upstream_starts=0,manual_resume='owner foreground once, then done; no retry'))
def publish(state,out):
    q=read(verified(state['request']));raw_header=read(verified(state['header']));shape=(raw_header['NAXIS2'],raw_header['NAXIS1'])
    structural={'SIMPLE','XTENSION','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND','PCOUNT','GCOUNT','CHECKSUM','DATASUM','EXTNAME'}
    h=fits.Header({k:v for k,v in raw_header.items() if k not in structural});receipt=yread(out/'receipt.yaml')
    require(receipt['request_sha256']==digest(verified(state['request'])) and receipt['observation']==state['identity'],'native receipt binding differs')
    def plane(n,dtype='<f8'):
        require(digest(out/n)==receipt['files'][n],'native output digest differs')
        return np.fromfile(out/n,dtype=dtype).reshape(shape)
    causes=plane('causes.u8','u1');valid=causes==0;m=plane('m.f64')
    require(valid.any() and np.isfinite(m[valid]).all(),'no finite certified JINC support')
    h['OBSID']=state['identity'];h['NUMPROF']=PROFILE;h['HISTORY']='Shared SCI-JINC r0.3; original FK5 frame. No AltAz conversion.'
    hdus=[fits.PrimaryHDU(np.where(valid,m,np.nan),h)]
    for name,file in [('NUMERATOR','N.f64'),('NORMALIZATION','C.f64'),('QUADRATIC','Q.f64'),('COEFFSQ_TIME','T.f64')]:
        ih=h.copy();ih['BUNIT']='mJy/beam' if name=='NUMERATOR' else ('s' if name=='COEFFSQ_TIME' else '1');hdus.append(fits.ImageHDU(plane(file),ih,name=name))
    hdus.append(fits.ImageHDU(valid.astype('u1'),name='LOCAL_VALID'))
    fits.HDUList(hdus).writeto(out/'map.fits',checksum=True)
    fits.HDUList([fits.PrimaryHDU(causes),fits.ImageHDU(plane('error-4.f64'),name='SIGNAL_ERROR'),fits.ImageHDU(plane('members.u64','<u8'),name='MEMBERS')]).writeto(out/'numerical-diagnostics.fits',checksum=True)
    save_json(out/'BUNDLE.json',dict(observation=state['identity'],array='a2000',frame='FK5',schema='oof-jinc-saved-comparison-v1',base_roles=['N','C','Q','N/C with LOCAL_VALID','coefficient-squared time'],map=bound(out/'map.fits'),diagnostics=bound(out/'numerical-diagnostics.fits'),receipt=bound(out/'receipt.yaml'),parent=state,profile=PROFILE,unit='mJy/nominal-beam',numerical_supported_pixels=int(valid.sum()),numerical_unavailable_pixels=int((~valid).sum()),not_authorized=['variance','precision','physical exposure','optical inference','feedback'],uniform_coefficients=1,upstream_reexecuted=False))
def run():
    p,storage=check();start=time.perf_counter()
    with (ROOT/'LAUNCHED.json').open('x') as f:json.dump(dict(source_revision=p['source_revision'],roles=list(OBS),time=time.time()),f)
    done=[]
    try:
        for state in p['states']:
            capacity();out=ROOT/str(state['observation']);save_json(ROOT/'STATUS.json',dict(state='RUNNING',observation=state['observation'],completed=done))
            with (ROOT/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(observation=state['observation'],request=state['request'],start=time.time()))+'\n')
            execution_identity(p)
            t=time.perf_counter();invoke(verified(state['request']),out);execution_identity(p);publish(state,out)
            done.append(dict(observation=state['observation'],seconds=time.perf_counter()-t,child_peak_rss_bytes_so_far=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,bundle=bound(out/'BUNDLE.json')))
        for b in p['bindings']:verified(b)
        execution_identity(p)
        save_json(ROOT/'STATUS.json',dict(state='COMPLETE_PENDING_OWNER_REVIEW',completed=done,seconds=time.perf_counter()-start,storage=capacity()))
        print('Three map-only JINC applications completed. Say done for saved-map assessment.')
    except BaseException as e:
        save_json(ROOT/'STATUS.json',dict(state='FAILED',completed=done,error=str(e)));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','freeze','check','run']);args=a.parse_args()
    if args.action=='check':check();print('Frozen launch checks passed; no maps launched.')
    else:globals()[args.action]()
