"""Bounded cached-only a2000 diagnosis. No scientific execution-route imports."""
import argparse,hashlib,json,os,platform,resource,shutil,subprocess,sys,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from astropy.io import fits
import yaml
from point_saved_diagnosis import (require,timing,paired,difference,magnitude,shape_components,
    aperture,split_candidate,memberships,score_audit,signed_response,signed_even,percent)

REPO=Path(__file__).resolve().parents[2]
V=REPO/'validation/point_152389_a2000_saved_diagnosis_2026-09-25'
PARENT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925')
POLICIES=('BASE-POINT','CORR','DIAG');BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder')
KEYS=('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    def native(v):
        if isinstance(v,np.ndarray):return native(v.tolist())
        if isinstance(v,np.generic):return native(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [native(x) for x in v]
        return v
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(native(value),indent=2,allow_nan=False)+'\n')


def revision():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'clean source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def ledger(path):
    return {(path.parent/n).resolve():s for s,n in (line.split('  ',1) for line in path.read_text().splitlines())}


def mean_path(products,segment):
    src=segment['source_segment'];n=next(n for n in src['files'] if n.endswith('-basis.f64'));index=n.split('-')[1]
    return products/f"network-{src['network']}-segment-{index}"/'mean.f64'


def prepare(out):
    start=time.perf_counter();rev=revision();require(not out.exists(),'fresh derivative output required')
    completion=json.loads((REPO/'validation/point_152389_guarded_feedback_comparison_2026-09-25/COMPLETION.json').read_text())
    lm=PARENT/'FILES_SHA256SUMS';require(sha(lm)==completion['root_manifest']['sha256'],'parent seal mismatch');expected=ledger(lm);consumed={}
    def bind(path):
        path=Path(path).resolve();require(path in expected,'no parent binding '+str(path));actual=sha(path);require(actual==expected[path],'parent input mismatch '+str(path));consumed[str(path)]=dict(sha256=actual,parent_entry=expected[path],bytes=path.stat().st_size);return path
    def read(path):return json.loads(bind(path).read_text())
    p=read(PARENT/'freeze/prospective.json');expected.update({Path(k).resolve():s for k,s in p['consumed_inputs'].items()})
    for ref in p['original_owner']['injections']['a2000'].values():expected[Path(ref['path']).resolve()]=ref['sha256'];bind(ref['path'])
    for name in ('ASSESSMENT.md','FINAL_REVIEW.md','COMPLETION.json','provenance/SOURCE.json','report/a2000/RESULTS.json'):bind(PARENT/name)
    corr=Path(p['correlation_root']);bind(corr/'freeze/a2000/inputs.npz');bind(corr/'freeze/a2000/geometry.json')
    for pol in POLICIES:
        for b in BRANCHES:
            branch=read(PARENT/'run'/pol/b/'a2000/SUMMARY.json')
            require(branch['completed_cycle']==10 and not branch['m10_applied'],'parent trajectory incomplete')
            for state in branch['histories']:
                for n in ('total','candidate','receipt'):bind(state[n]['path'])
                cd=Path(state['candidate']['path']).parent
                for n in ('admission.fits','admission.json'):bind(cd/n)
                if state['cycle_index']:
                    bind(state['request']['path']);products=Path(state['total']['path']).parent
                    for name in ('residual.fits','restored.fits'):bind(products/name)
                    rec=yaml.load(Path(state['receipt']['path']).read_text(),Loader=yaml.CSafeLoader)
                    for seg in rec['segments']:bind(mean_path(products,seg))
        for f in (PARENT/'report/a2000'/pol).rglob('*'):
            if f.is_file() and ('endpoint' not in f.parts) and f.suffix in ('.json','.fits'):bind(f)
    out.mkdir();shutil.copy2(V/'OWNER_DIRECTIVE.md',out/'OWNER_DIRECTIVE.md')
    sources={str(f):sha(f) for f in [Path(__file__),Path(__file__).with_name('point_saved_diagnosis.py')]}
    record=dict(identity='POINT152389-A2000-SAVED-DIAGNOSIS-v1',source_revision=rev,parent_root=str(PARENT),parent_manifest=dict(path=str(lm),sha256=sha(lm)),
        inputs=consumed,source_files=sources,directive_sha256=sha(out/'OWNER_DIRECTIVE.md'),policies=POLICIES,branches=BRANCHES,stages=list(range(11)),
        geometry=str(corr/'freeze/a2000/geometry.json'),frozen_arrays=str(corr/'freeze/a2000/inputs.npz'),thresholds=p['thresholds']['a2000'],
        injections=p['original_owner']['injections']['a2000'],injection_geometry=p['original_owner']['geometry']['a2000'],
        tolerances=p['tolerances'],accounting_atol=p['accounting_atol'],units='mJy/nominal beam; offsets arcsec; brightness-area mJy/nominal beam arcsec^2',
        exclusions='No fits, candidate or policy evaluations, covariance, native processing, projection, MAP, new fields or feedback.',
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        preparation_seconds=time.perf_counter()-start)
    save(out/'PROSPECTIVE.json',record);print(json.dumps(dict(stage='prepared',inputs=len(consumed),bytes=sum(x['bytes'] for x in consumed.values()),seconds=record['preparation_seconds'])))


class Cached:
    def __init__(self,p):self.p=p;self.images={};self.jsons={};self.yamls={};self.used=set()
    def path(self,path):
        path=Path(path).resolve();require(str(path) in self.p['inputs'],'unbound input '+str(path));self.used.add(str(path));return path
    def json(self,path):
        path=self.path(path)
        if path not in self.jsons:self.jsons[path]=json.loads(path.read_text())
        return self.jsons[path]
    def image(self,path):
        path=self.path(path)
        if path not in self.images:
            with fits.open(path,memmap=False) as h:self.images[path]=({x.name:np.array(x.data) for x in h if x.data is not None},dict(h[0].header))
        return self.images[path]
    def yaml(self,path):
        path=self.path(path)
        if path not in self.yamls:self.yamls[path]=yaml.load(path.read_text(),Loader=yaml.CSafeLoader)
        return self.yamls[path]
    def array(self,path):return np.fromfile(self.path(path),dtype='<f8')


def authenticate(p):
    start=time.perf_counter();lm=p['parent_manifest'];require(sha(lm['path'])==lm['sha256'],'parent seal changed')
    def check(item):
        name,ref=item;require(sha(name)==ref['sha256'],'changed input '+name)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,p['inputs'].items()))
    return dict(files=len(p['inputs']),bytes=sum(q['bytes'] for q in p['inputs'].values()),workers=8,seconds=time.perf_counter()-start)


def crop(v,h):
    row=int(round(h['CRPIX2']-1));col=int(round(h['CRPIX1']-1));require(abs(row-(h['CRPIX2']-1))<1e-8 and abs(col-(h['CRPIX1']-1))<1e-8,'noninteger saved origin')
    result=v[row-20:row+21,col-20:col+21];require(result.shape==(41,41),'truncated saved source stamp');return result


def compatible(ah,bh):
    require(all(ah[k]==bh[k] for k in KEYS),'saved grids differ; do not regrid')


def ids(mask):return np.flatnonzero(mask).tolist()


def annotation(a,mask):return dict(count=int(mask.sum()),pixel_ids=ids(mask))


def compare_measurements(a,b):
    require(a['fit']['status']==b['fit']['status']=='success','cached measurement unavailable')
    ap=np.array(a['fit']['parameters']);bp=np.array(b['fit']['parameters'])
    delta=ap-bp;out=dict(peak=delta[0],east=delta[1],north=delta[2],background=delta[6],
        peak_percent_own_REF=percent(delta[0],bp[0]),fixed_template=a['frozen_template_projection']-b['frozen_template_projection'])
    for name,index in (('FWHM_major',0),('FWHM_minor',1)):
        av=sorted(ap[3:5]*np.sqrt(8*np.log(2)),reverse=True)[index];bv=sorted(bp[3:5]*np.sqrt(8*np.log(2)),reverse=True)[index]
        out[name]=av-bv;out[name+'_percent_own_REF']=percent(av-bv,bv)
    def ellipse(p):
        c,s=np.cos(p[5]),np.sin(p[5]);return np.array([c*c*p[3]**2+s*s*p[4]**2,c*s*(p[3]**2-p[4]**2),s*s*p[3]**2+c*c*p[4]**2])
    out['ellipse_xx_xy_yy']=ellipse(ap)-ellipse(bp)
    out['apertures']=[dict(radius_arcsec=x['radius_arcsec'],source_relative=x['value_mJy_per_beam_arcsec2']-y['value_mJy_per_beam_arcsec2'],
        percent_own_REF=percent(x['value_mJy_per_beam_arcsec2']-y['value_mJy_per_beam_arcsec2'],y['value_mJy_per_beam_arcsec2']),
        raw=a['raw_apertures'][j]['value_mJy_per_beam_arcsec2']-b['raw_apertures'][j]['value_mJy_per_beam_arcsec2'],background=a['aperture_background_terms'][j]-b['aperture_background_terms'][j]) for j,(x,y) in enumerate(zip(a['apertures'],b['apertures']))]
    out['cached_flags']=dict(branch=a['fit'],REF=b['fit']);return out


def own_change(current,previous,kind):
    if kind=='map':
        compatible(current['header'],previous['header']);a,b=current['image'],previous['image'];value,mask='PRIMARY','SCI_SUPPORT'
    else:a,b=current['candidate'],previous['candidate'];value,mask='CANDIDATE','VALID'
    delta,valid,support=difference(a[value],a[mask].astype(bool),b[value],b[mask].astype(bool))
    return dict(magnitude=magnitude(delta,valid),support=support,comparison='current minus previous stage of this same branch')


def run(out):
    rev=revision();p=json.loads((out/'PROSPECTIVE.json').read_text());require(rev==p['source_revision'],'prospective code changed')
    require(not (out/'RESULTS.json').exists(),'completed diagnosis cannot rerun');before=authenticate(p);rd=Cached(p);start=time.perf_counter()
    geometry=rd.json(p['geometry']);z=dict(np.load(rd.path(p['frozen_arrays'])));x,y=z['x'],z['y'];radius=z['radius'];masks=[radius<=f*geometry['major'] for f in (1.,1.5,2.)]
    shape,sh=rd.image(p['injections']['C-shoulder']['path']);shape=shape['INJECTION'];shape_valid=np.isfinite(shape)
    require(np.array_equal(shape_valid,np.hypot(x,y)<=40),'archived seed domain changed')
    report=rd.json(PARENT/'report/a2000/RESULTS.json');states={};sums={};arrays={};allrows=[];seedrows=[];partrows=[];selection=[];cleaning=[];audits=[];evidence=[];first={};cross=[]
    for pol in POLICIES:
      for b in BRANCHES:
        summary=rd.json(PARENT/'run'/pol/b/'a2000/SUMMARY.json');sums[pol,b]=summary
        require([q['cycle_index'] for q in summary['histories']]==list(range(11)),'incomplete history')
        for state in summary['histories']:
            k=state['cycle_index'];require((state['array'],state['policy'],state['branch'])==('a2000',pol,b),'foreign history')
            image,h=rd.image(state['total']['path']);c,ch=rd.image(state['candidate']['path']);ad,ah=rd.image(Path(state['candidate']['path']).parent/'admission.fits');meta=rd.json(Path(state['candidate']['path']).parent/'admission.json')
            require(all(ch[n]==geometry['header'][n] for n in KEYS),'candidate geometry mismatch')
            require(np.array_equal(crop(image['PRIMARY'],h),ad['MEASURED'],equal_nan=True),'saved measurement image mismatch')
            entry=dict(state=state,image=image,header=h,candidate=c,ad=ad,meta=meta);states[pol,b,k]=entry
            for q in ('CORR','DIAG'):audits.append(dict(policy=pol,branch=b,stage=k,evidence_policy=q,**score_audit(ad,q)))
            previous=states.get((pol,b,k-1));mm=memberships(ad,previous['ad'] if previous else None,c['ADMITTED'],previous['candidate']['ADMITTED'] if previous else None)
            selection.append(dict(policy=pol,branch=b,**timing(k),imposed_seed=(b=='D-seed-shoulder' and k==0),masks={n:annotation(n,v) for n,v in mm.items()},
                actual_selected=annotation('actual',c['ADMITTED'].astype(bool)),normal_admission_summary=state['admission']))
            for n,a in ad.items():arrays[f'{pol}/{b}/{k}/evidence/{n}']=a
            arrays[f'{pol}/{b}/{k}/candidate']=c['CANDIDATE'];arrays[f'{pol}/{b}/{k}/actual_selected']=c['ADMITTED']
            # Cached guard measurements are copied; no descriptive fit is called.
            allrows.append(dict(policy=pol,branch=b,**timing(k),measurement=state['measurement'],admission=state['admission'],
                previous_own_map_change=own_change(entry,previous,'map') if previous else None,
                previous_own_candidate_change=own_change(entry,previous,'candidate') if previous else None,
                previous_own_measurement_change=compare_measurements(state['measurement'],previous['state']['measurement']) if previous else None))
    aggregate_errors=[];response=[];evenrows=[]
    for pol in POLICIES:
      first[pol]={}
      for k in range(11):
        ref=states[pol,'REF',k];seed=states[pol,'D-seed-shoulder',k];paired(seed['state'],ref['state']);compatible(seed['header'],ref['header'])
        si,ri=seed['image'],ref['image'];d,v,support=difference(si['PRIMARY'],si['SCI_SUPPORT'].astype(bool),ri['PRIMARY'],ri['SCI_SUPPORT'].astype(bool));stamp=crop(d,seed['header']);sv=crop(v,seed['header'])
        dm,mv,msupport=difference(seed['candidate']['CANDIDATE'],seed['candidate']['VALID'].astype(bool),ref['candidate']['CANDIDATE'],ref['candidate']['VALID'].astype(bool))
        mp,parallel,orth,domain=shape_components(stamp,sv,shape);cp,cparallel,corth,cdomain=shape_components(dm,mv,shape)
        old=report['policies'][pol]['branches']['D-seed-shoulder'][k]
        for actual,saved in [(mp['coefficient'],old['shoulder_projection']['value']),(cp['coefficient'],old['candidate_shoulder_projection']['value']),(magnitude(d,v)['rms'],old['difference_rms'])]:
            err=abs(actual-saved);require(err<=1e-8,'archived seed aggregate mismatch');aggregate_errors.append(err)
        row=dict(policy=pol,**timing(k),map_shape=mp,candidate_shape=cp,full_map=magnitude(d,v),full_map_support=support,candidate_support=msupport,
            measurement_difference=compare_measurements(seed['state']['measurement'],ref['state']['measurement']),candidate_apertures=aperture(dm,mv,masks),
            map_raw_apertures=aperture(stamp,sv,masks),source_parallel_peak_sampled=mp['parallel']['maximum_absolute'],
            parallel_peak_percent_original_REF=percent(mp['parallel']['maximum_absolute'],p['injection_geometry']['C-shoulder']['analytic_peak']/.05))
        if k:
            prev=seedrows[-1];dd=d-arrays[f'{pol}/seed/full_delta/{k-1}'];cv=v&arrays[f'{pol}/seed/full_valid/{k-1}'];cmd=dm-arrays[f'{pol}/seed/candidate_delta/{k-1}'];cmv=mv&arrays[f'{pol}/seed/candidate_valid/{k-1}']
            row['previous_seed_difference_change']=dict(full_map=magnitude(dd,cv),candidate=magnitude(cmd,cmv),map_coefficient=mp['coefficient']-prev['map_shape']['coefficient'],candidate_coefficient=cp['coefficient']-prev['candidate_shape']['coefficient'])
        seedrows.append(row)
        for n,a in dict(delta=stamp,valid=sv,parallel=parallel,orthogonal=orth,shape_domain=domain,candidate_delta=dm,candidate_valid=mv,candidate_parallel=cparallel,candidate_orthogonal=corth,full_delta=d,full_valid=v).items():arrays[f'{pol}/seed/{n}/{k}']=a
        if k==0:
            require(np.allclose(dm[mv],shape[mv],rtol=0,atol=1e-8),'imposed m0 is not exact archived shoulder difference');continue
        partition=split_candidate(seed['candidate'],ref['candidate'],seed['ad'],ref['ad'],seed['state']['measurement']['fit']['parameters'][6],ref['state']['measurement']['fit']['parameters'][6],k)
        terms={}
        for n,term in partition['terms'].items():
            terms[n]=dict(pixels=annotation(n,partition['masks'][n]),total=shape_components(term,mv,shape)[0],apertures=aperture(term,mv,masks),
                fitted_constant=shape_components(partition['background'][n],mv,shape)[0],constant_apertures=aperture(partition['background'][n],mv,masks),
                measured=shape_components(partition['measured'][n],mv,shape)[0],measured_apertures=aperture(partition['measured'][n],mv,masks))
            arrays[f'{pol}/partition/{k}/{n}']=term;arrays[f'{pol}/partition/{k}/{n}_constant']=partition['background'][n]
        partrows.append(dict(policy=pol,stage=k,applied_to_map=k+1 if k<10 else None,closure_error=partition['error'],terms=terms))
        sa,ra=seed['ad'],ref['ad']
        for label,n in [('baseline','BASE_POINT'),('proposed','PROPOSED'),('final','ADMITTED')]:
            mismatch=sa[n].astype(bool)^ra[n].astype(bool)
            if mismatch.any() and label not in first[pol]:first[pol][label]=dict(candidate_stage=k,affected_map=k+1 if k<10 else None,pixels=ids(mismatch),timing=timing(k))
        shared=partition['masks']['shared'];bd=sa['BRIGHTNESS']-ra['BRIGHTNESS']
        if np.any(bd[shared]!=0) and 'shared_brightness' not in first[pol]:first[pol]['shared_brightness']=dict(candidate_stage=k,affected_map=k+1 if k<10 else None,maximum_absolute=float(np.max(abs(bd[shared]))))
        for b in BRANCHES[1:]:
            st=states[pol,b,k];products=Path(st['state']['total']['path']).parent;refprod=Path(ref['state']['total']['path']).parent
            rec=rd.yaml(st['state']['receipt']['path']);rr=rd.yaml(ref['state']['receipt']['path']);require(len(rec['segments'])==len(rr['segments']),'segment count changed')
            res,rh=rd.image(products/'residual.fits');rest,_=rd.image(products/'restored.fits');rres,_=rd.image(refprod/'residual.fits');rrest,_=rd.image(refprod/'restored.fits')
            compatible(st['header'],rh);bv=st['image']['SCI_SUPPORT'].astype(bool)&ri['SCI_SUPPORT'].astype(bool)
            parts={n:np.where(bv,a-bb,np.nan) for n,a,bb in [('total',st['image']['PRIMARY'],ri['PRIMARY']),('cleaned',res['PRIMARY'],rres['PRIMARY']),('restored',rest['PRIMARY'],rrest['PRIMARY'])]}
            err=float(np.max(abs(parts['total'][bv]-parts['cleaned'][bv]-parts['restored'][bv])));require(err<=1e-8,'saved cleaning/restoration identity')
            previous=states[pol,b,k-1];refprevious=states[pol,'REF',k-1];prevdm,pv,_=difference(previous['candidate']['CANDIDATE'],previous['candidate']['VALID'].astype(bool),refprevious['candidate']['CANDIDATE'],refprevious['candidate']['VALID'].astype(bool))
            remap=float(np.max(abs(crop(parts['restored'],st['header'])[pv]-prevdm[pv])));require(remap<=1e-8,'saved restored map does not match applied m(k-1)')
            segments=[]
            for sg,rg in zip(rec['segments'],rr['segments']):
                a,bb=sg['source_segment'],rg['source_segment'];require(all(a[n]==bb[n] for n in ('network','scan','detector_indices','native_interval')),'unmatched segment identity')
                mm=rd.array(mean_path(products,sg));rm=rd.array(mean_path(refprod,rg));require(mm.shape==rm.shape,'centering shape');md=mm-rm
                segments.append(dict(network=a['network'],scan=a['scan'],mean_difference=magnitude(md,np.ones(md.shape,bool)),
                    source_mean_path=str(mean_path(products,sg)),reference_mean_path=str(mean_path(refprod,rg)),
                    state={n:sg[n] for n in ('converged','rank','iterations','retained','application_failed_times','input_eligibility_changed','output_cause_changed','projection_support_counts','basis_projector_Frobenius_difference')},
                    reference_state={n:rg[n] for n in ('converged','rank','iterations','retained','application_failed_times','input_eligibility_changed','output_cause_changed','projection_support_counts','basis_projector_Frobenius_difference')},
                    objective_terminal=sg['objective'][-1],reference_objective_terminal=rg['objective'][-1],mean_restoration='not restored by parent convention',basis_distance_scope='each distance is to original parent, not branch-minus-REF'))
            cleaning.append(dict(policy=pol,branch=b,**timing(k),accounting_error=err,restored_candidate_error=remap,segments=segments,
                parts={n:dict(full=magnitude(a,bv),seed_shape=shape_components(crop(a,st['header']),crop(bv,st['header']),shape)[0]) for n,a in parts.items()}))
            if b=='D-seed-shoulder':
                for n,a in parts.items():arrays[f'{pol}/cleaning/{k}/{n}']=crop(a,st['header'])
      # Existing signed products only. No containing-pixel projection is executed.
      for k in range(11):
        known={};deltas={};domains={}
        for b in BRANCHES[1:4]:
            file=PARENT/'report/a2000'/pol/b/f'cycle-{k:02d}.fits';im,h=rd.image(file);old=report['policies'][pol]['branches'][b][k];domain=im['BENCHMARK'].astype(bool)
            measured=signed_response(im['DELTA'],im['KNOWN'],domain)
            for n in ('g','eta','rho'):require(abs(measured[n]-old['benchmark_response'][n])<=1e-8,'cached signed response mismatch')
            st=states[pol,b,k];ref=states[pol,'REF',k];paired(st['state'],ref['state']);compatible(st['header'],ref['header']);compatible(h,st['header'])
            dd,dv,_=difference(st['image']['PRIMARY'],st['image']['SCI_SUPPORT'].astype(bool),ref['image']['PRIMARY'],ref['image']['SCI_SUPPORT'].astype(bool));require(np.allclose(dd[domain],im['DELTA'][domain],rtol=1e-11,atol=1e-9),'saved branch difference changed')
            response.append(dict(policy=pol,branch=b,**timing(k),response=measured,native_response=old['native_response'],benchmark_complete=old['benchmark_complete'],
                original_benchmark_pixels=old['original_benchmark_pixels'],benchmark_supported_pixels=old['benchmark_supported_pixels'],support=old['support'],benchmark_domain=old['benchmark_domain'],
                measurement_difference=compare_measurements(st['state']['measurement'],ref['state']['measurement']),ideal=old.get('ideal_measurement')))
            for n in ('DELTA','KNOWN','RESIDUAL','COMMON'):arrays[f'{pol}/response/{b}/{k}/{n}']=crop(im[n],h)
            known[b]=crop(im['KNOWN'],h);deltas[b]=crop(im['DELTA'],h);domains[b]=crop(im['COMMON'].astype(bool),h)
        valid=domains['C-core-plus']&domains['C-core-minus'];ev,ideal,nonlinear=signed_even(deltas['C-core-plus'],deltas['C-core-minus'],known['C-core-plus'],known['C-core-minus'],valid)
        stored,eh=rd.image(PARENT/'report/a2000'/pol/f'even-{k:02d}.fits');require(np.allclose(ev[valid],stored['EVEN'][valid],rtol=1e-11,atol=1e-9),'saved even mismatch')
        evenrows.append(dict(policy=pol,**timing(k),even=magnitude(ev,valid),ideal_even=magnitude(ideal,valid),even_minus_ideal=magnitude(nonlinear,valid)))
        arrays[f'{pol}/even/{k}']=ev;arrays[f'{pol}/ideal_even/{k}']=ideal
      for sel in POLICIES:
        d=PARENT/'report/a2000'/pol/'cross-candidates'/sel;ad,h=rd.image(d/'admission.fits');c,_=rd.image(d/'candidate.fits');meta=rd.json(d/'admission.json')
        require(np.array_equal(ad['MEASURED'],states[pol,'REF',10]['ad']['MEASURED'],equal_nan=True),'cross candidate map mismatch')
        cross.append(dict(producing_policy=pol,selecting_policy=sel,unapplied=True,summary=meta,
            audits={q:score_audit(ad,q) for q in ('CORR','DIAG')},pixels={n:ids(ad[n].astype(bool)) for n in ('BASE_POINT','PROPOSED','ADDED','ADMITTED')}))
        for n,a in ad.items():arrays[f'cross/{pol}/{sel}/{n}']=a
    # Complete evidence ledger for union of every within-policy differing final or
    # proposed membership in any branch/stage, not a hand-picked pixel subset.
    implicated=np.zeros((41,41),bool)
    for pol in POLICIES:
      for b in BRANCHES[1:]:
       for k in range(1,11):
        a,r=states[pol,b,k]['ad'],states[pol,'REF',k]['ad']
        for n in ('BASE_POINT','PROPOSED','ADMITTED'):implicated|=a[n].astype(bool)^r[n].astype(bool)
    for pol in POLICIES:
      for b in BRANCHES:
       for k in range(11):
        a=states[pol,b,k]['ad'];c=states[pol,b,k]['candidate'];B=states[pol,b,k]['state']['measurement']['fit']['parameters'][6]
        for idx in np.flatnonzero(implicated):
            row,col=divmod(int(idx),41);q=dict(policy=pol,branch=b,stage=k,pixel_id=int(idx),row=row,column=col,east=x[row,col],north=y[row,col],
                east_from_original_source=x[row,col]-z['parameters'][1],north_from_original_source=y[row,col]-z['parameters'][2],
                east_from_seed=x[row,col]-p['injection_geometry']['C-shoulder']['parameters'][1],north_from_seed=y[row,col]-p['injection_geometry']['C-shoulder']['parameters'][2],
                fitted_constant=B,pointwise_margin=a['BRIGHTNESS'][row,col]/z['base_scale'][row,col]-3,
                selected_brightness=c['CANDIDATE'][row,col],full_footprint='certified by true saved availability; false reason not cached',
                weight_concentration=1/a['NEFF_W'][row,col] if a['NEFF_W'][row,col]>0 else None)
            q.update({n:a[n][row,col] for n in a});q['excluded_by_core_or_baseline']=bool(a['K0'][row,col] or a['BASE_POINT'][row,col]);q['rejected_connectivity']=bool(a['PROPOSED'][row,col] and not a['ADDED'][row,col])
            for name in ('CORR','DIAG'):q[name+'_margin']=a[name+'_SCORE'][row,col]-p['thresholds'][name] if a[name+'_VALID'][row,col] else None
            evidence.append(q)
    arrays.update(x=x,y=y,shape=shape,radius=radius,implicated=implicated,original_template=z['q'])
    computational=time.perf_counter()-start
    result=dict(source_revision=rev,parent_manifest=p['parent_manifest'],seed=seedrows,partitions=partrows,selection=selection,measurements=allrows,
        first_divergence=first,response=response,even=evenrows,cleaning=cleaning,cross_candidates=cross,aggregate_max_error=max(aggregate_errors),score_audits=audits,
        implicated_pixels=ids(implicated),geometry=dict(header=geometry['header'],original_parameters=z['parameters'],major=geometry['major'],seed=p['injection_geometry']['C-shoulder']),
        unavailable=['individual neighbor weights/contributions','raw unnormalized numerator and weight normalizer','explicit full-footprint flags for unavailable tests','causal separation of changed cleaning from changed selection without a new intervention'],
        analysis_seconds=computational,input_verification_before=before)
    save(out/'RESULTS.json',result);save(out/'PIXEL_EVIDENCE.json',evidence);np.savez_compressed(out/'ARRAYS.npz',**arrays)
    render_start=time.perf_counter();draw(out,result,arrays);tables(out,result);render_seconds=time.perf_counter()-render_start
    after=authenticate(p);save(out/'RUNTIME.json',dict(analysis_seconds=computational,render_and_tables_seconds=render_seconds,verification_before=before,verification_after=after,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rss_scope='macOS process high-water RSS, including imported libraries; no child workers for analysis',
        consumed_paths=sorted(rd.used),bound_but_not_opened=sorted(set(p['inputs'])-rd.used)))
    print(json.dumps(dict(stage='complete',rows=len(allrows),seed_rows=len(seedrows),response_rows=len(response),evidence_pixels=len(ids(implicated)),analysis_seconds=computational,render_seconds=render_seconds)))


def draw(out,s,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'BASE-POINT':'#333333','CORR':'#0072b2','DIAG':'#d55e00'}
    fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for pol in POLICIES:
        rows=[r for r in s['seed'] if r['policy']==pol];ks=[r['map_stage'] for r in rows]
        for z,key,sub in [(ax[0,0],'map_shape','coefficient'),(ax[0,1],'candidate_shape','coefficient')]:z.plot(ks,[r[key][sub] for r in rows],'.-',label=pol,color=colors[pol])
        for key,style in [('parallel','-'),('orthogonal','--')]:ax[0,2].plot(ks,[r['map_shape'][key]['norm'] for r in rows],style,color=colors[pol],label=pol+' '+key)
        for n,style in [('shared','-'),('seed_only','--'),('REF_only',':')]:
            rows2=[r for r in s['partitions'] if r['policy']==pol];ax[1,0].plot([r['stage'] for r in rows2],[r['terms'][n]['total']['coefficient'] for r in rows2],style,color=colors[pol],label=pol+' '+n)
        data=[r for r in s['response'] if r['policy']==pol and r['branch']=='C-shoulder'];ax[1,1].plot(ks,[r['response']['g'] for r in data],'.-',color=colors[pol],label=pol)
        ax[1,2].plot(ks,[r['measurement_difference']['background'] for r in rows],'.-',color=colors[pol],label=pol)
    titles=['Seed map coefficient (shoulder units)','Candidate coefficient (m10 unapplied)','Map norms on finite seed domain','Candidate projection partition','Injected shoulder gain (own REF)','Seed fitted-constant difference [mJy/beam]']
    for z,t in zip(ax.flat,titles):z.set(title=t,xlabel='saved stage');z.grid(alpha=.2);z.legend(fontsize=6)
    for z in (ax[0,0],ax[0,1],ax[0,2]):z.set_yscale('symlog',linthresh=1e-6)
    fig.suptitle('a2000 saved histories: seed target is zero; data target is its injection reference');fig.savefig(out/'seed-history.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
    for j,(k,name) in enumerate([(1,'delta'),(1,'candidate_delta'),(10,'delta'),(10,'candidate_delta')]):
        limit=max(np.nanmax(abs(a[f'{pol}/seed/{name}/{k}'])) for pol in POLICIES)
        for i,pol in enumerate(POLICIES):
            z=ax[i,j];im=z.imshow(a[f'{pol}/seed/{name}/{k}'],origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-limit,vmax=limit)
            # Read archived actual memberships; no contour-derived selection.
            for branch,color in [('REF','black'),('D-seed-shoulder','limegreen')]:
                z.contour(a['x'],a['y'],a[f'{pol}/{branch}/{k}/actual_selected'],levels=[.5],colors=[color],linewidths=.65)
            z.set(title=f'{pol} '+('T' if name=='delta' else 'm')+str(k),xlabel='east [arcsec]',ylabel='north [arcsec]');fig.colorbar(im,ax=z,shrink=.7)
    fig.suptitle('Seed minus own REF [mJy/beam]; shared column scales; REF black / seed green\nT1 applies m0; T10 applies m9; m10 shown only as unapplied selection');fig.savefig(out/'seed-maps.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
    for i,name in enumerate(('C-core-plus','C-core-minus','even')):
        ims=[a[f'{pol}/even/10'] if name=='even' else a[f'{pol}/response/{name}/10/RESIDUAL'] for pol in POLICIES];limit=max(np.nanmax(abs(v)) for v in ims)
        for j,(pol,v) in enumerate(zip(POLICIES,ims)):
            z=ax[i,j];im=z.imshow(v,origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-limit,vmax=limit)
            ref=a[f'{pol}/REF/9/actual_selected'].astype(bool)
            branch='C-core-plus' if name=='even' else name;sel=a[f'{pol}/{branch}/9/actual_selected'].astype(bool);m=sel^ref
            z.scatter(a['x'][m],a['y'][m],facecolors='none',edgecolors='black',s=18,linewidths=.5)
            z.set(title=pol+' '+name,xlabel='east [arcsec]',ylabel='north [arcsec]');fig.colorbar(im,ax=z,shrink=.7)
    fig.suptitle('T10 compact residuals and even map [mJy/beam]; shared row scales\nCircles: applied m9 branch/REF membership differences (plus branch for even row)');fig.savefig(out/'compact-residuals.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for pol in POLICIES:
        for b,style in [('C-core-plus','-'),('C-core-minus','--')]:
            rows=[r for r in s['response'] if r['policy']==pol and r['branch']==b];ks=[r['map_stage'] for r in rows]
            ax[0,0].plot(ks,[r['response']['eta'] for r in rows],style,color=colors[pol],label=pol+' '+b)
            ax[0,1].plot(ks,[r['measurement_difference']['peak'] for r in rows],style,color=colors[pol])
            ax[0,2].plot(ks,[r['measurement_difference']['background'] for r in rows],style,color=colors[pol])
        rows=[r for r in s['even'] if r['policy']==pol];ax[1,0].plot([r['map_stage'] for r in rows],[r['even']['rms'] for r in rows],'.-',color=colors[pol],label=pol)
        rows=[r for r in s['selection'] if r['policy']==pol and r['branch']=='D-seed-shoulder'];ax[1,1].plot([r['map_stage'] for r in rows],[r['masks']['supplement']['count'] for r in rows],'.-',color=colors[pol],label=pol+' seed')
        rows=[r for r in s['selection'] if r['policy']==pol and r['branch']=='REF'];ax[1,1].plot([r['map_stage'] for r in rows],[r['masks']['supplement']['count'] for r in rows],':',color=colors[pol],label=pol+' REF')
        rows=[r for r in s['cleaning'] if r['policy']==pol and r['branch']=='D-seed-shoulder'];ks=[r['map_stage'] for r in rows]
        for name,style in [('cleaned','-'),('restored','--')]:ax[1,2].plot(ks,[r['parts'][name]['seed_shape']['coefficient'] for r in rows],style,color=colors[pol],label=pol+' '+name)
    titles=['Compact distortion eta (+ solid / − dashed)','Compact fitted-peak increment [mJy/beam]','Compact constant increment [mJy/beam]','Even RMS [mJy/beam]','Saved supplemental membership','Seed difference: cleaned / restored coefficient']
    for z,t in zip(ax.flat,titles):z.set(title=t,xlabel='saved stage');z.grid(alpha=.2)
    for z in (ax[0,0],ax[1,0],ax[1,1],ax[1,2]):z.legend(fontsize=6)
    fig.suptitle('Timing of existing compact asymmetry, selection and seed cleaning/restoration');fig.savefig(out/'timing-and-cleaning.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    labels=list(POLICIES);positions=np.arange(3)
    seed=[next(r for r in s['seed'] if r['policy']==pol and r['map_stage']==10) for pol in POLICIES]
    for j,(name,title) in enumerate([('peak_percent_own_REF','Seed fitted peak / REF peak [%]'),('east','Seed fitted east shift [arcsec]'),('background','Seed fitted constant shift [mJy/beam]')]):ax[0,j].bar(positions,[r['measurement_difference'][name] for r in seed],color=list(colors.values()));ax[0,j].set(title=title,xticks=positions,xticklabels=labels)
    for j,b in enumerate(BRANCHES[1:4]):
        for n,shift in [('peak',-.23),('fixed_template',0),('aperture_2',.23)]:
            vals=[]
            for pol in POLICIES:
                row=next(r for r in s['response'] if r['policy']==pol and r['branch']==b and r['map_stage']==10);q=next(q for q in row['ideal']['comparisons'] if q['name']==n);vals.append(percent(q['absolute_discrepancy'],q['ideal_increment']))
            ax[1,j].bar(positions+shift,vals,width=.23,label=n)
        ax[1,j].set(title=b+': (actual − ideal) / signed ideal [%]',xticks=positions,xticklabels=labels);ax[1,j].legend(fontsize=7);ax[1,j].axhline(0,color='gray',lw=.7)
    fig.suptitle('Different practical measures; denominators stated; no new uncertainty or significance');fig.savefig(out/'practical-scales.png',dpi=140);plt.close(fig)


def tables(out,s):
    f=lambda x:f'{x:.8g}' if x is not None else 'unavailable'
    lines=['# Saved a2000 numerical diagnosis','', 'All values derive from authenticated cached products. Brightness mJy/nominal beam; angular offsets arcsec; apertures brightness × arcsec². m(k−1) produces T(k); m10 is unapplied. Full machine-readable histories: RESULTS.json and PIXEL_EVIDENCE.json; mask identities and small arrays: ARRAYS.npz.','',
        '| Policy | Stage | Seed map coefficient | Candidate coefficient | Map seed-domain norm | Orthogonal norm | Full-map max | Peak difference | Peak / own REF % | Aperture-2 difference |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['seed']:
        if r['map_stage'] not in (0,1,3,5,9,10):continue
        m=r['measurement_difference'];values=[r['map_shape']['coefficient'],r['candidate_shape']['coefficient'],r['map_shape']['total']['norm'],r['map_shape']['orthogonal']['norm'],r['full_map']['maximum_absolute'],m['peak'],m['peak_percent_own_REF'],m['apertures'][2]['source_relative']]
        lines.append(f"| {r['policy']} | {r['map_stage']} | "+' | '.join(map(f,values))+' |')
    lines+=['','## First post-intervention divergence','', '| Policy | Record | Selected candidate | Could affect map | Pixel IDs or maximum brightness difference |','|---|---|---:|---|---|']
    for pol,records in s['first_divergence'].items():
        for key,r in records.items():lines.append(f"| {pol} | {key} | m{r['candidate_stage']} | T{r['affected_map']} | {r.get('pixels',r.get('maximum_absolute'))} |")
    lines+=['','## T10 candidate partition','', '| Policy | Term | Pixels | Projection | Constant contribution | Aperture-2 | Constant aperture-2 | Norm |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in s['partitions']:
        if r['stage']!=10:continue
        for name,q in r['terms'].items():lines.append(f"| {r['policy']} | {name} | {q['pixels']['count']} | "+' | '.join(f(v) for v in [q['total']['coefficient'],q['fitted_constant']['coefficient'],q['apertures'][2]['value'],q['constant_apertures'][2]['value'],q['total']['total']['norm']])+' |')
    lines+=['','## Availability','', '| Analysis | Availability / limit |','|---|---|',
        '| Stage histories, candidate partitions and practical measurements | Cached products and arithmetic; no refits |',
        '| Score audit | Saved EVIDENCE / sqrt(saved VAR) on saved valid tests |',
        '| Full-footprint validity | True saved score availability certifies combined predicate; reason for false unavailable |',
        '| Per-neighbor weights / signed contributions | Not serialized; no reconstruction or attribution |',
        '| Selection connectivity | Saved PROPOSED versus ADDED; no connectivity reevaluation |',
        '| Cleaning/centering | Saved residual/restored maps, detector means and receipt summaries; no new basis application |',
        '| Causal attribution or asymptotic behavior | Not established by saved observational histories |','']
    for name in ('seed-history','seed-maps','compact-residuals','timing-and-cleaning','practical-scales'):lines.append(f'![{name}]({out/name}.png)\n')
    (out/'NUMERICAL_REPORT.md').write_text('\n'.join(lines)+'\n')


def guarded_run(out):
    """Observe actual repository calls; reject any imported execution route."""
    import importlib.abc
    allowed={Path(__file__).resolve(),Path(__file__).with_name('point_saved_diagnosis.py').resolve()};calls={}
    class BlockScientificImports(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            name=fullname.split('.')[0]
            if name.startswith(('point_','run_point_','report_point_','inspect_point_')) and name!='point_saved_diagnosis':raise RuntimeError('forbidden scientific route import '+fullname)
    blocker=BlockScientificImports();sys.meta_path.insert(0,blocker)
    def trace(frame,event,arg):
        if event=='call':
            name=frame.f_code.co_filename
            if name.startswith(str(REPO/'tools/timestream_successor')):
                file=Path(name).resolve();require(file in allowed,'out-of-scope repository function '+name)
                key=file.name+':'+frame.f_code.co_name;calls[key]=calls.get(key,0)+1
    protected=json.loads((out/'PROSPECTIVE.json').read_text())['inputs']
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            name=str(Path(os.fsdecode(args[0])).resolve());flags=args[2]
            require(not ((name in protected or name.startswith(str(PARENT)+'/')) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)),'parent write attempted')
        if event=='subprocess.Popen':require(Path(str(args[0])).name=='git','non-Git subprocess forbidden')
    sys.addaudithook(audit);sys.setprofile(trace)
    completed=False;error=None
    try:
        run(out);completed=True
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        sys.setprofile(None);sys.meta_path.remove(blocker)
        save(out/'EXECUTION_ROUTE.json',dict(execution_completed=completed,error=error,repository_calls=calls,allowed_repository_files=[str(x) for x in sorted(allowed)],
            scientific_route_imports='blocked',non_Git_subprocesses='blocked',parent_writes='blocked',
            conclusion=('Completed cached-only diagnosis.' if completed else 'Diagnosis aborted; consult error and recorded calls, not a successful completion.')+' No fits, candidate/policy evaluation, projection, mapmaking, cleaning, fields or feedback are reachable through this route.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','analyze']);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    prepare(args.output.resolve()) if args.stage=='prepare' else guarded_run(args.output.resolve())
