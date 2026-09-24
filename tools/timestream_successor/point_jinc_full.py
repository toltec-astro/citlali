"""Bounded map-only POINT JINC execution and immutable prepared input binding.

Orchestration only: native AST owns coordinates, mapmaking owns deposition and
certificates. Saved z+operand histories are consumed; no PTC/FRUIT call exists.
"""
import argparse,json,os,shutil,subprocess,time,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import numpy as np
from scipy.ndimage import maximum_filter
from point_frozen_shoulder import ancestors,CD,ARRAYS,saved_context,close
from point_fruit_trajectory import bound,verified,manifest,verify_manifest,FixedMeasurements
from point_fruit_diagnostics import load
from point_additive_diagnostics import save_json
from point_empirical_io import write_fits
from point_empirical import coordinates,source_patch
from point_jinc_interval import enclose_cache
from inspect_forward_map import require,digest,read_yaml
from inspect_point_source import gaussian
from run_point_response_cd import source
from run_successor_observation import run as invoke

REPO=Path(__file__).resolve().parents[2]
PREP=CD.parent/'successor-point-152389-map-only-jinc-20260924'
PREP_HASH='ee1e289cc17168505aa40c29fecf4d8c638cb1e413bfefedfbfa9a796e40d908'
PROFILE='POINT-152389-JINC-NUMERICAL-ADEQUACY-v1'
MATRIX=(('REF',0),('REF',10),('C-core-plus',10),('C-core-minus',10),('C-shoulder',10))
HALF={'a1100':8,'a1400':10,'a2000':14}
PHASES=[(x,y) for y in (-1/3,0.,1/3) for x in (-1/3,0.,1/3)]

def read(p):return json.loads(Path(p).read_text())
def key(a,b,k):return f'{a}-{b}-{k:02d}'
def original_context(auth,owner,array):
    ctx=saved_context(auth,owner,array);f=FixedMeasurements.__new__(FixedMeasurements)
    f.header=ctx['header'];f.beam={'a1100':5.,'a1400':7.5,'a2000':10.}[array]
    f.parameters=ctx['p'];f.x=ctx['x'];f.y=ctx['y'];f.disk=np.hypot(f.x,f.y)<=40
    f.radius=ctx['radius'];f.major=ctx['major'];f.background=f.disk&(f.radius>3*f.major)
    f.edges=np.arange(0,42,2);f.weights=ctx['q'];f.scale=ctx['scale'];f.noise_valid=ctx['noise_valid']
    return ctx,f

def cache_prepare(args):
    array,header,helper,out=args;start=time.perf_counter();d=out/'cache'/array;d.mkdir(parents=True)
    index=ARRAYS.index(array);mm=(1.1,1.4,2.)[index];b=(1.67,2.17,3.17)[index]
    spec=dict(rows=int(header['NAXIS2']),columns=int(header['NAXIS1']),phases=16,pixel_matrix_rad=[header['CDELT1']*np.pi/180,0.,0.,header['CDELT2']*np.pi/180],scale_rad=(mm/1000)/45,a=1.1,b=b,c=2.,rmax=3.)
    q=dict(schema='point-152389-jinc-diagnostic-v1',mode='cache',spec=spec);save_json(d/'request.json',q)
    subprocess.run([str(helper),str(d/'request.json'),str(d/'native')],check=True)
    native=read_yaml(d/'native/receipt.yaml');require(native['half_width']==HALF[array],'selected square support')
    prod=np.fromfile(d/'native/cache.f64',dtype='<f8');t=time.perf_counter();enclosure=enclose_cache(prod,HALF[array],[header['CDELT1'],header['CDELT2']],str(mm),str(b));elapsed=time.perf_counter()-t
    for n in ('errors','lower','upper'):enclosure[n].astype('<f8').tofile(d/(n+'.f64'))
    record=dict(array=array,profile=PROFILE,spec=spec,exact_scale_formula=f'({mm}/1000)/45',exact_shape_decimals=['1.1',str(b),'2','3'],nsub=16,half=HALF[array],cache=bound(d/'native/cache.f64'),coefficient_errors=bound(d/'errors.f64'),lower=bound(d/'lower.f64'),upper=bound(d/'upper.f64'),
        entries=len(prod),max_coefficient_error=float(enclosure['errors'].max()),unique_radii=enclosure['unique_radii'],decimal_precision=enclosure['precision'],pi=enclosure['pi'],first_J1_root=enclosure['first_root'],enclosure_seconds=elapsed,wall_seconds=time.perf_counter()-start,native_environment=native,
        method=bound(REPO/'validation/point_152389_map_only_jinc_full_2026-09-24/JINC_NUMERICAL_ARGUMENT.md'),helper=bound(helper),status='all finite selected cache coefficients enclosed')
    require(np.all(enclosure['lower']<=enclosure['upper']),'coefficient interval order');save_json(d/'CERTIFICATE.json',record);manifest(d);return record

def save_array(p,a,dtype='<f8'):
    np.asarray(a,dtype=dtype).tofile(p);return bound(p)

def prepare(out,cli,helper):
    began=time.perf_counter();revision=source();require(not out.exists(),'fresh full-run bundle required')
    auth,owner=ancestors();auth.manifest(PREP,PREP_HASH);old=auth.json(PREP/'RECOVERY.json')
    binding=read(REPO/'validation/point_152389_map_only_jinc_2026-09-24/owner-numerical-binding.json')
    out.mkdir();(out/'prepared').mkdir();(out/'freeze').mkdir();(out/'ast').mkdir()
    shutil.copy2(REPO/'validation/point_152389_map_only_jinc_full_2026-09-24/OWNER_DIRECTIVE.md',out/'OWNER_DIRECTIVE.md')
    shutil.copy2(REPO/'validation/point_152389_map_only_jinc_full_2026-09-24/JINC_NUMERICAL_ARGUMENT.md',out/'NUMERICAL_ARGUMENT.md')
    arrays={};old_records={(r['array'],r['branch'],r['stage']):r for r in old['results']};headers={}
    # Exact selected parameters and inputs are recorded before any accumulation.
    for a in ARRAYS:
        r=old_records[a,'REF',0];data,h=auth.fits(r['archived_total']['path']);headers[a]=h
        require(binding['P'][a]['P_mJy_per_nominal_beam']==abs(owner['original_cycles'][a][10]['measurement']['fit']['parameters'][0]),'frozen peak mismatch')
        for n in ('source_measurement','source_total'):auth.check(binding['P'][a][n]['path'])
        ad=out/'prepared'/a;ad.mkdir();ctx,f=original_context(auth,owner,a)
        np.savez(ad/'fixed.npz',parameters=f.parameters,x=f.x,y=f.y,radius=f.radius,weights=f.weights,scale=f.scale,noise_valid=f.noise_valid)
        save_json(ad/'fixed.json',dict(major=f.major,beam=f.beam,header=dict(f.header),measurement_definition='saved seven-parameter Gaussian plus constant; original search/bounds/weights; original M0 template/apertures; 40arcsec physical fit disk'))
        ab=CD.parent/'successor-point-152389-post-fruit-ab-20260923'/f'companion/{a}'
        domains=auth.check(ab/'frozen-domains.npz');geometry=auth.json(ab/'frozen-domains.json')
        with np.load(domains) as z:
            base=maximum_filter(z['source_mask'].astype('u1'),size=2*(HALF[a]+1)+1,mode='constant',cval=0)>0
            larger=maximum_filter(z['larger_source_mask'].astype('u1'),size=2*(HALF[a]+1)+1,mode='constant',cval=0)>0
        np.savez(ad/'noise-masks.npz',base=base,larger=larger)
        parents=owner['parents'][a]
        plan=auth.yaml(parents['parent_plan']['path']);uniform={}
        for entry in plan['ordered_column_domain']:
            cp=Path(parents['parent_plan']['path']).parent/entry['coefficient_file'];auth.bind(cp,entry['coefficient_sha256']);cf=auth.yaml(cp)
            require(cf['scalar']==1 and cf['unit']==1 and cf['family']=='SCI-PTC:uniform_constant@draft-0.1' and cf['profile']=='SCI-PTC:uniform_coefficient_handoff@draft-0.1','uniform payload differs')
            require(cf['PTC_publication_sha256']==entry['publication_sha256'] and cf['stored_support_reference']==entry['support_reference'],'coefficient occurrence association differs')
            uniform[Path(entry['coefficient_file']).name.removesuffix('-coefficient.yaml')]=dict(payload=auth.ref(cp),occurrence=entry,scalar=cf['scalar'])
        ar=dict(schema='point-152389-jinc-ast-v1',map_request=parents['map_request'],parent_plan=parents['parent_plan'],archived_input_manifest=r['input_manifest'],array_index=ARRAYS.index(a),WCS=dict(h))
        save_json(out/'ast'/(a+'.json'),ar)
        arrays[a]=dict(uniform_payloads=uniform,header=dict(h),peak=binding['P'][a],mask_source=auth.ref(domains),mask_geometry=geometry,dilation_half_width=HALF[a]+1,compact_profile=owner['geometry'][a]['C-core-plus'],ast_request=bound(out/'ast'/(a+'.json')))
    save_json(out/'freeze/profile.json',dict(profile=PROFILE,authority=bound(out/'OWNER_DIRECTIVE.md'),argument=bound(out/'NUMERICAL_ARGUMENT.md'),frozen_peaks=binding['P'],parameters={a:dict(scale=['(1.1/1000)/45','(1.4/1000)/45','(2.0/1000)/45'][i],a='1.1',b=['1.67','2.17','3.17'][i],c='2',rmax='3',nsub=16) for i,a in enumerate(ARRAYS)},exact_tolerance_decimals=['0.000001','0.000001'],ancillary_relative_tolerance='0.000001',input_arithmetic='fixed authenticated binary64 sample/omega/frequency/native AST inputs',numerical_scope='all used JINC signal and Q/T values; local cause bits enforce separation/bounds/range',source=revision,helper=bound(helper),cli=bound(cli)))
    # Independent native AST exports run across arrays, each with one library thread.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={a:pool.submit(invoke,cli,out/'ast'/(a+'.json'),out/'ast'/a,1,'map') for a in ARRAYS}
        for a,j in jobs.items():require(j.result()==0,'native AST export failed '+a)
    with ProcessPoolExecutor(max_workers=3) as pool:cache=list(pool.map(cache_prepare,[(a,headers[a],helper,out) for a in ARRAYS]))
    cache={c['array']:c for c in cache};states=[];geometries={}
    for a in ARRAYS:
        h=headers[a];shape=(h['NAXIS2'],h['NAXIS1']);ast=read_yaml(out/'ast'/a/'products/receipt.yaml');seg_ast={s['name']:s for s in ast['segments']}
        for p,sha in read_yaml(out/'ast'/a/'products/input-manifest.yaml').items():auth.bind(p,sha);auth.check(p)
        ad=out/'prepared'/a
        for branch,k in MATRIX:
            oldstate=old_records[a,branch,k];products=Path(oldstate['archived_total']['path']).parent;r=auth.yaml(products/'receipt.yaml')
            for p,sha in auth.yaml(products/'input-manifest.yaml').items():auth.bind(p,sha);auth.check(p)
            previous=auth.json(CD/f'run/{branch}/{a}/SUMMARY.json')['histories'];req=auth.json(previous[k]['request']['path'])
            if k:require(req['candidate']==previous[9]['candidate'],'applied m9 differs')
            pixels=np.fromfile(auth.check(products/'operator-pixel.i32'),dtype='<i4');cursor=0;gs=[];zs=[];vs=[];nets=[];rows=[];tangents=[]
            for segment in r['segments']:
                original=segment['source_segment'];stem=next(p.removesuffix('-input.f64') for p in original['files'] if p.endswith('-input.f64'));name=f"network-{original['network']}-{stem}";astseg=seg_ast[name]
                require(original==astseg['source_segment'],'native AST segment/occurrence differs')
                nt,no,nd=(original[x] for x in ('scheduled_times','output_detectors','detectors'));columns=original['fit_columns_in_output']
                co=np.fromfile(verified(astseg['coordinates']),dtype='<f8').reshape(nt,no,4);cal=np.fromfile(verified(astseg['CAL_causes']),dtype='<u2').reshape(nt,no)
                clean=np.fromfile(auth.check(products/name/'cleaned.f64'),dtype='<f8').reshape(nt,no);causes=np.fromfile(auth.check(products/name/'causes.u8'),dtype='u1').reshape(nt,no)
                v=np.zeros((nt,no));v[:,columns]=np.fromfile(auth.check(products/name/'operand.f64'),dtype='<f8').reshape(nt,nd)
                retained=causes==0;count=int(retained.sum());require(count==segment['retained'],'saved retained domain')
                expected=pixels[cursor:cursor+count];cursor+=count
                eligible=retained&(cal==0)&np.isfinite(co[...,:2]).all(axis=-1)
                px=co[...,0][retained];py=co[...,1][retained];inside=(px>=.5)&(py>=.5)&(px<shape[1]+.5)&(py<shape[0]+.5)
                actual=np.full(count,-1,dtype=np.int64);use=eligible[retained]&inside;actual[use]=np.floor(py[use]-.5).astype('i8')*shape[1]+np.floor(px[use]-.5).astype('i8')
                require(np.array_equal(actual,expected),'continuous AST ordinary membership replay failed '+name)
                require(eligible.sum()==use.sum(),'saved full grid does not contain every eligible occurrence')
                require(np.isfinite(clean[eligible]).all() and np.isfinite(v[eligible]).all(),'invalid same-sample values')
                require(name in arrays[a]['uniform_payloads'],'missing coefficient parent')
                if branch=='REF' and k==10:tangents.append(co[...,2:4][eligible]*3600)
                g=np.column_stack([co[...,0][eligible],co[...,1][eligible],np.ones(int(eligible.sum())),np.full(int(eligible.sum()),astseg['frequency_hz'])])
                gs.append(g);zs.append(clean[eligible]);vs.append(v[eligible]);nets.append(np.full(len(g),original['network'],dtype='i2'))
                rows.append(dict(name=name,source_segment=original,retained=count,admitted=len(g),CAL_or_AST_excluded=count-len(g),coordinates=astseg['coordinates'],frequency_hz=astseg['frequency_hz'],cleaned=auth.ref(products/name/'cleaned.f64'),operand=auth.ref(products/name/'operand.f64'),causes=auth.ref(products/name/'causes.u8'),saved_mean=auth.ref(products/name/'mean.f64'),saved_basis=auth.ref(products/name/'basis.f64')))
            require(cursor==len(pixels),'all saved rows consumed');g=np.concatenate(gs);z=np.concatenate(zs);v=np.concatenate(vs);net=np.concatenate(nets);total=z+v
            name=key(a,branch,k);sd=ad/name;sd.mkdir();gr=save_array(sd/'geometry.f64',g)
            # Reuse exactly matching geometry bytes, retaining branch-local lineage.
            if gr['sha256'] in geometries:
                (sd/'geometry.f64').unlink();gr=geometries[gr['sha256']]
            else:geometries[gr['sha256']]=gr
            zr=save_array(sd/'clean.f64',z);vr=save_array(sd/'operand.f64',v);tr=save_array(sd/'total.f64',total)
            pixel=(np.floor(g[:,1]-.5).astype('i8')*shape[1]+np.floor(g[:,0]-.5).astype('i8'))
            n=np.zeros(np.prod(shape));hits=np.zeros(np.prod(shape));np.add.at(n,pixel,total);np.add.at(hits,pixel,1)
            archived,ah=auth.fits(products/'total.fits');valid=archived['SCI_SUPPORT'].astype(bool)
            require(np.array_equal(hits.reshape(shape),archived['HITS']) and np.array_equal(hits.reshape(shape),archived['NORMALIZATION']),'ordinary same-input normalization')
            signal=np.divide(n,hits,out=np.full(n.shape,np.nan),where=hits>0).reshape(shape)
            checks=dict(numerator=close(n.reshape(shape),archived['NUMERATOR'],'replayed MAP N',1e-7,1e-11),signal=close(signal[valid],archived['PRIMARY'][valid],'replayed MAP m',1e-9,1e-11),hits_exact=True,continuous_membership_exact=True)
            s=dict(key=name,array=a,branch=branch,stage=k,geometry=gr,clean=zr,operand=vr,total=tr,samples=len(g),segments=rows,archived_total=auth.ref(products/'total.fits'),archived_residual=auth.ref(products/'residual.fits'),archived_restored=auth.ref(products/'restored.fits'),measurement=oldstate['original_measurement'],replay=checks,original_request=auth.ref(previous[k]['request']['path']),applied_candidate=req['candidate'] if k else None)
            save_json(sd/'STATE.json',s);states.append(s)
            if branch=='REF' and k==10:
                save_array(sd/'network.i16',net,'<i2')
                refs={}
                for pert in ('C-core-plus','C-core-minus','C-shoulder'):
                    ir=auth.json(CD/f'run/{pert}/{a}/SUMMARY.json')['histories'][10];ip=Path(ir['receipt']['path']).parent;isr=auth.yaml(ip/'receipt.yaml');byname={f"network-{x['source_segment']['network']}-"+next(p.removesuffix('-input.f64') for p in x['source_segment']['files'] if p.endswith('-input.f64')):x for x in isr['segments']};pieces=[];originals=[]
                    for rr in rows:
                        sr=rr['source_segment'];name0=rr['name'];require(byname[name0]['source_segment']==sr,'injection parent differs')
                        nt,no,nd=(sr[x] for x in ('scheduled_times','output_detectors','detectors'));ca=np.fromfile(auth.check(products/name0/'causes.u8'),dtype='u1').reshape(nt,no);ac=seg_ast[name0];cal=np.fromfile(verified(ac['CAL_causes']),dtype='<u2').reshape(nt,no);co=np.fromfile(verified(ac['coordinates']),dtype='<f8').reshape(nt,no,4)
                        pi_path=ip/name0/'injection.f64';pi=np.fromfile(auth.check(pi_path),dtype='<f8').reshape(nt,nd);full=np.zeros((nt,no));full[:,sr['fit_columns_in_output']]=pi
                        eligible=(ca==0)&(cal==0)&np.isfinite(co[...,:2]).all(axis=-1);pieces.append(full[eligible]);originals.append(auth.ref(pi_path))
                    u=np.concatenate(pieces);require(len(u)==len(g),'injection REF occurrence count');refs[pert]=dict(values=save_array(sd/(pert+'-injection.f64'),u),parents=originals)
                arrays[a]['injection_references']=refs
                p=np.array(owner['geometry'][a]['C-core-plus']['parameters']);require(p[0]==.1*binding['P'][a]['P_mJy_per_nominal_beam'],'original compact amplitude differs')
                tangent=np.concatenate(tangents);tx,ty=tangent.T;arrays[a]['phase_native_tangent_arcsec']=save_array(sd/'tangent-arcsec.f64',tangent)
                fixture=[]
                for i,(ox,oy) in enumerate(PHASES):
                    x=tx-ox*h['CDELT1']*3600;y=ty-oy*h['CDELT2']*3600;r=np.hypot(x,y);taper=np.where(r<=30,1.,np.where(r<36,(1+np.cos(np.pi*(r-30)/6))/2,0.))
                    values=gaussian(p,x,y)*taper;fixture.append(dict(phase=i,offset_pixels=[ox,oy],offset_tangent_arcsec=[ox*h['CDELT1']*3600,oy*h['CDELT2']*3600],values=save_array(sd/f'phase-{i}.f64',values)))
                arrays[a]['phases']=fixture
        arrays[a]['cache']=cache[a]
    endhash=auth.verify_all();record=dict(schema='point-152389-jinc-prepared-v1',source_revision=revision,cli=bound(cli),helper=bound(helper),arrays=arrays,states=states,consumed_inputs=auth.consumed,authentication_seconds=auth.seconds,final_authentication_seconds=endhash,wall_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        status='inputs/cache/AST prepared; no observation JINC accumulation',uniform_coefficient='exact same gamma1; producer cause0 plus CAL cause0 and finite native AST; diagnostic rejoined descendant',comparison_source='original adaptive C+D only; no REFSEQ',new_Learn_or_Apply='none: native call graph exports AST only; Python reads saved z/operand')
    save_json(out/'PREPARED.json',record);manifest(out/'prepared');manifest(out/'ast');manifest(out/'cache');return record

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare']);p.add_argument('--output',type=Path,required=True);p.add_argument('--cli',type=Path,required=True);p.add_argument('--helper',type=Path,required=True);a=p.parse_args();prepare(a.output,a.cli,a.helper)
