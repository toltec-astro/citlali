"""Reload a complete successor MAP bundle and inspect its actual numerical products.

The sparse operator's column domain is the exact ordered PTC support in plan.yaml.
No map is complete when only its signal FITS plane is supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_yaml(p):
    with Path(p).open() as f:return yaml.load(f,Loader=yaml.CSafeLoader)

def require(ok, why):
    if not ok:raise ValueError(why)

HANDOFF_COMPONENTS=('publication','coefficient','AST_coordinate','AST_binding','coefficient_QC','MAP_evaluation','MAP_eligibility')
MAP_REGISTRY_SHA='ce027c3a5c0e71291e9ea509c3a2d1e2d9971fac64763e2afc676241df7d442b'
MAP_REGISTER_SHA='a30c1b71ada4793b6ae805699f9335e9f912399e066a4af6bb68ab81456d4555'
EVALUATION_BINDINGS={
    'allocation_sha256':'2afbf10026784ac4528aa799b9f4362bf5f2bf328f77d3a3edb187c3221523f4',
    'manifest_sha256':'57229670cd540ce2f8aeecf5d1a568414d0a85b6a9d6cf5a74b7e698b6ac5ec3',
    'shared_core_sha256':'3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31',
    'PTC_MAP_boundary_sha256':'58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8',
    'AST_original_footprint_boundary_sha256':'f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4',
}

def replay_handoffs(root,plan,numerator,hits,verify_operator=True,probe_fwhm_arcsec=None):
    """Verify closed handoffs and replay exact support-ordered PTC values.

    add.at preserves occurrence order, unlike a grouped/reassociated reduction.
    Only one segment's values/coordinates are materialized at a time.
    """
    root=Path(root).resolve();offset=0;cached={};receipts={};cal_arrays={}
    replay=np.zeros(numerator.size);counts=np.zeros(numerator.size,dtype=np.int64)
    probe=np.zeros(numerator.size) if probe_fwhm_arcsec else None
    pixels=np.memmap(root/'operator-pixel.i32',dtype='<i4',mode='r')
    def checked(path,sha):
        path=Path(path).resolve()
        if path not in cached:cached[path]=digest(path)
        require(cached[path]==sha,f'component digest mismatch: {path}')
        return path
    def source_receipt(parent,sha):
        path=checked(Path(parent)/'receipt.yaml',sha)
        if path not in receipts:receipts[path]=read_yaml(path)
        return receipts[path]
    for q in plan['ordered_column_domain']:
        require(q['first_input_row']==offset,'noncontiguous operator column domain')
        paths={}
        for key in HANDOFF_COMPONENTS:
            path=(root/q[key+'_file']).resolve();require(path.is_relative_to(root),'handoff escapes bundle')
            paths[key]=checked(path,q[key+'_sha256'])
        pub,coef,ast,qc,decision=(read_yaml(paths[k]) for k in ('publication','coefficient','AST_binding','coefficient_QC','MAP_evaluation'))
        require(q['MAP_evaluation_identity']=='sha256:'+q['MAP_evaluation_sha256'],'MAP decision immutable identity mismatch')
        require(all(decision[k]==v for k,v in EVALUATION_BINDINGS.items()),'MAP source tuple mismatch')
        n=q['input_rows'];support=q['support_reference']
        require(pub['identity']==q['PTC_publication'] and pub['support_reference']==support and pub['retained']==n,'PTC support mismatch')
        require(coef['scalar']==1 and coef['unit']=='1' and coef['stored_support_reference']==support,'coefficient/support mismatch')
        require(coef['PTC_publication_sha256']==q['publication_sha256']==qc['PTC_publication_sha256'],'coefficient/QC parent mismatch')
        require(qc['payload_sha256']==q['coefficient_sha256'] and qc['logical_occurrence_domain']==support,'QC payload/domain mismatch')
        require(qc['requested']=='requested' and qc['realization']=='realized' and qc['restrictions']=={f'UQ-{i:02}':'T' for i in range(1,8)},'coefficient QC did not pass')
        require(ast['PTC_publication']==pub['identity'] and ast['CAL_parent']==pub['CAL_parent'] and ast['RTC_grid']==pub['RTC_grid'],'AST exact parent mismatch')
        require(ast['sha256']==q['AST_coordinate_sha256'] and paths['AST_coordinate'].stat().st_size==n*16,'AST payload/domain mismatch')
        require(ast['coefficient_payload_sha256']==q['coefficient_sha256'] and ast['coefficient_QC_evaluation_sha256']==q['coefficient_QC_sha256'],'AST coefficient/QC binding mismatch')
        require(decision['registry_sha256']==MAP_REGISTRY_SHA and decision['source_register_sha256']==MAP_REGISTER_SHA and
                decision['profile']=='SCI-MAP:map_upstream_admission@2' and decision['binding']=='SCI-MAP-v0.2-r0.4-2026-09-08', 'foreign MAP evaluation generation')
        require(decision['role']==plan['role'] and decision['object_domain']==support and
                [decision[k] for k in ('requested','applicability','realization')]==['requested','applicable','realized'],'MAP decision axes/domain mismatch')
        for key in ('publication','coefficient','AST_binding','coefficient_QC'):require(decision[key+'_sha256']==q[key+'_sha256'],'MAP decision parent mismatch')
        require(decision['eligibility_sha256']==q['MAP_eligibility_sha256'],'MAP eligibility binding mismatch')
        packed=np.fromfile(paths['MAP_eligibility'],dtype='u1');require(len(packed)==(n+7)//8,'MAP eligibility domain size')
        unpacked=np.unpackbits(packed,bitorder='little');require(not unpacked[n:].any(),'nonzero decision padding')
        eligible=unpacked[:n].astype(bool);require(int(eligible.sum())==decision['eligible'] and n-int(eligible.sum())==decision['ineligible'],'MAP decision count mismatch')
        source=Path(pub['source_PTC_root']);ptc=source_receipt(source,pub['source_PTC_receipt_sha256'])
        segment=pub['source_segment'];stem=next(name[:-len('-cleaned.f64')] for name in segment['files'] if name.endswith('-cleaned.f64'))
        index=int(stem.split('-')[-1]);require(ptc['segments'][index]==segment,'PTC segment receipt differs')
        require(pub['identity']=='sha256:'+pub['source_PTC_receipt_sha256']+':'+stem,'PTC publication identity differs')
        for name,sha in segment['files'].items():checked(source/name,sha)
        cols=np.asarray(segment['output_detector_indices']);shape=(segment['scheduled_times'],segment['output_detectors'])
        causes=np.fromfile(source/(stem+'-causes.u8'),dtype='u1').reshape(shape)
        retained=causes==0;require(int(retained.sum())==n,'PTC retained membership differs')
        values=np.fromfile(source/(stem+'-cleaned.f64'),dtype='<f8').reshape(shape)[retained]
        require(np.isfinite(values).all(),'nonfinite retained PTC value')
        slots=np.fromfile(source/(stem+'-slots.i64'),dtype='<i8');require(len(slots)==shape[0] and np.all(np.diff(slots)>0),'RTC slot relation invalid')
        calroot=Path(pub['source_CAL_root']);cal=source_receipt(calroot,pub['source_CAL_receipt_sha256'])
        require(ptc['source_CAL_receipt_sha256']==pub['source_CAL_receipt_sha256'] and pub['CAL_parent']=='sha256:'+pub['source_CAL_receipt_sha256'],'CAL parent differs')
        require(np.all((slots>=0)&(slots<pub['RTC_scheduled_slots'])),'RTC slot out of range')
        cal_causes=[]
        for d in cols:
            det=cal['detectors'][int(d)];name='channel-'+str(det['channel'])+'-causes.u16'
            path=checked(calroot/name,det['files'][name])
            if path not in cal_arrays:cal_arrays[path]=np.memmap(path,dtype='<u2',mode='r')
            require(len(cal_arrays[path])==pub['RTC_scheduled_slots'],'CAL cause schedule differs')
            cal_causes.append(cal_arrays[path][slots])
        cause=np.column_stack(cal_causes)[retained]
        require(np.array_equal(eligible,cause==0),'MAP decision differs from exact CAL/PTC facts')
        chunk=np.asarray(pixels[offset:offset+n]);require(len(chunk)==n,'operator column domain truncated')
        if verify_operator:
            xy=np.fromfile(paths['AST_coordinate'],dtype='<f8').reshape(n,2);w=plan['WCS']
            x=w['CRPIX1']+xy[:,0]/w['CDELT1'];y=w['CRPIX2']+xy[:,1]/w['CDELT2']
            use=eligible & np.isfinite(x)&np.isfinite(y)&(x>=.5)&(y>=.5)&(x<w['columns']+.5)&(y<w['rows']+.5)
            expected=np.full(n,-1,dtype=np.int64)
            expected[use]=np.floor(y[use]-.5).astype(np.int64)*w['columns']+np.floor(x[use]-.5).astype(np.int64)
            require(np.array_equal(chunk,expected),'serialized operator differs from exact admission/AST placement')
            np.add.at(replay,chunk[use],values[use]);np.add.at(counts,chunk[use],1)
            if probe is not None:
                sigma=probe_fwhm_arcsec/np.sqrt(8*np.log(2))
                radius2=np.sum((xy*3600/sigma)**2,axis=1)
                injected=use & (radius2<=64) # explicit finite 8-sigma template
                np.add.at(probe,chunk[injected],np.exp(-radius2[injected]/2))
        offset+=n
    require((root/'operator-pixel.i32').stat().st_size==offset*4,'operator domain byte count mismatch')
    if verify_operator:
        require(np.array_equal(counts.reshape(hits.shape),hits),'persisted exact operator differs from hits')
        require(np.array_equal(replay.reshape(numerator.shape),numerator),'persisted operator PTC replay differs from numerator')
    return None if probe is None else probe.reshape(numerator.shape)


def summarize_map_probe(response,Q,science,wcs,fwhm_arcsec,hits):
    response=np.divide(response,Q,out=np.zeros_like(response),where=science)
    xx=(np.arange(Q.shape[1])+1-wcs['CRPIX1'])*wcs['CDELT1']*3600
    yy=(np.arange(Q.shape[0])+1-wcs['CRPIX2'])*wcs['CDELT2']*3600
    X,Y=np.meshgrid(xx,yy);mass=response.sum()
    if not mass>0:
        return dict(status='unavailable',cause='no supported probe footprint',
                    scope='optional MAP-only source diagnostic; base bundle and operator verification unaffected')
    cx=float((response*X).sum()/mass);cy=float((response*Y).sum()/mass)
    variance=float((response*((X-cx)**2+(Y-cy)**2)).sum()/mass/2)
    sigma=fwhm_arcsec/np.sqrt(8*np.log(2))
    area=abs(wcs['CDELT1']*wcs['CDELT2'])*3600**2
    return dict(status='measured',scope='MAP-only fixed realized operator; synthetic source inserted AFTER PTC; no RTC/CAL/PTC attenuation or measured-PSF claim',
      input='unit-peak circular Gaussian centered on declared tangent origin, truncated at 8 sigma',
      input_fwhm_arcsec=fwhm_arcsec,input_peak_mJy_per_nominal_beam=1,
      output_peak_mJy_per_nominal_beam=float(response.max()),integrated_ratio_to_template=float(mass*area/(2*np.pi*sigma*sigma)),
      output_centroid_tangent_arcsec=[cx,cy],output_second_moment_equivalent_fwhm_arcsec=float(np.sqrt(8*np.log(2)*variance)),
      constant_unit_response_max_error=float(np.max(np.abs(hits[science]/Q[science]-1))),
      full_procedure_response='unavailable: complete upstream source-response carrier not supplied')

def inspect(root, verify_operator=True):
    root=Path(root).resolve();b=read_yaml(root/'bundle.yaml')
    require(b['schema']=='citlali-forward-map-bundle-v1' and b['outcome']=='success','not a complete MAP bundle')
    require(b['role']=='SCI-MAP:base_observation_map@1' and b['base_parent']==b['product_identity'],'wrong role/base parent')
    require(b['reached_stages']==['requested','effective','observation_resolved','applied','realized'],'incomplete lifecycle')
    for name,sha in b['files'].items():require(digest(root/name)==sha,f'component digest mismatch: {name}')
    require(b['product_identity']=='SCI-MAP-base:'+b['plan_sha256']+':'+b['files']['map.fits'],'MAP product identity differs')
    p=read_yaml(root/'plan.yaml');require(digest(root/'plan.yaml')==b['plan_sha256'],'foreign plan')
    rows=np.fromfile(root/'output-rows.u64',dtype='<u8')
    with fits.open(root/'map.fits',checksum=True,memmap=False) as h:
        signal=np.asarray(h[0].data,dtype=float);N=np.asarray(h['NUMERATOR'].data,dtype=float)
        Q=np.asarray(h['NORMALIZATION'].data,dtype=float);hits=np.asarray(h['HITS'].data,dtype=np.int64)
        norm=np.asarray(h['NORM_SUPPORT'].data,dtype=bool);science=np.asarray(h['SCI_SUPPORT'].data,dtype=bool)
        header=h[0].header.copy()
        require(all(x.verify_checksum()==1 and x.verify_datasum()==1 for x in h),'FITS checksum failure')
    require(signal.shape==(p['WCS']['rows'],p['WCS']['columns']),'shape differs from plan')
    require(np.array_equal(rows,np.flatnonzero(science)),'scientific row domain mismatch')
    require(np.isfinite(N).all() and np.isfinite(Q).all() and np.all(Q>=0),'invalid accumulators')
    require(np.isfinite(signal[science]).all() and np.isnan(signal[~science]).all(),'unsupported pixel relabeled as sky')
    require(np.array_equal(Q,hits.astype(float)),'uniform normalization differs from contributing hits')
    positive=np.sort(Q[Q>0]);count=len(positive);k=((3*count)//4+count)//2
    reference=float(positive[k]) if count else 0
    require(reference==b['reference_normalization'],'reference order statistic mismatch')
    cut=b['coverage_cut'];require(np.array_equal(science,(Q>0)&(Q>=reference*cut)),'science support mismatch')
    require(np.array_equal(norm,(Q>0)&(Q>=reference*cut/10)),'normalization support mismatch')
    require(np.allclose(signal[science],N[science]/Q[science],rtol=0,atol=0),'signal is not the declared N/Q')
    for key in ('RADESYS','CTYPE1','CTYPE2','CUNIT1','CUNIT2'):require(header[key]==p['WCS'][key],f'WCS mismatch {key}')
    require(header['RADESYS']=='FK5' and header['EQUINOX']==2000 and header['CDELT1']<0<header['CDELT2'],'frame/orientation mismatch')
    # Preregistered comparison population: every pixel center and every vertex
    # along all four outer boundaries, in FITS 1-based continuous coordinates.
    yy,xx=np.indices(signal.shape,dtype=float);points=np.column_stack([xx.ravel()+1,yy.ravel()+1])
    ny,nx=signal.shape;vx=np.arange(nx+1)+.5;vy=np.arange(ny+1)+.5
    boundary=np.concatenate([np.column_stack([vx,np.full_like(vx,.5)]),np.column_stack([vx,np.full_like(vx,ny+.5)]),
                             np.column_stack([np.full_like(vy,.5),vy]),np.column_stack([np.full_like(vy,nx+.5),vy])])
    points=np.unique(np.concatenate([points,boundary]),axis=0)
    side=WCS(naxis=2);side.wcs.ctype=['RA---TAN','DEC--TAN'];side.wcs.cunit=['deg','deg'];side.wcs.radesys='FK5';side.wcs.equinox=2000
    side.wcs.crpix=[p['WCS']['CRPIX1'],p['WCS']['CRPIX2']];side.wcs.crval=[p['WCS']['CRVAL1'],p['WCS']['CRVAL2']]
    side.wcs.cdelt=[p['WCS']['CDELT1'],p['WCS']['CDELT2']]
    a=np.deg2rad(side.all_pix2world(points,1));c=np.deg2rad(WCS(header).all_pix2world(points,1))
    require(np.isfinite(a).all() and np.isfinite(c).all(),'invalid point in WCS comparison population')
    # Spherical haversine metric, stable for small representation differences.
    h=np.sin((a[:,1]-c[:,1])/2)**2+np.cos(a[:,1])*np.cos(c[:,1])*np.sin((a[:,0]-c[:,0])/2)**2
    max_arcsec=float(np.max(2*np.arcsin(np.sqrt(np.clip(h,0,1))))*180/np.pi*3600)
    require(max_arcsec<=.1,'FITS representation exceeds 0.1 arcsec finite-population bound')
    response=replay_handoffs(root,p,N,hits,verify_operator,b['nominal_beam_fwhm_arcsec'] if verify_operator else None)
    probe_report=None if response is None else summarize_map_probe(response,Q,science,p['WCS'],b['nominal_beam_fwhm_arcsec'],hits)
    return {'schema':'citlali-forward-map-inspection-v1','bundle':str(root),'product_identity':b['product_identity'],
            'array':b['array'],'shape_yx':list(signal.shape),'science_pixels':int(science.sum()),
            'contributing_occurrences':int(hits.sum()),'positive_and_negative_science_pixels':[int(np.sum(signal[science]>0)),int(np.sum(signal[science]<0))],
            'signal_min_max_mJy_per_nominal_beam':[float(signal[science].min()),float(signal[science].max())],
            'wcs_comparison_population':len(points),'wcs_max_arcsec':max_arcsec,'sparse_operator_verified':verify_operator,'exact_PTC_numerator_replay':verify_operator,'measured_MAP_local_source_probe':probe_report,
            'scope':'bundle/reload/operator/numerical and WCS representation checks; not full upstream source-response or production qualification'}

def plot(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with fits.open(Path(root)/'map.fits',memmap=False) as h:
        sky=h[0].data;Q=h['NORMALIZATION'].data;mask=h['SCI_SUPPORT'].data.astype(bool)
        W=WCS(h[0].header)
    fig=plt.figure(figsize=(12,5),layout='constrained')
    ax=fig.add_subplot(121,projection=W);finite=sky[np.isfinite(sky)];limit=np.percentile(np.abs(finite),99.5)
    im=ax.imshow(sky,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit);ax.set_title('Signed PTC → MAP signal')
    ax.set_xlabel('FK5 J2000 RA');ax.set_ylabel('FK5 J2000 Dec');fig.colorbar(im,ax=ax,label='mJy / nominal beam')
    bx=fig.add_subplot(122,projection=W);im=bx.imshow(np.where(Q>0,Q,np.nan),origin='lower',cmap='viridis')
    bx.contour(mask.astype(float),levels=[.5],colors='white',linewidths=.6);bx.set_title('Normalization and science-support boundary')
    bx.set_xlabel('FK5 J2000 RA');bx.set_ylabel('FK5 J2000 Dec');fig.colorbar(im,ax=bx,label='sum of uniform coefficients (not precision)')
    fig.savefig(output,dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--report',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();r=inspect(a.bundle)
    with a.report.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    if a.plot:plot(a.bundle,a.plot)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
