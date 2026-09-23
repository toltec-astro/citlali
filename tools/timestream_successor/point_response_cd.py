"""Fixed POINT C+D interventions and finite paired-response diagnostics.

All brightness is mJy/nominal beam; tangent offsets are east/north arcseconds.
This module never learns PTC or changes a science MAP. Signed differences are
linear diagnostics; only full science maps and ideal map-addition controls are
passed to the accepted positive-source seven-parameter fitter.
"""
from pathlib import Path
import numpy as np
from point_fruit_trajectory import load, candidate_guard, selection_difference
from point_fruit_diagnostics import patch_map
from point_admission import admit, fit_diagnostic, projection
from point_empirical import coordinates, aperture_measurements
from point_empirical_io import write_fits
from inspect_point_source import gaussian
from inspect_forward_map import require, read_yaml, digest

BRANCHES = ('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-low',
            'D-seed-high','D-seed-shoulder','D-threshold-5')
RATIO_FLOOR = 1e-8


def shapes(parameters, major0, header, valid):
    x,y=coordinates(valid.shape,header)
    require(header['CDELT1']<0 and header['CDELT2']>0,'east/north grid handedness')
    r=np.hypot(x,y);w=np.where(r<=30,1.,np.where(r<36,(1+np.cos(np.pi*(r-30)/6))/2,0.))
    require(np.all(valid[r<36]),'prescribed injection overlaps unavailable support')
    p=np.array(parameters,dtype=float);p[6]=0
    g=gaussian(p,x,y)*.1
    p[0]*=.05;p[1]+=.5*major0;p[3:5]*=2
    h=gaussian(p,x,y)
    results={};info={}
    for name,raw in [('C-core-plus',g),('C-core-minus',-g),('C-shoulder',h)]:
        v=raw*w
        results[name]=np.where(valid,v,np.nan)
        info[name]=dict(analytic_peak=float(parameters[0]*(.05 if name=='C-shoulder' else -.1 if name=='C-core-minus' else .1)),
            discrete_signed_peak=float(v.flat[np.argmax(abs(v))]),area_sum_arcsec2=float(v.sum()*4),
            taper_loss_area_sum_arcsec2=float((raw-v).sum()*4),
            untapered_raster_area_sum_arcsec2=float(raw.sum()*4),
            parameters=p.tolist() if name=='C-shoulder' else [parameters[0]*(-.1 if name=='C-core-minus' else .1),*parameters[1:6],0.],
            positive_first_axis='tangent east; decreasing FITS column, CDELT1<0',
            radius_origin='AST analysis center',taper_radii_arcsec=[30,36],renormalized=False)
    return results,info


def infer(fixed,signal,support,threshold=3.):
    fit=fit_diagnostic(signal,support,fixed.header,fixed.beam)
    if fit['status']!='success' or fit.get('constrained',False):return fit,None
    return fit,admit(signal,support,fixed.scale,fixed.noise_valid,fixed.header,
                     np.asarray(fit['parameters']),threshold,1.)


def save_candidate(path,header,candidate):
    write_fits(path,header,[('CANDIDATE',candidate['candidate'],'mJy/nominal beam'),
        ('ADMITTED',candidate['selected'].astype('u1'),'1'),('VALID',candidate['valid'].astype('u1'),'1'),
        ('RATIO',candidate['z'],'1')],companion='POINT-CD-UNAPPLIED')


def seed_candidate(original,branch,shoulder):
    c={k:np.array(v,copy=True) for k,v in original.items()}
    v=c['VALID'].astype(bool)
    if branch in ('D-seed-low','D-seed-high'):
        c['CANDIDATE'][v]*=.8 if branch=='D-seed-low' else 1.2
    elif branch=='D-seed-shoulder':
        require(np.all(np.isfinite(shoulder[v])),'unavailable shoulder seed')
        c['CANDIDATE'][v]+=shoulder[v]
        c['ADMITTED']=(v & (c['CANDIDATE']!=0)).astype('u1')
    else:raise ValueError('not a seed intervention')
    return c


def processing_guard(products,receipt):
    """Actual downstream changes are retained. Only new failures stop a branch."""
    data,h=load(products/'total.fits');v=data['SCI_SUPPORT'].astype(bool)
    require(np.isfinite(data['PRIMARY'][v]).all(),'nonfinite supported total')
    r,_=load(products/'residual.fits');a,_=load(products/'restored.fits')
    for part in (r,a):
        for n in ('SCI_SUPPORT','NORM_SUPPORT','NORMALIZATION','HITS'):
            require(np.array_equal(data[n],part[n]),'remove/rejoin support mismatch')
    err=float(np.max(abs((data['PRIMARY']-r['PRIMARY']-a['PRIMARY'])[v])))
    require(err<1e-8,'remove/rejoin accounting failed')
    transitions=[]
    for s in receipt['segments']:
        old=s['source_segment'];inherited=old['network']==3 and old['scan']==5 and not old['converged']
        require(s['rank']==10 and s['input_eligibility_changed']==0,'rank/upstream validity changed')
        require(s['projection_support_counts'][0]==0,'unsupported feedback projection')
        require(s.get('injection_projection_support_counts',[0])[0]==0,'unsupported injection projection')
        require(s['converged'] or inherited,'new failed processing segment')
        require(s['application_failed_times']<=old.get('application_failed_times',0),'new failed application time')
        if not s['converged'] or s['output_cause_changed'] or s['converged']!=old['converged']:
            transitions.append(dict(network=old['network'],scan=old['scan'],converged=s['converged'],
                inherited=inherited,stopping_reason=s['stopping_reason'],output_cause_changed=s['output_cause_changed']))
    return dict(accounting_max_abs=err,processing_states=transitions,uniform_gamma=1,
        actual_support=True,unchanged_parent_eligibility=True)


def direct_raster(raster,header,shape,out_header):
    """Independent containing-pixel P on arbitrary output pixel centers."""
    x,y=coordinates(shape,out_header)
    col=np.floor(x/(header['CDELT1']*3600)+header['CRPIX1']-.5).astype(int)
    row=np.floor(y/(header['CDELT2']*3600)+header['CRPIX2']-.5).astype(int)
    inside=(col>=0)&(col<raster.shape[1])&(row>=0)&(row<raster.shape[0])
    out=np.zeros(shape)
    out[inside]=raster[row[inside],col[inside]]
    # Outside the computational disk is no injection, with observed data retained.
    disk=np.hypot(x,y)<=40
    out[~disk]=0
    require(np.isfinite(out).all(),'unavailable injection reference')
    return out


def injection_reference(ref_products, injection_products, raster, header):
    """M_REF(P I), using REF retained-entry order, support, gamma=1 and norm.

    Production P operands are independently checked against grid-aligned direct
    raster evaluation at every REF-admitted output pixel. No PTC or centering.
    """
    ref,h=load(ref_products/'total.fits');rr=read_yaml(ref_products/'receipt.yaml')
    ir=read_yaml(injection_products/'receipt.yaml')
    require(len(rr['segments'])==len(ir['segments']),'projection segment mismatch')
    op=np.fromfile(ref_products/'operator-pixel.i32',dtype='<i4')
    numerator=np.zeros(ref['PRIMARY'].size);counts=np.zeros(numerator.size,np.int64)
    direct=direct_raster(raster,header,ref['PRIMARY'].shape,h).ravel()
    maxerr=0.;all_eligible=0;non_map_nonzero=0
    for rs,ins in zip(rr['segments'],ir['segments']):
        s=rs['source_segment'];require(s==ins['source_segment'],'foreign projection parent')
        stem=next(n[:-len('-input.f64')] for n in s['files'] if n.endswith('-input.f64'))
        name=f"network-{s['network']}-{stem}"
        pi_path=injection_products/name/'injection.f64'
        require(digest(pi_path)==ins['files']['injection.f64'],'changed native injection')
        pi=np.fromfile(pi_path,dtype='<f8').reshape(s['scheduled_times'],s['detectors'])
        eligible=np.fromfile(Path(rs['source_root'])/(stem+'-eligible.u8'),dtype='u1').reshape(pi.shape).astype(bool)
        all_eligible+=int(eligible.sum())
        require(np.isfinite(pi).all() and np.all(pi[~eligible]==0),'invalid-entry injection')
        full=np.zeros((s['scheduled_times'],s['output_detectors']))
        full[:,s['fit_columns_in_output']]=pi
        causes=np.fromfile(ref_products/name/'causes.u8',dtype='u1').reshape(full.shape)
        if not rs['retained']:
            non_map_nonzero+=int(np.count_nonzero(full));continue
        use=causes==0;values=full[use]
        require(values.size==rs['retained'],'REF publication membership mismatch')
        start=rs['MAP_first_input_row'];pixels=op[start:start+len(values)];good=pixels>=0
        non_map_nonzero+=int(np.count_nonzero(full[~use])+np.count_nonzero(values[~good]))
        if good.any():maxerr=max(maxerr,float(abs(values[good]-direct[pixels[good]]).max()))
        np.add.at(numerator,pixels[good],values[good]);np.add.at(counts,pixels[good],1)
    require(maxerr<1e-10,'native P differs from independent containing-pixel control')
    require(np.array_equal(counts.reshape(ref['HITS'].shape),ref['HITS']),'REF projection counts mismatch')
    require(np.array_equal(ref['HITS'],ref['NORMALIZATION']),'REF coefficients are not uniform gamma1')
    support=ref['SCI_SUPPORT'].astype(bool);out=np.full(ref['PRIMARY'].shape,np.nan)
    np.divide(numerator.reshape(out.shape),ref['NORMALIZATION'],out=out,where=support)
    error=float(np.max(abs(out[support]-direct.reshape(out.shape)[support])))
    require(error<1e-8,'normalized injection map mismatch')
    return out,support,h,dict(native_vs_independent_P_max_abs=maxerr,normalized_map_max_abs=error,
        eligible_parent_entries=all_eligible,non_MAP_nonzero_injected_entries=non_map_nonzero,
        REF_counts_normalization_exact=True,definition='fixed REF support/operator gamma1 normalization; no PTC/centering')


def ellipse(p):
    c,s=np.cos(p[5]),np.sin(p[5]);rot=np.array([[c,-s],[s,c]])
    return (rot@np.diag(np.square(p[3:5]))@rot.T).tolist()


def vector_measurement(m):
    if m['fit']['status']!='success':return None
    p=m['fit']['parameters'];cov=np.array(ellipse(p))
    return np.array([p[0],p[1],p[2],*sorted(np.array(p[3:5])*np.sqrt(8*np.log(2)),reverse=True),
        cov[0,0],cov[0,1],cov[1,1],p[6],m['frozen_template_projection'],
        *[q['value_mJy_per_beam_arcsec2'] for q in m['apertures']]],dtype=float)

MEASUREMENT_NAMES=['peak','east','north','FWHM_major','FWHM_minor','ellipse_xx','ellipse_xy','ellipse_yy',
                   'background','fixed_template','aperture_1','aperture_1p5','aperture_2']


def response_metrics(delta,known,common):
    use=common & np.isfinite(delta) & np.isfinite(known)
    d,i=delta[use],known[use];norm=float(np.sum(i*i))
    require(norm>0,'no common supported injection norm')
    return dict(pixels=int(use.sum()),g=float(np.sum(i*d)/norm),eta=float(np.linalg.norm(d-i)/np.sqrt(norm)),
        difference_rms=float(np.sqrt(np.mean(d*d))),residual_rms=float(np.sqrt(np.mean((d-i)**2))),
        common_domain='unweighted declared common supported pixels; zeros of known shape retained')


def measurement_response(fixed,reference,branch,support,known):
    base=fixed.measure(reference,support);actual=fixed.measure(branch,support)
    ideal=fixed.measure(reference+known,support)
    b,a,i=map(vector_measurement,(base,actual,ideal))
    result=dict(reference=base,actual=actual,ideal_map_addition=ideal,comparisons=None,
                ratio_floor_native_units=RATIO_FLOOR)
    if any(v is None for v in (b,a,i)):return result
    rows=[]
    for name,da,di in zip(MEASUREMENT_NAMES,a-b,i-b):
        rows.append(dict(name=name,actual_increment=da,ideal_increment=di,absolute_discrepancy=da-di,
            ratio=float(da/di) if np.isfinite(da+di) and abs(di)>RATIO_FLOOR else None))
    result['comparisons']=rows
    raw=[]
    for factor in (1.,1.5,2.):
        roi=fixed.radius<=factor*fixed.major
        if not np.all(support[roi]):raw.append(dict(factor=factor,available=False));continue
        raw.append(dict(factor=factor,available=True,actual_raw_increment=float(np.sum((branch-reference)[roi])*4),
            ideal_raw_increment=float(np.sum(known[roi])*4),
            actual_background_increment=float((a[8]-b[8])*roi.sum()*4),
            ideal_background_increment=float((i[8]-b[8])*roi.sum()*4)))
    result['raw_aperture_and_background_increments']=raw
    return result


def shape_projection(signal, support, shape):
    """Unweighted common finite shape-domain projection, retaining shape zeros."""
    valid=support & np.isfinite(signal) & np.isfinite(shape)
    norm=float(np.sum(shape[valid]**2))
    return dict(value=float(np.sum(signal[valid]*shape[valid])/norm) if norm>0 else None,
        pixels=int(valid.sum()),domain='common supported finite operand pixels; deliberate zero pixels retained')
