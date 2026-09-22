"""Immutable successor MAP/PTC input boundary and empirical companion serialization."""
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
from inspect_forward_map import inspect,read_yaml,digest,require
from inspect_point_source import measure,gaussian
from point_empirical import coordinates,source_patch,fixed_amplitude,aperture_measurements


def load_parent(root,measurement):
    root=Path(root).resolve();verification=inspect(root)
    bundle=read_yaml(root/'bundle.yaml');plan=read_yaml(root/'plan.yaml')
    require(measurement['MAP_product_identity']==bundle['product_identity'] and
            measurement['map_sha256']==bundle['files']['map.fits'],'measurement has foreign map parent')
    with fits.open(root/'map.fits',checksum=True,memmap=False) as f:
        data={h.name:np.array(h.data) for h in f};header=f[0].header.copy()
    require(bundle['observation']=='obs:152389_0_2' and bundle['array']==measurement['array'],'POINT observation/array mismatch')
    require(measurement['method']=='unweighted elliptical Gaussian plus constant; 40 arcsec radius around AST center','unsupported saved measurement method')
    beam=bundle['nominal_beam_fwhm_arcsec']
    require(beam==measurement['nominal_calibration_beam_fwhm_arcsec'],'nominal beam mismatch')
    desc,p,x,y=measure(data['PRIMARY'],data['SCI_SUPPORT'].astype(bool),header,beam)
    for key in ['fitted_amplitude_above_constant_mJy_per_nominal_beam','fitted_constant_mJy_per_nominal_beam',
                'centroid_tangent_east_north_arcsec','measured_fwhm_major_minor_arcsec']:
        require(np.allclose(desc[key],measurement[key],rtol=1e-10,atol=1e-10),f'saved descriptive method differs: {key}')
    return data,header,bundle,plan,desc,p,verification


def threshold(q,cut):
    positive=np.sort(q[q>0]);support=np.zeros(q.shape,bool)
    if not len(positive):return support,0.
    k=((3*len(positive))//4+len(positive))//2;reference=float(positive[k])
    return (q>0)&(q>=reference*cut),reference


def split_maps(root,plan,full,header,beam,parameters):
    """Partition exact already-admitted operator columns by scan parity.

    load_parent must validate every closed handoff before this read-only replay.
    No random sign, new coefficients, new eligibility or PTC learning occurs.
    """
    root=Path(root);shape=full['PRIMARY'].shape
    numerators=np.zeros((2,full['PRIMARY'].size));counts=np.zeros(numerators.shape,dtype=np.int64)
    operator=np.memmap(root/'operator-pixel.i32',dtype='<i4',mode='r');members=[]
    for q in plan['ordered_column_domain']:
        pub=read_yaml(root/q['publication_file']);segment=pub['source_segment']
        stem=next(n[:-len('-cleaned.f64')] for n in segment['files'] if n.endswith('-cleaned.f64'))
        source=Path(pub['source_PTC_root']);dimensions=(segment['scheduled_times'],segment['output_detectors'])
        causes=np.fromfile(source/(stem+'-causes.u8'),dtype='u1').reshape(dimensions)
        values=np.fromfile(source/(stem+'-cleaned.f64'),dtype='<f8').reshape(dimensions)[causes==0]
        offset=q['first_input_row'];pixels=np.asarray(operator[offset:offset+q['input_rows']]);admit=pixels>=0
        scan=int(segment['scan']);side=scan%2
        np.add.at(numerators[side],pixels[admit],values[admit]);np.add.at(counts[side],pixels[admit],1)
        members.append(dict(network=segment['network'],scan=scan,split='B' if side else 'A',contributions=int(admit.sum()),publication=pub['identity']))
    numerators=numerators.reshape((2,*shape));counts=counts.reshape((2,*shape))
    require(np.array_equal(counts.sum(axis=0),full['HITS']),'split counts do not recover parent')
    # Separate sums change floating addition order; use a declared conservative
    # accumulated roundoff bound, not equality of reassociated floating sums.
    error=float(np.max(np.abs(numerators.sum(axis=0)-full['NUMERATOR'])))
    scale=float(np.max(np.abs(full['NUMERATOR'])))
    require(error<=1e-10*max(1.,scale),'split numerator exceeds roundoff bound')
    signals=[];supports=[];descriptions=[];references=[]
    cut=read_yaml(root/'bundle.yaml')['coverage_cut']
    for i in range(2):
        valid,ref=threshold(counts[i],cut);references.append(ref);supports.append(valid)
        s=np.full(shape,np.nan);np.divide(numerators[i],counts[i],out=s,where=valid);signals.append(s)
        try:
            desc,_,_,_=measure(s,valid,header,beam);descriptions.append(dict(status='success',**desc))
        except (ValueError,RuntimeError) as e:descriptions.append(dict(status='unavailable',cause=str(e)))
    common=supports[0]&supports[1]&full['SCI_SUPPORT'].astype(bool)
    difference=np.where(common,signals[0]-signals[1],np.nan)
    balance=np.full(shape,np.nan);np.divide(counts[0],counts.sum(axis=0),out=balance,where=counts.sum(axis=0)>0)
    d,v,_,h,half=source_patch(difference,common,full['HITS'],header)
    x,y=coordinates(d.shape,h);p=np.array(parameters);p[0]=1;p[6]=0;t=gaussian(p,x,y)
    major=max(parameters[3:5])*np.sqrt(8*np.log(2));fixedroi=np.hypot(x-parameters[1],y-parameters[2])<=3*major
    residual=fixed_amplitude(d,v,t,fixedroi)
    for i in range(2):
        a,valid,_,_,_=source_patch(signals[i],supports[i],counts[i],header)
        descriptions[i]['fixed_template']=fixed_amplitude(a,valid,t,fixedroi)
        descriptions[i]['apertures']=aperture_measurements(a,valid,x,y,parameters[1:3],major,beam,abs(h['CDELT1']*3600)**2)
    summary=dict(split_definition={'A':'processing scans 0,2,4','B':'processing scans 1,3,5'},members=members,
        source_measurements=descriptions,difference_definition='A minus B; no equal-depth or independent-noise rescaling',
        source_shaped_difference=residual,
        source_shaped_difference_unavailable_cause='incomplete common support in fixed-template footprint' if residual is None else None,
        coverage_reference=references,
        common_pixels=int(common.sum()),coverage_A_fraction_quantiles=np.quantile(balance[common],[0,.1,.5,.9,1]).tolist() if common.any() else None,
        maximum_numerator_roundoff_error=error,counts_exact=True,
        caveat='one conditional split of existing cleaned products; shared calibration/processing and unequal coverage remain; not a full uncertainty ensemble')
    return dict(signals=signals,counts=counts,numerators=numerators,supports=supports,common=common,
                difference=difference,balance=balance,report=summary)


def write_fits(path,header,planes):
    hdus=[]
    for i,(name,values,unit) in enumerate(planes):
        h=header.copy()
        for key in ['CHECKSUM','DATASUM']:h.remove(key,ignore_missing=True)
        h['BUNIT']=unit;h['EXTNAME']=name;h['COMPAN']='POINT-EMPIRICAL'
        hdu=(fits.PrimaryHDU if i==0 else fits.ImageHDU)(values,header=h)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path,checksum=True,overwrite=False)


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
