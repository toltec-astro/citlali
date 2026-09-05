#!/usr/bin/env python3
"""Owner-approved EL-F13 offline predictor; no Citlali or outcome-reader imports.

Maps are [channel,row,col]: N,C,Q,absN,absC,absQ,term_count,unique_UIDs,sum_f.
Input terms are recorded binary64 values. No celestial truth is supplied here.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import sys
import time
from datetime import datetime, timezone

import numpy as np

U = 2.0**-53
CHANNELS = ('N', 'C', 'Q', 'abs_N', 'abs_C', 'abs_Q', 'terms', 'unique_UIDs', 'sum_f')
DISPOSITIONS = {'eligible','not_new','already_assigned','independent_exclusion','entry_hard_exclusion','horizon'}


def check(value, message):
    if not value:
        raise ValueError(message)


def file_record(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return dict(path=str(p.resolve()), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def decode_census(text, boundary):
    """Discard all old selection/score fields before constructing predictor keys."""
    lines = text.strip().splitlines()
    check(lines[0]=='SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001', 'state schema')
    check(lines[1]==f'H {boundary}', 'state boundary/arm')
    i = 2
    assignments = int(lines[i]); i += 1 + assignments
    count = int(lines[i]); i += 1
    census = []
    for line in lines[i:i+count]:
        fields = shlex.split(line)
        check(len(fields)==14, 'census schema')
        obs, array, uid, scan, disposition = fields[:5]
        check(obs=='123424' and disposition in DISPOSITIONS, 'census identity/disposition')
        array, uid, scan = int(array), int(uid), int(scan)
        check(0<=array<3 and uid>=0 and 0<=scan<12, 'census key')
        census.append(dict(observation=obs,array=array,uid=uid,scan=scan,disposition=disposition))
    check(len(census)==count, 'census truncation')
    identities = [(x['array'],x['uid'],x['scan']) for x in census]
    check(len(set(identities))==len(identities), 'duplicate census')
    check(sum(x['disposition']=='eligible' for x in census)<=16,'eligible cap')
    return census


class Stream:
    """A byte-only C++ accumulator: it cannot open any data file."""
    def __init__(self, library, shape, kernels, keys):
        self.shape = tuple(shape)
        self.lib = ct.CDLL(str(library))
        L = self.lib
        L.sa_error.restype = ct.c_char_p
        L.sa_create.argtypes = [ct.c_int,ct.c_int]; L.sa_create.restype = ct.c_void_p
        L.sa_destroy.argtypes = [ct.c_void_p]
        L.sa_kernel.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_double)]
        L.sa_key.argtypes = [ct.c_void_p,ct.c_int64,ct.c_int,ct.c_int]
        L.sa_feed.argtypes = [ct.c_void_p,ct.c_char_p,ct.c_size_t]
        L.sa_finish.argtypes = [ct.c_void_p,ct.c_uint64,ct.c_uint64]
        L.sa_map.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_int]
        L.sa_map.restype = ct.POINTER(ct.c_double)
        L.sa_scan_order.argtypes=[ct.c_void_p,ct.c_int]
        self.ptr = L.sa_create(*shape)
        check(bool(self.ptr), self.error())
        for a, bank in enumerate(kernels):
            for k, (coeff, square) in enumerate(bank):
                coeff=np.ascontiguousarray(coeff,dtype='float64'); square=np.ascontiguousarray(square,dtype='float64')
                self.call(L.sa_kernel,self.ptr,a,k,*coeff.shape,coeff.ctypes.data_as(ct.POINTER(ct.c_double)),square.ctypes.data_as(ct.POINTER(ct.c_double)))
        for key in keys:
            self.call(L.sa_key,self.ptr,key['uid'],key['array'],key['scan'])

    def error(self):
        return self.lib.sa_error().decode()

    def call(self, func, *args):
        rc=func(*args)
        check(rc==0,self.error())

    def feed(self, data):
        rc=self.lib.sa_feed(self.ptr,data,len(data))
        check(rc==0,self.error())

    def finish(self, records, occurrences):
        rc=self.lib.sa_finish(self.ptr,records,occurrences)
        check(rc==0,self.error())

    def plane(self, kind, index, scan=0):
        p=self.lib.sa_map(self.ptr,kind,index,scan)
        check(bool(p),self.error())
        return np.ctypeslib.as_array(p,shape=(9*math.prod(self.shape),)).reshape((9,*self.shape))

    def close(self):
        if self.ptr:
            self.lib.sa_destroy(self.ptr); self.ptr=None

    def scan_order(self):
        return [self.lib.sa_scan_order(self.ptr,i) for i in range(12)]


def gamma(j):
    j=np.asarray(j,dtype='float64')
    check(bool(np.all(j*U<1)), 'unbounded gamma')
    return j*U/(1-j*U)


def threshold(weight, cut):
    x=np.sort(weight[np.isfinite(weight)&(weight>0)])
    return float(x[(int(math.floor(.75*len(x)))+len(x))//2]*cut) if len(x) else 0.


def finalize(raw, scan_count):
    check(np.isfinite(raw).all(),'nonfinite accumulator')
    n,c,q=raw[:3]
    weight=np.zeros_like(n)
    ordinary=(np.abs(c)>1e-8)&(q>0)
    weight[ordinary]=c[ordinary]*c[ordinary]/np.maximum(q[ordinary],1e-30)
    check(np.isfinite(weight).all(),'coefficient overflow')
    provisional=(weight>0)&(weight>=threshold(weight,.01))
    signal=np.zeros_like(n); signal[provisional]=n[provisional]/c[provisional]
    weight[~provisional]=0
    support=(weight>0)&(weight>=threshold(weight,.1))
    # Inflate accumulated absolute sums by their summation/product bound before
    # rounding upward. 4n+4s+16 covers both levels and weighted product formation.
    g=gamma(4*raw[6]+4*scan_count+16)
    absolute=np.nextafter(raw[3:6]/(1-g),np.inf)
    bounds=np.nextafter(16*g*absolute,np.inf)
    cm=np.abs(c)-bounds[1]; qm=q-bounds[2]
    conditioned=(cm>1e-8)&(qm>0)
    error=np.zeros_like(n)
    m=np.zeros_like(n); m[conditioned]=n[conditioned]/c[conditioned]
    error[conditioned]=(bounds[0][conditioned]+np.abs(m[conditioned])*bounds[1][conditioned])/cm[conditioned]+16*U*np.abs(m[conditioned])
    error=np.nextafter(error,np.inf)
    check(np.isfinite(signal).all() and np.isfinite(error).all(),'finalization overflow')
    return dict(signal=signal,weight=weight,support=support,conditioned=conditioned,
                absolute_upper=absolute,bounds=bounds,signal_error=error,scan_count=scan_count)


def scaled_rms(values):
    values=np.asarray(values,dtype='float64')
    check(values.size>0 and np.isfinite(values).all(),'invalid RMS input')
    scale=float(np.max(np.abs(values)))
    return 0. if scale==0 else scale*math.sqrt(float(np.mean((values/scale)**2)))


def interval(model, ref, domain):
    m=model['signal'][domain]; r=ref['signal'][domain]
    e=scaled_rms(m-r)
    b1=scaled_rms(model['signal_error'][domain]+ref['signal_error'][domain])
    magnitude=scaled_rms(np.abs(m)+np.abs(r))
    b2=float(16*gamma(4*len(m)+16))*magnitude
    b=float(np.nextafter(b1+b2,np.inf))
    delta=m-r
    return dict(error=e,bound=b,map_bound_rms=b1,rms_magnitude=magnitude,
                rms_arithmetic_bound=b2,lower=max(0.,float(np.nextafter(e-b,-np.inf))),
                upper=float(np.nextafter(e+b,np.inf)),positive_peak=float(max(0.,delta.max())),
                negative_peak=float(min(0.,delta.min())),signed_sum=float(delta.sum()),
                positive_sum=float(delta[delta>0].sum()),negative_sum=float(delta[delta<0].sum()))


def assess(probes, references, footprint, memberships):
    """Pure approved scoring function. No paths, UID values or outcomes enter."""
    reasons=np.zeros(footprint.shape,dtype='uint16')
    for bit,ok in enumerate((probes[0]['support'],probes[0]['conditioned'],
            references[0]['support'],references[0]['conditioned'],references[0]['scan_count']>=2,
            references[1]['support'],references[1]['conditioned'],references[1]['scan_count']>=2)):
        reasons[footprint&~ok] |= 1<<bit
    domain=footprint&(reasons==0)
    nf,nv=int(footprint.sum()),int(domain.sum())
    status='available'
    if nf==0: status='no_contribution'
    elif any(len(x)<2 for x in memberships): status='unavailable_reference_scans'
    elif nv<256 or nv/nf<.90: status='unavailable_overlap'
    result=dict(status=status,footprint_pixels=nf,domain_pixels=nv,overlap_fraction=nv/nf if nf else None,
                uncovered_reasons={str(1<<i):int(np.count_nonzero(reasons&(1<<i))) for i in range(8)},probes=[])
    # Preserve maps/support accounting even when the scoring domain is unavailable.
    for j in (1,2):
        lost=probes[0]['support']&~probes[j]['support']; gained=~probes[0]['support']&probes[j]['support']
        good=probes[j]['support']&probes[j]['conditioned']
        p=dict(coefficient=j*.5,status=status,support_lost=int(lost.sum()),support_gained=int(gained.sum()),
               lost_domain_pixels=int(np.count_nonzero(domain&~good)),references=[])
        if status=='available':
            if p['lost_domain_pixels']: p['status']='unavailable_probe_support'
            else:
                for ref in references:
                    base=interval(probes[0],ref,domain); value=interval(probes[j],ref,domain)
                    disposition='no_gain'
                    if base['upper']<.1: disposition='insufficient_room'
                    elif value['upper']<=.9*base['lower'] and base['lower']-value['upper']>=.1: disposition='pass'
                    p['references'].append(dict(baseline=base,probe=value,status=disposition))
                p['status']='pass' if all(x['status']=='pass' for x in p['references']) else ('insufficient_room' if any(x['status']=='insufficient_room' for x in p['references']) else 'no_gain')
        result['probes'].append(p)
    return result,domain,reasons


def reference_scans(target_scan, order):
    return [[s for s in order if s!=target_scan and s%2==g] for g in (0,1)]


def combine(parts):
    check(bool(parts),'empty reference')
    result=np.zeros_like(parts[0]); counts=np.zeros_like(parts[0][0])
    for part in parts:
        result+=part
        counts+=(part[6]>0)
    # unique counts in combined maps are not a sum across scans. Per-scan counts
    # remain authoritative and full-observation unique counts are reconstructed
    # separately; mark the combined channel unavailable to prevent misuse.
    result[7]=0
    return result,counts


class BoundaryAccess:
    def __init__(self, records):
        self.allowed={str(Path(x['path']).resolve()):x for x in records}
        check(len(self.allowed)==6,'boundary must have exactly six inputs')
        self.log=[]

    def open_path(self, path, purpose):
        path=str(Path(path).resolve())
        check(path in self.allowed,'forbidden boundary data dependency')
        self.log.append(dict(path=path,purpose=purpose,utc=datetime.now(timezone.utc).isoformat()))
        return path

    def verify(self):
        for path,expected in self.allowed.items():
            self.open_path(path,'verify bound compressed/file bytes')
            check(file_record(path)==expected,'bound input identity changed: '+path)


def fits_orientation(value):
    """Exact native fits_io.h representation conversion, an involution.

    Ledger planes are in internal Eigen row/column orientation. Citlali's
    FITS writer reverses columns. There is no interpolation or fitted alignment.
    """
    value=np.asarray(value).squeeze()
    check(value.ndim==2,'expected a two-dimensional map plane')
    return value[:,::-1]


def save_map(group, name, value):
    value=np.asarray(value)
    if value.ndim==3:
        for i,x in enumerate(value): save_map(group,f'{name}_{i}',x)
        return
    dtype='f8' if value.dtype.kind=='f' else 'i4'
    group.createVariable(name,dtype,('row','col'),zlib=True,complevel=1)[:]=fits_orientation(value)


def save_final(group, raw, final):
    for name,x in zip(CHANNELS,raw): save_map(group,name,x)
    for name,x in final.items(): save_map(group,name,x)


def accounting(stream, ledgers, fits_paths, access):
    from astropy.io import fits
    reports=[]
    for a,ledger in enumerate(ledgers):
        full=stream.plane(2,a)
        for field,index in [('N',0),('C',1),('Q',2),('absolute_N_terms',3),('absolute_C_terms',4),('occurrence_pixel_count',6),('unique_detector_count',7)]:
            check(np.array_equal(full[index],ledger[field]),f'full reconstruction mismatch array {a} {field}')
        final=finalize(full,np.sum([stream.plane(0,a,s)[6]>0 for s in range(12)],axis=0))
        check(np.array_equal(final['signal'],ledger['signal']),'signal re-finalization mismatch')
        check(np.array_equal(final['weight'],ledger['formal_weight']),'formal coefficient re-finalization mismatch')
        support=(ledger['weight']>0)&(ledger['weight']>=threshold(ledger['weight'],.1))
        check(np.array_equal(support,ledger['science_support'].astype(bool)),'historical science support mismatch')
        with fits.open(access.open_path(fits_paths[a],'verify historical signal/formal weight/grid'),memmap=False) as hdul:
            for plane,field in [('signal_I','signal'),('weight_formal_I','formal_weight'),('weight_I','weight')]:
                check(np.array_equal(fits_orientation(hdul[plane].data),ledger[field]),'FITS/ledger plane mismatch '+plane)
            h=hdul['signal_I'].header
            wcs={key:h[key] for key in ('CTYPE1','CTYPE2','CUNIT1','CUNIT2','CDELT1','CDELT2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','BUNIT')}
            check(h['CTYPE1']=='AZOFFSET' and h['CTYPE2']=='ELOFFSET' and h['BUNIT']=='mJy/beam','FITS frame/unit')
            check(tuple(h[x] for x in ('CUNIT1','CUNIT2','CDELT1','CDELT2','CRPIX1','CRPIX2','CRVAL1','CRVAL2'))==('arcsec','arcsec',-1.,1.,179.,178.,0.,0.),'registered WCS')
        reports.append(dict(array=a,exact_fields=['N','C','Q','absolute_N_terms','absolute_C_terms','occurrence_pixel_count','unique_detector_count','signal','formal_weight','FITS signal/formal/weight','science_support'],wcs=wcs,science_pixels=int(support.sum())))
    return reports


def identity_check(full,target):
    # Inherited EL-F12 fixed-sample signed deletion identity, safety 64.
    n,c,q=full[:3];tn,tc,tq=target[:3]
    ok=(np.abs(c)>1e-8)&(np.abs(tc)>1e-8)&(np.abs(c-tc)>1e-8)&(q-tq>0)
    all_signal=n[ok]/c[ok]; t=tn[ok]/tc[ok]; deleted=(n[ok]-tn[ok])/(c[ok]-tc[ok])
    d=deleted-all_signal; predicted=(tc[ok]/c[ok])*(deleted-t)
    bound=64*U*np.maximum.reduce([np.ones_like(d),np.abs(d),np.abs(predicted),np.abs(all_signal),np.abs(t),np.abs(deleted)])
    residual=np.abs(d-predicted)
    check(np.all(residual<=bound),'inherited identity bound failure')
    return dict(pixels=int(ok.sum()),maximum_residual=float(residual.max()) if len(residual) else None,
                maximum_bound_fraction=float(np.max(residual/bound)) if len(residual) else None)


def run_boundary(config_path, library, output):
    import netCDF4
    start=time.monotonic(); output=Path(output); output.mkdir()
    cfg=json.loads(Path(config_path).read_text()); k=cfg['boundary']
    access=BoundaryAccess(cfg['files']); access.verify()
    paths={Path(x['path']).name:x['path'] for x in cfg['files']}
    spool=next(p for p in paths.values() if p.endswith('.gz'))
    checkpoint=paths['citlali_restart_checkpoint.nc']
    ledger_path=next(p for p in paths.values() if 'mapdiag_fruit_response' in p)
    with netCDF4.Dataset(access.open_path(checkpoint,'decode census; discard old selection and response fields')) as nc:
        state=str(np.asarray(nc['fruit_response_state'][:]).reshape(-1)[0])
    census=decode_census(state,k); keys=[x for x in census if x['disposition']=='eligible']
    with netCDF4.Dataset(access.open_path(ledger_path,'read only schema, state, array totals and kernel banks')) as nc:
        check(nc.getncattr('state')==state,'checkpoint/ledger state mismatch')
        for attr in ('schema','observation','iteration','spool_record_bytes','spool_records','science_occurrences','spool_byte_order','spool_int64_fields','spool_binary64_fields'):
            check(nc.getncattr(attr)==cfg['ledger_metadata'][attr], 'ledger schema '+attr)
        check(nc.getncattr('coverage_cut')==.1 and nc.getncattr('signal_unit')=='mJy/beam','finalizer metadata')
        ledgers=[]; kernels=[]
        for a in range(3):
            group=nc.groups[f'array_{a}']
            ledgers.append({name:np.asarray(v[:]) for name,v in group.variables.items()})
            bank=[]
            for name in sorted(group.groups,key=lambda x:int(x.split('_')[1])):
                bank.append(tuple(np.asarray(group.groups[name][field][:]) for field in ('coefficient','coefficient_square')))
            kernels.append(bank)
    shape=ledgers[0]['N'].shape
    check(all(x['N'].shape==shape for x in ledgers),'array grid mismatch')
    stream=Stream(library,shape,kernels,[])
    try:
        def read_spool(purpose):
            digest=hashlib.sha256(); size=0
            with gzip.open(access.open_path(spool,purpose),'rb') as f:
                for data in iter(lambda:f.read(104*8192),b''):
                    digest.update(data);size+=len(data);stream.feed(data)
            check(size==cfg['original_spool']['size_bytes'] and digest.hexdigest()==cfg['original_spool']['sha256'],'decompressed spool identity')
            stream.finish(cfg['ledger_metadata']['spool_records'],cfg['ledger_metadata']['science_occurrences'])
            return digest.hexdigest(),size
        digest,size=read_spool('first stream: original-order all-scan construction only; no decompressed output')
        fits_paths=[next(p for p in paths.values() if f'_{a}_' in p and p.endswith('.fits')) for a in ('a1100','a1400','a2000')]
        reconstructed=accounting(stream,ledgers,fits_paths,access)
        # No new reference partition is formed until all three arrays pass.
        if keys:
            stream.close()
            stream=Stream(library,shape,kernels,keys)
            digest,size=read_spool('second stream after exact reconstruction gates: direct probe and UID-excluded scan accumulation')
            for a in range(3):
                for j,name in enumerate(('N','C','Q')):
                    check(np.array_equal(stream.plane(2,a)[j],ledgers[a][name]),'second stream exact closure')
        order=stream.scan_order()
        results=[]
        with netCDF4.Dataset(output/'COMPONENT_MAPS_R0.1.nc','w') as nc:
            nc.createDimension('row',shape[0]); nc.createDimension('col',shape[1])
            nc.setncatts(dict(schema='SCI-FRUIT-EL-F13-SCAN-AGREEMENT-R0.1',boundary=k,
                             signal_unit='mJy/beam',frame='AZOFFSET/ELOFFSET',diagnostic_only=1,
                             storage_orientation='companion FITS; columns reversed from internal occurrence-ledger orientation',
                             wcs_json=json.dumps([x['wcs'] for x in reconstructed]),
                             reason_bits='1 baseline support;2 baseline conditioning;4 R0 support;8 R0 conditioning;16 R0 scan count;32 R1 support;64 R1 conditioning;128 R1 scan count'))
            for a in range(3):
                ag=nc.createGroup(f'array_{a}')
                fg=ag.createGroup('full_observation')
                full=stream.plane(2,a)
                full_final=finalize(full,np.sum([stream.plane(0,a,s)[6]>0 for s in order],axis=0))
                save_final(fg,full,full_final)
                save_map(fg,'historical_science_support',ledgers[a]['science_support'])
                for s in range(12):
                    sg=ag.createGroup(f'scan_{s}')
                    for name,x in zip(CHANNELS,stream.plane(0,a,s)):save_map(sg,name,x)
                    if any(key['array']==a for key in keys):
                        cg=sg.createGroup('eligible_UIDs_removed')
                        for name,x in zip(CHANNELS,stream.plane(1,a,s)):save_map(cg,name,x)
            for index,key in enumerate(keys):
                a,s=key['array'],key['scan']; target=stream.plane(4,index)
                identity=identity_check(stream.plane(2,a),target)
                members=reference_scans(s,order)
                contributing=[[scan for scan in group if np.any(stream.plane(1,a,scan)[6]>0)] for group in members]
                refs_raw=[]; refs=[]
                for group in members:
                    raw,counts=combine([stream.plane(1,a,scan) for scan in group])
                    refs_raw.append(raw);refs.append(finalize(raw,counts))
                probe_raw=[stream.plane(3,index,j) for j in range(3)]
                probes=[finalize(raw,(raw[6]>0).astype(int)) for raw in probe_raw]
                footprint=(target[4]>0)&ledgers[a]['science_support'].astype(bool)
                result,domain,reasons=assess(probes,refs,footprint,contributing)
                result.update(key=key,reference_scans=members,contributing_reference_scans=contributing,excluded_UIDs=sorted({x['uid'] for x in keys if x['array']==a}),identity_check=identity)
                results.append(result)
                kg=nc.createGroup(f'key_{index}');kg.setncattr('identity',json.dumps(key))
                save_map(kg,'footprint',footprint);save_map(kg,'domain',domain);save_map(kg,'uncovered_reasons',reasons)
                for name,x in zip(CHANNELS,target):save_map(kg,'target_'+name,x)
                for g in range(2):save_final(kg.createGroup(f'reference_{g}'),refs_raw[g],refs[g])
                for j in range(3):
                    pg=kg.createGroup(f'probe_{j}');save_final(pg,probe_raw[j],probes[j])
                    for g in range(2):save_map(pg,f'difference_R{g}',probes[j]['signal']-refs[g]['signal'])
                    save_map(pg,'support_lost_vs_deletion',probes[0]['support']&~probes[j]['support'])
                    save_map(pg,'support_gained_vs_deletion',~probes[0]['support']&probes[j]['support'])
        result=dict(boundary=k,construction='valid',census=census,keys=results,accounting=reconstructed,
                    scan_order=order,spool_sha256=digest,spool_bytes=size,stream_passes=2 if keys else 1,
                    records=cfg['ledger_metadata']['spool_records'],occurrences=cfg['ledger_metadata']['science_occurrences'],
                    inherited_identity_safety=64,finalization_safety=16,
                    combined_reference_unique_count='unavailable; per-scan and full-observation unique counts retained',
                    elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output/'PREDICTION_R0.1.json',result)
        print(json.dumps({key:result[key] for key in ('boundary','construction','keys','elapsed_seconds','peak_rss_bytes')}),flush=True)
    finally:
        stream.close()
        write_json(output/'ACCESS_LOG_R0.1.json',access.log)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    # macOS rejects some advisory RLIMIT_RSS settings despite reporting an
    # unlimited hard limit. The registered supervisor enforces the same 4 GiB
    # RSS ceiling by sampling the worker and verifies ru_maxrss on completion.
    try:run_boundary(args.input,args.library,args.output)
    except Exception as e:
        path=Path(args.output);path.mkdir(exist_ok=True)
        write_json(path/'INVALID_R0.1.json',dict(status='invalid',error=str(e),type=type(e).__name__))
        raise


if __name__=='__main__':main()
