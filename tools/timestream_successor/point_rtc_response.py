"""Saved pre-RTC finite differences and own-REF ordinary-MAP ideal controls."""
import argparse,json,re,time
from pathlib import Path
import numpy as np
from point_rtc_diagnostics import read,yread,arr,load_network
from point_fruit_diagnostics import load,patch_map
from point_empirical import source_patch,coordinates
from point_reference import context
from point_guarded_feedback import infer_base,ideal_comparison
from point_response_cd import direct_raster,response_metrics,vector_measurement,MEASUREMENT_NAMES
from point_additive_diagnostics import save_json
from inspect_forward_map import require

BRANCHES=('C-core-plus','C-core-minus','C-shoulder')
def map_path(out,k):return out/'M0/products/a2000/map.fits' if k==0 else out/f'run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/total.fits'

def map_values(ref,k,values,valid):
    product=ref/'M0/products/a2000' if k==0 else ref/f'run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products'
    if k==0:
        groups=[]
        for col in yread(product/'plan.yaml')['ordered_column_domain']:
            nw,sid=map(int,re.search(r'network-(\d+)-segment-(\d+)',col['publication_file']).groups());p=ref/f'upstream/products/network{nw}/donor-continuity/ptc';s=next(x for x in yread(p/'receipt.yaml')['segments'] if x['scan']==sid)
            groups.append((s,p/f'segment-{sid}-causes.u8',col['first_input_row']))
    else:groups=[(r['source_segment'],product/f"network-{r['source_segment']['network']}-segment-{r['source_segment']['scan']}/causes.u8",r['MAP_first_input_row']) for r in yread(product/'receipt.yaml')['segments']]
    pixels=arr(product/'operator-pixel.i32','<i4');v=np.full(len(pixels),np.nan);okay=np.zeros(len(pixels),bool)
    for s,causes,first in groups:
        nw=s['network'];nt=s['scheduled_times'];no=s['output_detectors'];p=ref/f'upstream/products/network{nw}/donor-continuity/ptc';rows=arr(p/f"segment-{s['scan']}-slots.i64",'<i8')*2;ids=np.asarray(s['output_detector_indices']);keep=arr(causes,'u1',(nt,no))==0
        require(int(keep.sum())==s['retained'],'MAP source support changed')
        x=values[nw][rows[:,None],ids][keep];good=valid[nw][rows[:,None],ids][keep]
        v[first:first+len(x)]=x;okay[first:first+len(x)]=good
    im,h=load(map_path(ref,k));shape=im['PRIMARY'].shape;include=pixels>=0
    count=np.bincount(pixels[include],minlength=np.prod(shape)).reshape(shape);require(np.array_equal(count,im['HITS']),'ideal ordinary MAP domain differs')
    include&=okay&np.isfinite(v);numerator=np.bincount(pixels[include],weights=v[include],minlength=np.prod(shape)).reshape(shape);actualcount=np.bincount(pixels[include],minlength=np.prod(shape)).reshape(shape)
    result=np.divide(numerator,count,out=np.zeros_like(numerator),where=count>0);support=im['SCI_SUPPORT'].astype(bool)&(actualcount==count)
    return result,support,dict(reference_contributions=int(count.sum()),available_contributions=int(actualcount.sum()),lost_contributions=int((count-actualcount).sum()),lost_science_pixels=int((im['SCI_SUPPORT'].astype(bool)&~support).sum()))

def response(root,family,dest):
    start=time.monotonic();dest.mkdir(parents=True,exist_ok=True);ref=root/f'{family}-REF-rep1';ctx=context(ref/'freeze/a2000');rs=read(ref/'run/BASE-POINT/REF/a2000/SUMMARY.json');networks={nw:load_network(root,family,nw) for nw in (11,12)};records=[];maps={};native=[]
    sigmas={x['name']:x['conditional_sigma'] for x in rs['endpoint_uncertainty']};require(all(x is not None and np.isfinite(x) for x in sigmas.values()),'nonfinite REF endpoint')
    for branch in BRANCHES:
        out=root/f'{family}-{branch}';bs=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json');known={};delta={};valid={};idealvalid={}
        for nw,a in networks.items():
            p=out/f'upstream/products/network{nw}';probe=p/'source-probe';pr=yread(probe/'receipt.yaml');increment=arr(probe/'increment-x.f64','<f8',a['shape']);known[nw]=increment*a['mult'];idealvalid[nw]=a['valid']&a['mapping'];delta[nw]=np.zeros(a['shape']);valid[nw]=np.zeros(a['shape'],bool);statechanges=0;centerchanges=0
            require(pr['original_x_sha256']==a['probe_receipt']['original_x_sha256'] and pr['original_r_sha256']==a['probe_receipt']['original_r_sha256'] and pr['r_unchanged'] and pr['input_validity_unchanged'],'raw parent changed')
            for j,c in enumerate(a['channels']):
                b=arr(p/f'donor-continuity/{c}-filtered.f64','<f8',(a['shape'][0],2));state=arr(p/f'donor-continuity/{c}-state.u8','u1');center=arr(p/f'donor-continuity/{c}-map-center.u8','u1');statechanges+=int((state!=a['state'][:,j]).sum());centerchanges+=int((center!=a['center'][:,j]).sum());delta[nw][:,j]=(b[:,0]-a['filtered'][:,j])*a['mult'][:,j];valid[nw][:,j]=((state&1)!=0)&((a['state'][:,j]&1)!=0)&a['mapping'][:,j]
            native.append(dict(branch=branch,network=nw,state_changed_cells=statechanges,map_center_changed_cells=centerchanges,calibration_only_max_abs_error=pr['calibration_only_max_abs_error_mJy_beam'],changed_cells=pr['changed_cells'],mapping_unavailable_cells=pr['mapping_unavailable_cells'],input_r_unchanged=True,input_validity_unchanged=True,own_RTC_learning=True))
        for k in (0,5):
            rim,rh=load(map_path(ref,k));bim,bh=load(map_path(out,k));knownmap,iv,ic=map_values(ref,k,known,idealvalid);rtcdelta,rv,rc=map_values(ref,k,delta,valid)
            raster,sh=load(out/f'freeze/shapes/{branch}.fits');direct=direct_raster(raster['INJECTION'],sh,knownmap.shape,rh)
            require(np.allclose(knownmap[iv],direct[iv],atol=1e-10,rtol=1e-12),'native sky/control projection mismatch')
            source,sv,norm,ph,_=source_patch(rim['PRIMARY'],rim['SCI_SUPPORT'].astype(bool),rim['NORMALIZATION'],rh);injected,bv,_,_,_=source_patch(bim['PRIMARY'],bim['SCI_SUPPORT'].astype(bool),bim['NORMALIZATION'],bh);ideal,ivs,_,_,_=source_patch(knownmap,iv,rim['NORMALIZATION'],rh);rtc,rvs,_,_,_=source_patch(rtcdelta,rv,rim['NORMALIZATION'],rh)
            idealmeasurement,_=infer_base(ctx,source+ideal,sv&ivs)
            actualmeasurement=bs['histories'][k]['measurement'];refmeasurement=rs['histories'][k]['measurement']
            comparison=ideal_comparison(ctx,refmeasurement,actualmeasurement,idealmeasurement)
            x,y=coordinates(source.shape,ph);disk=np.hypot(x,y)<=40
            rec=dict(family=family,branch=branch,cycle=k,ideal_scope='same exact retained physical raster projected at native occurrences, own unperturbed REF operator/coefficients/support; calibration-only roundtrip and direct raster agree',ideal_accounting=ic,RTC_delta_accounting=rc,measurement=comparison,spatial=response_metrics(injected-source,ideal,sv&bv&ivs&disk),RTC_spatial=response_metrics(rtc,ideal,sv&rvs&ivs&disk),endpoint_conditional_sigmas=sigmas if k==5 else {},M0_guard_sigma_A=refmeasurement['guard'].get('sigma_A'),support={name:dict(equal=bool(np.array_equal(rim[name],bim[name])),changed=int(np.count_nonzero(rim[name]!=bim[name]))) for name in ('HITS','NORMALIZATION','SCI_SUPPORT')})
            if k==0:
                rm,_=infer_base(ctx,source+rtc,sv&rvs);rec['conditional_RTC_only_measurement']=ideal_comparison(ctx,refmeasurement,rm,idealmeasurement);rec['conditional_RTC_only_scope']='RTC injected-minus-REF projected through fixed REF M0 MAP, added to REF M0 for common measurement; diagnostic only, not an independently reduced RTC sky map'
            records.append(rec)
            for name,value in [('REF',source),('injected',injected),('ideal',ideal),('RTC_delta',rtc),('residual',injected-source-ideal),('support',sv&bv&ivs)]:maps[f'{branch}_T{k}_{name}']=value
    np.savez_compressed(dest/'response-maps.npz',**maps);save_json(dest/'source-responses.json',records);save_json(dest/'native-source-accounting.json',native)
    save_json(dest/'response-extraction.json',dict(seconds=time.monotonic()-start,science_runs=0,endpoint_ensembles=0,scope='saved arrays/maps; existing ideal-addition guard only; conditional RTC replay explicitly labeled'))
    print(json.dumps([dict(branch=r['branch'],cycle=r['cycle'],spatial=r['spatial'],RTC_spatial=r['RTC_spatial'],support=r['support']) for r in records],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',default='A');p.add_argument('--output',type=Path,required=True);a=p.parse_args();response(a.root,a.family,a.output)
