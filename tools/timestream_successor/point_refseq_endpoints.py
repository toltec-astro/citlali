"""M0/T10-only native mean-aware history accounting, independent of closure."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import yaml
from run_point_refseq import *


def endpoints(out,binary):
    start=time.perf_counter();revision=source();version=subprocess.check_output([str(binary),'--version'],text=True).strip();check_native_version(version,revision)
    fr=json.loads((out/'FREEZE.json').read_text());w=json.loads(verified(fr['prospective']).read_text());auth=auth_work(w)
    oldwork=auth.json(FROZEN/'freeze/prospective.json');owner=auth.json(CD/'freeze/prospective.json');ctx,_=contexts(auth,owner)
    new={b:json.loads((out/'run'/b/'SUMMARY.json').read_text()) for b in BRANCHES};require(all(r['complete'] for r in new.values()),'both trajectories required')
    dest=out/'endpoints';dest.mkdir();results=[]
    for k in (0,10):
        d=dest/f'{k:02d}';d.mkdir();pair=next(p for p in oldwork['pairs'] if p['array']=='a2000' and p['cycle']==k)
        stages=[new[b]['stages'][k] for b in BRANCHES];products=[Path(p['receipt']['path']).parent for p in stages]
        maps,header=load(products[0]/'total.fits');hm,_=load(products[1]/'total.fits');oldmaps,_=auth.fits(FROZEN/f'pairs/a2000/{k:02d}/components.fits')
        domain=oldmaps['COMMON'].astype(bool);known=oldmaps['KNOWN'];shape=known.shape;normalization=maps['NORMALIZATION'];support=maps['SCI_SUPPORT'].astype(bool)
        def segment(s):
            name=s['name'];q=auth.yaml(s['request']['path']);q['reverse']=False
            inputs=q['inputs'];inputs['oldvr']=inputs['vr'];inputs['oldvh']=inputs['vh']
            for side,p in zip(('r','h'),products):
                for prefix,file in [('v','operand.f64'),('m','mean.f64'),('clean','cleaned.f64'),('cause','causes.u8'),('b','basis.f64')]:inputs[prefix+side]=bound(p/name/file)
            req=d/(name+'.yaml');req.write_text(yaml.safe_dump(q,sort_keys=False));output=d/name;t=time.perf_counter()
            cmd=[str(binary),str(req),str(output)];p=subprocess.run(cmd,text=True,capture_output=True)
            save_json(d/(name+'-invocation.json'),dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_seconds=time.perf_counter()-t))
            require(p.returncode==0,'endpoint native Apply failed '+name+': '+p.stderr)
            receipt=read_yaml(output/'receipt.yaml');a=np.fromfile(output/'numerators.f64',dtype='<f8').reshape(len(receipt['components']),*shape)
            return receipt,{n:v for n,v in zip(receipt['components'],a)}
        with ThreadPoolExecutor(max_workers=3) as pool:parts=list(pool.map(segment,pair['segments']))
        sums={n:sum(p[1][n] for p in parts) for n in parts[0][1]}
        components={n:np.divide(v,normalization,out=np.full(shape,np.nan),where=support) for n,v in sums.items()}
        delta=hm['PRIMARY']-maps['PRIMARY'];e=delta-oldmaps['DELTA'];history=components['HISTORY'];fc=components['FC'];dc=oldmaps['DC'];avoided=-oldmaps['CQ']
        checks=dict(direct_same=close(components['DC'][domain],dc[domain],'original DC identity'),
            seq_identity=close(delta[domain],(dc+fc)[domain],'new response centered identity'),
            history_identity=close(e[domain],(avoided+history)[domain],'avoided/history identity'),
            history_partition=close(history[domain],sum(components['HISTORY_H_'+n]-components['HISTORY_R_'+n] for n in ('RETAIN','ADD','REMOVE'))[domain],'paired history partition'))
        if k==0:
            checks['zero_fc']=close(fc[domain],np.zeros(domain.sum()),'M0 FC',0,0);checks['zero_history']=close(history[domain],np.zeros(domain.sum()),'M0 history',0,0)
        selected=dict(KNOWN=known,DELTA=delta,ADAPTIVE=oldmaps['DELTA'],E=e,DC=dc,FC=fc,AVOIDED=avoided,HISTORY=history,
            CLOSURE=e-avoided-history,SEQ_CLOSURE=delta-dc-fc,
            **{n:v for n,v in components.items() if n.startswith('HISTORY_')})
        measurements={n:map_measures(v,known,domain,ctx,header) for n,v in selected.items()}
        write_fits(d/'accounting.fits',header,[(n,v,'mJy/nominal beam') for n,v in selected.items()]+[('COMMON',domain.astype('u1'),'1')],companion='POINT-REFSEQ-HISTORY')
        record=dict(stage=k,checks=checks,measurements=measurements,native=[p[0] for p in parts],applied_candidate='none' if k==0 else 'm9')
        save_json(d/'result.json',record);results.append(record)
    save_json(dest/'SUMMARY.json',dict(source_revision=revision,binary=bound(binary),version=version,wall_seconds=time.perf_counter()-start,results=results,original_verification_seconds=auth.verify_all()))
    manifest(dest)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args();endpoints(a.output,a.binary)
