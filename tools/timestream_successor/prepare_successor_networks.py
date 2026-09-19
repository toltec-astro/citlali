#!/usr/bin/env python3
"""Prepare the bounded 152390 observation request from its exact existing inputs.

This is path/identity inventory only. The ordinary C++ route verifies the APT,
uses the existing KIDs producer and realizes native timing/support itself.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import netCDF4
from ptc_spod import digest, require

SUPPLIED = [0,1,2,3,4,5,7,8,9,11,12]


def binding(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=digest(p))


def write(p,data):
    with Path(p).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def csv_rows(p,delimiter=','):
    with p.open() as f:return list(csv.DictReader((line for line in f if not line.startswith('#')),delimiter=delimiter,skipinitialspace=True))

def finite(s):
    return bool(s) and math.isfinite(float(s))


def prepare(reference,output,repository):
    base=json.loads(reference.read_text());output.mkdir(parents=True,exist_ok=False)
    require(base['network']==12 and base['observation']==152390,'requires the preserved network12 reference')
    manifest=Path(base['manifest']['path']);require(digest(manifest)==base['manifest']['sha256'],'manifest changed')
    roles={r['role']:manifest.parent/r['relative_path'] for r in csv_rows(manifest)}
    # Use the manifested child, never a baseline/seed selected by filename glob.
    apt=csv_rows(roles['apt']);sources=csv_rows(roles['sources'])
    timing={r['network']:r for r in base['decision_apply']['timing_inputs']}
    require(sorted(timing)==SUPPLIED,'existing processing generation has a different network inventory')
    for key in ('telescope','effective_config','ast_acceptance'):
        require(digest(base[key]['path'])==base[key]['sha256'],key+' changed')
    inventory=[];requests=[]
    for nw in range(13):
        if nw not in timing:
            inventory.append(dict(network=nw,raw_available=False,reason='not-supplied',requested=False))
            continue
        raw=Path(timing[nw]['path']);require(digest(raw)==timing[nw]['sha256'],'raw digest mismatch')
        tune=Path(base['tune']['path']).with_name(Path(base['tune']['path']).name.replace('toltec12_',f'toltec{nw}_',1))
        tune_ref=binding(tune);tune_rows=csv_rows(tune,' ')
        source=next(r for r in sources if r['role']=='raw' and int(r['nw'])==nw)
        kmp=next(r for r in sources if r['role']=='kmp' and int(r['nw'])==nw)
        require(source['content_sha256']=='sha256:'+timing[nw]['sha256'] and int(source['byte_count'])==raw.stat().st_size,'raw/APT source identity')
        require(kmp['content_sha256']=='sha256:'+tune_ref['sha256'] and int(kmp['byte_count'])==tune.stat().st_size,'Tune/APT source identity')
        with netCDF4.Dataset(raw) as nc:
            def scalar(k):return float(nc.variables['Header.Toltec.'+k][...])
            require([int(scalar(k)) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex')]==[152390,0,2,nw],'raw header scope')
            rows,channels=nc.variables['Data.Toltec.Is'].shape
            require(channels==int(source['channel_count']),'raw/APT channel count')
            rate=scalar('SampleFreq');fpga=scalar('FpgaFreq');accum=scalar('AccumLen')
        members=sorted((r for r in apt if int(r['nw'])==nw),key=lambda r:int(r['kids_tone']))
        require([int(r['kids_tone']) for r in members]==list(range(channels)),'APT complete detector relation')
        arrays={int(r['array']) for r in members};require(len(arrays)==1,'network spans different arrays')
        array=arrays.pop();name=('a1100','a1400','a2000')[array]
        cfg=copy.deepcopy(base);cfg.update(network=nw,native_source='exact-raw-kids-v1',raw={k:timing[nw][k] for k in ('path','sha256')},tune=tune_ref)
        cfg.pop('audit_receipt',None)
        cfg['reassessment_selection']=dict(base['reassessment_selection'],
            authority='owner-2026-09-19-full-available-network-directive',
            purpose='complete supplied detector population; preserve individual unavailable states and existing treatment',
            positive_rationale='bounded RTC/CAL/ALS10 coverage and scheduling comparison; no new treatment or qualification')
        existing={d['channel']:d for d in base['detectors']}
        cfg['detectors']=[dict(channel=d,filter=existing.get(d,{}).get('filter','lowpass') if nw==12 else 'lowpass') for d in range(channels)]
        if nw!=12:
            cfg.pop('finite_notch',None);cfg.pop('finite_notch_identity',None)
        cfg['filter_plan']=binding(repository/'validation/rtc_development_lowpass_2026-09-19'/f'{name}.json')
        path=output/f'network{nw}-input.json';write(path,cfg)
        requests.append(dict(network=nw,input=binding(path)))
        inventory.append(dict(network=nw,array=name,raw_available=True,requested=True,requested_detectors=channels,native_rows=rows,
            nominal_sampling_hz=rate,integration_seconds=accum/fpga,output_sampling_hz=rate/(2 if array==2 else 1),
            raw=cfg['raw'],tune=cfg['tune'],APT_manifest=base['manifest'],filter_plan=cfg['filter_plan'],
            detectors=[dict(channel=int(r['kids_tone']),selected_APT_uid=int(r['uid']),array=int(r['array']),flxscale=r['flxscale'],
                producer_tune_valid=float(tune_rows[int(r['kids_tone'])]['flag'])==0,
                APT_position_available=finite(r['x_t']) and finite(r['y_t']),
                APT_factor_available=finite(r['flxscale']) and float(r['flxscale'])!=0) for r in members],
            timing_and_physical_runs='verified and published by native C++ ingress; no common-cadence substitution',
            connected_execution='prepared; not yet executed',stage_availability='pending execution',diagnostic_support='pending separate assessment'))
    write(output/'inventory.json',dict(schema='citlali-152390-network-inventory-v1',reference=binding(reference),networks=inventory))
    full=dict(schema='citlali-observation-networks-v1',observation=152390,subobservation=0,scan=2,networks=requests)
    write(output/'observation.json',full)
    reference_cfg=json.loads((output/'network12-input.json').read_text())
    reference_cfg['reassessment_selection']=base['reassessment_selection']
    reference_cfg['detectors']=[{k:v for k,v in d.items() if k!='samples'} for d in base['detectors']]
    path=output/'network12-reference-input.json';write(path,reference_cfg)
    write(output/'observation-reference.json',dict(full,networks=[dict(network=12,input=binding(path))]))
    for nw in (0,7,12):write(output/f'observation-network{nw}.json',dict(full,networks=[e for e in requests if e['network']==nw]))
    return inventory


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();rows=prepare(a.reference,a.output.resolve(),a.repository.resolve())
    print(json.dumps([{k:v for k,v in r.items() if k in ('network','array','raw_available','requested_detectors','output_sampling_hz')} for r in rows],indent=2))
