"""Post-freeze artifact verification and display; never feeds a prediction."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f13-scan-agreement-feasibility-r0.1')
REPO=Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
sys.path.insert(0,str(ROOT/'setup_r0.3'))
import scan_agreement as sa

freeze=json.loads((ROOT/'PREDICTION_FREEZE_R0.1.json').read_text())
for r in freeze['files']:assert sa.file_record(r['path'])==r
summary=json.loads((ROOT/'PAIRED_REPORT_R0.1.json').read_text())
assert all(x['opened_utc']>freeze['frozen_utc'] for x in summary['evaluations'])
audits=[]; csvrows=[]
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for k in range(6):
    out=ROOT/f'boundary_{k}'
    prediction=json.loads((out/'PREDICTION_R0.1.json').read_text())
    access=json.loads((out/'ACCESS_LOG_R0.1.json').read_text())
    allowed={x['path'] for x in json.loads((ROOT/'setup_r0.3'/f'boundary_{k}_input.json').read_text())['files']}
    assert {x['path'] for x in access}==allowed
    audit=dict(boundary=k,current_boundary_access_only=True,term_presence_equals_nonzero_coefficient=True,keys=[])
    with netCDF4.Dataset(out/'COMPONENT_MAPS_R0.1.nc') as nc:
        # All admitted recorded kernels are nonzero. Confirm the corresponding
        # actual pixel contribution property in every retained scan and cleaned
        # reference scan, rather than assuming a stencil hit contributes.
        for a in range(3):
            for s in range(12):
                group=nc.groups[f'array_{a}'].groups[f'scan_{s}']
                for g in [group,*group.groups.values()]:
                    assert np.array_equal(np.asarray(g['terms'][:])>0,np.asarray(g['abs_C'][:])>0)
        for i,key in enumerate(prediction['keys']):
            group=nc.groups[f'key_{i}']
            # Return serialized FITS planes to predictor arithmetic order.
            def internal(g,name):return np.asarray(g[name][:])[:,::-1]
            domain=internal(group,'domain').astype(bool);footprint=internal(group,'footprint').astype(bool)
            assert domain.sum()==key['domain_pixels'] and footprint.sum()==key['footprint_pixels']
            def model(g):return {n:internal(g,n) for n in ('signal','signal_error','support','conditioned','scan_count')}
            refs=[model(group.groups[f'reference_{g}']) for g in range(2)]
            baseline=model(group.groups['probe_0'])
            intervals=[sa.interval(baseline,ref,domain) for ref in refs]
            # These already-defined baseline errors remain meaningful when a
            # retention probe is unavailable; no replacement/shrunken domain.
            ka=dict(identity=key['key'],baseline_intervals=intervals,probes=[])
            for j,p in enumerate(key['probes'],start=1):
                m=model(group.groups[f'probe_{j}']);valid=m['support'].astype(bool)&m['conditioned'].astype(bool)
                assert np.count_nonzero(domain&~valid)==p['lost_domain_pixels']
                entry=dict(coefficient=p['coefficient'],status=p['status'])
                if p['references']:
                    for g in range(2):
                        assert sa.interval(m,refs[g],domain)==p['references'][g]['probe']
                        assert intervals[g]==p['references'][g]['baseline']
                    entry['probe_intervals']=[x['probe'] for x in p['references']]
                else:
                    entry['probe_intervals']=None
                    entry['unavailable_reason']='No error or benefit decision on an incomplete fixed domain.'
                ka['probes'].append(entry)
                csvrows.append(dict(boundary=k,array=key['key']['array'],uid=key['key']['uid'],scan=key['key']['scan'],coefficient=p['coefficient'],footprint_pixels=int(footprint.sum()),domain_pixels=int(domain.sum()),domain_loss_pixels=p['lost_domain_pixels'],full_support_lost=p['support_lost'],full_support_gained=p['support_gained'],status=p['status'],baseline_R0=intervals[0]['error'],baseline_R1=intervals[1]['error'],probe_R0=p['references'][0]['probe']['error'] if p['references'] else '',probe_R1=p['references'][1]['probe']['error'] if p['references'] else ''))
            audit['keys'].append(ka)
            row=k-1
            F=footprint[:,::-1];V=domain[:,::-1]
            for col in range(3):
                display=np.zeros(F.shape);display[F]=1;display[V]=2
                lost=np.zeros(F.shape,bool)
                if col:
                    g=group.groups[f'probe_{col}']
                    valid=np.asarray(g['support'][:]).astype(bool)&np.asarray(g['conditioned'][:]).astype(bool)
                    lost=V&~valid;display[lost]=3
                ax=axes[row,col]
                ax.imshow(display,origin='lower',interpolation='nearest',cmap=ListedColormap(['white','#d5d5d5','#3978a8','#c13735']),vmin=0,vmax=3,extent=(178.5,-178.5,-177.5,177.5),aspect='equal')
                if lost.any():
                    rr,cc=np.where(lost);ax.scatter(178-cc,rr-177,s=18,facecolors='none',edgecolors='#c13735',linewidths=.7)
                title=f'Fixed domain: {int(V.sum()):,} / {int(F.sum()):,}' if col==0 else f'f = {col*.5:g}: {int(lost.sum())} required pixels lost'
                ax.set_title(title,fontsize=11);ax.set_xlabel('AZ offset (arcsec)')
                if col==0:ax.set_ylabel(f'Boundary {k}, scan {key["key"]["scan"]}\nEL offset (arcsec)')
                ax.tick_params(labelsize=8)
    audits.append(audit)
fig.suptitle('EL-F13 fixed comparison support\nBlue: required domain · Gray: footprint outside domain · Red: unavailable required pixels',fontsize=14)
fig.savefig(ROOT/'SUPPORT_DOMAINS_R0.1.png',dpi=170)
plt.close(fig)
sa.write_json(ROOT/'POST_FREEZE_ARTIFACT_AUDIT_R0.1.json',dict(prediction_freeze_unchanged=True,all_outcome_access_after_freeze=True,boundaries=audits,scope='Verification and baseline-error completion from frozen component maps only; no changed prediction, domain, threshold or outcome input.'))
with (ROOT/'PROBE_SUMMARY_R0.1.csv').open('x') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
bindings=json.loads((REPO/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F13_BOUND_INPUTS_R0.1.json').read_text())
checked=[]
for record in [r for b in bindings['decision_boundaries'] for r in b['files']]+bindings['archives']:
    actual=sa.file_record(record['path']);assert actual==record;checked.append(actual)
sa.write_json(ROOT/'INPUT_PRESERVATION_R0.1.json',dict(status='all 36 input files and both archives unchanged',files=checked))
print(json.dumps(dict(audits=len(audits),probe_rows=len(csvrows),inputs_preserved=len(checked)),indent=2))
