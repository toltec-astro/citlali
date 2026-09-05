"""Present all frozen EL-F12 measurements without changing their gates."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
results={a:json.loads((ROOT/'analysis'/f'{a}_RESULT.json').read_text()) for a in ('Half','Hold')}
resources=json.loads((out/'RESOURCE_SUMMARY_R0.1.json').read_text())
times={(r['arm'],r['injection']):r['native_iteration_seconds'] for r in resources}
metric_rows=[];check_rows=[];convergence=[];dispositions=[];quality=[]

def flatten(value,prefix=''):
 for key,item in value.items():
  name=f'{prefix}.{key}' if prefix else key
  if isinstance(item,dict):yield from flatten(item,name)
  elif isinstance(item,(int,float,str,bool)) or item is None:yield name,item

for arm,result in results.items():
 missing=[{'iteration':r['iteration'],'array':r['array'],'failure':r['failure']} for r in result['rows'] if r.get('status')=='unavailable']
 checks=[r.get('protections',{}) for r in result['rows']]
 restarts=[]
 for injection in ('uninjected','injected'):
  path=ROOT/arm/f'restart-{injection}'/'EXECUTION_RECEIPT.json'
  restarts.append(json.loads(path.read_text())['status'] if path.exists() else 'not_run')
 if missing:status='unavailable'
 elif result['selected_opportunities']==0:status='no_selected_opportunity'
 elif not result['promising_before_required_restarts']:status='not_promising'
 elif restarts==['pass','pass']:status='eligible_for_replication_only'
 else:status='required_restart_not_passed'
 dispositions.append(dict(arm=arm,status=status,all_protections_pass=result['all_protections_pass'],
   prioritized_leakage_pass=result['prioritized_leakage_pass'],selected_opportunities=result['selected_opportunities'],
   failed_protection_checks=sum(not x for row in checks for x in row.values()),
   total_protection_checks=sum(len(row) for row in checks),unavailable_metrics=missing,restarts=restarts,
   independent_pointing_replication='not authorized or performed',policy='not established'))
 for row in result['rows']:
  ident=dict(arm=arm,iteration=row['iteration'],array=row['array'])
  if 'H' not in row:continue
  for metric,h in flatten(row['H']):
   c=dict(flatten(row['candidate']))[metric]
   metric_rows.append(dict(**ident,metric=metric,H=h,candidate=c))
  for check,passed in row['protections'].items():check_rows.append(dict(**ident,check=check,passed=passed))
  for name,value in flatten(row['support']):metric_rows.append(dict(**ident,metric='support.'+name,H=None,candidate=value))
  for name,value in flatten(row['convergence']):metric_rows.append(dict(**ident,metric='convergence.'+name,H=None,candidate=value))
 for array in ('a1100','a1400','a2000'):
  rows=[r for r in result['rows'] if r['array']==array and 'H' in r]
  if len(rows)!=6:continue
  for branch in ('H','candidate'):
   for metric in ('central_recovery','whole_kernel_recovery'):
    values=np.array([r[branch][metric] for r in rows]);delta=np.diff(values);signs=np.sign(delta[delta!=0])
    convergence.append(dict(arm=arm,branch=branch,array=array,metric=metric,
       iteration_1_to_6_signed_change=float(values[-1]-values[0]),
       consecutive_nonzero_change_sign_reversals=int((signs[1:]!=signs[:-1]).sum()),
       iteration_5_to_6_signed_change=float(delta[-1]),values=values.tolist()))
 # A descriptive trace against the already declared terminal source targets.
 # This adds no pass condition and is never used to stop or select an arm.
 for k in range(1,7):
  rows=[r for r in result['rows'] if r['iteration']==k and 'candidate' in r]
  source_targets=len(rows)==3 and all(
    abs(r['candidate']['central_recovery']-1)<=.05 and abs(r['candidate']['whole_kernel_recovery']-1)<=.05
    and abs(r['candidate']['major_ratio']-1)<=.03 and abs(r['candidate']['minor_ratio']-1)<=.03
    and r['candidate']['centroid_error']<=.1 for r in rows)
  quality.append(dict(arm=arm,iteration=k,all_arrays_meet_declared_terminal_source_targets=source_targets,
    uninjected_cumulative_native_seconds=sum(float(v) for i,v in times[(arm,'uninjected')].items() if int(i)<=k),
    injected_cumulative_native_seconds=sum(float(v) for i,v in times[(arm,'injected')].items() if int(i)<=k)))

for name,rows in [('PAIRED_METRICS_R0.1.csv',metric_rows),('PROTECTION_CHECKS_R0.1.csv',check_rows),('QUALITY_TRACE_R0.1.csv',quality)]:
 with (out/name).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name,data in [('DISPOSITION_R0.1.json',dispositions),('CONVERGENCE_TRACE_R0.1.json',convergence)]:
 with (out/name).open('x') as f:f.write(json.dumps(data,indent=2,allow_nan=False)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
for col,arm in enumerate(('Half','Hold')):
 rows=[r for r in results[arm]['rows'] if r['array']=='a1400' and 'H' in r]
 for ri,region in enumerate(('annulus','neptune')):
  ax=axes[ri,col];x=[r['iteration'] for r in rows]
  h=np.array([r['H']['regions'][region]['transfer']['rms'] for r in rows])
  c=[r['candidate']['regions'][region]['transfer']['rms'] for r in rows]
  ax.plot(x,h,'o-',color='#222222',label='Historical H');ax.plot(x,c,'s--',color={'Half':'#1976b9','Hold':'#c76816'}[arm],label=arm)
  ax.plot(x,np.minimum(.8*h,h-.1),':',color='#777777',label='Maximum for both improvement targets')
  ax.axvspan(4.8,6.1,color='#777777',alpha=.08);ax.set_title(f'{arm}: a1400 {region}');ax.set_ylabel('Total-response RMS (mJy/beam)')
  ax.set_xticks(range(1,7));ax.grid(alpha=.2);ax.legend(fontsize=8)
  if ri==1:ax.set_xlabel('Absolute iteration')
fig.suptitle('EL-F12 fixed leakage endpoints\nTargets apply at iterations 5 and 6, together with every protection')
fig.savefig(out/'PRIORITIZED_LEAKAGE_R0.1.png',dpi=160);fig.savefig(out/'PRIORITIZED_LEAKAGE_R0.1.pdf');plt.close(fig)

fig,axes=plt.subplots(3,3,figsize=(12,10),sharex=True,layout='constrained')
for col,array in enumerate(('a1100','a1400','a2000')):
 for arm,color in (('Half','#1976b9'),('Hold','#c76816')):
  rows=[r for r in results[arm]['rows'] if r['array']==array and 'H' in r];x=[r['iteration'] for r in rows]
  axes[0,col].plot(x,[r['candidate']['central_recovery'] for r in rows],'o-',color=color,label=arm+' central')
  axes[0,col].plot(x,[r['candidate']['whole_kernel_recovery'] for r in rows],'s--',color=color,label=arm+' whole kernel')
  axes[1,col].plot(x,[max(abs(r['candidate']['major_ratio']-1),abs(r['candidate']['minor_ratio']-1)) for r in rows],'o-',color=color,label=arm)
  axes[2,col].plot(x,[r['support']['lost'] for r in rows],'o-',color=color,label=arm)
 axes[0,col].axhspan(.95,1.05,color='#777777',alpha=.12,label='Terminal recovery range')
 axes[1,col].axhline(.03,linestyle=':',color='#777777',label='Terminal width limit')
 axes[2,col].axhline(0,linestyle=':',color='#777777');axes[0,col].set_title(array)
 for row in range(3):axes[row,col].grid(alpha=.2);axes[row,col].set_xticks(range(1,7));axes[row,col].legend(fontsize=7)
 axes[2,col].set_xlabel('Absolute iteration')
axes[0,0].set_ylabel('Recovered fraction');axes[1,0].set_ylabel('Largest width deviation from kernel');axes[2,0].set_ylabel('Lost pixels from H paired support')
fig.suptitle('EL-F12 all-array recovery, morphology and support\nFull H comparisons and remaining protections are retained in the paired tables')
fig.savefig(out/'ALL_ARRAY_PROTECTIONS_R0.1.png',dpi=160);fig.savefig(out/'ALL_ARRAY_PROTECTIONS_R0.1.pdf');plt.close(fig)
print(json.dumps(dispositions,indent=2))
