"""Saved-only plots; no originals, scientific fitting or learning."""
from pathlib import Path
import json,sys,textwrap
import numpy as np
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else root;out.mkdir(parents=True,exist_ok=True);windows=json.loads((root/'windows.json').read_text());z=np.load(root/'windows.npz')
for kind in ('event','unflagged'):
 chosen=[q for q in windows if q['kind']==kind]
 for stage in ('original','stages'):
  fig,axes=plt.subplots(8,2,figsize=(15,24),layout='constrained');axes=axes.ravel()
  for ax,q in zip(axes,chosen):
   key=q['key'];get=lambda k:z[key+'_'+k];t=get('time')-get('time')[0];center=(get('native_row')==q['center']);ct=t[center][0]
   for row in q.get('physical_run_boundaries',[]):ax.axvline(t[row-q['first']],color='red',ls='--',lw=1,label='physical boundary')
   for row in q.get('processing_boundaries',[]):ax.axvline(t[row-q['first']],color='teal',ls=':',lw=.8,label='processing science boundary')
   if stage=='original':
    for k,color in [('raw_x','C0'),('raw_r','C1')]:
     v=np.where(get('valid_x' if k=='raw_x' else 'valid_r'),get(k),np.nan);base=np.nanmedian(v[t<ct]);scale=1.4826*np.nanmedian(abs(np.diff(v)-np.nanmedian(np.diff(v))));scale=scale if scale>0 else 1
     ax.plot(t,(v-base)/scale,lw=.7,color=color,label=k+' / own diff MAD')
    peer=get('peer_x');base=np.nanmedian(peer[t<ct],axis=0);sc=1.4826*np.nanmedian(abs(np.diff(peer,axis=0)-np.nanmedian(np.diff(peer,axis=0),axis=0)),axis=0);sc=np.where(sc>0,sc,1);ax.plot(t,(peer-base)/sc,color='gray',lw=.4,alpha=.4)
    ax.set_ylabel('own pre-median removed; diff-MAD units')
   else:
    # Same physical window; each stage has a labelled display-only median offset.
    for k,color in [('CAL','C0'),('M0','C2'),('T5_residual','C3'),('T5_total','C4')]:
     v=get(k);good=np.isfinite(v)
     if good.any():ax.plot(t[good],v[good]-np.nanmedian(v),'.',ms=1,color=color,label=k)
    raw=np.where(get('valid_x'),get('raw_x'),np.nan);rtc=get('RTC_x');valid=(get('state')&1)!=0;good=np.isfinite(rtc)&valid
    if np.isfinite(get('CAL')).any():
     value=np.where(valid,rtc*get('CAL_multiplier'),np.nan);good=np.isfinite(value)
     if good.any():ax.plot(t[good],value[good]-np.nanmedian(value),'.',ms=1,color='C5',label='RTC x × CAL multiplier')
    if not np.isfinite(get('CAL')).any():
     ax.plot(t,raw-np.nanmedian(raw),color='C0',lw=.6,label='raw x only; no downstream support');ax.set_ylabel('raw Δf/f; median removed')
    else:ax.set_ylabel('mJy/beam; own-stage median removed')
    ax.fill_between(t,0,1,where=~valid,color='red',alpha=.1,transform=ax.get_xaxis_transform())
   ax.fill_between(t,0,1,where=get('transition'),color='purple',alpha=.2,transform=ax.get_xaxis_transform());ax.fill_between(t,.94,1,where=get('fitting_guard'),color='gray',alpha=.3,transform=ax.get_xaxis_transform());ax.axvline(ct,color='k',ls=':',lw=.7);ax.fill_between(t,0,1,where=get('source'),color='gold',alpha=.15,transform=ax.get_xaxis_transform());ax.set_title(f"{key}: channel{q['channel']} event{q['event']}\n{textwrap.fill(q['rule'],65)}",fontsize=8);ax.set_xlabel('seconds from excerpt start');ax.legend(fontsize=6,loc='best')
  fig.suptitle(root.name+' '+kind+' '+stage+'; full samples, no decimation. Gold=geometric source context. Gray=original x peers; purple=accepted transition; top gray=fit guard.',fontsize=11)
  fig.savefig(out/f'{kind}-{stage}.png',dpi=110);plt.close(fig)
