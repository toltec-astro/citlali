#!/usr/bin/env python3
"""Static native-timestream audit figures. No flags, maps or treatment outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from rtc_disturbance_burden import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads(a.selection.read_text()); index=json.loads((a.analysis/'inspection-index.json').read_text())
    lookup={(x['observation'],x['network'],x['detector']):x for x in index}
    cache={}
    def load(o,n):
        k=(o,n)
        if k not in cache:
            folder=a.campaign/f'{o}-{n:02d}';r=json.loads((folder/'receipt.json').read_text())
            s=np.memmap(folder/'spectra.f64',dtype='<f8',mode='r',shape=(r['channels'],2,4,r['bins']))
            cache[k]=(folder,r,s,np.arange(r['bins'])/(r['fft_samples']*r['measured_interval']))
        return cache[k]
    draws=[x for x in selected if x['selection']=='independent-hash' and (a.campaign/f"{x['observation']}-{x['network']:02d}").exists()]
    draws.sort(key=lambda x:(x['observation'],x['network'],x['channel']))
    if draws:
        fig,axes=plt.subplots(1,2,figsize=(13,12),sharey=True)
        for c,ax in enumerate(axes):
            matrix=[]
            for x in draws:
                _,_,s,f=load(x['observation'],x['network']);v=np.array(s[x['channel'],c,0]);total=np.nansum(v)
                matrix.append(np.log10(np.maximum(v/total,1e-7)) if total>0 else np.full(v.shape,np.nan))
            im=ax.imshow(np.array(matrix),aspect='auto',origin='lower',extent=(0,f[-1],0,len(draws)),vmin=-5,vmax=-.5,cmap='magma',interpolation='nearest')
            ax.set_title(('x','r')[c]+' — independent channels');ax.set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Fixed hash-selected detector occurrence; sorted observation/network/channel')
        ticks=[i for i,x in enumerate(draws) if i==0 or draws[i-1]['observation']!=x['observation']]
        axes[0].set_yticks(ticks,[str(draws[i]['observation']) for i in ticks]);fig.colorbar(im,ax=axes,label='log10 fraction per stored PSD bin',shrink=.7)
        fig.suptitle('Independent spectral census — original native x/r; unavailable rows blank')
        fig.savefig(a.output/'independent-spectral-atlas.png',dpi=130);plt.close(fig)
    # One preselected channel in each of three fixed networks (one per array)
    # per observation. Membership is independent of every spectral descriptor.
    contacts=[];seen=set()
    for x in selected:
        key=(x['observation'],x['network'])
        if x['selection']=='independent-hash' and x['network'] in (0,7,11) and key not in seen and (*key,x['channel']) in lookup:
            contacts.append((x['observation'],x['network'],x['channel'],'independent'));seen.add(key)
    supplement=[]
    for x in selected:
        key=(x['observation'],x['network'],x['channel'])
        if x['selection'].startswith('prior-case-') and key in lookup:supplement.append((*key,x['selection']))
    # Include strong narrow, crowded/profile-sensitive and bursty examples from
    # exported raw inspections; explicitly label this supplementary selection.
    exported=[x for x in index if (a.campaign/f"{x['observation']}-{x['network']:02d}"/f"samples-{x['detector']}.f64").exists()]
    used=set(x[:3] for x in supplement)
    for criterion in ('narrow_fraction','burst_ratio','profile_sensitive_review','broad_or_crowded_review'):
        ranked=sorted(exported,key=lambda x:x['coordinates'][0].get(criterion) or 0,reverse=True)
        chosen=0
        for x in ranked:
            key=(x['observation'],x['network'],x['detector'])
            if key in used:continue
            supplement.append((*key,'supplement:'+criterion));used.add(key);chosen+=1
            if chosen==3:break
    def pages(items,prefix):
        for page,start in enumerate(range(0,len(items),6)):
            group=items[start:start+6];fig,axes=plt.subplots(len(group),3,figsize=(16,2.7*len(group)),squeeze=False)
            for axesrow,(o,n,d,label) in zip(axes,group):
                folder,r,s,f=load(o,n);values=np.memmap(folder/f'samples-{d}.f64',dtype='<f8',mode='r',shape=(r['rows'],4));t=np.fromfile(folder/'native-time.f64',dtype='<f8');item=lookup[(o,n,d)]
                for c,color in [(0,'tab:blue'),(1,'tab:orange')]:
                    v=np.array(values[:,c]);v[values[:,c+2]==0]=np.nan
                    finite=np.isfinite(v);scale=np.nanstd(v) if finite.any() else 1;center=np.nanmedian(v) if finite.any() else 0
                    normalized=(v-center)/scale if scale>0 else v-center
                    axesrow[0].plot(t,normalized+5*c,color=color,lw=.35,label=('x','r')[c])
                    middle=t[len(t)//2];mask=np.abs(t-middle)<=2
                    axesrow[1].plot(t[mask]-middle,normalized[mask]+5*c,color=color,lw=.6)
                    psd=s[d,c,0];total=np.nansum(psd)
                    if total>0:
                        axesrow[2].semilogy(f,psd/total,color=color,lw=.8,label=('x','r')[c])
                        axesrow[2].semilogy(f,s[d,c,2]/total,color=color,lw=.6,ls='--',alpha=.6)
                axesrow[0].set_title(f'{o} / nw {n} / ch {d} — {label}\nAPT quality {item["apt_good"]}; eligible {item["eligible"]}',fontsize=9)
                axesrow[0].set_xlabel('Native time (s)');axesrow[0].set_ylabel('Centered / own rms; r offset +5')
                axesrow[1].set_title('Fixed middle 4 seconds; original samples',fontsize=9);axesrow[1].set_xlabel('Seconds from fixed midpoint')
                axesrow[2].set_title('Original PSD; dashed accepted ±2 Hz background',fontsize=9);axesrow[2].set_xlabel('Frequency (Hz)');axesrow[2].set_ylim(1e-7,1)
                for ax in axesrow:ax.grid(alpha=.2)
            fig.suptitle('Inert native audit — exploratory inspections; no exclusions or corrections applied',fontsize=12)
            fig.tight_layout(rect=(0,0,1,.975));fig.savefig(a.output/f'{prefix}-{page+1:02d}.png',dpi=120);plt.close(fig)
    pages(contacts,'independent');pages(supplement,'supplement')
    (a.output/'contact-selection.json').write_text(json.dumps(dict(independent=contacts,supplement=supplement),indent=2)+'\n')
    summary=json.loads((a.analysis/'summary.json').read_text())
    fam=summary['families'][:20]
    if fam:
        fig,ax=plt.subplots(figsize=(11,4));ax.bar([str(round(x['frequency_hz'],2)) for x in fam],[x['eligible_us']/1e6/3600 for x in fam]);ax.set_ylabel('Detector-hours in selected occurrences');ax.set_xlabel('Strong narrow x peak family (Hz), ordered by exposure');ax.tick_params(axis='x',rotation=60);ax.set_title('Exploratory ≥10% narrow-excess screen — whole-observation exposure, not interference duration');fig.tight_layout();fig.savefig(a.output/'frequency-families.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
