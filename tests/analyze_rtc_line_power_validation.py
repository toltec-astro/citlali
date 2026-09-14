#!/usr/bin/env python3
"""Summarize exported C++ line-power fixtures; does not estimate a new PSD.

Run with the TolTEC venv and a headless Matplotlib backend. Empirical percentiles
are descriptive results from the bounded seeds, never admission thresholds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(args.input/'cases.csv');p=pd.read_csv(args.input/'spectra.csv');w=pd.read_csv(args.input/'windows.csv');raw=pd.read_csv(args.input/'raw.csv')
    regions=pd.read_csv(args.input/'regions.csv')
    regions['peak_hz']=regions.first_hz+(regions.peak_bin-regions.first_bin)*regions.bin_span_hz/(regions.past_last_bin-regions.first_bin)
    keys=['family','support','seed','radius_hz']
    ranked=regions.sort_values(keys+['positive_excess'],ascending=[True]*4+[False],kind='stable').copy()
    ranked['rank']=ranked.groupby(keys).cumcount()+1
    ranked[ranked['rank']<=4].to_csv(args.output/'candidate-ranks.csv',index=False)
    assert len(c)==873 and len(c[c.family.str.startswith('null_')])==576
    assert set(c.radius_hz)=={1,2,4}
    null=c[c.family.str.startswith('null_')].copy()
    null['positive_fraction']=null.positive_excess_sum/null.total_stored_power
    null_rows=[]
    for (family,support,radius),a in null.groupby(['family','support','radius_hz'],sort=True):
        assert len(a)==32
        null_rows.append(dict(family=family,support=support,radius_hz=float(radius),seeds=len(a),windows=int(a.windows.iloc[0]),
            band_background_to_mean_psd=float(a.band_background_power.mean()/a.band_stored_power.mean()),
            positive_fraction_median=float(a.positive_fraction.median()),positive_fraction_p95=float(a.positive_fraction.quantile(.95)),
            strongest_fraction_median=float(a.strongest_fraction.median()),strongest_fraction_p95=float(a.strongest_fraction.quantile(.95)),
            region_count_median=float(a.region_count.median())))
    injections=c[c.family.str.startswith(('white_','sloping_'))].copy();injection_rows=[]
    for (family,radius),a in injections.groupby(['family','radius_hz'],sort=True):
        assert len(a)==8
        detected=a.matched_regions_power/a.injected_estimator_band_power
        raw_increment=(a.band_stored_power-a.paired_null_raw_band)/a.injected_estimator_band_power
        signed_increment=(a.band_signed_residual-a.paired_null_signed_band)/a.injected_estimator_band_power
        signed=a.band_signed_residual/a.injected_estimator_band_power
        entry=dict(family=family,radius_hz=float(radius),seeds=len(a),input_mean_square=float(a.injected_input_mean_square.mean()),injected_estimator_band_power=float(a.injected_estimator_band_power.mean()))
        selected=regions[(regions.family==family)&(regions.radius_hz==radius)&regions.peak_hz.between(2,4)]
        counts=selected.groupby('seed').size().reindex(a.seed,fill_value=0)
        entry.update(descriptive_band_region_count_median=float(counts.median()),
                     descriptive_band_region_count_min=int(counts.min()),descriptive_band_region_count_max=int(counts.max()))
        for name,v in [('detected_regions',detected),('raw_band_increment',raw_increment),('signed_band_increment',signed_increment),('signed_band',signed)]:
            entry[name+'_median']=float(v.median());entry[name+'_p16']=float(v.quantile(.16));entry[name+'_p84']=float(v.quantile(.84))
        injection_rows.append(entry)
    nr=pd.DataFrame(null_rows);ir=pd.DataFrame(injection_rows)
    nr.to_csv(args.output/'null-summary.csv',index=False);ir.to_csv(args.output/'injection-summary.csv',index=False)
    summary=dict(null_profiles=576,injection_profiles=288,transient_profiles=6,retained_real_profiles=3,
        null=null_rows,injection=injection_rows,
        limits=['32 null seeds and 8 paired injection seeds; empirical distributions only',
                'Trial median is not calibrated to mean; positive-region power is selection-biased',
                'Injection band is fixed at 2–4 Hz inclusive of bin centers; matched regions have representative peak in that band and may extend outside it',
                'Raw band increment subtracts the matched noise realization; signed increment also includes changes in trial background',
                'Known infinite stationary input PSD differs from its finite centered/windowed/padded estimator image',
                '16 Hz null/injection controls are spectral fixtures; transient joint controls use 128 Hz to meet the existing 256-difference noise minimum',
                'Descriptive region counts record merging/splitting of connected positive support, not a recovered oscillator count',
                'Retained Case E values use an explicitly reindexed test parent and fixture cadence bound, not an operational realization',
                'Window witness uses independent direct DFT to audit actual stored windows; overlapping windows are not independent',
                'No line significance, contamination power, intrinsic width, persistence admission or notch benefit inferred'])
    (args.output/'measurement-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    groups=[('null_white','short'),('null_white','long'),('null_white','gap_padding'),('null_sloping','short'),('null_sloping','long'),('null_sloping','gap_padding')]
    labels=['White\nshort','White\nlong','White\ngap/pad','Sloping\nshort','Sloping\nlong','Sloping\ngap/pad']
    fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
    x=np.arange(6)
    for radius,color,offset in [(1,'#279e68',-.24),(2,'#245d9b',0),(4,'#cd7b20',.24)]:
        vals=[nr[(nr.family==f)&(nr.support==s)&(nr.radius_hz==radius)].band_background_to_mean_psd.iloc[0] for f,s in groups]
        axes[0].bar(x+offset,vals,width=.22,label=f'±{radius} Hz',color=color)
    axes[0].axhline(1,color='black',lw=1);axes[0].set_ylabel('Mean trial background /\nmean measured PSD in 2–4 Hz');axes[0].legend(ncol=3);axes[0].set_title('Noise only: background bias and positive-selection floor')
    for ax,col,label in [(axes[1],'positive_fraction','Sum of positive excess /\ntotal stored PSD power'),(axes[2],'strongest_fraction','Strongest connected region /\ntotal stored PSD power')]:
        data=[null[(null.family==f)&(null.support==s)&(null.radius_hz==2)][col].to_numpy() for f,s in groups]
        ax.boxplot(data,positions=x,widths=.45,showfliers=True)
        for j,v in enumerate(data):ax.scatter(j+np.linspace(-.13,.13,len(v)),v,s=9,alpha=.5,color='#245d9b')
        ax.set_ylabel(label);ax.set_title('Initial ±2 Hz profile; all 32 fixed realizations shown')
    for ax in axes:ax.set_xticks(x,labels);ax.set_xlim(-.6,5.6)
    fig.savefig(args.output/'null-behavior.png',dpi=150);plt.close(fig)
    shapes=['on_bin','off_bin','weak','neighboring','chirp','dense']
    fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    for ax,background in zip(axes,['white','sloping']):
        for metric,color,offset,label in [('detected_regions','#cd7b20',-.23,'Detected regions'),('signed_band_increment','#279e68',0,'Signed band increment'),('raw_band_increment','#245d9b',.23,'Raw band increment')]:
            vals=[ir[(ir.family==background+'_'+s)&(ir.radius_hz==2)].iloc[0] for s in shapes]
            med=np.array([v[metric+'_median'] for v in vals]);lo=np.array([v[metric+'_p16'] for v in vals]);hi=np.array([v[metric+'_p84'] for v in vals])
            ax.errorbar(x+offset,med,yerr=[med-lo,hi-med],fmt='o',capsize=4,color=color,label=label)
        ax.axhline(1,color='black',lw=1);ax.axhline(0,color='grey',lw=.7);ax.set_xticks(x,[s.replace('_',' ') for s in shapes]);ax.set_ylabel('Measured power / known injected\nestimator power in fixed 2–4 Hz band');ax.set_title(background.capitalize()+' background — median and 16–84% range over 8 paired seeds')
    axes[0].legend(ncol=3);fig.suptitle('Initial ±2 Hz profile: region selection and a predeclared band give different answers')
    fig.savefig(args.output/'injection-recovery.png',dpi=150);plt.close(fig)
    families=[('persistent_tone','white_on_bin'),('impulse','impulse'),('level_shift','level_shift'),('retained_case_e','retained_case_e')]
    fig,axes=plt.subplots(4,3,figsize=(16,12),constrained_layout=True)
    for row,(name,spectral_name) in enumerate(families):
        a=raw[raw.family==name];b=p[(p.family==spectral_name)&(p.radius_hz==2)&(p.seed==0)];v=w[w.family==name]
        axes[row,0].plot(a.relative_time,a.x,lw=.65);axes[row,0].set_title(name.replace('_',' '));axes[row,0].set_xlabel('Native relative time (s)');axes[row,0].set_ylabel('Original x')
        axes[row,1].semilogy(b.frequency_hz,b.psd,label='Accepted PSD',lw=1);axes[row,1].semilogy(b.frequency_hz,b.background,label='Trial background',lw=1)
        axes[row,1].fill_between(b.frequency_hz,b.background,b.psd,where=b.psd>b.background,alpha=.25,color='#cd7b20');axes[row,1].set_xlabel('Frequency (Hz)');axes[row,1].set_ylabel('x units² / Hz')
        t=(v.support_begin+v.support_end)/2-1000
        for center,(_,entry) in zip(t,v.iterrows()):axes[row,2].plot([entry.support_begin-1000,entry.support_end-1000],[entry.band_stored_power]*2,color='#a9b7c7',lw=2,alpha=.8)
        axes[row,2].scatter(t,v.band_stored_power,c=np.where(v.contains_control_event,'#c43c39','#245d9b'),s=20,zorder=3)
        axes[row,2].set_xlabel('Window support and center (native s)');axes[row,2].set_ylabel('Predeclared-band window power');axes[row,2].set_title(f'{len(v)} windows; shared support retained')
    axes[0,1].legend();fig.suptitle('Original values, descriptive spectral features, and time-resolved window audit\nOverlapping windows are correlated; no persistence classification is made')
    fig.savefig(args.output/'window-witnesses.png',dpi=150);plt.close(fig)
    print(nr[nr.radius_hz==2].to_string(index=False));print(ir[ir.radius_hz==2][['family','detected_regions_median','signed_band_increment_median','raw_band_increment_median']].to_string(index=False))

if __name__=='__main__':main()
