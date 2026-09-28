"""Bounded original-I/Q excerpts at retained survey maxima; no RTC or new detector."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,netCDF4
from astropy.table import Table
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W))
from tools.diagnostics.pointing_iq_event_coherence import _apt_arrays,_robust_sigma
R=Path(__file__).resolve().parent
P=Path('/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/pointing-iq-event-survey-20260729/pointing_raw_iq_event_survey.json')
survey=json.loads(P.read_text());records=[];packed={}
for src in survey['inputs']:
 obs=src['obsnum'];assert obs in (152420,152432,152434)
 apt=Table.read(src['apt_path']);fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained');panel=0
 for nw in (11,12):
  candidates=sorted([x for x in survey['rows'] if x['obsnum']==obs and x['network']==nw],key=lambda x:(-x['n_strong_phase_tones'],x['citlali_scan']))[:2]
  path=Path(src['raw_paths'][str(nw)])
  with netCDF4.Dataset(path) as nc:
   time=np.asarray(nc['Data.Toltec.RecvTime'][:]);n,d=nc['Data.Toltec.Is'].shape
   uid,f,usable=_apt_arrays(apt,network=nw,n_tones=d,raw_tone_frequency_hz=np.asarray(nc['Header.Toltec.ToneFreq'][0,:]))
   for k,c in enumerate(candidates):
    center=c['raw_event_absolute_sec'];idx=np.flatnonzero(abs(time-center)<=1.5);tt=time[idx]-center
    z=np.asarray(nc['Data.Toltec.Is'][idx,:],float)+1j*np.asarray(nc['Data.Toltec.Qs'][idx,:],float)
    pre=(tt>=-.25)&(tt<=-.05);post=(tt>=.05)&(tt<=.25)
    phase=np.unwrap(np.angle(z),axis=0);shift=phase[post].mean(0)-phase[pre].mean(0)
    ranked=np.flatnonzero(usable)[np.argsort(-abs(shift[usable]),kind='stable')];tones=ranked[:4]
    phase-=np.median(phase[pre],axis=0);amp=abs(z)/np.median(abs(z)[pre],axis=0)-1
    key=f'o{obs}_n{nw}_s{c["citlali_scan"]}'
    packed[key+'_time']=tt;packed[key+'_phase']=phase[:,tones];packed[key+'_amp']=amp[:,tones];packed[key+'_tones']=tones
    for col,values in enumerate((phase*1000,amp*100)):
     for tone in tones:axs[panel,col].plot(tt,values[:,tone],lw=.65,label=f'tone{tone},uid{uid[tone]}')
     axs[panel,col].axvline(0,color='k',ls=':',lw=.7);axs[panel,col].set_title(f'{obs} nw{nw} survey scan{c["citlali_scan"]} '+('phase (mrad)' if col==0 else 'amplitude (%)'));axs[panel,col].legend(fontsize=6);axs[panel,col].set_xlabel('seconds from retained raw survey maximum')
    records.append(dict(obsnum=obs,network=nw,scan=c['citlali_scan'],raw=str(path),raw_bytes=path.stat().st_size,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),header_identity=[int(nc['Header.Toltec.'+x][...]) for x in ('ObsNum','SubObsNum','ScanNum','RoachIndex')],tune=[int(nc['Header.Toltec.TargSweep'+x][...]) for x in ('ObsNum','SubObsNum','ScanNum')],native_rows=n,tones=d,usable_tones=int(usable.sum()),duration_recv_seconds=float(time[-1]-time[0]),detector_seconds_recv=float(usable.sum()*(time[-1]-time[0])),survey_strong_tones=c['n_strong_phase_tones'],survey_max_fraction=c['coherent_same_sign_fraction'],center_unix=center,excerpt_key=key,review_tones=tones.tolist(),pre_post_phase_mrad=(shift[tones]*1000).tolist(),review_selection='two survey maxima per network ranked by strong-tone count; four largest APT-usable absolute raw phase shifts; full 3s original window, no cleaned outputs',limitation='survey maxima/phase shifts are not step identifications; descriptive receiver time, not reconstructed ALIGN timing'))
    panel+=1
 fig.savefig(R/f'candidate-{obs}.png',dpi=120);plt.close(fig)
np.savez_compressed(R/'candidate-original-excerpts.npz',**packed)
(R/'CANDIDATE_DISCOVERY.json').write_text(json.dumps(dict(candidate_pool=[152420,152432,152434],survey_binding=dict(path=str(P),sha256=hashlib.sha256(P.read_bytes()).hexdigest()),records=records,RTC_executions=0,new_maps_inspected=False),indent=2)+'\n')
print('saved 12 original windows, no RTC execution')
