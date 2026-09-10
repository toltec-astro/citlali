"""Complete declared influence accounting using saved coefficients/maps only."""
from pathlib import Path
import numpy as np
from core import write_json
from synthetic import CASES
V=Path(__file__).resolve().parent;A=V/'attempt_03';R=V/'report'
rows=[]
for name in CASES:
 with np.load(A/f'T1_{name}.npz') as z:
  g=z['gamma'];e=z['e'];dp=np.zeros((16,81));np.add.at(dp,(z['detector'],z['pixel']),1)
  q=z['Q'];q2=np.einsum('tsad,dp->tsap',g*g,dp);max_d=np.zeros(q.shape)
  for d in range(16):max_d=np.maximum(max_d,g[:,:,:,d,None]*dp[d])
  max_share=max_d/q;conc=q*q/q2
  np.savez_compressed(R/f'T1_{name}_influence.npz',count=dp.sum(axis=0),sum_gamma=q,sum_gamma_squared=q2,concentration=conc,max_detector_share=max_share,total_gamma=np.sum(g*e,axis=-1),total_gamma_squared=np.sum(g*g*e,axis=-1),max_total_detector_share=np.max(g*e,axis=-1)/np.sum(g*e,axis=-1))
  rows.append(dict(case=name,mean_null_max_pixel_detector_share_U=float(max_share[:,0,0].mean()),mean_null_max_pixel_detector_share_N4U=float(max_share[:,0,1].mean()),meaning='coefficient influence only; no independence or inverse-variance claim'))
arrays=[]
for obs in [123424,152389]:
 for array in ['a1100','a1400','a2000']:
  for scope in ['full','window0','window1','window2']:
   for arm in ['U','N4U']:
    with np.load(A/f'T2_{obs}_{array}_{scope}_{arm}.npz') as z:
     q=float(z['Q'].sum());q2=float(z['Q2'].sum())
     arrays.append(dict(obsnum=obs,array=array,scope=scope,policy=arm,count=int(z['count'].sum()),sum_gamma=q,sum_gamma_squared=q2,concentration=q*q/q2,max_detector_share=float(z['detector_Q'].max()/q),fallback_count=int(z['fallback_count'].sum()),fallback_coefficient_mass=float(z['fallback_Q'].sum())))
write_json(R/'WHOLE_POPULATION_INFLUENCE.json',dict(T1=rows,T2=arrays,source='immutable attempt_03 only',new_maps_or_input_access=False))
print('Saved complete influence diagnostics: 9 T1 cases and 48 T2 native maps.')
