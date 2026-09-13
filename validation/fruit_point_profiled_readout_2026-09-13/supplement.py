"""Saved-parameter boundary and practical-precision diagnostics; no fitting."""
import numpy as np
from common import *
from profiled import QSCALE,LOW,HIGH
rows=read(HERE/'RESULTS.json');problems={r['id']:r for r in read(HERE/'PROBLEMS.json')}
witness_ids={r['id'] for r in read(HERE/'WITNESS_RESULTS.json')}
with np.load(OUT/'geometry.npz') as z:x,y,central=z['x'],z['y'],z['central']
records=[]
for r in rows:
    pr=problems[r['id']];old=pr['original_fit'];p=np.array(old['parameters']);q=p[1:6]
    with np.load(pr['map_path']) as z:total=z['total'][pr['array_index']]
    mask=central[pr['array_index']]&(np.hypot(x,y)<=r['radius'])&np.isfinite(total)
    residual=gaussian(p,x[mask],y[mask],True)-total[mask]
    jac=model_jac(p,x[mask],y[mask]);scale=r['repaired']['normalization_scale']
    grad=np.sum(jac[:,1:6]*residual[:,None],axis=0)/scale**2
    scaled=QSCALE*grad;cost=.5*float(np.sum(residual**2))/scale**2
    active_low=q-LOW<=1e-8*QSCALE;active_high=HIGH-q<=1e-8*QSCALE
    projected=np.where((active_low&(scaled>0))|(active_high&(scaled<0)),0.,scaled)
    records.append(dict(id=r['id'],primary=r['primary'],witness=r['id'] in witness_ids,
        original_angle_rad=p[5],angle_upper_active=bool(active_high[4]),angle_lower_active=bool(active_low[4]),
        original_raw_scaled_geometry_gradient=scaled,original_relative_unprojected_gradient=float(np.max(abs(scaled))/max(cost,1)),
        original_relative_projected_gradient=float(np.max(abs(projected))/max(cost,1)),
        original_outward_angle_gradient=bool((active_high[4] and grad[4]<0) or (active_low[4] and grad[4]>0)),
        original_status=old['solver_status'],new_angle_rad=r['repaired']['selected_fit']['parameters'][5]))
comparisons=read(HERE/'FIT_COMPARISON.json');measurements=read(HERE/'MEASUREMENTS.json')
selected_keys={(r['case'],r['arm']) for r in measurements}
old_receipts=[r for r in read(OUT/'RECEIPTS.json') if r['pass_index']==6 and (r['case'],r['arm']) in selected_keys]
summary=dict(old_witnesses_at_angular_upper_bound=sum(r['witness'] and r['angle_upper_active'] for r in records),
    old_witnesses_with_outward_angle_gradient=sum(r['witness'] and r['original_outward_angle_gradient'] for r in records),
    original_geometry_max_abs_relative_peak_change=max(abs(r['linear']['peak']/r['original']['peak']-1) for r in comparisons if r['primary']),
    original_geometry_max_relative_objective_improvement=max(r['coefficient_relative_objective_improvement'] for r in comparisons if r['primary']),
    original_geometry_max_background_change=max(float(np.max(abs(np.array(r['linear']['background'])-r['original']['background']))) for r in comparisons if r['primary']),
    original_terminal_evaluation_seconds=sum(r['evaluation_seconds'] for r in old_receipts),
    old_timing_role='archived full evaluator timing; not a controlled fit-only benchmark',
    all_start_peak_spread_gt_5pct=sum(r['all_start_spread']['relative_peak_span']>.05 for r in comparisons),
    all_start_centroid_spread_gt_1arcsec=sum(r['all_start_spread']['centroid_diameter']>1 for r in comparisons),
    max_similar_cost_peak_span=max(r['similar_cost_spread']['relative_peak_span'] for r in comparisons),
    max_similar_cost_centroid_diameter=max(r['similar_cost_spread']['centroid_diameter'] for r in comparisons),
    primary_max_similar_cost_peak_span=max(r['similar_cost_spread']['relative_peak_span'] for r in comparisons if r['primary']),
    primary_max_similar_cost_centroid_diameter=max(r['similar_cost_spread']['centroid_diameter'] for r in comparisons if r['primary']),
    first_order_misses=[dict(id=r['id'],relative_gradient=r['repaired']['attempts'][r['repaired']['selected_start']]['diagnostics']['relative_scaled_projected_gradient']) for r in rows if not r['repaired']['attempts'][r['repaired']['selected_start']]['diagnostics']['numerical_complete']])
write(HERE/'SUPPLEMENTAL_DIAGNOSTICS.json',dict(summary=summary,original_boundary_checks=records))
print(summary)
print('witness bound checks',[(r['id'],r['original_relative_projected_gradient'],r['original_relative_unprojected_gradient']) for r in records if r['witness']])
