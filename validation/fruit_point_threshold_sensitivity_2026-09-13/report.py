"""Pure saved-JSON arithmetic and Markdown reporting; standard library only."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'fruit_point_profiled_readout_2026-09-13'
THRESHOLDS=[5.,7.5,10.,15.]
KINDS=['original','linear','repaired','truth_template']
ARRAYS=['a1100','a1400','a2000']
LABELS={'absolute_peak':'Absolute peak','matched_ratio':'Matched H/D ratio','crossed_ratio':'Crossed H/D ratio','original':'Original free fit','linear':'Original-geometry coefficients','repaired':'Repaired free fit','truth_template':'Injected-profile coefficient'}

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['case'],r['arm'],r['array'],r['radius']
def present(v):return v is not None and math.isfinite(v)
def sign(v):return 'undefined' if v is None else f'{v:+.4f}'
def flag(v):return '—' if v is None else ('yes' if v else 'no')
def idtext(r):
    if r['quantity']=='absolute_peak':return r['case']
    return f"H{r['numerator_seed']}/D{r['denominator_seed']}"

def summarize(rows):
    finite=[r for r in rows if present(r['signed_error_pct'])]
    usable=[r for r in rows if r['usable_at_60'] is True]
    numeric_flags=[r['numerically_complete'] for r in rows]
    applicable=any(r['usable_at_60'] is not None for r in rows)
    sorted_rows=sorted(finite,key=lambda r:(abs(r['signed_error_pct']),r['id']))
    return dict(required=len(rows),numeric_available=len(finite),usable_available=len(usable) if applicable else None,
        numerical_complete=None if all(v is None for v in numeric_flags) else sum(v is True for v in numeric_flags),
        numerical_incomplete=None if all(v is None for v in numeric_flags) else sum(v is False for v in numeric_flags),
        thresholds=[dict(percent=t,raw=sum(present(r['error_fraction']) and abs(r['error_fraction'])<=t/100 for r in rows),
            usable=sum(present(r['error_fraction']) and abs(r['error_fraction'])<=t/100 and r['usable_at_60'] is True for r in rows) if applicable else None) for t in THRESHOLDS],
        median_signed_pct=statistics.median(r['signed_error_pct'] for r in finite) if finite else None,
        median_absolute_pct=statistics.median(abs(r['signed_error_pct']) for r in finite) if finite else None,
        worst=None if not finite else dict(id=sorted_rows[-1]['id'],array=sorted_rows[-1]['array'],case=idtext(sorted_rows[-1]),signed_error_pct=sorted_rows[-1]['signed_error_pct']),
        sorted_magnitudes=[dict(id=r['id'],array=r['array'],case=idtext(r),magnitude_pct=abs(r['signed_error_pct']),
            signed_error_pct=r['signed_error_pct'],usable_at_60=r['usable_at_60'],numerically_complete=r['numerically_complete']) for r in sorted_rows])

def main():
    files=[PRIOR/n for n in ['FIT_COMPARISON.json','RATIOS.json','MEASUREMENTS.json','RESULT_MANIFEST.json','PRESERVATION_START.json']]
    manifest=read(PRIOR/'RESULT_MANIFEST.json');bound={r['path']:r for r in manifest['files']}
    for p in files[:3]:assert digest(p)==bound[p.name]['sha256']
    states=read(PRIOR/'PRESERVATION_START.json')['protected_worktrees']
    for w in states:assert subprocess.check_output(['git','-C',w['path'],'status','--short'],text=True)==w['status']
    start=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=str(p),sha256=digest(p)) for p in files],protected_worktrees=states,
        source_script_sha256=digest(HERE/'report.py'),scope_sha256=digest(HERE/'SCOPE.md'))
    write(HERE/'INPUT_BINDINGS.json',start)
    comp=read(files[0]);ratios=read(files[1]);measure=read(files[2])
    lookup={key(r):r for r in comp};ml={(r['case'],r['arm'],r['array']):r for r in measure}
    def flags(q,kind):
        related=lookup[q['case'],q['arm'],q['array'],52 if q['radius']==60 else 60]
        fit=q[kind];m=ml[q['case'],q['arm'],q['array']]
        diag=kind in ['linear','truth_template']
        record=None if diag else m[kind]
        judgment=None if diag else record['judgments']
        return dict(fit_available=q['new_fit_available'] if kind=='repaired' else bool(fit and fit.get('available',True)),
            usable_at_60=None if diag else judgment['peak_response_usable'],
            availability_domain=60 if not diag else None,
            numerical_complete=q['new_numerical_complete'] if kind=='repaired' else None,
            paired_domain_numerical_complete=related['new_numerical_complete'] if kind=='repaired' else None,
            numerical_flag_meaning='saved independent first-order criterion' if kind=='repaired' else 'not assessed here',
            judgments=judgment,
            fit_solver_status=None if diag else (record['original'].get('fit') if q['radius']==60 else record['fixed_inner_domain_probe'].get('fit') or {}).get('solver_status'),
            score=None if diag else record['original'].get('empirical_peak_score'),
            inner_probe_available=None if diag else record['fixed_inner_domain_probe']['available'],
            peak_domain_change=None if diag else record['fixed_inner_domain_probe'].get('peak_relative_difference'))
    rows=[]
    for q in comp:
        for kind in KINDS:
            fit=q.get(kind)
            if fit is None:continue  # This readout was not defined for that safeguard; no fit is discarded.
            value=fit.get('amplitude') if kind=='truth_template' else fit.get('peak')
            error=fit.get('peak_error')
            f=flags(q,kind)
            rows.append(dict(id=q['id']+'_'+kind,quantity='absolute_peak',case=q['case'],arm=q['arm'],array=q['array'],
                noise_seed=int(q['case'].rsplit('_',1)[1]),radius=q['radius'],primary=q['primary'],readout=kind,
                value=value,error_fraction=error,signed_error_pct=None if error is None else 100*error,
                usable_at_60=f['usable_at_60'],numerically_complete=f['numerical_complete'],members=[f],
                measurement_role='coefficient against known injected profile' if kind=='truth_template' else 'continuous Gaussian peak'))
    abslookup={(r['case'],r['arm'],r['array'],r['radius'],r['readout']):r for r in rows}
    for q in ratios:
        hn=f"H_{q['numerator_seed']}";dn=f"D_{q['denominator_seed']}"
        h=abslookup[hn,q['arm'],q['array'],q['radius'],q['readout']]
        d=abslookup[dn,q['arm'],q['array'],q['radius'],q['readout']]
        expected=h['value']/d['value'];assert math.isclose(expected,q['ratio'],rel_tol=2e-15,abs_tol=2e-15)
        error=expected/(100/90)-1;assert math.isclose(error,q['error'],rel_tol=2e-13,abs_tol=2e-15)
        usable=None if h['usable_at_60'] is None else h['usable_at_60'] and d['usable_at_60']
        assert usable==q['full_domain_pair_usable']
        assert q['raw_within_5pct']==(abs(q['error'])<=.05)
        numerical=None if h['numerically_complete'] is None else h['numerically_complete'] and d['numerically_complete']
        rows.append(dict(id=f"{q['arm']}_{q['array']}_{q['radius']}_H{q['numerator_seed']}_D{q['denominator_seed']}_{q['readout']}",
            quantity=q['pairing']+'_ratio',case=None,arm=q['arm'],array=q['array'],numerator_seed=q['numerator_seed'],
            denominator_seed=q['denominator_seed'],radius=q['radius'],primary=True,readout=q['readout'],value=q['ratio'],
            error_fraction=q['error'],signed_error_pct=100*q['error'],usable_at_60=usable,numerically_complete=numerical,
            members=[dict(case=hn,**h['members'][0]),dict(case=dn,**d['members'][0])],measurement_role=h['measurement_role']))
    write(HERE/'ERROR_RECORDS.json',rows)
    summaries=[]
    for population in ['H_D','safeguards']:
        for radius in [60,52]:
            for kind in KINDS:
                for arm in ['P','C']:
                    for quantity in ['absolute_peak','matched_ratio','crossed_ratio']:
                        for array in ['all']+ARRAYS:
                            rr=[r for r in rows if r['primary']==(population=='H_D') and r['radius']==radius and r['readout']==kind and r['arm']==arm and r['quantity']==quantity and (array=='all' or r['array']==array)]
                            if rr:summaries.append(dict(population=population,radius=radius,readout=kind,arm=arm,quantity=quantity,array=array,**summarize(rr)))
    write(HERE/'SUMMARY.json',summaries)
    def numflag(q):
        if q['readout']!='repaired':return 'not assessed'
        return '; '.join((('H: ' if i==0 else 'D: ') if len(q['members'])==2 else '')+flag(v['numerical_complete'])+' / '+flag(v['paired_domain_numerical_complete']) for i,v in enumerate(q['members']))
    def countcell(q,t):
        c=next(c for c in q['thresholds'] if c['percent']==t)
        return f"{c['raw']}/{q['required']} ; "+('—' if c['usable'] is None else f"{c['usable']}/{q['required']}")
    def counttable(qq):
        lines=['| Quantity | Arm | Readout | Available for peak use | 5% raw ; usable | 7.5% raw ; usable | 10% raw ; usable | 15% raw ; usable |',
               '| --- | --- | --- | ---: | --- | --- | --- | --- |']
        for q in qq:
            avail='—' if q['usable_available'] is None else f"{q['usable_available']}/{q['required']}"
            lines.append(f"| {LABELS[q['quantity']]} | {q['arm']} | {LABELS[q['readout']]} | {avail} | "+' | '.join(countcell(q,t) for t in THRESHOLDS)+' |')
        return '\n'.join(lines)
    def errortable(qq):
        lines=['| Array | State or H/D noise pairing | Arm | Signed error % | Peak usable at 60″ | Numerical complete: fit / other domain | Warnings / limitations |',
               '| --- | --- | --- | ---: | --- | --- | --- |']
        def order(r):
            if r['quantity']=='absolute_peak':
                state=r['case'].rsplit('_',1)[0]
                return (r['array'],r['noise_seed'],{'H':0,'D':1,'H-shift':2,'N':3,'E':4}.get(state,5),['P','C'].index(r['arm']))
            return (r['array'],r['numerator_seed'],r['denominator_seed'],['P','C'].index(r['arm']))
        for q in sorted(qq,key=order):
            limitations=[]
            for v in q['members']:
                j=v['judgments']
                if j:
                    desc=list(j['limitations'])
                    if j['shape_warning']:desc.append('shape warning')
                    if j['support_warning']:desc.append('support warning')
                    if not j['peak_score_pass']:desc.append('low peak score')
                    if not j['peak_stability_pass']:desc.append('peak domain check')
                    if desc:limitations.append((v.get('case','')+': ' if v.get('case') else '')+', '.join(desc))
            lines.append(f"| {q['array']} | {idtext(q)} | {q['arm']} | {sign(q['signed_error_pct'])} | {flag(q['usable_at_60'])} | {numflag(q)} | {'; '.join(limitations) or 'none'} |")
        return '\n'.join(lines)
    # Preserve every source record, including original and coefficient diagnostics, in a separate human-readable appendix.
    appendix=['# Every saved error and flag','',
        'All percentages are signed. No threshold, availability judgment or numerical flag is altered. The primary H/D population and safeguard populations are kept separate. All original judgments are preserved verbatim in ERROR_RECORDS.json. Each source fit is also linked by its original id. Coefficient readouts have no operational peak usability.','']
    for radius in [60,52]:
        for kind in KINDS:
            for quantity in ['absolute_peak','matched_ratio','crossed_ratio']:
                rr=[q for q in rows if q['radius']==radius and q['readout']==kind and q['quantity']==quantity]
                if not rr:continue
                appendix += [f'## {radius} arcsec — {LABELS[kind]} — {LABELS[quantity]}','',errortable(rr),'']
    (HERE/'EVERY_ERROR.md').write_text('\n'.join(appendix)+'\n')
    detail=['# Counts and sorted errors','',
        'Counts use full denominators. Null errors are undefined and never treated as successes; safeguard aggregates remain separate from H/D. At 52 arcsec, usable counts describe the fixed 60-arcsec-usable subset only. No 52-arcsec policy is being evaluated. Each ordered list contains every finite error with its identity, including withheld and numerically incomplete results.','']
    for radius in [60,52]:
        for kind in KINDS:
            qq=[q for q in summaries if q['population']=='H_D' and q['radius']==radius and q['readout']==kind and q['array']=='all']
            detail += [f'## H/D: {radius} arcsec — {LABELS[kind]}','',counttable(qq),'']
            for q in qq:
                detail += [f"**{q['arm']} {LABELS[q['quantity']]}**: median signed {sign(q['median_signed_pct'])}%; median absolute {sign(q['median_absolute_pct'])}%; worst {sign(q['worst']['signed_error_pct'])}% ({q['worst']['array']}, {q['worst']['case']}).",'',
                    '| Sorted magnitude % | Signed error % | Array | Case | Usable at 60″ | Numerical complete |',
                    '| ---: | ---: | --- | --- | --- | --- |']
                for v in q['sorted_magnitudes']:
                    detail.append(f"| {v['magnitude_pct']:.4f} | {sign(v['signed_error_pct'])} | {v['array']} | {v['case']} | {flag(v['usable_at_60'])} | {flag(v['numerically_complete'])} |")
                detail.append('')
    detail+=['## Per-array summaries, all finite H/D results','',
        '| Domain | Readout | Arm | Quantity | Array | n | Median signed % | Median absolute % | Worst signed % |',
        '| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for q in summaries:
        if q['population']=='H_D' and q['array']!='all':detail.append(f"| {q['radius']} | {LABELS[q['readout']]} | {q['arm']} | {LABELS[q['quantity']]} | {q['array']} | {q['required']} | {sign(q['median_signed_pct'])} | {sign(q['median_absolute_pct'])} | {sign(q['worst']['signed_error_pct'])} |")
    detail += ['', '## Safeguard counts, separate from H/D', '',
        'Every shifted/null/boundary state is retained. These per-state counts do not enter the primary H/D totals. Null has no defined percentage error, so a peak-error success count is not meaningful; its saved source/availability flags remain in EVERY_ERROR.md.', '',
        '| Domain | State | Arm | Readout | Numeric errors defined | Peak usable at 60″ | 5% raw ; usable | 7.5% raw ; usable | 10% raw ; usable | 15% raw ; usable |',
        '| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |']
    for radius in [60,52]:
        for case in ['H-shift_20260911','N_20260911','E_20260911']:
            for arm in ['P','C']:
                for kind in ['original','repaired']:
                    selected=[r for r in rows if r['case']==case and r['arm']==arm and r['readout']==kind and r['radius']==radius]
                    summary=summarize(selected)
                    cells=' | '.join(countcell(summary,t) for t in THRESHOLDS) if summary['numeric_available'] else 'undefined | undefined | undefined | undefined'
                    detail.append(f"| {radius} | {case} | {arm} | {LABELS[kind]} | {summary['numeric_available']}/{summary['required']} | {summary['usable_available']}/{summary['required']} | {cells} |")
    (HERE/'COUNTS_AND_ORDERED_ERRORS.md').write_text('\n'.join(detail)+'\n')
    primary=[q for q in summaries if q['population']=='H_D' and q['array']=='all' and q['radius']==60 and q['readout']=='repaired']
    output=['# POINT peak-error threshold sensitivity','',
        '2026-09-13 · SCI-FRUIT v0.1 development · r0.1','',
        '## Program adherence and prior-work recovery','',
        'Follow the [charter](../../doc/scientific_contracts/README.md), [reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md), [reporting scope](SCOPE.md) and [saved readout result](../fruit_point_profiled_readout_2026-09-13/SCIENTIFIC_REPORT.md). This check is JSON arithmetic only. All prior results and the parked POINT candidate remain unchanged.','',
        '## How to read the counts','',
        'The primary tables use the repaired free-Gaussian readout at 60″. P is the pixelwise reference; C is starlet. Absolute peak error is 100 × (peak/declared peak − 1); H has declared peak 100 and D has 90. Gain-ratio error is 100 × [(H/D)/(100/90) − 1]. Matched and crossed pairings stay separate. The seeds are 20260911 and 20260912; crossed pairings reuse those two realizations.','',
        'Raw counts include all finite errors. Usable counts additionally require the unchanged saved peak-use judgment, for both members of a ratio. Both retain all 12 peak or six ratio cases in the denominator. Numerical flags are displayed independently, not used to remove cases. “No” numerical completion does not silently change a saved usability judgment. Counts use unrounded errors; displayed errors have four decimal places. These four cutoffs are descriptive, not selected acceptance requirements. Numerical flags show the displayed domain / other domain; ratio flags identify the H numerator and D denominator separately.','',
        counttable(primary),'',
        'The wider cutoffs expose different limitations: C matched ratios are already all inside 5%, but only four pairs are usable. All C crossed ratios are inside 10%; absolute peaks still show larger errors. One C/a1400/D_20260911 absolute error is +15.0004%, just above the literal 15% boundary. The 52″ value is inside 15%, accounting for its one extra raw success there; that peak is withheld in both tables. This sensitivity is reported rather than used to choose a domain or cutoff.','',
        '## Error distributions: every primary result retained','',
        '| Quantity | Arm | Median signed % | Median absolute % | Worst signed error % | Worst array and realization | Sorted error magnitudes % |',
        '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for q in primary:
        output.append(f"| {LABELS[q['quantity']]} | {q['arm']} | {sign(q['median_signed_pct'])} | {q['median_absolute_pct']:.4f} | {sign(q['worst']['signed_error_pct'])} | {q['worst']['array']} {q['worst']['case']} | "+', '.join(f"{v['magnitude_pct']:.4f}" for v in q['sorted_magnitudes'])+' |')
    for quantity,title in [('absolute_peak','Absolute peaks'),('matched_ratio','Matched-noise H/D ratios'),('crossed_ratio','Crossed-noise H/D ratios')]:
        output += ['',f'## {title}: signed errors at 60 arcsec','',errortable([q for q in rows if q['primary'] and q['radius']==60 and q['readout']=='repaired' and q['quantity']==quantity])]
    output += ['','## 52-arcsec sensitivity, kept separate','',
        'The next counts use 52″ errors on every corresponding case. The usable column conditions on the same saved 60″ judgment; it is not independent 52″ availability and cannot be substituted to improve the primary result.', '',
        counttable([q for q in summaries if q['population']=='H_D' and q['array']=='all' and q['radius']==52 and q['readout']=='repaired']), '',
        'All 52″ signed errors, original free fits, fixed-original-geometry coefficients, truth-template coefficients and safeguards remain in [every error and flag](EVERY_ERROR.md). [Counts and ordered errors](COUNTS_AND_ORDERED_ERRORS.md) retain the complete sorted lists with case identities and per-array summaries. The truth-template coefficient remains a different measurement from the free-profile peak; no diagnostic coefficient is assigned operational usability.','',
        '## Limits and preservation','',
        'These summaries describe small, dependent saved populations. No new cutoff is recommended. They do not qualify amplitude bias, uncertainty, or production performance. Source presence, centroid use, shape/support warnings, peak-score and domain checks remain exactly as saved in each contributing measurement. Null safeguards retain undefined peak percentage error. The one H/D independent numerical-completion miss (C/H_20260912/a2000 at 60″) remains flagged, including both ratios containing that numerator. The earlier runtime failures and incomplete numerical qualification remain.','',
        'No map or timestream was read; no fitter, optimizer, PTC, feedback or new observation was run. [Input bindings](INPUT_BINDINGS.json), [all machine-readable errors](ERROR_RECORDS.json), [summary](SUMMARY.json) and [verification](VERIFICATION.json) preserve the arithmetic and provenance. This reporting packet leaves all previous packets untouched.']
    (HERE/'REPORT.md').write_text('\n'.join(output)+'\n')
    # Independent arithmetic/count identities, not numerical experiments.
    assert len(comp)==84 and len(ratios)==192 and len(measure)==42
    assert len(rows)==456 and len({r['id'] for r in rows})==456
    for summary in summaries:
        counts=summary['thresholds']
        assert [q['raw'] for q in counts]==sorted(q['raw'] for q in counts)
        for q in counts:
            assert 0<=q['raw']<=summary['numeric_available']<=summary['required']
            if q['usable'] is not None:assert q['usable']<=min(q['raw'],summary['usable_available'])
        assert len(summary['sorted_magnitudes'])==summary['numeric_available']
    for r in rows:
        if r['quantity']=='absolute_peak' and r['primary']:
            target=100 if r['case'].startswith('H_') else 90
            assert math.isclose(r['value']/target-1,r['error_fraction'],rel_tol=1e-12,abs_tol=2e-15)
    for f in start['files']:assert digest(Path(f['path']))==f['sha256']
    for w in states:assert subprocess.check_output(['git','-C',w['path'],'status','--short'],text=True)==w['status']
    write(HERE/'VERIFICATION.json',dict(source_fit_records=84,source_ratios=192,source_measurements=42,
        output_error_records=456,original_source_records_retained=True,ratio_identities_verified=192,
        raw_primary_absolute_identities_verified=192,summary_groups=len(summaries),threshold_monotonicity_verified=True,
        source_input_hashes_unchanged=True,protected_archive_statuses_unchanged=True,archive_contents_read=False,
        map_reads=0,PTC_calls=0,feedback_fits=0,source_fits=0,optimization_calls=0,
        new_observations=0,registered_threshold_changes=0,cutoff_selected=False))
    print(json.dumps(primary,indent=2))

if __name__=='__main__':main()
