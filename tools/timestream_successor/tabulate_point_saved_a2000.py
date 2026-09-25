"""Tabulate only an existing bounded diagnosis; no parent inference or processing."""
import argparse,hashlib,json,time
from pathlib import Path


def tables(out):
    start=time.perf_counter();inputs={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('RESULTS.json','PIXEL_EVIDENCE.json')}
    r=json.loads((out/'RESULTS.json').read_text());e=json.loads((out/'PIXEL_EVIDENCE.json').read_text())
    lookup={(q['policy'],q['branch'],q['stage'],q['pixel_id']):q for q in e}
    sel={(q['policy'],q['branch'],q['map_stage']):q for q in r['selection']}
    def pixels(pol,b,k):return set(sel[pol,b,k]['actual_selected']['pixel_ids'])
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for h in headers)+'|']+['| '+' | '.join(f'{v:.8g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in rows])+'\n'
    overlaps=[]
    for pol in ('BASE-POINT','CORR','DIAG'):
      for k in range(1,11):
        ref=pixels(pol,'REF',k);seed=pixels(pol,'D-seed-shoulder',k)^ref
        for b in ('C-core-plus','C-core-minus'):
          mismatch=pixels(pol,b,k)^ref
          overlaps.append(dict(policy=pol,branch=b,candidate=k,affected_map=k+1 if k<10 else None,seed_difference=sorted(seed),compact_difference=sorted(mismatch),intersection=sorted(seed&mismatch)))
    terminal=[]
    for q in r['measurements']:
        if q['map_stage']!=10:continue
        m=q['previous_own_measurement_change'];a=q['previous_own_map_change'];b=q['previous_own_candidate_change']
        terminal.append([q['policy'],q['branch'],a['magnitude']['norm'],a['magnitude']['maximum_absolute'],b['magnitude']['norm'],b['magnitude']['maximum_absolute'],m['peak'],m['background'],m['apertures'][2]['source_relative']])
    lines=['# Practical measurements from cached a2000 products','', 'Brightness: mJy/nominal beam. Centroid/FWHM: arcsec. Apertures: mJy/nominal beam arcsec², not calibrated total flux. A2 radius = 18.868344095 arcsec, frozen center; all three apertures and physical ellipse components remain in RESULTS.json. No new measurements were fitted. Signed values retain decrement meaning.','', '## Own T9-to-T10 changes','',table(['Policy','Branch','Map norm','Map max abs','Candidate norm','Candidate max abs','Peak','Constant','A2 source-relative'],terminal),'The branch changes above use their own preceding stage; the percentage fields inherited in previous_own_measurement_change are normalized to that preceding measurement, not a different REF branch.','', '## T10 seed minus own REF','']
    rows=[]
    for q in r['seed']:
        if q['map_stage']!=10:continue
        m=q['measurement_difference'];rows.append([q['policy'],m['peak'],m['east'],m['north'],m['FWHM_major'],m['FWHM_minor'],m['background'],m['fixed_template'],m['apertures'][2]['raw'],m['apertures'][2]['background'],m['apertures'][2]['source_relative']])
    lines.append(table(['Policy','Peak','East','North','FWHM major','FWHM minor','Constant','Fixed template','A2 raw','A2 constant term','A2 source-relative'],rows))
    lines+=['','## T10 response versus saved ideal addition','', 'Rows report the actual branch-minus-own-REF increment, the cached ideal-addition increment, and their signed discrepancy. No ratios to tiny centroid/shape increments are displayed. The percentages shown for peak/template/aperture divide by the explicitly listed signed ideal increment.','']
    rows=[];raw=[]
    for q in r['response']:
        if q['map_stage']!=10:continue
        for a in q['ideal']['comparisons']:
            if a['name'].startswith('ellipse'):continue
            pct=100*a['absolute_discrepancy']/a['ideal_increment'] if a['name'] in ('peak','fixed_template','aperture_1','aperture_1p5','aperture_2') and abs(a['ideal_increment'])>1e-8 else '—'
            rows.append([q['policy'],q['branch'],a['name'],a['actual_increment'],a['ideal_increment'],a['absolute_discrepancy'],pct])
        raw.append(dict(policy=q['policy'],branch=q['branch'],data=q['ideal']['raw_aperture_and_background_increments']))
    lines.append(table(['Policy','Branch','Measure','Actual increment','Ideal increment','Actual − ideal','Discrepancy / signed ideal %'],rows))
    lines+=['','Raw aperture and fitted-constant pieces for all nine ideal comparisons are copied without reinterpretation in INTERPRETIVE_TABLES.json (ideal_raw_apertures). Guard/constrained flags remain in each cached measurement in RESULTS.json.']
    (out/'MEASUREMENT_TABLES.md').write_text('\n'.join(lines)+'\n')
    # Compact ledger includes every terminal CORR seed/REF membership difference,
    # plus all three T1 CORR-only versus DIAG seed selections, without cherry-picking.
    terminal_ids=pixels('CORR','D-seed-shoulder',10)^pixels('CORR','REF',10)
    first_ids=pixels('CORR','D-seed-shoulder',1)^pixels('DIAG','D-seed-shoulder',1)
    chosen=sorted(terminal_ids|first_ids);rows=[]
    for idx in chosen:
        a=lookup['CORR','D-seed-shoulder',10,idx];b=lookup['CORR','REF',10,idx]
        rows.append([idx,a['east'],a['north'],a['east_from_original_source'],a['north_from_original_source'],a['east_from_seed'],a['north_from_seed'],a['BRIGHTNESS'],b['BRIGHTNESS'],a['CORR_margin'],b['CORR_margin'],a['DIAG_margin'],a['NEFF_W']])
    lines=['# Short implicated-pixel ledger','', 'Pixel ID = row × 41 + column (zero based). East/north are saved J2000 tangent offsets in arcsec. Source-relative coordinates use the frozen measurement/aperture center (4.6970338404, −5.2035241123), distinct from the artificial seed center (9.3214385638, −5.2001178518). All 100 implicated pixels, every branch/policy/stage, all saved fields including negative EVIDENCE and availability, are retained in PIXEL_EVIDENCE.json (16,500 rows).','', 'Displayed set is the complete union of nine terminal CORR seed-only pixels and the three CORR-versus-DIAG seed m1 membership differences (one overlaps). Values below are T10; null scores mean unavailable, never recomputed.','',table(['ID','East','North','ΔE source','ΔN source','ΔE seed','ΔN seed','b seed','b REF','CORR margin seed','CORR margin REF','DIAG margin on seed state','N_eff seed'],rows),'', '## Complete m1-to-m10 selection strings for this set','', '1 selected, 0 excluded, ordered m1…m10. These candidates affect only the next map; m10 remains unapplied.','']
    rows=[]
    for idx in chosen:
        rows.append([idx]+[''.join(str(int(idx in pixels(pol,b,k))) for k in range(1,11)) for pol,b in [('CORR','D-seed-shoulder'),('CORR','REF'),('DIAG','D-seed-shoulder'),('DIAG','REF'),('CORR','C-core-plus'),('CORR','C-core-minus')]])
    lines.append(table(['ID','CORR seed','CORR REF','DIAG seed','DIAG REF','CORR plus','CORR minus'],rows))
    (out/'PIXEL_LEDGER.md').write_text('\n'.join(lines)+'\n')
    payload=dict(input_sha256=inputs,selection_overlaps=overlaps,terminal_drift=terminal,short_pixel_ids=chosen,ideal_raw_apertures=raw,seconds=time.perf_counter()-start)
    (out/'INTERPRETIVE_TABLES.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seconds=payload['seconds'],short_pixels=len(chosen),complete_pixel_records=len(e))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);tables(parser.parse_args().output)
