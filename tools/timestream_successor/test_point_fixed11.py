import copy,json
import numpy as np
import pytest
import point_fixed11_selection as s
import point_fixed11_experiment as e
from point_processing import rtc_experiment,experiment_consume

def test_selection_requires_resolved_target_and_independent_time_evidence():
    f=np.fft.rfftfreq(488,.008192062377929688);p=np.ones_like(f);p[np.argmin(abs(f-s.CENTER))]=20
    w=[dict(start=0,value=20),dict(start=488,value=20)]
    assert s.disposition(p,f,w,488)[0]=='selected'
    assert s.disposition(p,f,w[:1],488)[0]=='uncertain'
    assert s.disposition(p,f,[w[0],dict(start=244,value=20)],488)[0]=='uncertain'
    assert s.disposition(p*np.nan,f,w,488)[0]=='uncertain'
    q=np.ones_like(p);q[np.argmin(abs(f-12))]=20
    assert s.disposition(q,f,w,488)[0]=='uncertain'
    assert s.disposition(np.ones_like(p),f,[dict(start=0,value=2)],488)[0]=='unselected'

def test_frozen_set_configuration_and_unselected_identity():
    m=json.loads((e.RECORD/'APPLICABILITY.json').read_text());d=json.loads((e.RECORD/'DESIGN.json').read_text())
    assert s.coeff_sha(d['finite_notch'])==s.NOTCH_SHA
    for n in m['networks']:
        cfg=json.loads((s.PRIOR/f"A-REF-rep1/inputs/network{n['network']}-input.json").read_text())
        b=e.configure(copy.deepcopy(cfg),m,d,'B');c=e.configure(copy.deepcopy(cfg),m,d,'C')
        assert all(x['filter']=='lowpass' for x in b['detectors'])
        assert sum(x['filter']=='w1-t3' for x in c['detectors'])==n['counts']['selected']
        for i,x in enumerate(c['detectors']):
            if n['detectors'][i]['disposition']!='selected':assert x==cfg['detectors'][i]
        cfg['tune']['sha256']='wrong'
        with pytest.raises(ValueError):e.configure(cfg,m,d,'C')

def test_new_scope_limits_and_duplicate_failure(tmp_path):
    out=tmp_path/'run';(out/'freeze').mkdir(parents=True)
    (out/'freeze/prospective.json').write_text(json.dumps(dict(campaign_scope='rtc-fixed11-v1')))
    (out/'freeze/budget.json').write_text(json.dumps(dict(campaign_root=str(tmp_path))))
    assert rtc_experiment(out)
    for k in range(45):experiment_consume(out,'feedback',dict(cycle=k))
    with pytest.raises(ValueError):experiment_consume(out,'feedback',dict(cycle=45))
    e.consume(tmp_path,'rtc_only',dict(id=1));e.consume(tmp_path,'rtc_only',dict(id=2))
    with pytest.raises(ValueError):e.consume(tmp_path,'rtc_only',dict(id=3))
    e.consume(tmp_path,'trajectory',dict(id=1))
    with pytest.raises(ValueError):e.consume(tmp_path,'trajectory',dict(id=1))
    (tmp_path/'DEFECT-REPLACEMENT.json').write_text(json.dumps(dict(maximum_replacements=1,implementation_defect='fixture',corrected_source='recorded')))
    for k in range(45,50):experiment_consume(out,'feedback',dict(cycle=k))
    with pytest.raises(ValueError):experiment_consume(out,'feedback',dict(cycle=50))
    assert e.ORDER==[('B',1),('C',1),('B',2),('C',2),('B',3),('C',3)]
