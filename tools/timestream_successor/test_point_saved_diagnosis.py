"""Focused deterministic bookkeeping controls. No random fields or scientific fits."""
import ast
from pathlib import Path
import numpy as np
import pytest
from point_saved_diagnosis import (timing,paired,difference,shape_components,split_candidate,
    score_audit,signed_response,signed_even,percent,aperture,memberships)
from diagnose_point_saved_a2000 import Cached


def test_signed_reference_even_control_and_zero_seed_target():
    shape=np.array([[1.,-2.],[0.,0.]]);valid=np.ones((2,2),bool)
    zero=shape_components(np.zeros((2,2)),valid,shape)[0]
    assert zero['coefficient']==0 and zero['orthogonal']['norm']==0
    assert signed_response(-shape,-shape,valid)==dict(g=1.,eta=0.,rho=0.,pixels=4)
    e,i,n=signed_even(shape,-shape,shape,-shape+.1,valid)
    assert np.all(e==0) and np.allclose(i,.1) and np.allclose(n,-.1)


def test_shape_domain_and_total_norm_are_distinct():
    shape=np.array([[1.,0.],[np.nan,np.nan]]);v=np.array([[2.,3.],[100.,200.]])
    r,p,o,d=shape_components(v,np.ones((2,2),bool),shape)
    assert r['coefficient']==2 and r['total']['norm']==np.sqrt(13)
    assert r['orthogonal']['norm']==3 and np.isnan(p[1]).all()


def test_unavailable_never_zero_filled_and_aperture_requires_support():
    a=np.array([[0.,np.nan],[2.,3.]]);av=np.isfinite(a);b=np.ones((2,2));bv=np.ones((2,2),bool)
    d,v,s=difference(a,av,b,bv)
    assert d[0,0]==-1 and np.isnan(d[0,1]) and s['lost']==1
    assert aperture(d,v,[bv])[0]['value'] is None
    with pytest.raises(ValueError):difference(a,bv,b,bv)


def test_disjoint_partition_and_fitted_constant_bookkeeping():
    valid=np.array([True,True,True,True,False]);ms=np.array([1,1,0,0,0],bool);mr=np.array([1,0,1,0,0],bool)
    bs=np.array([8.,-2.,5.,1.,np.nan]);br=np.array([6.,3.,-4.,2.,np.nan])
    def c(m,b):return dict(VALID=valid,ADMITTED=m,CANDIDATE=np.where(valid,np.where(m,b,0.),np.nan))
    q=split_candidate(c(ms,bs),c(mr,br),dict(BRIGHTNESS=bs),dict(BRIGHTNESS=br),2.,-1.,1)
    assert np.array_equal(q['delta'][:4],[2.,-2.,4.,0.]) and q['error']==0
    assert np.array_equal(sum(q['background'].values())[:4],[-3.,-2.,-1.,0.])
    assert np.allclose((sum(q['measured'].values())+sum(q['background'].values()))[:4],q['delta'][:4])
    with pytest.raises(ValueError,match='m0'):split_candidate(c(ms,bs),c(mr,br),{}, {},2.,-1.,0)


def test_matching_and_m9_T10_m10_order():
    r=dict(array='a2000',policy='CORR',cycle_index=10,branch='REF');b=r|dict(branch='D-seed-shoulder')
    paired(b,r)
    with pytest.raises(ValueError,match='policy'):paired(b,r|dict(policy='DIAG'))
    with pytest.raises(ValueError,match='cycle_index'):paired(b,r|dict(cycle_index=9))
    assert timing(10)['applied_candidate']==9 and timing(10)['next_map'] is None
    assert timing(0)['applied_candidate'] is None and timing(1)['applied_candidate']==0
    with pytest.raises(ValueError):timing(11)


def test_saved_score_audit_preserves_negative_evidence():
    a=dict(CORR_VALID=np.array([1,1,0],bool),EVIDENCE=np.array([-6.,8.,np.nan]),CORR_VAR=np.array([4.,16.,np.nan]),CORR_SCORE=np.array([-3.,2.,np.nan]))
    assert score_audit(a,'CORR')['negative_signed_evidence']==1
    a['CORR_SCORE'][0]=3
    with pytest.raises(ValueError):score_audit(a,'CORR')
    assert percent(1.,0.) is None


def test_membership_transitions_use_saved_records():
    prev=dict(BASE_POINT=np.array([1,0,0],bool),ADDED=np.array([0,1,0],bool),ADMITTED=np.array([1,1,0],bool))
    now=dict(BASE_POINT=np.array([0,1,0],bool),ADDED=np.array([0,0,0],bool),PROPOSED=np.array([0,0,1],bool),ADMITTED=np.array([0,1,0],bool))
    r=memberships(now,prev)
    assert np.array_equal(r['normal_admission_supplement_to_baseline'],[0,1,0]) and np.array_equal(r['rejected_connectivity'],[0,0,1]) and r['dropped'].sum()==1


def test_no_unbound_cache_fallback_or_scientific_execution_import():
    with pytest.raises(ValueError,match='unbound'):Cached(dict(inputs={})).json('/missing/cached.fit')
    root=Path(__file__).parent
    for name in ('point_saved_diagnosis.py','diagnose_point_saved_a2000.py'):
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                mod=node.module or ''
                if mod.startswith(('point_','run_point_','report_point_','inspect_point_')):assert mod=='point_saved_diagnosis'
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('fit','least_squares','scores','svd','infer','select','learn')


def test_imposed_seed_turnover_uses_actual_previous_candidate():
    ad=dict(BASE_POINT=np.array([1,0,0],bool),ADDED=np.array([0,0,0],bool),PROPOSED=np.array([0,0,0],bool),ADMITTED=np.array([1,0,0],bool))
    r=memberships(ad,ad,ad['ADMITTED'],np.array([1,1,1],bool))
    assert r['dropped'].tolist()==[False,True,True]
    assert r['shared_selected'].tolist()==[True,False,False]


def test_own_candidate_drift_preserves_support_and_nonzero_shared_drift():
    from diagnose_point_saved_a2000 import own_change
    a=dict(candidate=dict(CANDIDATE=np.array([4.,5.,np.nan]),VALID=np.array([1,1,0],bool)))
    b=dict(candidate=dict(CANDIDATE=np.array([1.,1.,8.]),VALID=np.array([1,0,1],bool)))
    r=own_change(a,b,'candidate')
    assert r['support']==dict(common=1,lost=1,gained=1)
    assert r['magnitude']['norm']==3 and r['magnitude']['maximum_absolute']==3
