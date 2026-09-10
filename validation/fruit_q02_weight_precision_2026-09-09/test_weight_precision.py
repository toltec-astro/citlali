#!/usr/bin/env python3
"""Deterministic verification required by the approved Q02 r0.5 protocol."""
import json
from pathlib import Path
import numpy as np


def verify(m):
    checks = []
    def passed(name):
        checks.append(dict(name=name, status='PASS'))
    def equal(x, y, **kw):
        np.testing.assert_allclose(x, y, rtol=kw.get('rtol', 1e-12), atol=kw.get('atol', 1e-14), equal_nan=True)

    scans = np.array([[0,288],[305,593],[610,898],[915,1203],[1220,1508],
                      [1525,1813],[1830,2118],[2135,2423],[2440,2728],
                      [2745,3033],[3050,3338],[3339,3627]])
    expected = np.array([0,289,594,899,1204,1509,1814,2119,2424,2729,3034,3339,3628])
    np.testing.assert_array_equal(m.recover_edges(scans), expected)
    tr, ev, chunks, early = m.split_rows(expected)
    assert len(np.intersect1d(tr, ev)) == 0
    np.testing.assert_array_equal(np.sort(np.r_[tr, ev]), np.arange(3628))
    assert len(tr) == 1808 and len(ev) == 1820
    assert np.all(np.diff(tr)[chunks[:-1] != chunks[1:]] > 1)
    assert np.sum(early[chunks == 0]) == 72
    passed('Recovered labels; all rows retained; disjoint first-half/evaluation split')

    t = np.array([0., .07, .21, .48, 2.9, 3.04, 5.8, 6.0])
    x = np.array([[1.,1.4,0.,0.],[3.,3.8,0.,0.],[-2.,-1.9,0.,0.],[.5,.2,0.,0.],
                  [0.,0.,2.,1.],[0.,0.,-1.,-.7],[0.,0.,.8,2.6],[0.,0.,4.,3.]])
    valid = np.array([[True,True,False,False]]*4 + [[False,False,True,True]]*4)
    n, mu, v = m.moments(x, valid)
    w, c, ok = m.inverse_and_score(x, valid, (n,mu,v))
    e = np.array([2,5,3,7])
    gamma, wbar, state, unc = m.joint_state(w, e, ok)
    assert state == unc == 'available'
    equal(np.sum(e*gamma)/e.sum(), 1.)
    equal(gamma, m.normalize(w*7.3, e)[0])
    n2, mu2, v2 = m.moments(3.2*x, valid)
    w2, _, _ = m.inverse_and_score(3.2*x, valid, (n2,mu2,v2))
    equal(gamma, m.normalize(w2,e)[0])
    # Distinct time-window normalizations would not reproduce this whole-array identity.
    assert not np.allclose(gamma[:2], m.normalize(w[:2],e[:2])[0])
    passed('Whole-array occurrence normalization; common weight/input-scale cancellation')

    cbar = c @ (e/e.sum())
    h = (c - cbar[:,None]*gamma[None,:])/wbar
    equal(np.sum(e[None,:]*h,axis=1)/e.sum(), np.zeros(len(t)))
    perturb = np.array([.4,-.9,.2,1.1,-.3,.6,-.8,.7])
    expected_derivative = h.T @ perturb
    def weighted_gamma(eps):
        masses = valid*(1+eps*perturb[:,None])
        den = masses.sum(axis=0)
        center = np.sum(masses*x,axis=0)/den
        scatter = np.sum(masses*(x-center)**2,axis=0)/den
        weights = 1/scatter
        return weights / np.sum(e*weights)*e.sum()
    derivative_errors = []
    for step in [1e-4,1e-5,1e-6]:
        derivative = (weighted_gamma(step)-weighted_gamma(-step))/(2*step)
        derivative_errors.append(dict(step=step,max_absolute_error=float(np.max(np.abs(derivative-expected_derivative)))))
    equal(derivative, expected_derivative, rtol=1e-6, atol=1e-8)
    passed('Exact joint first-order derivative; all three prescribed finite-difference steps')

    for tau in [.125,.25,.5,1.,2.,4.]:
        kernel = np.maximum(1-np.abs(t[:,None]-t[None,:])/tau,0.)
        equal(kernel,kernel.T)
        assert np.linalg.eigvalsh(kernel).min() > -1e-12
        direct = np.einsum('ij,ij->j',h,kernel@h)
        equal(m.kernel_quad(h,m.kernel_plan(t,tau),batch=1), direct)
        equal(m.kernel_quad(h,m.kernel_plan(t,tau),batch=3), direct)
        equal(m.kernel_quad(h,m.kernel_plan(t+1000,tau),batch=2), direct)
    zero_between = np.maximum(1-np.abs(t[:,None]-t[None,:])/.5,0.)
    assert zero_between[3,4] == 0
    compact = np.arange(len(t))*.07
    assert not np.allclose(m.kernel_quad(h,m.kernel_plan(compact,.5)),m.kernel_quad(h,m.kernel_plan(t,.5)))
    passed('Irregular-time positive kernel; gaps; direct versus bounded batches; time translation')

    kernel = np.maximum(1-np.abs(t[:,None]-t[None,:])/1.,0.)
    cov = c.T @ kernel @ c
    jac = (np.eye(4)-gamma[:,None]*(e/e.sum())[None,:])/wbar
    joint = np.diag(jac @ cov @ jac.T)
    independent = (jac*jac) @ np.diag(cov)
    equal(joint,m.kernel_quad(h,m.kernel_plan(t,1.)))
    assert np.max(np.abs(joint-independent)) > 1e-5
    passed('Cross-detector dependence changes normalized uncertainty from independent-error propagation')

    bad = w.copy(); bad[2] = np.nan
    assert m.normalize(bad,e)[2] == 'unavailable_required_weight'
    assert np.isnan(m.normalize(bad,e)[0]).all()
    counts = e.copy(); counts[2] = 0
    assert m.normalize(bad,counts)[2] == 'available'
    assert m.normalize(w,np.zeros(4))[2] == 'unavailable_empty_population'
    invalid_score = ok.copy(); invalid_score[1] = False
    assert m.joint_state(w,e,invalid_score)[3] == 'unresolved_required_score'
    assert np.isfinite(m.joint_state(w,e,invalid_score)[0]).all()
    passed('Missing required weight or score propagates to whole array; empty and nonrequired populations')

    xx = np.array([[1.,1.,1.,2.,np.nan],[2.,4.,1.,2.,np.inf],[3.,5.,1.,2.,3.],[4.,8.,1.,2.,4.]])
    vv = np.array([[False,True,True,True,True],[False,False,True,True,True],
                   [False,False,False,True,True],[False,False,False,True,True]])
    nn, mm, ss = m.moments(xx,vv)
    ww, cc, available = m.inverse_and_score(xx,vv & np.isfinite(xx),(nn,mm,ss))
    np.testing.assert_array_equal(nn,[0,1,2,4,2])
    assert np.isnan(ww[:4]).all() and np.isfinite(ww[4])
    assert not available.any()
    assert np.all(cc == 0)
    two = np.array([[1.],[7.]])
    nt,mt,vt = m.moments(two,np.ones_like(two,bool))
    wt,ct,ot = m.inverse_and_score(two,np.ones_like(two,bool),(nt,mt,vt))
    assert np.isfinite(wt).all() and not ot.any() and np.all(ct == 0)
    passed('Zero/one/two rows, constant signal, nonfinite values and degenerate uncertainty')

    bt = np.array([0.,.249,.25,.499,.5,2.])
    bm = np.array([[1,0],[1,1],[0,1],[1,0],[1,1],[0,1]],bool)
    ids, count = m.bin_counts(bt,bm,.25)
    np.testing.assert_array_equal(ids,[0,1,2,8])
    np.testing.assert_array_equal(count,[[2,1],[1,1],[1,1],[0,1]])
    passed('Original-time half-open bin membership and exact counts including gaps')

    lt = np.array([0.,.1,.3,2.,2.2,2.5])
    lc = np.array([0,0,0,1,1,1])
    le = [0,.125,.25,.5,1,2,4,8]
    mats = m.lag_matrices(lt,lc,le)
    lv = np.array([[1,1],[1,0],[1,1],[1,1],[0,1],[1,1]],bool)
    ls = np.array([[.1,1.],[.2,-.6],[-.3,.7],[.4,.8],[.8,-1.],[1.,.2]])
    for ib,(lo,hi) in enumerate(zip(le[:-1],le[1:])):
        for category in range(2):
            n_pair,corr = m.pair_statistics(ls,lv,mats[ib][category])
            for d in range(2):
                pairs = [(i,j) for i in range(6) for j in range(i+1,6)
                         if lv[i,d] and lv[j,d] and lo <= lt[j]-lt[i] < hi
                         and ((lc[i] == lc[j]) == (category == 0))]
                assert n_pair[d] == len(pairs)
                if pairs:
                    left = np.array([ls[i,d] for i,j in pairs]);right = np.array([ls[j,d] for i,j in pairs])
                    equal(corr[d],np.sum(left*right)/np.sqrt(np.sum(left*left)*np.sum(right*right)))
                else:
                    assert np.isnan(corr[d])
    passed('Within/cross-chunk lag pairs against independent explicit pair enumeration')

    class Variable:
        def __init__(self): self.reads=[]
        def __getitem__(self, key):
            self.reads.append(key)
            return np.ma.array([[1.,np.inf,3.],[4.,5.,np.nan]], mask=[[0,0,1],[0,0,0]])
    fake = Variable()
    f, masked, nonfinite = m.signal_finiteness(fake,slice(10,12))
    np.testing.assert_array_equal(f,[[1,0,0],[1,1,0]])
    assert all(a.dtype == bool for a in (f,masked,nonfinite))
    assert fake.reads == [(slice(10,12),slice(None))]
    passed('Evaluation reader returns finite/mask predicates only for specified rows')
    assert m.quantiles([3,1,2],[1,8,1]) == [1.,1.,2.]
    summ = m.series_summary(np.array([.1,.3,np.nan]),np.array([2,3,5]))
    assert summ['below_goal_fraction_of_all_occurrences'] == .2
    assert summ['unknown_fraction_of_all_occurrences'] == .5
    assert summ['below_goal_fraction_of_available_occurrences'] == .4
    passed('Declared quantiles and all-population/available-population denominators')
    return dict(status='PASS', checks=checks, finite_difference_errors=derivative_errors,
                cross_detector_joint_variance=joint.tolist(), independent_error_variance=independent.tolist(),
                statistical_coverage_test=False, stochastic_trials=0, external_signal_access=False)


if __name__ == '__main__':
    import run_weight_precision as m
    result = verify(m)
    print(json.dumps(result,indent=2))
