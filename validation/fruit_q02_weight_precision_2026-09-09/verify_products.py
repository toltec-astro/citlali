#!/usr/bin/env python3
"""Read-only identity and population checks for a completed Q02 diagnostic."""
from pathlib import Path
import json
import sys
import numpy as np
import run_weight_precision as m


def verify(attempt):
    root = m.HERE/attempt
    manifest = json.loads((root/'PRODUCT_MANIFEST.json').read_text())
    assert m.digest(root/'PRODUCT_MANIFEST.json') == (root/'PRODUCT_MANIFEST.json.sha256').read_text().split()[0]
    for item in manifest['products']:
        p = root/item['path']
        assert p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size == item['bytes'] and m.digest(p) == item['sha256'], item['path']
    start = json.loads((root/'RUN_START.json').read_text())
    for item in start['sources']:
        assert m.digest(m.REPO/item['path']) == item['sha256'], item['path']
    end = json.loads((root/'COMPLETION.json').read_text())
    assert end['status'] == 'complete'
    assert end['elapsed_seconds'] <= start['settings']['wall_time_limit_seconds']
    assert end['peak_aggregate_process_bytes'] <= start['settings']['aggregate_memory_limit_bytes']
    assert manifest['payload_bytes'] <= start['settings']['output_limit_bytes']
    for item in start['settings']['inputs']:
        assert Path(item['path']).stat().st_size == item['bytes']
        assert m.digest(item['path']) == item['sha256']
    summary = json.loads((root/'SUMMARY.json').read_text())
    assert len(summary['windows']) == 18
    assert len(summary['lags']) == 84
    counts = []
    for obs in [123424,152389]:
        with np.load(root/f'{obs}_chunk_diagnostics.npz',allow_pickle=False) as chunks:
            for k in [1,2,4]:
                with np.load(root/f'{obs}_K{k}_groups.npz',allow_pickle=False) as d, np.load(root/f'{obs}_K{k}_training_bins.npz',allow_pickle=False) as bins:
                    assert d['e'].shape[0] == 12//k
                    np.testing.assert_array_equal(d['e'],chunks['evaluation_counts'].reshape(12//k,k,-1).sum(axis=1))
                    np.testing.assert_array_equal(d['central_e'],chunks['central_evaluation_counts'].reshape(12//k,k,-1).sum(axis=1))
                    np.testing.assert_array_equal(d['n'],chunks['n'].reshape(12//k,k,-1).sum(axis=1))
                    np.testing.assert_array_equal(d['near_n']+d['far_n'],d['n'])
                    for j in range(12//k):
                        for h in range(6):
                            nbin = bins[f'window{j}_tau{h}_counts']
                            np.testing.assert_array_equal(nbin.sum(axis=0),d['n'][j])
                            np.testing.assert_array_equal((nbin>0).sum(axis=0),d['occupied_bins'][j,:,h])
                    raw_available = d['score_ok'] & (d['e']>0)
                    np.testing.assert_array_equal(np.isfinite(d['p_raw']).all(axis=2),raw_available)
                    for a,array in enumerate(m.ARRAYS):
                        columns = d['array'] == a
                        e, w = d['e'][:,columns],d['w'][:,columns]
                        req = e>0
                        s = next(s for s in summary['windows'] if (s['obsnum'],s['K'],s['array']) == (obs,k,array))
                        assert s['required_groups'] == req.sum()
                        assert s['evaluation_occurrences'] == e.sum()
                        assert s['weight_unavailable']['groups'] == np.sum(req & ~np.isfinite(w))
                        g,_,ws,ps = m.joint_state(w,e,d['score_ok'][:,columns])
                        np.testing.assert_allclose(d['gamma'][:,columns],g,rtol=1e-12,atol=1e-14,equal_nan=True)
                        assert (s['weight_state'],s['normalized_uncertainty_state']) == (ws,ps)
                        p = d['p_normalized'][:,columns]
                        if ps == 'available':
                            assert np.isfinite(p[req]).all()
                            np.testing.assert_allclose(np.sum(e[req]*g[req])/e.sum(),1.,rtol=1e-12,atol=1e-14)
                        else:
                            assert np.isnan(p).all()
                        for h in range(6):
                            assert m.clean(m.series_summary(p[:,:,h],e)) == s['precision'][h]['normalized']
                        counts.append(dict(obsnum=obs,array=array,K=k,groups=int(req.sum()),
                                           evaluation_occurrences=int(e.sum()),normalized_precision_available=ps=='available'))
        with np.load(root/f'{obs}_lag_diagnostics.npz',allow_pickle=False) as lags:
            assert lags['correlations'].shape[1:] == (7,2,2)
            assert lags['pair_counts'].shape[1:] == (7,2)
            for h in range(7):
                for category in range(2):
                    no_pairs = lags['pair_counts'][:,h,category] == 0
                    assert np.isnan(lags['correlations'][no_pairs,h,category]).all()
    return dict(status='PASS',attempt=attempt,products_verified=len(manifest['products']),
                pre_signal_source_digests_verified=len(start['sources']),
                post_run_discovery_input_hashes='both unchanged',
                reserved_129081='not opened',groups=counts,
                checks='Exact population, bin counts, availability, joint normalization, summary denominators, lag missingness and resource limits')


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]),indent=2))
