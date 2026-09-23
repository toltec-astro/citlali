"""Bounded continuation controls, lineage and failure behavior."""
import tempfile
import json
import yaml
import unittest
from pathlib import Path
import numpy as np
from run_point_fruit_ten import continue_array, make_request, original_parents
from point_fruit_trajectory import candidate_guard, selection_difference, verified, bound, manifest, verify_manifest, difference
from point_admission import admit


class TenCycleControls(unittest.TestCase):
    def test_exact_nine_cycles_seed_one_and_stop_ten(self):
        applied = []
        def execute(k, prior):
            self.assertEqual(prior['cycle_index'], k-1)
            applied.append(prior['candidate'])
            return dict(cycle_index=k, ready=True, candidate=f'm{k}')
        result = continue_array(dict(cycle_index=1, ready=True, candidate='m1'), execute)
        self.assertEqual([r['cycle_index'] for r in result], list(range(2, 11)))
        self.assertEqual(applied, [f'm{k}' for k in range(1, 10)])
        self.assertNotIn('m10', applied)
        with self.assertRaises(ValueError): continue_array({}, execute, last=11)
        with self.assertRaises(ValueError): continue_array({}, execute, first=1)

    def test_failure_preserved_and_unaffected_array_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def execute(k, prior):
                (root/f'T{k}').write_text(f'original y minus {prior["candidate"]}')
                return dict(cycle_index=k, candidate=f'm{k}', ready=k!=4, stop_reason='boundary contact')
            result = continue_array(dict(ready=True, candidate='m1'), execute)
            self.assertEqual([s['cycle_index'] for s in result], [2,3,4])
            self.assertEqual((root/'T2').read_text(), 'original y minus m1')
            self.assertFalse((root/'T5').exists())
            other = continue_array(dict(ready=True), lambda k,p:dict(cycle_index=k,ready=True))
            self.assertEqual(other[-1]['cycle_index'], 10)

    def test_requests_reuse_original_parent_and_bound_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for n in ['original','plan','owner','m1','m2','state1','state2']:(r/n).write_text(n)
            parents=dict(map_request=bound(r/'original'),parent_plan=bound(r/'plan'))
            for k in [2,3]:
                request=make_request('a1100',k,r/f'm{k-1}',r/f'state{k-1}',parents,r/'owner')
                self.assertEqual(request['map_request'],parents['map_request'])
                self.assertEqual(request['parent_plan'],parents['parent_plan'])
                self.assertEqual(request['candidate'],bound(r/f'm{k-1}'))
            with self.assertRaises(ValueError):make_request('a1100',11,r/'m2',r/'state2',parents,r/'owner')
            ref=bound(r/'m1');(r/'m1').write_text('mutated')
            with self.assertRaises(ValueError):verified(ref)

    def test_original_request_and_external_ledger_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cycle/reference/a1100/products').mkdir(parents=True)
            for n in ['request','plan','original-y']:(root/n).write_text(n)
            refs=dict(map_request=bound(root/'request'),parent_plan=bound(root/'plan'))
            (root/'cycle/reference-a1100.json').write_text(json.dumps(refs))
            ledger=root/'cycle/reference/a1100/products/input-manifest.yaml'
            ledger.write_text(yaml.safe_dump({str(root/'original-y'):bound(root/'original-y')['sha256']}))
            self.assertEqual(original_parents(root,'a1100'),refs)
            (root/'request').write_text('changed config')
            with self.assertRaises(ValueError):original_parents(root,'a1100')
            (root/'request').write_text('request');(root/'original-y').write_text('changed input')
            with self.assertRaises(ValueError):original_parents(root,'a1100')

    def test_manifest_preserves_intermediates_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'T2').write_text('two');(root/'T3').write_text('three')
            manifest(root);self.assertEqual(verify_manifest(root),2)
            (root/'T2').write_text('overwritten')
            with self.assertRaises(ValueError):verify_manifest(root)

    def test_candidate_replacement_removal_and_guard_states(self):
        h=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600)
        p=np.array([10,0,0,2,2,0,0]);v=np.ones((41,41),bool);s=np.zeros((41,41));s[20,20:22]=10
        def candidate(signal):
            c=admit(signal,v,np.ones_like(signal),v,h,p,3.,1.)
            return dict(CANDIDATE=c['candidate'],ADMITTED=c['selected'],VALID=c['valid'])
        old=candidate(s);s[20,21]=0;new=candidate(s);candidate_guard(new,h)
        self.assertEqual(selection_difference(new,old)['removed'],1)
        bad={k:a.copy() for k,a in new.items()};bad['ADMITTED'][20,0]=1;bad['CANDIDATE'][20,0]=10
        with self.assertRaisesRegex(ValueError,'boundary'):candidate_guard(bad,h)
        bad={k:a.copy() for k,a in new.items()};bad['VALID'][20,20]=0
        with self.assertRaisesRegex(ValueError,'unsupported'):candidate_guard(bad,h)
        bad={k:a.copy() for k,a in new.items()};bad['CANDIDATE'][20,21]=2
        with self.assertRaisesRegex(ValueError,'zero'):candidate_guard(bad,h)

    def test_two_cycle_differences_expose_alternation(self):
        domain=np.ones((2,2),bool);a=np.zeros((2,2));b=a+2
        self.assertEqual(difference(b,a,domain,10)['rms_over_M0_fitted_amplitude'],.2)
        self.assertEqual(difference(a,a,domain,10)['rms'],0)

if __name__ == '__main__': unittest.main()
