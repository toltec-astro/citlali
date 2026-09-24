"""Independent bounded sequence/request and centered history identities."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from run_point_refseq import stage_request,BRANCHES,trajectory_complete,check_native_version
from point_frozen_shoulder import close

class RefSequenceControls(unittest.TestCase):
    def test_native_version_separates_dependency_provenance(self):
        check_native_version('v1-g123456789\nkids abc-dirty','123456789abcdef')
        for v in ('v1-g123456789-dirty\nkids abc','v1-g987654321\nkids 123456789',''):
            with self.assertRaises(ValueError):check_native_version(v,'123456789abcdef')

    def test_stage_index_and_current_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'owner.json';p.write_text('{}');state=Path(d)/'state.json';state.write_text('{}')
            w=dict(original_initial_grid_control={'path':'unused-original-grid'},parents={'a2000':{}},injections={'a2000':{'C-shoulder':{'path':'exact-injection'}}})
            r=stage_request(w,p,BRANCHES[0],0,None);self.assertEqual(r['mode'],'zero');self.assertNotIn('injection',r)
            for k in range(1,11):
                prior=dict(ready=True,cycle_index=k-1,candidate={'path':f'own-m{k-1}'},state_path=str(state))
                for b in BRANCHES:
                    r=stage_request(w,p,b,k,prior);self.assertEqual(r['candidate']['path'],f'own-m{k-1}');self.assertEqual(r['cycle_index'],k)
            with self.assertRaises(ValueError):stage_request(w,p,BRANCHES[0],11,prior)
            prior['cycle_index']=2
            with self.assertRaises(ValueError):stage_request(w,p,BRANCHES[0],10,prior)

    def test_terminal_candidate_failure_preserves_completed_maps(self):
        h=[dict(numerical_passed=True,ready=True) for k in range(11)]
        h[10]['ready']=False
        self.assertTrue(trajectory_complete(h))
        h[9]['ready']=False
        self.assertFalse(trajectory_complete(h))
        h[9]['ready']=True;h[10]['numerical_passed']=False
        self.assertFalse(trajectory_complete(h))
        self.assertFalse(trajectory_complete(h[:10]))

    def test_failed_saved_terminal_candidate_is_not_revalidated(self):
        from report_point_refseq import new_candidate_evidence
        stage=dict(ready=False,candidate_available=True,candidate={'path':'must-not-open'},stop_reason='anchor below threshold')
        candidate,record=new_candidate_evidence(stage,None)
        self.assertIsNone(candidate);self.assertEqual(record['status'],'unavailable')
        self.assertEqual(record['preserved_candidate'],stage['candidate'])

    def test_centered_history_noop_simple_and_exact_protection(self):
        rng=np.random.default_rng(152389);u=rng.normal(size=12);dold=rng.normal(size=12);dnew=rng.normal(size=12)
        j=np.ones((12,12))/12;c=np.eye(12)-j;b=np.eye(12)[:,:2];q=np.eye(12)-b@b.T
        for l in (np.eye(12),c,q@c):
            dc=l@u;fc=(np.eye(12)-l)@dnew;oldcq=.2*rng.normal(size=12)
            old=dc+(np.eye(12)-l)@dold+oldcq;new=dc+fc;history=(np.eye(12)-l)@(dnew-dold)
            close(new-old,-oldcq+history,'history identity',1e-12)
            close(l@u+(np.eye(12)-l)@u,u,'exact protection',1e-12)
            close((np.eye(12)-l)@np.zeros(12),np.zeros(12),'M0 zero',0,0)
        close((np.eye(12)-np.eye(12))@dnew,np.zeros(12),'no-op feedback',0,0)

if __name__=='__main__':unittest.main()
