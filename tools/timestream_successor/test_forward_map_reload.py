"""Small exact-parent fixtures for the actual MAP reload/replay reader."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from inspect_forward_map import digest, replay_handoffs, MAP_REGISTRY_SHA, MAP_REGISTER_SHA, EVALUATION_BINDINGS, summarize_map_probe

class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='map-reload-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'ptc';self.source.mkdir()
        self.cal=self.root/'cal';self.cal.mkdir();(self.root/'handoff').mkdir()
        self.stem='segment-0';self.shape=(2,3)
        self.values=np.array([[2,-3,7],[4,9,100]],dtype='<f8');self.causes=np.array([[0,0,0],[0,0,1]],dtype='u1')
        self.values.tofile(self.source/'segment-0-cleaned.f64');self.causes.tofile(self.source/'segment-0-causes.u8')
        np.array([0,1],dtype='<i8').tofile(self.source/'segment-0-slots.i64')
        det=[]
        for i,v in enumerate(([0,0],[2,0],[0,0])):
            name=f'channel-{i}-causes.u16';np.array(v,dtype='<u2').tofile(self.cal/name)
            det.append(dict(channel=i,files={name:digest(self.cal/name)}))
        self.write(self.cal/'receipt.yaml',dict(detectors=det));calsha=digest(self.cal/'receipt.yaml')
        seg=dict(output_detector_indices=[0,1,2],scheduled_times=2,output_detectors=3,
                 files={p.name:digest(p) for p in self.source.iterdir()})
        self.write(self.source/'receipt.yaml',dict(source_CAL_receipt_sha256=calsha,segments=[seg]));ptcsha=digest(self.source/'receipt.yaml')
        pub=dict(identity='sha256:'+ptcsha+':segment-0',support_reference='support',retained=5,CAL_parent='sha256:'+calsha,
                 RTC_grid='rtc',source_PTC_root=str(self.source),source_PTC_receipt_sha256=ptcsha,source_segment=seg,
                 source_CAL_root=str(self.cal),source_CAL_receipt_sha256=calsha,RTC_scheduled_slots=2)
        self.q=dict(first_input_row=0,input_rows=5,PTC_publication=pub['identity'],support_reference='support')
        self.bind('publication',pub)
        self.bind('coefficient',dict(scalar=1,unit=1,stored_support_reference='support',PTC_publication_sha256=self.q['publication_sha256']))
        self.bind('coefficient_QC',dict(PTC_publication_sha256=self.q['publication_sha256'],payload_sha256=self.q['coefficient_sha256'],
                  logical_occurrence_domain='support',requested='requested',realization='realized',restrictions={f'UQ-{i:02}':'T' for i in range(1,8)}))
        xy=np.array([[0,0],[0,0],[-1,0],[0,0],[-1,0]],dtype='<f8');self.bind('AST_coordinate',xy)
        ast=dict(PTC_publication=pub['identity'],CAL_parent=pub['CAL_parent'],RTC_grid='rtc',sha256=self.q['AST_coordinate_sha256'],
                 coefficient_payload_sha256=self.q['coefficient_sha256'],coefficient_QC_evaluation_sha256=self.q['coefficient_QC_sha256'])
        self.bind('AST_binding',ast);self.bind('MAP_eligibility',np.array([29],dtype='u1'))
        decision=dict(registry_sha256=MAP_REGISTRY_SHA,source_register_sha256=MAP_REGISTER_SHA,profile='SCI-MAP:map_upstream_admission@2',
                      binding='SCI-MAP-v0.2-r0.4-2026-09-08',role='SCI-MAP:base_observation_map@1',object_domain='support',
                      requested='requested',applicability='applicable',realization='realized',eligible=4,ineligible=1,
                      eligibility_sha256=self.q['MAP_eligibility_sha256'])
        for k in ('publication','coefficient','AST_binding','coefficient_QC'):decision[k+'_sha256']=self.q[k+'_sha256']
        decision.update(EVALUATION_BINDINGS);self.bind('MAP_evaluation',decision)
        self.plan=dict(role='SCI-MAP:base_observation_map@1',ordered_column_domain=[self.q],
                       WCS=dict(rows=1,columns=2,CRPIX1=1,CRPIX2=1,CDELT1=-1,CDELT2=1))
        self.pixels=np.array([0,-1,1,0,1],dtype='<i4');self.pixels.tofile(self.root/'operator-pixel.i32')
        self.N=np.array([[6.,16.]]);self.hits=np.array([[2,2]])
    def write(self,path,node):path.write_text(yaml.safe_dump(node,sort_keys=False))
    def bind(self,key,value):
        path=self.root/'handoff'/(key+('.bin' if isinstance(value,np.ndarray) else '.yaml'))
        if isinstance(value,np.ndarray):value.tofile(path)
        else:self.write(path,value)
        self.q[key+'_file']=str(path.relative_to(self.root));self.q[key+'_sha256']=digest(path)
        if key=='MAP_evaluation':self.q['MAP_evaluation_identity']='sha256:'+digest(path)
    def run_replay(self):replay_handoffs(self.root,self.plan,self.N,self.hits)
    def test_complete_exact_parent_replay_and_ineligible_retained_column(self):
        self.run_replay();self.assertEqual(self.pixels[1],-1);self.assertEqual(self.causes[0,1],0)
    def test_local_probe_uses_same_membership_and_operator(self):
        response=replay_handoffs(self.root,self.plan,self.N,self.hits,True,2.)
        summary=summarize_map_probe(response,self.hits,np.ones_like(self.hits,dtype=bool),self.plan['WCS'],2.,self.hits)
        self.assertEqual(summary['status'],'measured');self.assertEqual(summary['constant_unit_response_max_error'],0.)
        self.assertEqual(response[0,0],2.) # two admitted occurrences at origin
        self.assertEqual(response[0,1],0.) # distant template excluded by declared 8-sigma truncation
    def test_absent_optional_probe_footprint_does_not_fail_base_inspection(self):
        self.run_replay()
        result=summarize_map_probe(np.zeros_like(self.N),self.hits,np.ones_like(self.hits,dtype=bool),self.plan['WCS'],2.,self.hits)
        self.assertEqual(result['status'],'unavailable');self.assertEqual(result['cause'],'no supported probe footprint')
    def test_missing_required_ast_and_qc_never_pass(self):
        for key in ('AST_coordinate','AST_binding','coefficient_QC','MAP_evaluation','MAP_eligibility'):
            path=self.root/self.q[key+'_file'];saved=path.read_bytes();path.unlink()
            with self.assertRaises(FileNotFoundError):self.run_replay()
            path.write_bytes(saved)
    def test_same_hit_counts_do_not_validate_permuted_operator(self):
        swapped=self.pixels.copy();swapped[[0,2]]=swapped[[2,0]];swapped.tofile(self.root/'operator-pixel.i32')
        with self.assertRaisesRegex(ValueError,'operator differs'):self.run_replay()
    def test_wrong_numerator_fails_exact_value_replay(self):
        self.N[0,0]+=1
        with self.assertRaisesRegex(ValueError,'PTC replay'):self.run_replay()
    def test_dimensionless_unit_accepts_native_scalar_one_only(self):
        path=self.root/self.q['coefficient_file'];original=yaml.safe_load(path.read_text())
        for unit in (1.0,True,False,'mJy','inverse_variance'):
            coefficient=dict(original,unit=unit);self.bind('coefficient',coefficient)
            with self.assertRaisesRegex(ValueError,'coefficient/support mismatch'):self.run_replay()
        self.bind('coefficient',original);self.run_replay()
    def test_wrong_qc_parent_fails_even_when_own_hash_updated(self):
        path=self.root/self.q['coefficient_QC_file'];qc=yaml.safe_load(path.read_text());qc['PTC_publication_sha256']='foreign';self.bind('coefficient_QC',qc)
        with self.assertRaisesRegex(ValueError,'QC parent'):self.run_replay()
    def test_ineligible_cannot_be_promoted_by_saved_bit(self):
        self.bind('MAP_eligibility',np.array([31],dtype='u1'))
        path=self.root/self.q['MAP_evaluation_file'];d=yaml.safe_load(path.read_text());d.update(eligibility_sha256=self.q['MAP_eligibility_sha256'],eligible=5,ineligible=0);self.bind('MAP_evaluation',d)
        with self.assertRaisesRegex(ValueError,'exact CAL/PTC facts'):self.run_replay()
    def test_foreign_source_generation_cannot_be_rebound(self):
        path=self.root/self.q['MAP_evaluation_file'];d=yaml.safe_load(path.read_text());d['registry_sha256']='foreign';self.bind('MAP_evaluation',d)
        with self.assertRaisesRegex(ValueError,'evaluation generation'):self.run_replay()
    def test_decision_identity_changes_with_parent_even_when_bits_match(self):
        old=self.q['MAP_evaluation_identity'];path=self.root/self.q['MAP_evaluation_file']
        d=yaml.safe_load(path.read_text());d['coefficient_sha256']='new-coefficient';self.bind('MAP_evaluation',d)
        self.assertNotEqual(old,self.q['MAP_evaluation_identity'])
        with self.assertRaisesRegex(ValueError,'decision parent mismatch'):self.run_replay()
    def test_byte_mismatch_and_nonzero_padding_fail(self):
        self.bind('MAP_eligibility',np.array([157],dtype='u1'))
        path=self.root/self.q['MAP_evaluation_file'];d=yaml.safe_load(path.read_text());d['eligibility_sha256']=self.q['MAP_eligibility_sha256'];self.bind('MAP_evaluation',d)
        with self.assertRaisesRegex(ValueError,'padding'):self.run_replay()

if __name__=='__main__':unittest.main()
