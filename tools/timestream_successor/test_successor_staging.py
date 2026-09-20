"""Exact resident-input binding without copying raw measurements."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import stage_successor_inputs as staging


class ResidentInputs(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.original=self.root/'original';self.original.mkdir()
        self.packet=self.root/'packet';self.resident=self.root/'unity-mirror'
        def ref(name,data):
            p=self.original/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            return dict(path=str(p),sha256=staging.digest(p))
        self.base=dict(native_source='exact-raw-kids-v1',tune=ref('toltec12_tune.txt',b'tune'),
            telescope=ref('tel.nc',b'telescope'),effective_config=ref('config.yaml',b'config'),
            ast_acceptance=ref('ast.json',b'ast'),manifest=ref('apt/manifest.ecsv',b'apt'),
            filter_plan={'path':'source-owned-filter','sha256':'not-a-copied-asset'},
            decision_apply=dict(timing_inputs=[ref('toltec12.nc',b'raw')],
                processing_provenance=ref('provenance.yaml',b'provenance')))
        ref('apt/member.ecsv',b'member')
        self.reference=self.root/'reference.json';self.reference.write_text(json.dumps(self.base))

    def stage(self):
        with patch.object(staging,'SUPPLIED',[12]):
            staging.stage(self.reference,self.packet,'projects/science/data')
        for entry in json.loads((self.packet/'EXISTING_INPUTS.json').read_text())['files']:
            dst=self.resident/entry['relative_path'];dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes((self.original/dst.name).read_bytes())

    def test_reuses_exact_raw_and_telescope_with_no_netcdf_payload(self):
        self.stage();self.assertEqual(list(self.packet.rglob('*.nc')),[])
        checked=staging.verify_existing(self.packet,self.resident)
        self.assertEqual(len(checked),2)
        output=self.root/'materialized'
        with patch('prepare_successor_networks.prepare') as prepare:
            staging.materialize(self.packet,output,self.root,self.resident)
            bound=json.loads(prepare.call_args.args[0].read_text())
        self.assertEqual(bound['decision_apply']['timing_inputs'][0]['sha256'],self.base['decision_apply']['timing_inputs'][0]['sha256'])
        self.assertEqual(bound['telescope']['sha256'],self.base['telescope']['sha256'])
        self.assertTrue(Path(bound['telescope']['path']).is_relative_to(self.resident))
        self.assertTrue(Path(bound['tune']['path']).is_relative_to(self.packet))
        self.assertNotIn('filter_plan',bound)

    def test_missing_root_or_file_fails_before_materialization(self):
        self.stage()
        with self.assertRaisesRegex(ValueError,'existing-root'):staging.verify_existing(self.packet,None)
        (self.resident/'projects/science/data/toltec12.nc').unlink()
        with patch('prepare_successor_networks.prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'missing resident input'):
                staging.materialize(self.packet,self.root/'output',self.root,self.resident)
            prepare.assert_not_called()
        self.assertFalse((self.root/'output-reference.json').exists())

    def test_same_size_wrong_raw_is_rejected(self):
        self.stage();(self.resident/'projects/science/data/toltec12.nc').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):staging.verify_existing(self.packet,self.resident)

    def test_traversal_and_absolute_resident_paths_are_rejected(self):
        self.stage();path=self.packet/'EXISTING_INPUTS.json';original=json.loads(path.read_text())
        for invalid in ('../outside.nc','/outside.nc'):
            original['files'][0]['relative_path']=invalid;path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'contained relative'):staging.verify_existing(self.packet,self.resident)

    def test_unbound_token_and_packet_escape_are_rejected(self):
        self.stage();path=self.packet/'reference-template.json';base=json.loads(path.read_text())
        for invalid in ('@existing/unknown.nc','../outside'):
            base['telescope']['path']=invalid;path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,'unbound resident|contained relative'):
                staging.materialize(self.packet,self.root/'output',self.root,self.resident)

    def test_historical_copied_packet_still_materializes(self):
        with patch.object(staging,'SUPPLIED',[12]):staging.stage(self.reference,self.packet)
        with patch('prepare_successor_networks.prepare') as prepare:
            staging.materialize(self.packet,self.root/'output',self.root)
            bound=json.loads(prepare.call_args.args[0].read_text())
        self.assertTrue(Path(bound['telescope']['path']).is_relative_to(self.packet))
        self.assertEqual(staging.verify_existing(self.packet,None),{})


if __name__=='__main__':unittest.main()
