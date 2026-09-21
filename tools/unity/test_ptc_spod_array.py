"""Focused scheduling tests; numerical diagnostic remains unchanged."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

spec = importlib.util.spec_from_file_location('ptc_spod_array', Path(__file__).with_name('ptc_spod_array.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ArrayPacket(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve(); self.root = self.base/'new'; self.root.mkdir()
        self.prior = self.base/'serial'; self.prior.mkdir()
        for name, value in [('ROOT',self.root),('PRIOR',self.prior)]:
            p=patch.object(m,name,value);p.start();self.addCleanup(p.stop)
        self.source=self.base/'products'; self.source.mkdir()
        self.networks=[];self.snapshots={};self.values={}
        for n in (0,1):
            config=self.source/f'config{n}.json';config.write_text(json.dumps({'network':n,'dt':.008192*(n+1)}))
            source=self.source/f'network{n}';source.mkdir()
            values=np.sin(np.arange(256)*(n+1)*.1);self.values[n]=np.fft.rfft(values)
            np.save(source/'original.npy',values)
            self.networks.append(dict(network=n,output=str(source),input=dict(path=str(config),sha256=m.sha(config))))
            self.snapshots[n]=dict(root=str(source))
        self.data=dict(networks=self.networks,binding={'fixed':'parallel'},serial_binding={'fixed':'serial'})
        def check(files):
            for p,h in files.items():m.require(m.sha(p)==h,'input changed')
        def validate(result,network,snapshot):
            m.require(result['network']==network['network'],'network changed')
            m.require(result['config_sha256']==network['input']['sha256'],'config changed')
        def stage(name,command,attempt):
            network=self.networks[int(name.replace('network',''))]
            out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
            values=np.load(Path(network['output'])/'original.npy')
            np.save(out/'modes.npy',np.fft.rfft(values))
            (out/'result.json').write_text(json.dumps(dict(network=network['network'],config_sha256=network['input']['sha256'])))
        self.serial=SimpleNamespace(require=m.require,check_inputs=check,validate_result=validate,stage=Mock(side_effect=stage),
            check_tools=Mock(),TOOLS=self.base/'tools',resume_or_run=Mock(),source_state=Mock(return_value={}))
        p=patch.object(m,'snapshot_for',side_effect=lambda serial,n:self.snapshots[n]);p.start();self.addCleanup(p.stop)
        self.attempt=self.root/'attempt';self.attempt.mkdir()

    def run_one(self,n,generation='trial'):
        return m.run_network({},self.serial,self.data,n,self.attempt/generation,generation)

    def test_concurrent_networks_have_exact_values_and_isolated_outputs(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(self.run_one,(0,1)))
        self.assertEqual([r['network'] for r in results],[0,1])
        self.assertNotEqual(results[0]['output'],results[1]['output'])
        for r in results:np.testing.assert_array_equal(np.load(Path(r['output'])/'modes.npy'),self.values[r['network']])
        self.assertEqual(self.serial.stage.call_count,2)
        for call in self.serial.stage.call_args_list:
            command=call.args[1];self.assertEqual(command[-5:],['--fft-seconds','2','8','--pool-seconds','120'])

    def test_resume_rehashes_and_does_not_repeat_science(self):
        first=self.run_one(0);second=self.run_one(0,'retry')
        self.assertEqual(first['output'],second['output']);self.assertEqual(self.serial.stage.call_count,1)
        (Path(first['output'])/'modes.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'input changed'):self.run_one(0,'retry2')
        self.assertEqual(self.serial.stage.call_count,1)

    def test_serial_completed_network_reused_without_relocation(self):
        output=self.prior/'runs/old/network0';output.mkdir(parents=True)
        (output/'result.json').write_text(json.dumps(dict(network=0,config_sha256=self.networks[0]['input']['sha256'])))
        record=dict(network=0,binding=self.data['serial_binding'],output=str(output),outputs_sha256={'result.json':m.sha(output/'result.json')})
        m.sealed(self.prior/'completed/network0.json',record)
        self.serial.resume_or_run.return_value=record
        result=self.run_one(0)
        self.assertEqual(result['origin'],'serial-reused');self.assertEqual(Path(result['output']),output)
        self.serial.stage.assert_not_called()
        self.assertEqual(self.serial.resume_or_run.call_args.args[2],self.data['serial_binding'])
        self.run_one(0,'retry');self.assertEqual(self.serial.resume_or_run.call_count,1)

    def test_unfinished_serial_attempt_is_preserved_and_restarted_elsewhere(self):
        partial=self.prior/'runs/old/network0';partial.mkdir(parents=True);(partial/'partial.npz').write_bytes(b'partial')
        result=self.run_one(0)
        self.assertEqual((partial/'partial.npz').read_bytes(),b'partial')
        self.assertNotEqual(Path(result['output']),partial);self.serial.resume_or_run.assert_not_called()

    def test_failed_execution_cannot_publish_completion(self):
        self.serial.stage.side_effect=RuntimeError('diagnostic failed')
        with self.assertRaisesRegex(RuntimeError,'diagnostic failed'):self.run_one(0)
        self.assertFalse((self.root/'completed/network0.json').exists())

    def test_failure_signal_prevents_further_scientific_launches(self):
        m.write(self.root/'submissions/trial/failures/network1.json',{'reason':'failed'})
        with self.assertRaisesRegex(RuntimeError,'another task failed'):self.run_one(0)
        self.serial.stage.assert_not_called()

    def test_unbound_assignment_and_config_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'unbound'):self.run_one(99)
        Path(self.networks[0]['input']['path']).write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'config changed'):self.run_one(0)
        self.serial.stage.assert_not_called()

    def test_lock_excludes_writers_and_allows_distinct_network_readers(self):
        p=self.root/'shared.lock'
        with m.lock(p):
            with self.assertRaisesRegex(RuntimeError,'another writer'):
                with m.lock(p,shared=True):pass
        with m.lock(p,shared=True),m.lock(p,shared=True):
            with self.assertRaisesRegex(RuntimeError,'another writer'):
                with m.lock(p):pass
        with self.assertRaisesRegex(RuntimeError,'absent'):
            with m.lock(self.prior/'missing',shared=True):pass
        self.assertFalse((self.prior/'missing').exists())

    def test_prepare_binding_and_checkpoint_corruption_rejected(self):
        path=self.root/'PREPARED.json';m.sealed(path,{'binding':'original'})
        with patch.object(m,'binding',return_value='changed'),patch.object(m,'environment',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'prepared environment'):m.prepared({},self.serial)
        obj=json.loads(path.read_text());obj['record']['binding']='altered';path.write_text(json.dumps(obj))
        with self.assertRaisesRegex(RuntimeError,'damaged'):m.sealed(path)

    def test_missing_network_prevents_final_success(self):
        self.run_one(0)
        with self.assertRaises(FileNotFoundError):m.finalize({},self.serial,self.data,self.attempt,'trial')
        self.assertFalse((self.root/'STATUS.json').exists())
        self.run_one(1)
        m.write(self.root/'submissions/trial/JOBS.json',{'networks':'400'})
        meta=dict(reporting_revision='source',runtime_revision='runtime')
        with self.assertRaisesRegex(RuntimeError,'missing current task success'):
            m.finalize(meta,self.serial,self.data,self.attempt,'trial')
        for n in (0,1):
            m.write(self.root/f'attempts/trial/network-{n}/FINAL.json',dict(mode='network',network=n,disposition='PASS',
                job=str(401+n),array_job='400',**meta))
        with patch.object(m.subprocess,'check_output',return_value='400_0|401|COMPLETED|0:0\n400_1|402|OUT_OF_MEMORY|0:9\n'),patch.object(m.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'Slurm array task outcomes'):
                m.task_outcomes(meta,self.data,self.attempt,'trial')
        with patch.object(m.subprocess,'check_output',return_value='400_0|401|COMPLETED|0:0\n400_1|402|COMPLETED|0:0\n'):
            m.task_outcomes(meta,self.data,self.attempt,'trial')
        self.assertTrue((self.attempt/'array-outcomes.json').exists())

    def test_submission_wires_prepare_array_and_finalizer(self):
        with patch.object(m.subprocess,'check_output',side_effect=['','101\n','102\n','103\n']) as call,contextlib.redirect_stdout(io.StringIO()):
            m.submit({'networks':[0,1,2,3,4,5,7,8,9,11,12]})
        commands=[c.args[0] for c in call.call_args_list]
        self.assertIn('--dependency=afterok:101',commands[2]);self.assertIn('--array=0,1,2,3,4,5,7,8,9,11,12%11',commands[2])
        self.assertIn('--kill-on-invalid-dep=yes',commands[2]);self.assertIn('--dependency=afterany:102',commands[3])
        jobs=json.loads(next(self.root.glob('submissions/*/JOBS.json')).read_text())
        self.assertEqual(jobs,dict(prepare='101',networks='102',finalize='103'))

    def test_partial_submission_cancels_only_its_created_jobs(self):
        with patch.object(m.subprocess,'check_output',side_effect=['','201\n',RuntimeError('submission failed')]),patch.object(m.subprocess,'run') as cancel,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'submission failed'):m.submit({'networks':[0,1]})
        self.assertEqual(cancel.call_args.args[0],['scancel','201'])

    def test_active_array_submission_not_duplicated(self):
        m.write(self.root/'submissions/old/JOBS.json',{'networks':'301'})
        with patch.object(m.subprocess,'check_output',return_value='301_[0-12]\n') as call:
            with self.assertRaisesRegex(RuntimeError,'still active'):m.submit({'networks':[0,1]})
        self.assertEqual(call.call_count,1)

    def test_compact_return_excludes_numerical_arrays(self):
        self.run_one(0)
        with contextlib.redirect_stdout(io.StringIO()):m.collect()
        import tarfile
        with tarfile.open(next(self.root.glob('results-*.tar.gz'))) as archive:
            names=archive.getnames()
        self.assertIn('diagnostics/network0/result.json',names)
        self.assertFalse(any(n.endswith('.npy') for n in names))
        published=set(self.root.glob('results-*.tar.gz'));original_write=m.write
        def fail_ready(path,value):
            if path.name.endswith('.ready.json'):raise OSError('ready publication failed')
            original_write(path,value)
        with patch.object(m,'write',side_effect=fail_ready),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError,'ready publication failed'):m.collect()
        self.assertEqual(set(self.root.glob('results-*.tar.gz')),published)
        (self.prior/'diagnostics.lock').touch()
        meta=dict(reporting_revision='source',runtime_revision='runtime')
        with patch.object(m,'controls',return_value=meta),patch.object(m,'load_prior',return_value=self.serial),patch.object(m,'prepared',return_value={}),patch.object(m,'finalize',side_effect=lambda *a:m.write(self.root/'STATUS.json',{'state':'PASS'})),patch.object(m,'collect',side_effect=RuntimeError('collection failed')),patch.object(m.signal,'signal'),patch.dict(m.os.environ,SLURM_JOB_ID='123'),patch.object(m.sys,'argv',['runner','finalize','collectionfailure']):
            with self.assertRaisesRegex(RuntimeError,'collection failed'):m.main()
        final=json.loads(next(self.root.glob('attempts/collectionfailure/*/FINAL.json')).read_text())
        self.assertEqual(final['disposition'],'FAIL');self.assertIn('collection failed',final['collection_failure'])
        self.assertEqual(json.loads((self.root/'STATUS.json').read_text())['state'],'FAIL')


if __name__=='__main__':unittest.main()
