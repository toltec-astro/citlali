"""Check fixed scheduling coverage and failure reporting without running science."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import run_successor_campaign as campaign


class CampaignScheduling(unittest.TestCase):
    def exercise(self,repeat=False,failed_worker=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=root/'input.json'
            request.write_text(json.dumps(dict(networks=[dict(network=i) for i in range(11)])))
            output=root/'campaign';calls=[];comparisons=[]
            def run(binary,input_path,dest,workers):
                calls.append((dest.name,workers));dest.mkdir()
                result=dict(workers=workers,wall_seconds=120/workers,output_bytes=1000,
                    exit_code=1 if workers==failed_worker else 2,samples=[])
                (dest/'RUN.json').write_text(json.dumps(result))
                return result['exit_code']
            def compare(left,right,out,ids):
                comparisons.append((left.name,right.name,ids));return True
            args=['campaign','--binary',str(root/'binary'),'--input',str(request),'--output',str(output)]
            if repeat:args.append('--repeat')
            with patch.object(sys,'argv',args),patch.object(campaign,'run',run),patch.object(campaign,'compare',compare),patch.object(campaign.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)):
                if failed_worker is None:campaign.main()
                else:
                    with self.assertRaisesRegex(RuntimeError,'execution failure'):campaign.main()
            record=json.loads((output/'CAMPAIGN.json').read_text())
            return calls,comparisons,record

    def test_all_five_settings_are_compared_and_partial_outcomes_remain_explicit(self):
        calls,comparisons,record=self.exercise()
        self.assertEqual([w for _,w in calls],[1,2,4,8,12])
        self.assertEqual([b for _,b,_ in comparisons],['workers-2','workers-4','workers-8','workers-12'])
        self.assertTrue(all(a=='workers-1' and len(ids)==11 for a,_,ids in comparisons))
        self.assertEqual(record['requested_worker_settings'],[1,2,4,8,12])
        self.assertEqual(record['internal_threads_per_worker'],1)
        self.assertEqual(record['state'],'completed-equivalent-with-partial-results')

    def test_optional_repeat_includes_fastest_extended_setting(self):
        calls,comparisons,record=self.exercise(repeat=True)
        self.assertEqual([w for _,w in calls],[1,2,4,8,12,1,12])
        self.assertEqual([r['workers'] for r in record['runs'] if r.get('repeat')],[1,12])
        self.assertEqual(len(comparisons),6)

    def test_failure_at_eight_workers_cannot_become_equivalence_pass(self):
        calls,comparisons,record=self.exercise(failed_worker=8)
        self.assertEqual([w for _,w in calls],[1,2,4,8])
        self.assertFalse(comparisons);self.assertNotIn('state',record)
        self.assertEqual(record['runs'][-1]['exit_code'],1)


if __name__=='__main__':unittest.main()
