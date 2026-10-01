import sys,json,tempfile,unittest,importlib.util
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parents[1]/'tools/timestream_successor'));sys.path.insert(0,str(D))
from point_quality_experiment import consume,ROLES,LIMITS
from point_processing import experiment_consume
class Quality(unittest.TestCase):
 def test_bounded_accounting_actual_dispatch(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for role in ROLES:
    out=root/role;(out/'freeze').mkdir(parents=True)
    (out/'freeze/prospective.json').write_text(json.dumps(dict(campaign_scope='rtc-point-transfer-v1',accounting_scope='point-map-quality-v1')))
    (out/'freeze/budget.json').write_text(json.dumps(dict(campaign_root=str(root))))
    consume(root,'parent',dict(run=role));consume(root,'trajectory',dict(run=role,branch='REF',cycle=0))
    for k in range(1,6):experiment_consume(out,'feedback',dict(run=role,branch='REF',cycle=k))
    experiment_consume(out,'endpoint',dict(run=role,cycle=5))
   rows=[json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()]
   self.assertEqual({k:sum(x['kind']==k for x in rows) for k in LIMITS},LIMITS)
   for k in LIMITS:
    with self.assertRaises(ValueError):consume(root,k,dict(run='B-REF'))
 def test_disallow_foreign_roles_cycles_and_retries(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   for role in ('Q-REF','B-C-shoulder','R-REF','B-REF-retry'):
    with self.assertRaises(ValueError):consume(root,'parent',dict(run=role))
   with self.assertRaises(ValueError):consume(root,'feedback',dict(run='B-REF',branch='REF',cycle=6))
   consume(root,'parent',dict(run='B-REF'))
   with self.assertRaises(ValueError):consume(root,'parent',dict(run='B-REF'))
 def test_asymmetry_even_image_and_missing_reflected_support(self):
  import numpy as np
  from saved_scorecard import asymmetry
  x,y=np.meshgrid(np.arange(-50.,51.,2.),np.arange(-50.,51.,2.));signal=100*np.exp(-(x*x+y*y)/100)+3
  d=dict(signal=signal,support=np.ones(x.shape,bool),x=x,y=y);p=np.array([100,0,0,5,5,0,3.])
  r=asymmetry(d,p);self.assertAlmostEqual(r['half_absolute_odd_fraction'],0)
  d['support'][x>10]=False;q=asymmetry(d,p);self.assertLess(q['paired_pixels'],r['paired_pixels']);self.assertAlmostEqual(q['half_absolute_odd_fraction'],0)
 def test_actual_current_primary_shape_and_legacy_extension(self):
  import numpy as np
  from point_rtc_response import injection_plane,load
  path=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/B-REF/freeze/shapes/C-core-plus.fits')
  self.assertTrue(path.is_file(), 'required real shape fixture absent')
  data,_=load(path);self.assertIn('PRIMARY',data);self.assertNotIn('INJECTION',data)
  self.assertTrue(np.array_equal(injection_plane(data),data['PRIMARY']))
  image=np.array([[3.,0.],[0.,1.]])
  self.assertTrue(np.array_equal(injection_plane({'PRIMARY':None,'INJECTION':image}),image))
  with self.assertRaises(ValueError):injection_plane({'VALID':image})
 def test_missing_support_is_unavailable_and_common_is_fixed(self):
  import numpy as np
  from saved_scorecard import common_pair,finite_stats
  x,y=np.meshgrid(np.arange(-110.,111.,2.),np.arange(-110.,111.,2.));a=dict(signal=x+y,support=np.ones(x.shape,bool),x=x,y=y);b={**a,'signal':x+y+3,'support':a['support'].copy()};b['support'][:,0]=False
  r=common_pair(a,b);self.assertEqual(r['lost_from_M0'],len(y));self.assertEqual(r['gained_after_M0'],0)
  self.assertTrue(all(abs(q['paired_delta']['median']-3)<1e-8 for q in r['scales']))
  self.assertEqual(len({q['M0']['n'] for q in r['scales']}),1)
  self.assertFalse(finite_stats([np.nan])['available'])
if __name__=='__main__':unittest.main()
