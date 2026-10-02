"""Actual native FRUIT admission, synthetic bound files, no observational arrays."""
import argparse,copy,hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np,yaml
from astropy.io import fits
CLI=None
class Contract(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='fruit-contract-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.serial=0
 def bound(self,name,value):
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return self.binding(p)
 def binding(self,p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def fixture(self,scope='rtc-retained-residuals-v1',cycle=1,array='a1100',obs=152430,networks=None):
  self.serial+=1;prefix=str(self.serial);nets=networks or ([1,2,3,4] if array=='a1100' else [11,12]);self.prefix=prefix
  p=dict(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope=scope,observation=obs,subobservation=0,scan=2,array=array,networks=nets,ptc=dict(rank=10,method='observed-als'),experimental_method='warm-observed-als',feedback_count=5,cycles=5,coverage_policy='high-coverage-informational')
  mo=self.bound(prefix+'/map-owner.json',{});sources=[];nr=[]
  for n in nets:
   root=self.root/prefix/f'network{n}';inp=self.bound(prefix+f'/input{n}.json',dict(network=n,observation=obs));rt=self.bound(prefix+f'/network{n}/receipt.yaml',dict(network=n,observation=obs));pt=self.bound(prefix+f'/network{n}/donor-continuity/ptc/receipt.yaml',dict(segments=[dict(network=n)]));sources.append(dict(network=n,root=str(root),PTC_receipt_sha256=pt['sha256']));nr.append(dict(root=str(root),input=inp,receipt_sha256=rt['sha256'],ptc_receipt_sha256=pt['sha256']))
  wcs=dict(CDELT1=-2/3600,CDELT2=2/3600,CRVAL1=176.,CRVAL2=40.)
  parent=dict(array=array,observation=f'obs:{obs}_0_2',coverage_cut=.5,owner_binding='sha256:'+mo['sha256'],sources=sources,WCS=wcs)
  mr=dict(schema='citlali-map-saved-ptc-v1',coverage_cut=.5,pixel_size_arcsec=2,owner_binding=mo,networks=nr)
  owner=dict(identity='POINT-SPEED-PCA-v1',observation=obs,outside_disk_no_feedback=True,direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=5,policy='BASE-POINT',threshold=3,source_method='POINT-EXPECTED-v1')
  candidate=self.root/prefix/'candidate.fits';candidate.parent.mkdir(parents=True,exist_ok=True);h=fits.Header();h.update(wcs);h.update(CRPIX1=21,CRPIX2=21,RADESYS='FK5',EQUINOX=2000.,BUNIT='mJy/nominal beam');fits.HDUList([fits.PrimaryHDU(np.zeros((41,41)),h),fits.ImageHDU(np.ones((41,41),dtype='u1'),name='VALID')]).writeto(candidate)
  req=dict(schema='citlali-point-speed-pca-v1',method='point-speed-pca-v1',mode='reference',array=array,branch='REF',cycle_index=cycle,candidate=self.binding(candidate))
  return dict(p=p,parent=parent,mr=mr,owner=owner,req=req)
 def submit(self,f,terminal='point-fruit-validate',corrupt=None):
  pre=self.prefix;req=copy.deepcopy(f['req']);owner=copy.deepcopy(f['owner']);owner['prospective']=self.bound(pre+'/prospective.json',f['p']);req['parent_plan']=self.bound(pre+'/parent.json',f['parent']);req['map_request']=self.bound(pre+'/map.json',f['mr']);owner['parents']={req['array']:dict(parent_plan=req['parent_plan'],map_request=req['map_request'])};req['owner_binding']=self.bound(pre+'/owner.json',owner)
  prev=dict(schema='point-fruit-ready-state-v1',array=req['array'],cycle_index=req['cycle_index']-1,ready=True,candidate=req['candidate'],original_plan_sha256=req['parent_plan']['sha256'],original_map_request_sha256=req['map_request']['sha256'],branch=req['branch'],owner_sha256=req['owner_binding']['sha256'],total=self.bound(pre+'/total.json',{}),receipt=self.bound(pre+'/receipt.json',{}))
  if corrupt:corrupt(req,prev)
  req['previous_state']=self.bound(pre+'/previous.json',prev);inp=self.bound(pre+'/request.json',req);out=self.root/pre/'out';config=self.bound(pre+'/development.json',dict(schema='citlali-development-v1',terminal=terminal,input=inp,output=str(out)))
  result=subprocess.run([CLI,config['path']],text=True,capture_output=True,timeout=30)
  return result,out
 def accept(self,f):
  result,out=self.submit(f);self.assertEqual(result.returncode,0,result.stdout+result.stderr);v=yaml.safe_load((out/'validation.yaml').read_text());self.assertEqual(v['outcome'],'request-contract-accepted');self.assertFalse(v['scientific_processing_executed']);self.assertFalse(v['detector_arrays_loaded']);self.assertEqual(v['cycle_index'],f['req']['cycle_index']);self.assertEqual(list(out.iterdir()),[out/'validation.yaml'])
 def reject(self,f,reason,**kw):
  result,out=self.submit(f,**kw);self.assertNotEqual(result.returncode,0);self.assertIn(reason,result.stdout+result.stderr);self.assertFalse(out.exists())
 def test_every_retained_and_previous_four_network_cycle(self):
  for scope in ('rtc-retained-residuals-v1','rtc-step-consequences-v1'):
   for cycle in range(1,6):
    with self.subTest(scope=scope,cycle=cycle):self.accept(self.fixture(scope,cycle))
 def test_transfer_all_five_native_cycles(self):
  for cycle in range(1,6):self.accept(self.fixture('rtc-point-transfer-v1',cycle,'a1100',152420))
 def test_approved_runtime_profile_all_cycles_and_rejects_unbound_caps(self):
  for cycle in range(1,6):
   f=self.fixture('rtc-point-transfer-v1',cycle,'a1100',152420);f['p']['runtime_profile']='point-transfer-runtime-2026-09-29-v1';f['p']['scientific_recipe']={'PTC':{'max_iterations':500}};self.accept(f)
  for scope,obs,profile,cap in [('rtc-point-transfer-v1',152420,'foreign',500),('rtc-retained-residuals-v1',152430,'point-transfer-runtime-2026-09-29-v1',500),('rtc-point-transfer-v1',152420,'point-transfer-runtime-2026-09-29-v1',100),('rtc-point-transfer-v1',152420,None,500)]:
   f=self.fixture(scope,1,'a1100',obs);f['p']['scientific_recipe']={'PTC':{'max_iterations':cap}}
   if profile:f['p']['runtime_profile']=profile
   self.reject(f,'FRUIT')
 def test_transfer_rejects_foreign_identity_and_cycles(self):
  for obs in (152430,152389):self.reject(self.fixture('rtc-point-transfer-v1',1,'a1100',obs),'FRUIT')
  for cycle in (0,6):self.reject(self.fixture('rtc-point-transfer-v1',cycle,'a1100',152420),'FRUIT')
  for key,value in [('subobservation',1),('scan',3),('networks',[1,2,3,5]),('cycles',10),('feedback_count',6)]:
   f=self.fixture('rtc-point-transfer-v1',1,'a1100',152420);f['p'][key]=value;self.reject(f,'FRUIT')
  f=self.fixture('rtc-point-transfer-v1',1,'a1100',152420);f['owner']['observation']=152430;self.reject(f,'FRUIT')
  f=self.fixture('rtc-point-transfer-v1',1,'a1100',152420);f['parent']['observation']='obs:152430_0_2';self.reject(f,'FRUIT')
  self.reject(self.fixture('rtc-point-transfer-v1',1,'a2000',152420),'FRUIT')
 def test_altaz_parent_cannot_enter_historical_fk5_feedback(self):
  f=self.fixture();f['parent']['WCS']={'ASTFRAME':'delta-source-AltAz'};self.reject(f,'FRUIT AltAz feedback requires separate frame-aware validation')
 def test_previous_a2000_routes(self):
  for scope,obs in [('rtc-cost-response-v1',152430),('rtc-fixed11-v1',152430),('rtc-step-point-v1',152420),('rtc-step-point-v1',152430)]:
   for cycle in (1,5):
    with self.subTest(scope=scope,obs=obs,cycle=cycle):self.accept(self.fixture(scope,cycle,'a2000',obs))
 def test_foreign_bound_identifiers_and_recipes(self):
  mutations=[('observation',152420),('subobservation',1),('scan',3),('array','a2000'),('networks',[1,2,3,5]),('networks',[1,2,3]),('networks',[1,2,3,3]),('feedback_count',6),('cycles',10),('experimental_method','observed-als')]
  for key,value in mutations:
   with self.subTest(key=key,value=value):
    f=self.fixture();f['p'][key]=value;self.reject(f,'FRUIT')
  f=self.fixture();f['p']['ptc']['rank']=15;self.reject(f,'FRUIT')
  f=self.fixture();f['owner']['observation']=152420;self.reject(f,'FRUIT')
  f=self.fixture();f['req']['array']='a2000';self.reject(f,'FRUIT')
  f=self.fixture();f['parent']['observation']='obs:152430_1_2';self.reject(f,'FRUIT foreign array/observation')
 def test_unsupported_cycles_and_predecessor(self):
  for cycle in (-1,0,6,11):
   with self.subTest(cycle=cycle):self.reject(self.fixture(cycle=cycle),'FRUIT')
  self.reject(self.fixture(),'FRUIT predecessor unavailable',corrupt=lambda req,prev:prev.update(cycle_index=3))
  self.reject(self.fixture(),'FRUIT predecessor candidate differs',corrupt=lambda req,prev:prev.update(candidate=dict(path=prev['candidate']['path'],sha256='0'*64)))
 def test_native_domain_and_binding_checks(self):
  f=self.fixture();f['parent']['sources'][3]['network']=5;self.reject(f,'FRUIT four-network')
  f=self.fixture();f['mr']['networks'].pop();self.reject(f,'FRUIT four-network declared networks')
  f=self.fixture();f['mr']['networks'][0]['input']['sha256']='0'*64;self.reject(f,'digest mismatch')
  f=self.fixture();f['p']['campaign_scope']='unknown-scope';self.reject(f,'FRUIT foreign array/observation')
 def test_same_guards_in_processing_route(self):
  f=self.fixture();f['p']['subobservation']=7;self.reject(f,'FRUIT bounded rank experiment scope',terminal='point-fruit')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);a,rest=p.parse_known_args();CLI=str(a.binary.resolve());unittest.main(argv=[__file__,*rest])
