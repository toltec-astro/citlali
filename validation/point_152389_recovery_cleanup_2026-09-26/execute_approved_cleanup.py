from pathlib import Path
import concurrent.futures, gzip, hashlib, json, os, stat, subprocess, time, datetime
W=Path('/private/tmp/citlali-point-additive-20260923')
E=W/'validation/point_152389_recovery_cleanup_2026-09-26'
BASE=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs')
EXPECTED=[BASE/'successor-full-network-ready-ae4842e-20260919/unity-packet/assets', BASE/'successor-full-network-ready-ae4842e-20260919/unity-packet/local-reference',Path('/private/tmp/point-fruit-one-cycle-20260923')]
RETAINED=[BASE/'successor-full-network-checkpoint-98380090e-20260919/unity-packet/assets',BASE/'successor-full-network-checkpoint-98380090e-20260919/unity-packet/local-reference',BASE/'successor-point-152389-fruit-one-cycle-20260923/cycle']
def require(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stamp(p):
 s=p.lstat();require(stat.S_ISREG(s.st_mode),'Not regular: '+str(p))
 return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def inventory(root):
 require(root.resolve()==root and root.is_dir() and not root.is_symlink(),'Unsafe/missing root '+str(root))
 files=set();dirs=[]
 for directory,ds,fs in os.walk(root,followlinks=False):
  d=Path(directory);dirs.append(d)
  for name in ds:require(not (d/name).is_symlink(),'Symlink directory '+str(d/name))
  for name in fs:
   p=d/name;stamp(p);files.add(str(p.relative_to(root)))
 return files,dirs
def save(p,x):
 with p.open('w') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def free():return subprocess.check_output(['df','-k','/Users/gwilson'],text=True)
def checkpoint():
 rows=json.loads((E/'CHECKPOINT_AUTHENTICATION.json').read_text())['checks']
 for r in rows:require(sha(Path(r['path']))==r['expected'],'Checkpoint mismatch '+r['path'])
 return len(rows)
proposal_path=E/'CLEANUP_PROPOSAL.json'
require(sha(proposal_path)=='47d9ea6deeab053a907bfebf2144623aff032cccafeb32507a865b1235b7de8e','Proposal changed')
p=json.loads(proposal_path.read_text())
require([Path(x['path']) for x in p['targets']]==EXPECTED,'Wrong targets')
require([Path(x['retained_path']) for x in p['targets']]==RETAINED,'Wrong preservation paths')
require(sha(E/'PACKET_DUPLICATE_VERIFICATION.json.gz')==p['packet_ledger_gzip_sha256'],'Packet ledger changed')
raw=gzip.decompress((E/'PACKET_DUPLICATE_VERIFICATION.json.gz').read_bytes())
require(hashlib.sha256(raw).hexdigest()==p['packet_ledger_sha256'],'Packet ledger uncompressed mismatch')
require(sha(E/'SCRATCH_DUPLICATE_VERIFICATION.json')==p['scratch_ledger_sha256'],'Scratch ledger changed')
packet=json.loads(raw);scratch=json.loads((E/'SCRATCH_DUPLICATE_VERIFICATION.json').read_text())
rows=[(Path(x['target']),Path(x['preservation']),x['target_sha256'],x['bytes']) for x in packet['pairs']]
rows += [(Path(x['source']),Path(x['retained']),x['source_sha256'],x['bytes']) for x in scratch['entries']]
require(len(rows)==7039 and sum(r[3] for r in rows)==12804081472,'Wrong counts')
require(len({x[0] for x in rows})==len(rows),'Duplicate targets')
dirs=[]
for root,keep in zip(EXPECTED,RETAINED):
 selected=[r for r in rows if r[0].is_relative_to(root)]
 for target,preserved,_,_ in selected:require(preserved==keep/target.relative_to(root),'Wrong counterpart')
 rel={str(r[0].relative_to(root)) for r in selected}
 inv,ds=inventory(root);require(inv==rel,'Target inventory changed '+str(root));dirs+=ds
 inv,_=inventory(keep);require(inv==rel,'Preserved inventory changed '+str(keep))
receipt=E/'EXECUTION_RECEIPT.json';require(not receipt.exists(),'Execution receipt already exists; do not rerun')
log=E/'DELETED_PATHS.jsonl';require(not log.exists(),'Deletion log already exists; do not rerun')
old_patch=subprocess.check_output(['git','-C',str(W),'diff','--','tools/timestream_successor/point_corr_admission.py'])
require(old_patch==(E/'UNFINISHED_CUT3.patch').read_bytes(),'Unfinished patch changed')
parent=EXPECTED[0].parent
metadata={str(f):sha(f) for f in parent.iterdir() if f.is_file() and not f.is_symlink()}
check_count=checkpoint();t0=time.monotonic()
record={'status':'verifying approved operation','owner_authorization':'User: yes, go ahead and delete those files','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proposal_sha256':sha(proposal_path),'script_sha256':sha(Path(__file__)),'free_space_before':free(),'targets':[str(x) for x in EXPECTED],'retained':[str(x) for x in RETAINED],'checkpoint_checks_before':check_count,'deleted_files':0,'deleted_logical_bytes':0}
save(receipt,record)
def verify(row):
 target,preserved,digest,size=row;a=stamp(target);b=stamp(preserved)
 require(a[2]==b[2]==size,'Size mismatch '+str(target))
 require(sha(target)==sha(preserved)==digest,'Digest mismatch '+str(target))
 require(stamp(target)==a and stamp(preserved)==b,'Changed during verification')
 return str(target),(a,b)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:stamps=dict(ex.map(verify,rows))
 record['fresh_pairs_verified']=len(stamps);record['verification_seconds']=time.monotonic()-t0
 print('Verified all 7039 target/retained pairs. Checking open files before removal.',flush=True)
 for root in EXPECTED:
  proc=subprocess.run(['/usr/sbin/lsof','-nP','-Fpcn','+D',str(root)],capture_output=True,text=True)
  require(proc.returncode==1 and not proc.stdout.strip() and not proc.stderr.strip(),'Open-file check did not establish idle target: '+str(root)+' '+proc.stdout[:1000]+proc.stderr[:1000])
 for root,keep in zip(EXPECTED,RETAINED):
  inv,_=inventory(root);expected={str(r[0].relative_to(root)) for r in rows if r[0].is_relative_to(root)}
  require(inv==expected,'Target changed before removal')
  inv,_=inventory(keep);require(inv==expected,'Retained changed before removal')
 for target,preserved,_,_ in rows:
  a,b=stamps[str(target)];require(stamp(target)==a and stamp(preserved)==b,'File changed before removal')
 record['status']='all pre-deletion checks passed; deleting only enumerated files';save(receipt,record)
 with log.open('x') as f:
  for target,preserved,digest,size in rows:
   a,b=stamps[str(target)];require(stamp(target)==a and stamp(preserved)==b,'File changed at deletion')
   f.write(json.dumps({'target':str(target),'retained':str(preserved),'sha256':digest,'bytes':size,'action':'unlink about to execute'})+'\n');f.flush()
   target.unlink();record['deleted_files']+=1;record['deleted_logical_bytes']+=size
  os.fsync(f.fileno())
 for d in sorted(dirs,key=lambda x:len(x.parts),reverse=True):d.rmdir()
 require(all(not x.exists() for x in EXPECTED),'Target remains')
 for target,preserved,_,_ in rows:require(stamp(preserved)==stamps[str(target)][1],'Preserved file changed')
 for path,digest in metadata.items():require(sha(Path(path))==digest,'Original packet metadata changed')
 require(subprocess.check_output(['git','-C',str(W),'diff','--','tools/timestream_successor/point_corr_admission.py'])==old_patch,'Unfinished patch changed')
 record['checkpoint_checks_after']=checkpoint()
 notice=parent/'CLEANUP_RETIREMENT_2026-09-26.md';require(not notice.exists(),'Retirement notice already exists')
 notice.write_text('# Historical packet retired from self-contained use — 2026-09-26\n\nOwner-approved duplicate cleanup removed only assets/ and local-reference/.\nDo not use the historical submission/readiness instructions on this layout.\nThe original PREPARATION, scripts, reviews and manifests remain unchanged;\nthe original full-payload verification now reports the retired paths missing.\n\nIdentical payloads remain at:\n'+''.join('- '+str(k)+'\n' for k in RETAINED[:2])+'\nRestore those two trees to their original names and verify the full per-file\nSHA256 pair ledger before considering a historical layout replay. The packet\nwas superseded by the 2026-09-20 resident-input submission design.\n\nExact cleanup proposal, pair ledgers and execution receipt:\n'+str(E)+'\n\nNo current POINT map, source input, retained checkpoint, or ready/local-evidence\nwas removed. No old scientific manifest was rewritten.\n')
 record.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-t0,free_space_after=free(),all_targets_absent=True,retained_file_stats_unchanged=True,original_packet_metadata_unchanged=True,unfinished_patch_unchanged=True,retirement_notice=str(notice),retirement_notice_sha256=sha(notice),deletion_log_sha256=sha(log))
 save(receipt,record);print(json.dumps(record,indent=2),flush=True)
except BaseException as exc:
 record.update(status='stopped; inspect receipt and deletion log before any further action',error=repr(exc),free_space_at_stop=free());save(receipt,record);raise
