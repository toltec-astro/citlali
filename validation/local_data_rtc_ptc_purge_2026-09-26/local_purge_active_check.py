from pathlib import Path
import subprocess,json,gzip,time,hashlib,argparse
V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26');R=Path('/Users/gwilson/work_toltec/local_data')
ap=argparse.ArgumentParser();ap.add_argument('--precheck',action='store_true');args=ap.parse_args()
x=subprocess.run(['/usr/sbin/lsof','-nP','-F','pfn'],capture_output=True,text=True);targets={str(R/r['path'])for r in json.load(gzip.open(V/'TARGETS.json.gz','rt'))};opened=[];pid=None;fd=None
for line in x.stdout.splitlines():
 if line.startswith('p'):pid=line[1:]
 elif line.startswith('f'):fd=line[1:]
 elif line.startswith('n')and line[1:].startswith(str(R)+'/'):opened.append(dict(pid=pid,descriptor=fd,path=line[1:]))
r=dict(epoch=time.time(),command='/usr/sbin/lsof -nP -F pfn',returncode=x.returncode,stderr=x.stderr,incomplete=x.returncode!=0 or bool(x.stderr.strip()),local_data_open_paths=opened,target_open_paths=[r for r in opened if r['path']in targets],all_open_output_sha256=hashlib.sha256(x.stdout.encode()).hexdigest(),scope='Current-host open descriptor census; only local_data entries retained in record. No processes stopped.')
(V/('ACTIVE_USE_PRECHECK.json'if args.precheck else 'ACTIVE_USE_CHECK.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
