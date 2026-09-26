from pathlib import Path
import os,stat,json,gzip,time,collections,subprocess
ROOT=Path('/Users/gwilson/work_toltec/local_data').resolve()
OUT=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26')
start=time.time();dev=ROOT.stat().st_dev;rows=[];links=[];errors=[];dirs=[];inodes={};classes=collections.Counter();tops=collections.Counter();allocated=collections.Counter();names=collections.Counter()
def error(p,e):errors.append(dict(path=str(p),error=str(e)))
def walk(d):
 try:entries=list(os.scandir(d))
 except OSError as e:error(d,e);return
 for e in entries:
  p=Path(e.path)
  try:s=e.stat(follow_symlinks=False)
  except OSError as ex:error(p,ex);continue
  rel=p.relative_to(ROOT).as_posix()
  if stat.S_ISLNK(s.st_mode):links.append(dict(path=rel,target=os.readlink(p)));continue
  if s.st_dev!=dev:errors.append(dict(path=rel,error='different-device/mount skipped'));continue
  if stat.S_ISDIR(s.st_mode):dirs.append(rel);walk(p);continue
  if not stat.S_ISREG(s.st_mode):errors.append(dict(path=rel,error='nonregular file'));continue
  row=dict(path=rel,bytes=s.st_size,blocks=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
  rows.append(row);top=rel.split('/')[0];tops[top]+=s.st_size;allocated[top]+=s.st_blocks*512;classes[p.suffix.lower()]+=s.st_size;names[p.name]+=s.st_size
  inodes.setdefault((s.st_dev,s.st_ino),[]).append(row)
walk(ROOT)
with gzip.open(OUT/'INVENTORY_BEFORE.json.gz','wt') as f:json.dump(rows,f,separators=(',',':'))
for name,obj in [('SYMLINKS.json',links),('TRAVERSAL_EXCEPTIONS.json',errors),('HARDLINKS.json',[[r['path'] for r in v] for v in inodes.values() if len(v)>1 or v[0]['nlink']>1]),('DIRECTORIES.json.gz',dirs)]:
 if name.endswith('.gz'):
  with gzip.open(OUT/name,'wt') as f:json.dump(obj,f)
 else:(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
summary=dict(root=str(ROOT),files=len(rows),directories=len(dirs),symlinks=len(links),errors=len(errors),logical_bytes=sum(tops.values()),allocated_file_blocks_bytes=sum(allocated.values()),unique_inode_logical_bytes=sum(v[0]['bytes'] for v in inodes.values()),unique_inode_allocated_bytes=sum(v[0]['blocks'] for v in inodes.values()),by_top_level=[dict(path=k,bytes=v,blocks=allocated[k]) for k,v in tops.most_common()],by_extension=classes.most_common(),largest_basenames=names.most_common(70),largest_files=sorted(rows,key=lambda r:-r['bytes'])[:50],start_epoch=start,seconds=time.time()-start)
(OUT/'SUMMARY_BEFORE.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'FREE_SPACE_BEFORE.txt').write_text(subprocess.check_output(['df','-k',str(ROOT)],text=True))
print(json.dumps({k:v for k,v in summary.items() if k not in ('largest_files','by_extension','largest_basenames')},indent=2));print('LARGEST BASENAMES',names.most_common(35));print('EXTENSIONS',classes.most_common(25))
