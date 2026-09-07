"""Compile/render only this new document package; no scientific computation."""
from pathlib import Path
import argparse, hashlib, os, shutil, subprocess
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--bundle',type=Path,default=Path('/Users/gwilson/Library/Caches/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c'))
parser.add_argument('--render',action='store_true')
args=parser.parse_args()
cache=root/'qa/cache'; bundle=cache/'texbundle'
for p in [root/'qa/build',root/'qa/renders',cache/'mpl',cache/'xdg',cache/'tectonic',root/'pdf']:
    p.mkdir(parents=True,exist_ok=True)
if not (bundle/'SHA256SUM').exists():
    if not args.bundle.is_dir(): raise SystemExit('Supply an existing local TeX resource directory with --bundle; no network fetching is performed.')
    shutil.copytree(args.bundle,bundle,dirs_exist_ok=True)
    h=hashlib.sha256()
    for p in sorted(bundle.iterdir()):
        if p.is_file() and p.name!='SHA256SUM': h.update(p.name.encode()+b'\0'+p.read_bytes())
    (bundle/'SHA256SUM').write_text(h.hexdigest()+'\n')
env=os.environ.copy()
env.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(cache/'mpl'),XDG_CACHE_HOME=str(cache/'xdg'),TECTONIC_CACHE_DIR=str(cache/'tectonic'),SOURCE_DATE_EPOCH='1788739200')
for view in ['scientist','engineering']:
    command=['tectonic','--bundle',str(bundle),'--only-cached','--keep-logs','--keep-intermediates','--outdir',str(root/'qa/build'),view+'.tex']
    result=subprocess.run(command,cwd=root/'src',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (root/f'qa/build/{view}.console.txt').write_text(result.stdout)
    if result.returncode: raise SystemExit(result.stdout)
    log=(root/f'qa/build/{view}.log').read_text(errors='replace')
    rejected=['Overfull','Undefined control sequence','undefined references','Missing character','LaTeX Error']
    if any(x in log for x in rejected): raise SystemExit(f'{view}: document diagnostics need repair; inspect qa/build/{view}.log')
    shutil.copyfile(root/f'qa/build/{view}.pdf',root/f'pdf/{view}.pdf')
    print(view+': compiled; no rejected TeX diagnostics')
    if args.render:
        subprocess.run(['pdftoppm','-r','100','-png',str(root/f'pdf/{view}.pdf'),str(root/f'qa/renders/{view}')],env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        print(view+': all pages rendered with Poppler at 100 dpi')
