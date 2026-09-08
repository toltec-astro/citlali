"""Build only SCI-FRUIT r0.4 documents from local TeX resources; no science runs."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
VIEWS=('core','rationale','ecs')
CORE_ID='SCI-FRUIT-NORMATIVE-CORE v0.1/r0.4'
EPOCH='1788825600'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory(paths): return ''.join(f'{sha(ROOT/p)}  {p}\n' for p in sorted(paths))
def split_hash(value): return r'\allowbreak '.join(value[i:i+8] for i in range(0,len(value),8))
def bind_sources():
    normative=[p.relative_to(ROOT).as_posix() for p in (ROOT/'src/common').glob('*.tex')]+['src/core_body.tex']
    assert len(normative)==7
    (ROOT/'identities/CORE_SOURCES.sha256').write_text(inventory(normative))
    controls={
      'StageAHash':ROOT/'inputs/r0.2/inputs/scientific_core/r0.4-amended/SCI-FRUIT-v0.1-stage-a-r0.4-amended-owner-review.tar.gz',
      'DirectiveHash':ROOT/'inputs/OWNER_DIRECTIVE_R0.4.txt',
      'ApprovalHash':ROOT/'inputs/r0.2/inputs/scientific_core/SCI_FRUIT_STAGE_A_R0.4_AMENDED_OWNER_APPROVAL_2026-09-07.md',
      'PriorHash':ROOT/'inputs/r0.2/SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz',
      'FreezeHash':ROOT/'inputs/r0.2/inputs/scientific_core/SCI_FRUIT_STAGE_A_R0.4_AMENDED_OWNER_FREEZE_2026-09-07.md',
      'CoreHash':ROOT/'identities/CORE_SOURCES.sha256'}
    for view in VIEWS:
        closure=sorted(set(normative+[f'src/{view}.tex',f'src/{view}_body.tex','src/preamble.tex','src/cover.tex']))
        p=ROOT/f'identities/{view.upper()}_DOCUMENT_SOURCES.sha256';p.write_text(inventory(closure))
        controls[view.title()+'SourceHash']=p
    text='% Deterministic derived digest bindings; excluded from their own source inventories.\n'
    text+=''.join(r'\newcommand{'+chr(92)+name+'}{'+split_hash(sha(p))+'}\n' for name,p in controls.items())
    (ROOT/'src/bindings.tex').write_text(text)
    return {name:sha(p) for name,p in controls.items()}
def resources(source):
    bundle=ROOT/'qa/cache/texbundle'
    if not (bundle/'SHA256SUM').exists():
        if not source.is_dir():raise SystemExit('An existing local TeX resource directory is required; no network fetch is permitted.')
        shutil.copytree(source,bundle,dirs_exist_ok=True)
    rows=[]; digest=hashlib.sha256()
    for p in sorted(bundle.iterdir()):
        if p.is_file() and p.name!='SHA256SUM':
            rows.append(f'{sha(p)}  {p.name}\n');digest.update(p.name.encode()+b'\0'+p.read_bytes())
    (bundle/'SHA256SUM').write_text(digest.hexdigest()+'\n')
    (ROOT/'identities/TEX_RESOURCES.sha256').write_text(''.join(rows))
    return bundle,digest.hexdigest(),len(rows)
def env_for(cache):
    for name in ('mpl','xdg','tectonic'): (cache/name).mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(cache/'mpl'),XDG_CACHE_HOME=str(cache/'xdg'),TECTONIC_CACHE_DIR=str(cache/'tectonic'),SOURCE_DATE_EPOCH=EPOCH,FORCE_SOURCE_DATE='1')
    return env
def compile_views(src,out,cache,bundle):
    out.mkdir(parents=True,exist_ok=True);env=env_for(cache);res={}
    for view in VIEWS:
        command=['tectonic','--bundle',str(bundle),'--only-cached','--keep-logs','--keep-intermediates','--outdir',str(out),view+'.tex']
        result=subprocess.run(command,cwd=src,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/f'{view}.console.txt').write_text(result.stdout)
        if result.returncode:raise SystemExit(result.stdout)
        log=(out/f'{view}.log').read_text(errors='replace')
        diagnostics=[line for line in log.splitlines() if any(x in line for x in ['Overfull','Underfull','Undefined control sequence','undefined references','Missing character','LaTeX Error','LaTeX Warning','Package hyperref Warning'])]
        res[view]={'sha256':sha(out/f'{view}.pdf'),'diagnostics':diagnostics}
        print(view, 'compiled',len(diagnostics),'diagnostic lines',flush=True)
    return res
def tool_identity(cmd,args):
    exe=Path(shutil.which(cmd) or cmd);result=subprocess.run([str(exe)]+args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    return {'path':str(exe),'resolved_path':str(exe.resolve()),'sha256':sha(exe),'version':result.stdout.strip()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path('/Users/gwilson/Library/Caches/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c'));ap.add_argument('--render',action='store_true');ap.add_argument('--clean',action='store_true');args=ap.parse_args()
    for name in ['identities','pdf','reports','qa/cache','qa/build','qa/renders']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    bindings=bind_sources();bundle,bhash,bcount=resources(args.bundle)
    built=compile_views(ROOT/'src',ROOT/'qa/build',ROOT/'qa/cache',bundle)
    for view in VIEWS:shutil.copyfile(ROOT/f'qa/build/{view}.pdf',ROOT/f'pdf/{view}.pdf')
    report={'identity':'SCI-FRUIT-DOC-BUILD-R0.4-2026-09-08','scope':'Document build only; no numerical conformance or experiments.','source_date_epoch':EPOCH,'bindings':bindings,'tex_bundle_content_digest':bhash,'tex_resource_count':bcount,'tex_resource_inventory_sha256':sha(ROOT/'identities/TEX_RESOURCES.sha256'),'tools':{name:tool_identity(name,flags) for name,flags in [('tectonic',['--version']),('pdftoppm',['-v']),('pdfinfo',['-v'])]},'python':{'executable':sys.executable,'resolved_executable':str(Path(sys.executable).resolve()),'executable_sha256':sha(Path(sys.executable)),'version':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['pypdf','Pillow']}},'views':built}
    if args.clean:
        clean=ROOT/'qa/clean'
        if clean.exists():shutil.rmtree(clean)
        (clean/'src').mkdir(parents=True)
        closure=set()
        for view in VIEWS:
            for line in (ROOT/f'identities/{view.upper()}_DOCUMENT_SOURCES.sha256').read_text().splitlines():closure.add(line.split('  ',1)[1])
        closure.add('src/bindings.tex')
        for rel in sorted(closure):
            p=clean/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
        cb=compile_views(clean/'src',clean/'build',clean/'cache',bundle)
        report['clean_build']={'copied_source_files':len(closure),'independent_source_and_build_directories':True,'fresh_format_cache':True,'views':cb,'all_pdf_bytes_identical':all(cb[v]['sha256']==built[v]['sha256'] for v in VIEWS)}
        if not report['clean_build']['all_pdf_bytes_identical']:raise SystemExit('Clean-build PDF byte mismatch; inspect before any claim of reproducibility.')
    if args.render:
        for view in VIEWS:
            for old in (ROOT/'qa/renders').glob(view+'-[0-9]*.png'):old.unlink()
            subprocess.run(['pdftoppm','-r','100','-png',str(ROOT/f'pdf/{view}.pdf'),str(ROOT/f'qa/renders/{view}')],env=env_for(ROOT/'qa/cache'),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        report['render']={'tool':'pdftoppm','dpi':100,'views':list(VIEWS),'visual_inspection':'not established by build; separate page record required'}
    (ROOT/'reports/BUILD_RECORD.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Build record written; visual inspection is separate.',flush=True)
if __name__=='__main__':main()
