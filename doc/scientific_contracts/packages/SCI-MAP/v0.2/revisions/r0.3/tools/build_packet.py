#!/usr/bin/env python3
"""Document-only identity preparation and independent PDF inspection.

No application/configuration/schema/test/operational paths are consumed.
A build attempt is immutable: prepare into a previously nonexistent directory.
The entry source and shared-core digests exclude generated cover fragments;
the final delivery manifest binds the complete transitive sources and outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

VIEWS = {
    'formal': ('formal-scientific-engineering-contract', 'SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.2'),
    'rationale': ('scientific-rationale', 'SCI-MAP-SCIENTIFIC-RATIONALE-v0.2'),
    'engineering': ('engineering-conformance', 'SCI-MAP-ENGINEERING-CONFORMANCE-v0.2'),
}
CORE = ['SCI-MAP-v0.2_SHARED_AUTHORITY_r0.3.tex'] + [f'common/{x}.tex' for x in ('notation','definitions','equations','assumptions','requirements','edge_cases')]
CORE_ID = 'SCI-MAP-v0.2-SHARED-AUTHORITY/r0.3'
MANIFEST = 'doc/scientific_contracts/studies/SCI_MAP_POST_FREEZE_RECONCILIATION_2026-09-07/AUTHOR_REFERENCES.json'
DIRECTIVE = 'owner/SCI_MAP_R03_OWNER_DIRECTIVE.txt'
TECTONIC = '/opt/homebrew/bin/tectonic'
POPPLER = '/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def core_digest(src):
    h = hashlib.sha256()
    for name in CORE:
        blob = (src/name).read_bytes()
        h.update(name.encode() + b'\0' + str(len(blob)).encode('ascii') + b'\0' + blob + b'\0')
    return h.hexdigest()


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {'argv': command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def tool_info(path, args):
    result = run([path] + args)
    return {'invoked_path': path, 'resolved_path': str(Path(path).resolve()), 'sha256': sha(path), 'version': result}


def textext(s):
    if ' ' in s:
        replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
        return r'\texttt{' + ''.join(replacements.get(c,c) for c in s) + '}'
    return r'\nolinkurl{' + s + '}'


def digest_tex(s):
    return r'\texttt{' + r'\allowbreak{}'.join(s[i:i+16] for i in range(0,64,16)) + '}'


def prepare(args):
    package, attempt = Path(args.package).resolve(), Path(args.attempt).resolve()
    if attempt.exists():
        raise SystemExit('Refusing to overwrite an existing build attempt')
    src = package/'src'
    inventory = json.loads((package/'AUTHOR_PACKET_INVENTORY.json').read_text())
    for name, expected in inventory['input_files'].items():
        if sha(package/'inputs'/name) != expected:
            raise SystemExit(f'Input digest mismatch: {name}')
    attempt.mkdir(parents=True)
    shutil.copytree(src, attempt/'src')
    (attempt/'src'/'identity').mkdir(exist_ok=True)
    authority_sha = core_digest(src)
    manifest_sha = sha(package/'inputs'/MANIFEST)
    manifest_identity = json.loads((package/'inputs'/MANIFEST).read_text())['identity']
    directive_sha = sha(package/'inputs'/DIRECTIVE)
    inventory_sha = sha(package/'AUTHOR_PACKET_INVENTORY.json')
    build_id = 'SCI-MAP-R03-DOCUMENT-BUILD/' + attempt.name
    data = {
        'identity': build_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'document build attempt; no scientific acceptance or application evidence',
        'shared_authority': {'identity': CORE_ID, 'sha256': authority_sha,
            'algorithm': 'For each ordered core path: UTF8(path), NUL, ASCII(raw byte length), NUL, raw bytes, NUL; SHA256 of concatenation.',
            'ordered_sources': {p: sha(src/p) for p in CORE}},
        'original_author_manifest': {'identity': manifest_identity, 'path': 'inputs/'+MANIFEST, 'sha256': manifest_sha},
        'author_packet_inventory': {'identity': inventory['identity'], 'path': 'AUTHOR_PACKET_INVENTORY.json', 'sha256': inventory_sha},
        'owner_directive': {'identity': inventory['owner_directive_identity'], 'path': 'inputs/'+DIRECTIVE, 'sha256': directive_sha},
        'entry_sources': {}, 'generated_cover_sources': {},
        'environment': {'platform': platform.platform(), 'python': sys.version, 'python_executable': sys.executable, 'python_executable_sha256': sha(sys.executable)},
        'tools': {'tectonic': tool_info(TECTONIC, ['--version']),
                  'pdftoppm': tool_info(POPPLER+'/pdftoppm',['-v']),
                  'pdfinfo': tool_info(POPPLER+'/pdfinfo',['-v'])},
        'recipe_sha256': sha(__file__),
        'compile_recipe': [TECTONIC, '--only-cached', '--untrusted', '--keep-logs', '--keep-intermediates', '--makefile-rules', '<attempt>/<view>/dependencies.mk', '--outdir', '<attempt>/<view>', '<attempt>/src/<entry>.tex'],
        'compile_environment': {'SOURCE_DATE_EPOCH': '1788825600', 'TZ':'UTC'},
        'reproducibility_claim': 'Exact observed tools, source bytes, generated fragments, commands and outputs are bound. Cached TeX assets are separately inventoried from .fls if emitted; no cross-host bit reproducibility is asserted.'
    }
    for view, (entry, _) in VIEWS.items():
        source_sha = sha(src/(entry+'.tex'))
        data['entry_sources'][view] = {'path': 'src/'+entry+'.tex', 'sha256':source_sha}
        rows = [
            ('Scientific owner', 'Grant Wilson'),
            ('Contract / document', 'SCI-MAP v0.2 / r0.3'),
            ('Status / date', 'Candidate for scientific-owner disposition / 2026-09-08'),
            ('Shared authority', textext(CORE_ID)),
            ('Core bundle SHA-256', digest_tex(authority_sha)),
            ('Original author manifest', textext(data['original_author_manifest']['identity'])+' (56 references)'),
            ('Manifest SHA-256',digest_tex(manifest_sha)),
            ('Packet addendum',textext(inventory['identity'])),
            ('Addendum SHA-256',digest_tex(inventory_sha)),
            ('Owner directive',textext(inventory['owner_directive_identity'])),
            ('Directive SHA-256',digest_tex(directive_sha)),
            ('Document entry source',textext('src/'+entry+'.tex')),
            ('Entry-source SHA-256',digest_tex(source_sha)),
            ('Build record',textext(build_id)),
        ]
        # The shared-core/entry hashes are acyclic; final source manifest seals this fragment.
        cover = '% Mechanically generated; scientific prose is owned by the independent author.\n'
        cover += '\\begin{tabularx}{\\linewidth}{@{}p{0.27\\linewidth}X@{}}\n'
        cover += '\n'.join(a+' & '+b+r'\\[0.2em]' for a,b in rows)
        cover += '\n\\end{tabularx}\n'
        cover_path=attempt/'src'/'identity'/(view+'.tex')
        cover_path.write_text(cover)
        data['generated_cover_sources'][view]={'path':'src/identity/'+view+'.tex','sha256':sha(cover_path)}
        (attempt/view).mkdir()
    for source in (attempt/'src').rglob('*.tex'):
        for name in re.findall(r'\\input\{([^{}]+)\}',source.read_text()):
            target=attempt/'src'/name
            if not target.suffix: target=target.with_suffix('.tex')
            if not target.is_file() or not target.resolve().is_relative_to(attempt/'src'):
                raise SystemExit(f'Unresolved or out-of-packet TeX input: {source.name}: {name}')
    dump(attempt/'BUILD_ATTEMPT.json',data)
    print(json.dumps({'attempt':str(attempt),'build_identity':build_id,'core_sha256':authority_sha}))


def inspect(args):
    import pypdf, PIL
    from pypdf import PdfReader
    from PIL import Image, ImageOps, ImageDraw
    attempt=Path(args.attempt).resolve()
    record=json.loads((attempt/'BUILD_ATTEMPT.json').read_text())
    report={'inspection_libraries': {'pypdf': {'version':pypdf.__version__, 'module_sha256':sha(pypdf.__file__)}, 'Pillow': {'version':PIL.__version__, 'module_sha256':sha(PIL.__file__)}}, 'identity':record['identity']+'/inspection','build_attempt_sha256':sha(attempt/'BUILD_ATTEMPT.json'),'views':{},'interpretation':'Document compilation/rendering evidence only; no prospective ECS result is executed by this inspection.'}
    for view,(entry,name) in VIEWS.items():
        out=attempt/view
        pdf=out/(entry+'.pdf')
        log=out/(entry+'.log')
        if not pdf.exists():
            raise SystemExit(f'Missing output: {pdf}')
        reader=PdfReader(pdf,strict=True)
        texts=[p.extract_text() or '' for p in reader.pages]
        (out/'all-pages.txt').write_text('\n\f\n'.join(texts))
        logtext=log.read_text(errors='replace') if log.exists() else ''
        bad=[line for line in logtext.splitlines() if re.search(r'Overfull|undefined references|Reference .+ undefined|Citation .+ undefined|multiply.?defined|Missing character|^!',line,re.I)]
        metadata={str(k):str(v) for k,v in reader.metadata.items()}
        info=run([POPPLER+'/pdfinfo',str(pdf)])
        dump(out/'pdfinfo.json',info)
        renders=out/'render'
        renders.mkdir(exist_ok=False)
        rendered=run([POPPLER+'/pdftoppm','-r','100','-png',str(pdf),str(renders/'page')])
        dump(out/'render-command.json',rendered)
        if rendered['exit_code']:
            raise SystemExit(f'Render failed: {view}')
        pages=sorted(renders.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
        if len(pages)!=len(reader.pages):
            raise SystemExit(f'Render page count mismatch: {view}')
        # Every page is retained at 100 dpi; six-page sheets support complete visual inspection.
        for start in range(0,len(pages),6):
            sheet=Image.new('RGB',(1260,1740),'#d8dee5')
            draw=ImageDraw.Draw(sheet)
            for j,p in enumerate(pages[start:start+6]):
                im=Image.open(p).convert('RGB')
                im.thumbnail((610,540))
                x=(j%2)*630+(630-im.width)//2
                y=(j//2)*580+30
                sheet.paste(im,(x,y))
                draw.text(((j%2)*630+20,(j//2)*580+8),f'{view} page {start+j+1}',fill='black')
            sheet.save(out/f'contact-{start+1:03d}.png')
        # Bound consumed filesystem inputs from recorder when available, without asserting a bundled TeX closure.
        recorder={}
        for fls in out.glob('*.fls'):
            for line in fls.read_text(errors='replace').splitlines():
                if line.startswith('INPUT '):
                    p=Path(line[6:]); p=p if p.is_absolute() else attempt/'src'/p
                    if p.is_file(): recorder[str(p)]=sha(p)
        dependency_sources=[]
        makefile=out/'dependencies.mk'
        if makefile.exists():
            raw_rules=makefile.read_text()
            inputs=raw_rules.split(' : ',1)[1].replace('\\\n',' ').split()
            for item in inputs:
                reported=Path(item)
                resolved=reported
                if not resolved.is_file() and reported.is_relative_to(out):
                    resolved=attempt/'src'/reported.relative_to(out)
                if not resolved.is_file() or not resolved.is_relative_to(attempt/'src'):
                    raise SystemExit(f'Unresolved or out-of-source document dependency: {item}')
                dependency_sources.append({'compiler_reported_path':item,'source_path':'src/'+resolved.relative_to(attempt/'src').as_posix(),'sha256':sha(resolved)})
        report['views'][view]={'dependency_sources':dependency_sources,'dependency_rule_note':'Raw compiler makefile retains its outdir-relative spellings. Those spellings are explicitly resolved against the recorded source working directory; only existing in-packet sources are admitted. Cached TeX resources are tool-environment inputs and are not asserted as a portable bundle closure.', 'pdf_path':str(pdf),'delivery_filename':name+'.pdf','pdf_sha256':sha(pdf),'pages':len(texts),'nonempty_pages':all(t.strip() for t in texts),'metadata':metadata,'tex_errors_or_overfull':bad,'pdfinfo_exit_code':info['exit_code'],'render_exit_code':rendered['exit_code'],'rendered_pages':len(pages),'render_sha256':{p.name:sha(p) for p in pages},'recorder_inputs':recorder,'log_sha256':sha(log) if log.exists() else None,'visual_inspection':'Separate all-page visual review required; this automated record does not assert that result.'}
    dump(attempt/'PDF_BUILD_INSPECTION.json',report)
    print(json.dumps({k:{'pages':v['pages'],'tex_errors_or_overfull':v['tex_errors_or_overfull'],'metadata':v['metadata']} for k,v in report['views'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('prepare'); p.add_argument('--package',required=True); p.add_argument('--attempt',required=True); p.set_defaults(func=prepare)
    p=subs.add_parser('inspect'); p.add_argument('--attempt',required=True); p.set_defaults(func=inspect)
    args=parser.parse_args(); args.func(args)
