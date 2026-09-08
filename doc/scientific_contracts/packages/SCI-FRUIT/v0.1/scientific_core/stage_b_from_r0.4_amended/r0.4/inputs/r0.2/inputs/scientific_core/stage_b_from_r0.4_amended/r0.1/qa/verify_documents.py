"""Bounded artifact checks only. No codebase test or scientific computation."""
from pathlib import Path
from pypdf import PdfReader
import hashlib,json,re,unicodedata
root=Path(__file__).resolve().parents[1]
packet=root.parents[1]/'r0.4-amended'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=(packet/'AUTHOR_INPUT_MANIFEST.md').read_bytes()
assert len(manifest)==18734 and hashlib.sha256(manifest).hexdigest()=='8d4099d31f48cad1e5bab0a04acffb377520ae2e619896b59e19ef08fd35c37d'
assert (packet/'AUTHOR_INPUT_MANIFEST.sha256').read_text()=='8d4099d31f48cad1e5bab0a04acffb377520ae2e619896b59e19ef08fd35c37d  AUTHOR_INPUT_MANIFEST.md\n'
allowed=[]
for line in manifest.decode().splitlines():
    if line.startswith('| `') and 'Author may read only after' in line:
        c=[x.strip() for x in line.split('|')[1:-1]];p=packet/c[0].strip('`')
        assert p.stat().st_size==int(c[2]) and sha(p)==c[3].strip('`')
        allowed.append(p.resolve())
assert len(allowed)==11
# Exact author-recorded source/control identities; no manager-only payload opens.
controls=[packet/'AUTHOR_INPUT_MANIFEST.md',packet/'AUTHOR_INPUT_MANIFEST.sha256',packet.parent/'SCI_FRUIT_STAGE_A_R0.4_AMENDED_OWNER_APPROVAL_2026-09-07.md']
for name,size,h in re.findall(r'\| \[([^]]+)\]\([^)]*\) \| (\d+) \| `([0-9a-f]{64})` \|',(root/'SOURCE_IDENTITIES.md').read_text()):
    paths=[p for p in allowed+controls if p.name==name];assert len(paths)==1
    assert paths[0].stat().st_size==int(size) and sha(paths[0])==h
assert (root/'src/PTC_APPLICATION_REFERENCE.tex').read_bytes()==(packet/'PTC_APPLICATION_REFERENCE.tex').read_bytes()
review=(root/'review/ROUND_2.md').read_text()
reviewed=re.findall(r'\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|',review)
assert len(reviewed)==12
for name,size,h in reviewed:
    p=root/name;assert p.stat().st_size==int(size) and sha(p)==h,(name,'changed since independent review')
modules=['notation','definitions','equations','assumptions','requirements','edge_cases']
pre=(root/'src/preamble.tex').read_text()
assert sorted(re.findall(r'\\input\{common/([^}]+)\.tex\}',pre))==sorted(modules)
texts={n:(root/f'src/common/{n}.tex').read_text() for n in modules}
assert re.findall(r'\\Pred\{(\d{3})\}',texts['edge_cases'])==[f'{n:03}' for n in range(1,21)]
assert re.findall(r'\\hypertarget\{req-(\d{3})\}',texts['requirements'])==[f'{n:03}' for n in range(1,25)]
assert len(re.findall(r'^\| SCI-FRUIT-REQ-\d{3} \|',(root/'REQUIREMENT_CROSSWALK.md').read_text(),re.M))==24
assert all(f'RF-{n:02}' in texts['equations'] for n in range(1,8))
# All canonical content units appear once in each wrapper's expanded view.
narrative=['OpeningPage','TypePage','CompositionPage','SupportPage','ResponseLocalPage','ResponseOuterPage','UncertaintyPage','StatePage','ContinuationPage','CompletionPage']
appendices=['PredictionsAppendix','NotationAppendix','PremiseAppendix','OperationsAppendix','TerminalAppendix','BoundaryAppendix','PTCAppendix']
expected=narrative+appendices+['RequirementsAppendix','MethodAppendix','AuthorityAppendix']
for v in ['scientist','engineering']:
    wrapper=(root/f'src/{v}.tex').read_text()
    assert wrapper.count(r'\input{preamble.tex}')==1
    expanded=wrapper.replace(r'\ScientificNarrative',''.join('\\'+x for x in narrative)).replace(r'\ScientificAppendices',''.join('\\'+x for x in appendices))
    assert all(expanded.count('\\'+x)==1 for x in expected)
    for match in re.finditer(r'\\input\{([^}]+)\}',wrapper+pre):
        assert (root/'src'/match.group(1)).is_file(),match.group(1)
# No unresolved references in final TeX logs; exact canonical body parity in PDFs.
pdfs={v:PdfReader(root/f'pdf/{v}.pdf') for v in ['scientist','engineering']}
def normalized_page(page):
    lines=unicodedata.normalize('NFKC',page.extract_text()).splitlines()[1:]
    lines=[l for l in lines if not l.startswith('Owner-review draft;')]
    lines[0]=re.sub(r'^\S+\s+','',lines[0])
    text=re.sub(r'(Section|Appendix)\s+[A-Z0-9]+\b',r'\1 REF','\n'.join(lines))
    return re.sub(r'\s+','',text)
pairs=[(i,i+7) for i in range(1,22)]+[(i,i-21) for i in range(22,29)]
for v,pdf in pdfs.items():
    assert len(pdf.pages)==28
    log=(root/f'qa/build/{v}.log').read_text(errors='replace')
    assert not any(t in log for t in ['Warning','Overfull','Underfull','Undefined control sequence','undefined references','Missing character','LaTeX Error'])
    alltext=unicodedata.normalize('NFKC','\n'.join(p.extract_text() for p in pdf.pages))
    for n in range(1,21): assert f'SCI-FRUIT-PRED-{n:03}:' in alltext
    assert '??' not in alltext
    assert 'unavailable_pending_separate_owner_approval' in re.sub(r'\s+','',alltext)
    assert len(pdf.named_destinations)>=28
    for dest in pdf.named_destinations.values():
        assert 0<=pdf.get_destination_page_number(dest)<28
for sp,ep in pairs:
    assert normalized_page(pdfs['scientist'].pages[sp-1])==normalized_page(pdfs['engineering'].pages[ep-1]),(sp,ep)
assert 'Conditional predictions: composition limits' in pdfs['scientist'].pages[10].extract_text()
# Local Markdown links only: outside-package targets must be exact permitted inputs/controls.
links=0
for p in list(root.glob('*.md'))+list((root/'review').glob('*.md')):
    for url in re.findall(r'\[[^]]*\]\(([^)]+)\)',p.read_text()):
        if url.startswith(('http:','https:','mailto:')): raise AssertionError(('unexpected external link',p.name,url))
        target=(p.parent/url.split('#')[0]).resolve()
        assert target.is_relative_to(root) or target in [x.resolve() for x in allowed+controls],(p.name,url,'outside allowed link set')
        # Packaging controls are created in the final packaging step, then checked there.
        links+=1
        if target.name in ['DOCUMENT_CHECKS.json','ARTIFACT_MANIFEST.md','ARTIFACT_MANIFEST.sha256','SCI-FRUIT-v0.1-stage-b-r0.1-owner-review.tar.gz','SCI-FRUIT-v0.1-stage-b-r0.1-owner-review.tar.gz.sha256'] and not target.exists():continue
        assert target.exists(),(p.name,url,'missing link')
visual=json.loads((root/'qa/PAGE_INSPECTION.json').read_text())
for v in pdfs:
    assert visual[v]['pdf_sha256']==sha(root/f'pdf/{v}.pdf')
    assert visual[v]['inspected_pages']==list(range(1,29))
    assert visual[v]['open_layout_defects']==0
report={'scope':'document verification only; no numerical qualification','permitted_payloads_verified':11,'reviewed_snapshot_files_verified':len(reviewed),'canonical_modules':6,'stable_requirements':24,'stable_predictions':20,'pdf_pages':{'scientist':28,'engineering':28},'scientist_main_narrative_pages':10,'canonical_content_pairs_equal':28,'source_and_pdf_references_resolved':True,'local_links_checked':links,'all_56_pages_visually_inspected':True,'open_layout_defects':0,'tex_diagnostics':0,'pdf_sha256':{v:sha(root/f'pdf/{v}.pdf') for v in pdfs}}
(root/'qa/DOCUMENT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
