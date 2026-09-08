"""Document-only source/authority, PDF, index and extracted-link verification."""
from pathlib import Path
import hashlib,json,re,unicodedata
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def inventory(base,p):
    rows=[]
    for line in p.read_text().splitlines():
        h,rel=line.split('  ',1);q=(base/rel).resolve()
        assert q.is_relative_to(base.resolve()) and q.is_file() and sha(q)==h,(p,rel)
        rows.append(rel)
    assert rows==sorted(set(rows)),p
    return rows
def main():
    inputs=inventory(ROOT,ROOT/'identities/INPUT_FILES.sha256')
    assert set(inputs)=={str(p.relative_to(ROOT)) for p in (ROOT/'inputs').rglob('*') if p.is_file()}
    hist=ROOT/'inputs/r0.2'
    assert sha(hist/'SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz')=='6d552e49510c92fff65d9c835f8f31cd21f8ef50bed4f1a24ff355cfce7523a4'
    assert (hist/'SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz').stat().st_size==2519780
    hrows=re.findall(r'^\| `([^`]+)` \| [^|]+ \| (\d+) \| `([0-9a-f]{64})` \|$',(hist/'ARTIFACT_MANIFEST.md').read_text(),re.M)
    assert len(hrows)==107
    for rel,size,h in hrows:p=hist/rel;assert p.stat().st_size==int(size) and sha(p)==h,rel
    for p in (hist/'identities').glob('*.sha256'):
        if p.name!='TEX_RESOURCES.sha256':inventory(hist,p)
    sidecars=0
    for p in (ROOT/'inputs').rglob('*.sha256'):
        if 'identities' in p.relative_to(ROOT).parts:continue
        for line in p.read_text().splitlines():
            h,rel=line.split(None,1);q=(p.parent/rel.strip()).resolve()
            assert q.is_relative_to(ROOT) and q.is_file() and sha(q)==h,(p,rel)
            sidecars+=1
    ptc=hist/'inputs/scientific_core/r0.4-amended/PTC_APPLICATION_REFERENCE.tex'
    assert sha(ptc)=='75116261a33ec1adbb092d38da43a590624a08428e51cac32ee0c4a34f216a59'
    prior=ROOT/'inputs/r0.3'
    assert sha(prior/'SCI-FRUIT-v0.1-stage-b-r0.3-owner-review.tar.gz')=='bb6beeaaa613017c7a683c1ed3a520129c5dc9d6e93109881b8881000e3977ff'
    assert (prior/'SCI-FRUIT-v0.1-stage-b-r0.3-owner-review.tar.gz').stat().st_size==5491010
    for p in (prior/'identities').glob('*.sha256'):inventory(prior,p)
    boundary=hist/'inputs/scientific_core/r0.4-amended/BOUNDARIES_AND_CONVENTIONS.md'
    assert sha(boundary)=='0321336a895348c29f94e61521b8523325fe8a598c81d0f3477a1def82828c7e'
    for rel in ['src/common/equations.tex','src/common/assumptions.tex','src/common/edge_cases.tex','src/core_body.tex']:
        assert (ROOT/rel).read_bytes()==(prior/rel).read_bytes(),rel
    original=(prior/'src/common/definitions.tex').read_text()
    expected=original.replace('calibration; PTC retains cleaning, operator families and application state;',
        'calibration; PTC retains its cleaning operators, analysis/gridding-coefficient\nfamilies and fitted/resolved/applied state;')
    expected=expected.replace('A STOKES label supplies no authority. Processed occurrence identity requires\nexact PTC output/modal-application-coordinate/QC and compatible coordinates, not row/time labels.\nMAP permission does not imply JINC operator-family permission.',
        'A STOKES label supplies no authority. A selected MAP/JINC parent route retains\nexact PTC transformed-output identity, the analysis/gridding-coefficient and QC\nidentities required by that route, and compatible coordinate association.\nRow/time labels do not establish the same processed occurrence. Internal modal\napplication coordinates remain separate and do not substitute for those\nhandoff facts. Permission for MAP to use a PTC analysis/gridding-coefficient\nfamily does not imply permission for JINC to use that family.')
    assert expected==(ROOT/'src/common/definitions.tex').read_text()
    oldreq=(prior/'src/common/requirements.tex').read_text()
    assert oldreq.replace('sole current r0.3 review-candidate','sole current r0.4 review-candidate').replace('supersedes r0.2 as the review candidate','supersedes r0.3 as the review candidate')==(ROOT/'src/common/requirements.tex').read_text()
    oldn=(prior/'src/common/notation.tex').read_text();newn=(ROOT/'src/common/notation.tex').read_text()
    assert oldn[oldn.index(r'\textbf{A}'):]==newn[newn.index(r'\textbf{A}'):]
    oldr=(prior/'src/rationale_body.tex').read_text();newr=(ROOT/'src/rationale_body.tex').read_text()
    marker=r'\Unit{Three model roles'
    assert oldr[oldr.index(marker):]==newr[newr.index(marker):]
    olde=(prior/'src/ecs_body.tex').read_text()
    assert olde.replace('v0.1/r0.3','v0.1/r0.4').replace('Complete r0.3 review-candidate core','Complete r0.4 review-candidate core').replace('not relabeled as r0.3 output','not relabeled as r0.4 output')==(ROOT/'src/ecs_body.tex').read_text()
    counts={n:len(inventory(ROOT,ROOT/'identities'/n)) for n in ['CORE_SOURCES.sha256','CORE_DOCUMENT_SOURCES.sha256','RATIONALE_DOCUMENT_SOURCES.sha256','ECS_DOCUMENT_SOURCES.sha256']}
    assert counts['CORE_SOURCES.sha256']==7
    mods={p.stem:p.read_text() for p in (ROOT/'src/common').glob('*.tex')};assert len(mods)==6
    req=re.findall(r'\\Req\{(\d{3})\}',mods['requirements']);pred=re.findall(r'\\Pred\{(\d{3})\}',mods['edge_cases']);asm=re.findall(r'\\item\[SCI-FRUIT-ASM-(\d{3}):',mods['assumptions'])
    assert req==[f'{i:03}' for i in range(1,25)] and pred==[f'{i:03}' for i in range(1,21)] and asm==[f'{i:03}' for i in range(1,7)]
    macros=re.findall(r'\\newcommand\{\\(Norm\w+)\}','\n'.join(mods.values()));body=(ROOT/'src/core_body.tex').read_text()
    assert sorted(re.findall(r'\\(Norm\w+)',body))==sorted(macros)
    preamble=(ROOT/'src/preamble.tex').read_text();assert sorted(re.findall(r'\\input\{common/([^}]+)\.tex\}',preamble))==sorted(mods)
    for p in (ROOT/'src').rglob('*.tex'):
        assert all(ord(c)>=32 or c in '\r\n\t' for c in p.read_text()),p
        for inc in re.findall(r'\\input\{([^}]+)\}',p.read_text()):assert (ROOT/'src'/(inc if inc.endswith('.tex') else inc+'.tex')).is_file(),(p,inc)
    rb=(ROOT/'src/rationale_body.tex').read_text();eb=(ROOT/'src/ecs_body.tex').read_text();eqs=re.findall(r'\\newcommand\{\\(Eq\w+)\}',mods['equations']);invoked=re.findall(r'\\(Eq\w+)\b',rb)
    assert len(eqs)==9 and len(invoked)==8 and len(set(invoked))==8 and set(invoked)<=set(eqs)
    assert not re.search(r'\\(?:newcommand|renewcommand)\{\\(?:Eq|Norm)',rb+eb)
    data=json.loads((ROOT/'evidence/INDEX_DATA.json').read_text());vocab=['pass','fail','blocked','not_applicable','not_assessed']
    assert data['result_vocabulary']==vocab
    assert [x['id'] for x in data['requirements']]==['SCI-FRUIT-REQ-'+i for i in req]
    assert [x['id'] for x in data['predictions']]==['SCI-FRUIT-PRED-'+i for i in pred]
    assert [x['fixture'] for x in data['predictions']]==['SCI-FRUIT-FIX-'+i for i in pred]
    assert all(x['result']=='blocked' and x['evidence_artifact_realization']=='not_produced' for x in data['requirements']+data['predictions'])
    assert all(x['fixture_payload'] is None for x in data['predictions'])
    for row in data['requirements']:
        i=row['id'][-3:];assert re.search(r'^'+i+r' & [^\n]*'+re.escape(row['observable']),eb,re.M)
        match=re.search(r'^'+i+r' & ([^&]+) & ',eb,re.M);assert match and [s.strip() for s in match[1].split(',')]==[p[-3:] for p in row['procedures']]
    for row in data['predictions']:assert re.search(r'^'+row['id'][-3:]+r' & '+re.escape(row['design']),eb,re.M)
    template=json.loads((ROOT/'evidence/CONFORMANCE_RESULT_TEMPLATE.json').read_text())
    assert template['result']=='blocked' and template['allowed_result_values']==vocab and all(v is None for v in template['candidate'].values())
    assert template['evidence_artifact_realization']['state']=='not_produced' and template['evidence_artifact_realization']['allowed_values']==['realized','incomplete','failed','not_produced']
    cross=(ROOT/'R0.3_TO_R0.4_CROSSWALK.md').read_text()
    for prefix,ids in [('REQ',req),('ASM',asm),('PRED',pred)]:
        for i in ids:assert cross.count('SCI-FRUIT-'+prefix+'-'+i)==1,(prefix,i)
    for p in (ROOT/'src/common').glob('*.tex'):assert sha(p) in cross and sha(prior/'src/common'/p.name) in cross
    b=json.loads((ROOT/'reports/BUILD_RECORD.json').read_text());assert b['identity']=='SCI-FRUIT-DOC-BUILD-R0.4-2026-09-08' and b['clean_build']['all_pdf_bytes_identical']
    corehash=sha(ROOT/'identities/CORE_SOURCES.sha256');assert b['bindings']['CoreHash']==corehash
    metadata={};texts={}
    for view,expected,title in [('core',28,'SCI-FRUIT-NORMATIVE-CORE'),('rationale',10,'SCI-FRUIT-SCIENTIFIC-RATIONALE'),('ecs',13,'SCI-FRUIT-ENGINEERING-CONFORMANCE')]:
        p=ROOT/f'pdf/{view}.pdf';reader=PdfReader(p);assert len(reader.pages)==expected
        assert sha(p)==b['views'][view]['sha256']==b['clean_build']['views'][view]['sha256']
        assert not b['views'][view]['diagnostics'] and not b['clean_build']['views'][view]['diagnostics']
        meta=dict(reader.metadata);assert meta['/Title']==title+' v0.1/r0.4' and meta['/Author']=='Grant Wilson'
        assert 'review candidate' in meta['/Subject'] and meta['/CreationDate'].startswith('D:20260908')
        cover=norm(reader.pages[0].extract_text())
        for key in ['CoreHash','StageAHash','DirectiveHash','ApprovalHash','FreezeHash','PriorHash',view.title()+'SourceHash']:assert b['bindings'][key] in cover,(view,key)
        assert 'GrantWilson' in cover and '2026-09-08' in cover and 'notownerapproved,frozenoractivated' in cover
        for dest in reader.named_destinations.values():assert 0<=reader.get_destination_page_number(dest)<expected
        texts[view]='\n'.join(p.extract_text() for p in reader.pages);assert '\ufffd' not in texts[view]
        metadata[view]={'pages':expected,'sha256':sha(p),'metadata':meta}
    cn=norm(texts['core'])
    for prefix,ids in [('REQ',req),('ASM',asm),('PRED',pred)]:
        for i in ids:assert cn.count('SCI-FRUIT-'+prefix+'-'+i)==1,(prefix,i)
    base=['ITERATION-RESULT','ITERATION-BUNDLE','SUCCESSOR-STATE','CONTINUATION-STATE','PATH-SUPPORT','INFORMATION-ORIGIN','TERMINAL-SELECTION','TERMINAL-PRODUCT','MODEL-PATH-TRANSFER','MODEL-INDUCED-OUTPUT-SHIFT','MODEL-INDUCED-BIAS','MODEL-MISMATCH']
    for role in base:assert 'SCI-FRUIT/'+role in cn
    for i in range(1,8):assert f'SCI-FRUIT/RESPONSE/RF-{i:02}' in cn
    tokens=['MEASURED-PARENT-COVARIANCE','CANDIDATE-MODEL','CANDIDATE-SELECTION','ACCEPTED-MODEL-COVARIANCE','ACCEPTANCE-TO-APPLICATION','APPLIED-MODEL-COVARIANCE','PARENT-APPLIED-CROSS-COVARIANCE','OPERATOR-STATE','SUPPORT-THRESHOLD-SELECTION','STOPPING-TERMINAL-SELECTION','EXTERNAL-PRIOR','EMPIRICAL-REPEATABILITY','NOI-PRODUCT-REFERENCE']
    assert 'SCI-FRUIT/UNCERTAINTY/' in cn
    for token in tokens:assert token in cn
    stages=['requested','effective','application_scope_resolved','applied','iteration_realized','complete_iteration_candidate','completion_decided','completed_iteration','published']
    for state in ['not_requested']+stages+['available','unavailable','realized','failed','not_produced']:assert state in cn
    # These checks corroborate terminology propagation, not scientific/numerical fidelity.
    assert 'completesolecurrentStageBreview-candidatenormativecore' in cn
    for bad in ['every numerical coefficient','registered base products','registered response products','registered uncertainty family','sole active Stage B']:assert norm(bad) not in cn,bad
    for marker in ['input-dependent internal application coordinates','fixed affine','operator-defining','does not silently','PTC owns']:assert norm(marker).lower() in cn.lower(),marker
    for marker in ['analysis/gridding-coefficient families and fitted/resolved/applied state','exact PTC transformed-output identity','analysis/gridding-coefficient and QC','do not substitute for those handoff facts','does not imply permission for JINC to use that family']:
        assert norm(marker) in cn,marker
    assert norm('output/modal-application-coordinate/QC') not in cn
    links=0
    # Only manifest controls may be absent before packaging; no outer self-link exemption.
    pending={ROOT/'ARTIFACT_MANIFEST.md',ROOT/'ARTIFACT_MANIFEST.sha256'}
    outer={ROOT/'SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz',ROOT/'SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz.sha256'}
    for p in ROOT.rglob('*.md'):
        if p.relative_to(ROOT).parts[0]=='qa':continue
        for url in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in url or url.startswith('#'):continue
            q=(p.parent/url.split('#')[0]).resolve();assert q.is_relative_to(ROOT),(p,url,'outside tree')
            assert q not in outer,(p,url,'outer self-link forbidden');links+=1
            assert q.exists() or q in pending,(p,url,'missing in extracted tree')
    visual=json.loads((ROOT/'reports/PAGE_INSPECTION.json').read_text())
    for v,m in metadata.items():
        q=visual['views'][v];assert q['pdf_sha256']==m['sha256'] and q['inspected_pages']==list(range(1,m['pages']+1)) and q['open_visual_defects']==0
    report={'scope':'Document checks only; numerical comparisons blocked, evidence artifacts not produced.','core_identity':'SCI-FRUIT-NORMATIVE-CORE v0.1/r0.4','core_source_inventory_sha256':corehash,'source_inventory_counts':counts,'preserved_input_files':len(inputs),'prior_r02_payloads_verified':len(hrows),'prior_r03_archive_verified':True,'exact_b2_b3_boundary_source_verified':True,'restored_normownership_spans':2,'accepted_r03_science_byte_preservation_verified':True,'preserved_sidecars_verified':sidecars,'exact_ptc_source_verified':True,'requirements':24,'assumptions':6,'predictions':20,'canonical_modules':6,'normative_macros':len(macros),'core_declared_products':32,'base_model_roles':12,'response_roles':7,'uncertainty_grammar_tokens':13,'successful_producer_stages':stages,'canonical_equation_macros':eqs,'rationale_equation_imports':invoked,'evidence_result_vocabulary':vocab,'numerical_fixtures_instantiated':0,'numerical_results_pass':0,'source_and_view_identity_verified':True,'all_clean_build_pdf_bytes_identical':True,'tex_diagnostics':0,'in_tree_local_links_checked':links,'outer_self_links':0,'new_independent_review_rounds':0,'visual_record_verified':True,'pdfs':metadata}
    (ROOT/'reports/DOCUMENT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pdfs'},indent=2))
if __name__=='__main__':main()
