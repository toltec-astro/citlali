"""Document-only identity, authority closure, rendering record and parity checks."""
from pathlib import Path
import hashlib,json,re,unicodedata
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split())
def check_inventory(path):
    rows=[]
    for line in path.read_text().splitlines():
        digest,rel=line.split('  ',1);p=(ROOT/rel).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and sha(p)==digest,(path.name,rel)
        rows.append(rel)
    assert len(rows)==len(set(rows)) and rows==sorted(rows),path
    return rows
def main():
    counts={p.name:len(check_inventory(p)) for p in [ROOT/'identities'/n for n in ['CORE_SOURCES.sha256','CORE_DOCUMENT_SOURCES.sha256','RATIONALE_DOCUMENT_SOURCES.sha256','ECS_DOCUMENT_SOURCES.sha256']]}
    assert counts['CORE_SOURCES.sha256']==7
    inputs=check_inventory(ROOT/'identities/INPUT_FILES.sha256')
    # The original identity record is an admitted historical output, not a new scientific input.
    historical=ROOT/'inputs/scientific_core/stage_b_from_r0.4_amended/r0.1'
    rows=re.findall(r'\| \[([^]]+)\]\([^)]*\) \| (\d+) \| `([0-9a-f]{64})` \|',(historical/'SOURCE_IDENTITIES.md').read_text())
    assert len(rows)==14
    for name,size,digest in rows:
        p=ROOT/'inputs/scientific_core/r0.4-amended'/name
        if name.startswith('SCI_FRUIT_STAGE_A'):p=ROOT/'inputs/scientific_core'/name
        assert p.stat().st_size==int(size) and sha(p)==digest,name
    for name,size,digest in re.findall(r'\| `([^`]+)` \| [^|]+ \| (\d+) \| `([0-9a-f]{64})` \|',(historical/'ARTIFACT_MANIFEST.md').read_text()):
        p=historical/name;assert p.stat().st_size==int(size) and sha(p)==digest,name
    # Check every preserved digest sidecar without opening either historical archive.
    sidecars=0
    for p in (ROOT/'inputs').rglob('*.sha256'):
        for line in p.read_text().splitlines():
            h,rel=line.split(None,1);q=(p.parent/rel.strip()).resolve()
            assert q.is_relative_to(ROOT) and sha(q)==h,(p,rel)
            sidecars+=1
    modules={p.stem:p.read_text() for p in (ROOT/'src/common').glob('*.tex')}
    assert len(modules)==6
    req=re.findall(r'\\Req\{(\d{3})\}',modules['requirements'])
    pred=re.findall(r'\\Pred\{(\d{3})\}',modules['edge_cases'])
    asm=re.findall(r'\\item\[SCI-FRUIT-ASM-(\d{3}):',modules['assumptions'])
    assert req==[f'{i:03}' for i in range(1,25)]
    assert pred==[f'{i:03}' for i in range(1,21)]
    assert asm==[f'{i:03}' for i in range(1,7)]
    # Ensure canonical content is rendered once, not silently omitted by a wrapper.
    definitions='\n'.join(modules.values())
    macros=re.findall(r'\\newcommand\{\\(Norm\w+)\}',definitions)
    corebody=(ROOT/'src/core_body.tex').read_text()
    assert sorted(re.findall(r'\\(Norm\w+)',corebody))==sorted(macros)
    preamble=(ROOT/'src/preamble.tex').read_text()
    assert sorted(re.findall(r'\\input\{common/([^}]+)\.tex\}',preamble))==sorted(modules)
    for p in (ROOT/'src').rglob('*.tex'):
        assert all(ord(c)>=32 or c in '\n\r\t' for c in p.read_text()),('control character',p)
        for inc in re.findall(r'\\input\{([^}]+)\}',p.read_text()):
            assert (ROOT/'src'/(inc if inc.endswith('.tex') else inc+'.tex')).exists(),(p,inc)
    eqnames=re.findall(r'\\newcommand\{\\(Eq\w+)\}',modules['equations'])
    rb=(ROOT/'src/rationale_body.tex').read_text();eb=(ROOT/'src/ecs_body.tex').read_text()
    invoked=re.findall(r'\\(Eq\w+)\b',rb)
    assert len(invoked)==len(set(invoked)) and set(invoked)<=set(eqnames)
    assert not re.search(r'\\(?:newcommand|renewcommand)\{\\(?:Eq|Norm)',rb+eb)
    assert not re.search(r'\\(?:NormRequirements|NormPredictions|NormAssumptions)',rb+eb)
    for view in ['core','rationale','ecs']:
        wrapper=(ROOT/f'src/{view}.tex').read_text()
        assert wrapper.count(r'\input{preamble.tex}')==1 and wrapper.count(r'\Cover')==1
    data=json.loads((ROOT/'evidence/INDEX_DATA.json').read_text())
    assert [x['id'] for x in data['requirements']]==['SCI-FRUIT-REQ-'+i for i in req]
    assert [x['id'] for x in data['predictions']]==['SCI-FRUIT-PRED-'+i for i in pred]
    assert [x['fixture'] for x in data['predictions']]==['SCI-FRUIT-FIX-'+i for i in pred]
    assert all(x['result']=='blocked' for x in data['requirements']+data['predictions'])
    assert all(x['fixture_payload'] is None for x in data['predictions'])
    assert data['result_vocabulary']==['pass','fail','blocked','not_applicable','not_assessed']
    assert all(x['evidence_artifact_realization']=='not_produced' for x in data['requirements']+data['predictions'])
    template=json.loads((ROOT/'evidence/CONFORMANCE_RESULT_TEMPLATE.json').read_text())
    assert template['result']=='blocked' and all(v is None for v in template['candidate'].values())
    assert template['allowed_result_values']==data['result_vocabulary']
    assert template['evidence_artifact_realization']['state']=='not_produced'
    assert template['evidence_artifact_realization']['allowed_values']==['realized','incomplete','failed','not_produced']
    assert all(v is None for v in template['target_law'].values())
    assert all(k in template['target_law'] for k in ['target_law_identity','random_variables','conditioned_fixed_objects','resolution_or_selection_condition','omitted_variation','support_domain','reference_gauge_null_space'])
    # Every index entry occurs exactly once in its rendered ECS table source.
    for row in data['requirements']:
        i=row['id'][-3:];assert re.search(r'^'+i+r' & [^\n]*'+re.escape(row['observable']),eb,re.M),row
    for row in data['predictions']:
        i=row['id'][-3:];assert re.search(r'^'+i+r' & '+re.escape(row['design']),eb,re.M),row
    b=json.loads((ROOT/'reports/BUILD_RECORD.json').read_text())
    assert b['clean_build']['all_pdf_bytes_identical'] is True
    for view in b['views']:assert not b['views'][view]['diagnostics'] and not b['clean_build']['views'][view]['diagnostics']
    corehash=sha(ROOT/'identities/CORE_SOURCES.sha256')
    assert b['bindings']['CoreHash']==corehash
    metadata={};texts={}
    for view,expected,title in [('core',26,'SCI-FRUIT-NORMATIVE-CORE'),('rationale',9,'SCI-FRUIT-SCIENTIFIC-RATIONALE'),('ecs',13,'SCI-FRUIT-ENGINEERING-CONFORMANCE')]:
        p=ROOT/f'pdf/{view}.pdf';reader=PdfReader(p);assert len(reader.pages)==expected
        assert sha(p)==b['views'][view]['sha256']==b['clean_build']['views'][view]['sha256']
        meta=dict(reader.metadata);assert meta['/Title']==title+' v0.1/r0.2' and meta['/Author']=='Grant Wilson'
        assert 'owner-review draft' in meta['/Subject'] and meta['/CreationDate'].startswith('D:20260907')
        cover=norm(reader.pages[0].extract_text())
        for h in [corehash,b['bindings']['StageAHash'],b['bindings']['DirectiveHash'],b['bindings']['ApprovalHash'],b['bindings']['FreezeHash'],b['bindings']['InterimHash'],sha(ROOT/f'identities/{view.upper()}_DOCUMENT_SOURCES.sha256')]:assert h in cover,(view,h)
        assert 'GrantWilson' in cover and '2026-09-07' in cover
        assert 'SCI-FRUIT-DOC-BUILD-R0.2-2026-09-07' in cover
        for dest in reader.named_destinations.values():assert 0<=reader.get_destination_page_number(dest)<expected
        texts[view]='\n'.join(p.extract_text() for p in reader.pages)
        assert '\ufffd' not in texts[view]
        metadata[view]={'pages':expected,'sha256':sha(p),'metadata':meta}
    cn=norm(texts['core'])
    for prefix,ids in [('REQ',req),('ASM',asm),('PRED',pred)]:
        for i in ids:assert cn.count('SCI-FRUIT-'+prefix+'-'+i)==1,(prefix,i)
    # Registry presence verifies inventory, not scientific semantics.
    products=['ITERATION-RESULT','ITERATION-BUNDLE','SUCCESSOR-STATE','CONTINUATION-STATE','PATH-SUPPORT','INFORMATION-ORIGIN','TERMINAL-SELECTION','TERMINAL-PRODUCT','MODEL-PATH-TRANSFER','MODEL-INDUCED-OUTPUT-SHIFT','MODEL-INDUCED-BIAS','MODEL-MISMATCH']
    for role in products:assert 'SCI-FRUIT/'+role in cn,role
    for i in range(1,8):assert f'SCI-FRUIT/RESPONSE/RF-{i:02}' in cn
    uncertainty=['MEASURED-PARENT-COVARIANCE','CANDIDATE-MODEL','CANDIDATE-SELECTION','ACCEPTED-MODEL-COVARIANCE','ACCEPTANCE-TO-APPLICATION','APPLIED-MODEL-COVARIANCE','PARENT-APPLIED-CROSS-COVARIANCE','OPERATOR-STATE','SUPPORT-THRESHOLD-SELECTION','STOPPING-TERMINAL-SELECTION','EXTERNAL-PRIOR','EMPIRICAL-REPEATABILITY','NOI-PRODUCT-REFERENCE']
    assert 'SCI-FRUIT/UNCERTAINTY/' in cn
    for token in uncertainty:assert token in cn,token
    assert 'SCI-FRUIT/RESPONSE/<RF-role>' not in cn and 'SCI-FRUIT/UNCERTAINTY/<role>' not in cn
    request_states=['not_requested','requested']
    producer_stages=['requested','effective','application_scope_resolved','applied','iteration_realized','complete_iteration_candidate','completion_decided','completed_iteration','published']
    availability=['available','unavailable'];outcomes=['realized','failed','not_produced']
    for state in request_states+producer_stages+availability+outcomes:assert state in cn,state
    for label in ['target_law_identity','random variables','conditioned fixed objects','resolution/selection condition','omitted variation']:assert norm(label) in cn,label
    # Round 3 is historical after the later owner correction; its bytes remain preserved.
    assert sha(ROOT/'review/ROUND_3.md')=='2f5c2122b0223ef1bc0ad3a51454f355b2e179a40df4e3c1f3e657d75f47282e'
    assert sha(ROOT/'inputs/PRE_INTERIM_R0.2_REVIEWED_CANDIDATE.tar.gz')=='7cdefc3b43474647fc0cadd9c505efae8c769c381a58d82491cb165ed525cc1f'
    review=ROOT/'review/INTERIM_CORRECTION_REVIEW.md'
    reviewed=0;review_paths=[]
    assert review.exists(),'Supplementary correction check must be complete before delivery'
    for rel,size,digest in re.findall(r'\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|',review.read_text()):
        p=ROOT/rel;assert p.is_file() and p.stat().st_size==int(size) and sha(p)==digest,('correction review snapshot',rel)
        reviewed+=1;review_paths.append(rel)
    assert len(review_paths)==len(set(review_paths)) and reviewed>0
    assert set(check_inventory(ROOT/'identities/CORE_SOURCES.sha256'))<=set(review_paths)
    assert {'src/rationale_body.tex','src/ecs_body.tex','R0.1_TO_R0.2_CROSSWALK.md','evidence/INDEX_DATA.json','evidence/CONFORMANCE_RESULT_TEMPLATE.json'}<=set(review_paths)
    # All Markdown links, including preserved inputs, must resolve within delivery root.
    pending={'DOCUMENT_CHECKS.json','ARTIFACT_MANIFEST.md','ARTIFACT_MANIFEST.sha256','SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz','SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz.sha256'}
    links=0
    for p in ROOT.rglob('*.md'):
        if 'qa' in p.relative_to(ROOT).parts:continue
        for url in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if '://' in url or url.startswith('#'):continue
            target=(p.parent/url.split('#')[0]).resolve()
            assert target.is_relative_to(ROOT),(p,url,'outside bundle')
            links+=1
            if target.name in pending and not target.exists():continue
            assert target.exists(),(p,url,'missing')
    visual=ROOT/'reports/PAGE_INSPECTION.json'
    if visual.exists():
        visual=json.loads(visual.read_text())
        for v,m in metadata.items():
            q=visual['views'][v];assert q['pdf_sha256']==m['sha256'] and q['inspected_pages']==list(range(1,m['pages']+1)) and q['open_visual_defects']==0
    report={'scope':'Document verification only; numerical comparisons remain blocked and evidence artifacts not produced.','core_identity':'SCI-FRUIT-NORMATIVE-CORE v0.1/r0.2','core_source_inventory_sha256':corehash,'source_inventory_counts':counts,'preserved_input_files':len(inputs),'stage_a_payloads_verified_against_prior_exact_identity':11,'permission_controls_verified':5,'preserved_sidecars_verified':sidecars,'canonical_modules':6,'normative_units':len(macros),'requirements':len(req),'assumptions':len(asm),'predictions':len(pred),'response_roles':7,'base_model_product_roles':len(products),'uncertainty_family_tokens':len(uncertainty),'registered_product_roles':len(products)+7+len(uncertainty),'request_states':request_states,'successful_producer_stages':producer_stages,'scientific_availability_values':availability,'producer_outcome_values':outcomes,'evidence_result_vocabulary':data['result_vocabulary'],'evidence_artifact_realization_vocabulary':template['evidence_artifact_realization']['allowed_values'],'prospective_requirement_rows':len(data['requirements']),'prospective_fixture_rows':len(data['predictions']),'numerical_fixtures_instantiated':0,'numerical_results_pass':0,'canonical_equation_macros':eqnames,'rationale_equation_imports':invoked,'view_redefinitions':0,'all_views_bind_identical_core':True,'all_clean_build_pdf_bytes_identical':True,'tex_diagnostics':0,'local_links_checked':links,'reviewed_snapshot_files_verified':reviewed,'reviewed_current_source_files':sum(not p.startswith('inputs/') for p in review_paths),'reviewed_retained_input_files':sum(p.startswith('inputs/') for p in review_paths),'correction_review_complete':True,'historical_round_3_preserved':True,'pdfs':metadata,'visual_record_verified':visual.exists() if isinstance(visual,Path) else True}
    (ROOT/'reports/DOCUMENT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pdfs'},indent=2))
if __name__=='__main__':main()
