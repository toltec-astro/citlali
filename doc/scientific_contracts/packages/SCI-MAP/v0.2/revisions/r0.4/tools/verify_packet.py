#!/usr/bin/env python3
"""Read-only SCI-MAP r0.4 document-packet checks, not application evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
from pypdf import PdfReader
from jsonschema import Draft202012Validator
import jsonschema
from importlib.metadata import version
from build_packet import CORE, VIEWS, MANIFEST, core_digest, sha


def verify(root):
    failures=[]; checks={}
    def check(name, result, details=None):
        checks[name]={'pass':bool(result),'details':details}
        if not result: failures.append(name)
    manifest_path=root/'SOURCE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    expected=manifest['files']
    mismatches=[p for p,d in expected.items() if not (root/p).is_file() or sha(root/p)!=d]
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    excluded={'SOURCE_MANIFEST.json','SOURCE_MANIFEST.sha256'}
    check('complete_package_file_inventory', actual==set(expected)|excluded, sorted(actual^(set(expected)|excluded)))
    check('all_bound_file_hashes',not mismatches,mismatches)
    byte_counts=manifest['file_bytes']
    check('all_bound_file_byte_counts',set(byte_counts)==set(expected) and all((root/name).stat().st_size==size for name,size in byte_counts.items()))
    check('source_manifest_sidecar', (root/'SOURCE_MANIFEST.sha256').read_text().strip()==sha(manifest_path)+'  SOURCE_MANIFEST.json')
    inv=json.loads((root/'AUTHOR_PACKET_INVENTORY.json').read_text())
    check('199_exact_author_packet_files', len(inv['input_files'])==199 and all(sha(root/'inputs'/p)==d for p,d in inv['input_files'].items()))
    original=json.loads((root/'inputs'/MANIFEST).read_text())
    check('56_reference_manifest_exact_sources',len(original['references'])==56 and all(sha(root/'inputs'/r['path'])==r['sha256'] for r in original['references']))
    check('exact_current_owner_directive_binding',sha(root/'inputs/owner/SCI_MAP_R04_OWNER_DIRECTIVE.txt')==inv['owner_directive_sha256'])
    check('exact_predecessor_packet_binding',sha(root/'inputs/basis-r0.3/AUTHOR_PACKET_INVENTORY.json')==inv['predecessor_packet_sha256'])
    predecessor=json.loads((root/'inputs/basis-r0.3/AUTHOR_PACKET_INVENTORY.json').read_text())
    retained={name:('basis-r0.3/'+name if name.startswith('manager/') else name) for name in predecessor['input_files']}
    check('all_136_predecessor_inputs_preserved_with_manager_recovery_relocated',len(retained)==136 and all(sha(root/'inputs'/retained[name])==digest for name,digest in predecessor['input_files'].items()))

    current_prose=[p for directory in ['src','records','profiles','bindings'] for p in (root/directory).rglob('*') if p.is_file() and p.suffix in {'.tex','.md'}]
    generic_zero=[]; vague_cause=[]
    for p in current_prose:
        compact=' '.join(p.read_text().split())
        if re.search(r'exact.zero, non.finite, overflowed, or unrepresentable aggregate/index never',compact,re.I): generic_zero.append(str(p.relative_to(root)))
        if 'like-named cause' in compact: vague_cause.append(str(p.relative_to(root)))
    check('no_generic_zero_aggregate_index_rejection',not generic_zero,generic_zero)
    check('no_like_named_cause_in_current_prose',not vague_cause,vague_cause)
    req=(root/'src/common/requirements.tex').read_text()
    pred=(root/'src/common/edge_cases.tex').read_text()
    reqids=re.findall(r'\\SCIMapRequirement\{(SCI-MAP-REQ-\d{3})\}',req)
    predids=re.findall(r'\\SCIMapPrediction\{(SCI-MAP-PRED-\d{3})\}',pred)
    check('52_unique_gap_free_requirements', sorted(reqids)==[f'SCI-MAP-REQ-{i:03d}' for i in range(1,53)],len(reqids))
    check('25_unique_gap_free_predictions', sorted(predids)==[f'SCI-MAP-PRED-{i:03d}' for i in range(1,26)],len(predids))
    check('core_aggregate_hash',core_digest(root/'src')==manifest['shared_authority']['sha256'])
    definitions=(root/'src/common/definitions.tex').read_text()
    check('distinct_full_empty_rule_and_short_reference',r'\providecommand{\SCIMapEmptySupportDispositionBody}' in definitions and r'\providecommand{\SCIMapEmptySupportReference}' in definitions and r'\label{rule:empty-support-disposition}' in definitions and r'\SCIMapEmptySupportDispositionBody' not in req)
    check('fixture_A_to_D_mapping_present',(root/'records/PRED025_QUANTITY_SPECIFIC_ZERO_FIXTURES.md').is_file() and 'PRED025_QUANTITY_SPECIFIC_ZERO_FIXTURES.md' in (root/'PREDICTION_CROSSWALK.md').read_text())
    ecs=' '.join((root/'src/engineering-conformance.tex').read_text().split())
    creation=ecs[ecs.index('To produce a new VAL decision'):ecs.index('When consuming an existing decision')]
    check('ECS_new_decision_order_is_non_circular',creation.index('first verify exact profile semantics') < creation.index('perform the evaluation') < creation.index('resulting decision-artifact identity') < creation.index('MAP consume the decision') and 'decision-artifact' not in creation[:creation.index('perform the evaluation')])
    existing=ecs[ecs.index('When consuming an existing decision'):ecs.index('Only requested/applicable/eligible/realized passes')]
    check('ECS_existing_decision_binding_and_realization_before_admission',all(word in existing for word in ['profile/source binding','object identity','four-axis state','causes','realization before MAP admission']))
    old_eq=(root/'inputs/basis-r0.3/src/common/equations.tex').read_text()
    new_eq=(root/'src/common/equations.tex').read_text()
    def support_selector(source):
        start=source.index(r'N&=|\mathcal P|')
        end=source.index(r'\label{eq:threshold-selector}',start)
        return ''.join(source[start:end].split())
    check('inherited_order_statistic_and_N0_convention',support_selector(old_eq)==support_selector(new_eq))
    old_owner=(root/'inputs/basis-r0.3/src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.3.tex').read_text()
    new_owner=(root/'src/SCI-MAP-v0.2_OWNER_DECISION_REGISTER_r0.4.tex').read_text()
    check('nine_settled_rendered_owner_decisions_preserved',old_owner.replace('v0.2/r0.3','v0.2/r0.4',1)==new_owner)
    for name,prefix,count in [('CROSSWALK.md','REQ',52),('PREDICTION_CROSSWALK.md','PRED',25)]:
        ids=re.findall(r'^\| (SCI-MAP-'+prefix+r'-\d{3}) \|',(root/name).read_text(),re.M)
        check(name+'_complete_unique_rows',sorted(ids)==[f'SCI-MAP-{prefix}-{n:03d}' for n in range(1,count+1)])
    texinputs=[]
    for p in (root/'src').rglob('*.tex'):
        for s in re.findall(r'\\input\{([^{}]+)\}',p.read_text()):
            q=root/'src'/s
            if not q.suffix: q=q.with_suffix('.tex')
            if not q.is_file() or not q.resolve().is_relative_to(root/'src'): texinputs.append((str(p.relative_to(root)),s))
    check('all_explicit_tex_inputs_resolve',not texinputs,texinputs)
    mdlinks=[]
    # Frozen/reference snapshots preserve original bytes; links in newly authored files are checked.
    for p in root.rglob('*.md'):
        if p.is_relative_to(root/'inputs'): continue
        for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',p.read_text()):
            link=link.strip('<>')
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:',link) or link.startswith('#'): continue
            path=unquote(link.split('#')[0])
            if path and not (p.parent/path).exists(): mdlinks.append([str(p.relative_to(root)),link])
    check('authored_markdown_file_links_resolve',not mdlinks,mdlinks)
    authored=json.loads((root/'records/SCIENTIFIC_SOURCE_DIGEST_REPORT.json').read_text())
    authored_mismatches=[r['path'] for r in authored['files'] if not (root/r['path']).is_file() or sha(root/r['path'])!=r['sha256'] or (root/r['path']).stat().st_size!=r['bytes']]
    check('all_author_output_file_digests',bool(authored['files']) and len({r['path'] for r in authored['files']})==len(authored['files']) and not authored_mismatches,{'files':len(authored['files']),'mismatches':authored_mismatches})
    check('author_core_digest_matches_delivered_core',authored['shared_authority']['aggregate_sha256']==core_digest(root/'src'))
    inventory_stream=b''.join(row['path'].encode()+b'\0'+str(row['bytes']).encode()+b'\0'+row['sha256'].encode()+b'\0' for row in authored['files'])
    inventory_record=authored['authored_file_inventory']
    check('author_inventory_aggregate_and_count',hashlib.sha256(inventory_stream).hexdigest()==inventory_record['sha256'] and len(authored['files'])==inventory_record['files_excluding_this_report'] and [row['path'] for row in authored['files']]==sorted(row['path'] for row in authored['files']))
    # These are new prospective document templates, not application schemas.
    templates=root/'templates'
    template_errors=[]
    expected_templates={'REQUIREMENT_RESULT_RECORD.schema.json','MAP_LIFECYCLE_AND_PRODUCT_ROLE_RECORD.schema.json','COVERAGE_CUT_OVERRIDE_RECORD.schema.json','FULL_PROCEDURE_COMPARISON_RECORD.schema.json','WCS_COMPARISON_RECORD.schema.json','VAL_HANDOFF_DECISION_RECORD.schema.json'}
    check('six_prospective_record_templates_present',{p.name for p in templates.glob('*.schema.json')}==expected_templates)
    for p in sorted(templates.glob('*.schema.json')):
        try: Draft202012Validator.check_schema(json.loads(p.read_text()))
        except Exception as e: template_errors.append([p.name,str(e)])
    check('prospective_record_template_structure',not template_errors,template_errors)
    schema=json.loads((templates/'REQUIREMENT_RESULT_RECORD.schema.json').read_text())
    initial=json.loads((templates/'REQUIREMENT_RESULTS_INITIAL.json').read_text())
    rows=initial['requirements']
    row_errors=[{'id':row.get('requirement_id'),'error':e.message} for row in rows for e in Draft202012Validator(schema).iter_errors(row)]
    check('52_machine_readable_results_match_template',len(rows)==52 and not row_errors,row_errors)
    check('52_machine_readable_results_gap_free',sorted(r['requirement_id'] for r in rows)==[f'SCI-MAP-REQ-{n:03d}' for n in range(1,53)])
    check('all_machine_results_prospective',all(r['result']=='not_assessed' and r['artifact_realization']=='not_produced' and not r['evidence_artifacts'] and r['reviewer'] is None for r in rows))
    binding_rows=[]; bad_bindings=[]
    # A source-path/hash row must match the raw file named on that row.
    for p in sorted((root/'bindings').glob('*.md')):
        for line in p.read_text().splitlines():
            if not line.startswith('|'): continue
            tokens=re.findall(r'`([^`]+)`',line)
            paths=[t for t in tokens if t.startswith(('src/','profiles/','bindings/','inputs/')) and not t.endswith('/')]
            digests=[t for t in tokens if re.fullmatch('[0-9a-f]{64}',t)]
            for rel in paths:
                if digests:
                    binding_rows.append([str(p.relative_to(root)),rel])
                    if not (root/rel).is_file() or sha(root/rel) not in digests: bad_bindings.append([str(p.relative_to(root)),rel,sha(root/rel) if (root/rel).is_file() else 'missing',digests])
    check('candidate_source_binding_path_hash_rows',bool(binding_rows) and not bad_bindings,{'checked_rows':len(binding_rows),'mismatches':bad_bindings})
    registry=root/'bindings/SCI-VAL_PROFILE_REGISTRY_SCI-MAP_v0.2_r0.4_CANDIDATE.md'
    register=root/'bindings/SCI-VAL_SOURCE_BINDING_REGISTER_SCI-MAP_v0.2_r0.4_CANDIDATE.md'
    check('VAL_register_binds_exact_paired_registry',sha(registry) in register.read_text())
    checks['template_reader_identity']={'version':version('jsonschema'),'module':jsonschema.__file__,'module_sha256':sha(Path(jsonschema.__file__)),'limit':'Local structural validation of prospective document records; semantic conformance is not inferred.'}
    build=json.loads((root/'build/BUILD_ATTEMPT.json').read_text())
    inspection=json.loads((root/'build/PDF_BUILD_INSPECTION.json').read_text())
    for view,(entry,filename) in VIEWS.items():
        text=(root/'src'/(entry+'.tex')).read_text()
        check(view+'_one_full_empty_support_rule',text.count(r'\SCIMapEmptySupportDispositionBody')==1)
        check(view+'_one_complete_lifecycle_table',text.count(r'\SCIMapLifecycleTable')==1 and 'The exact MAP lifecycle is' not in definitions)
        previous=(root/'inputs/basis-r0.3/src'/(entry+'.tex')).read_text()
        design_patterns=[r'\\documentclass[^\n]*',r'\\usepackage\[margin=[^\n]*',r'\\setlength\{\\parskip\}[^\n]*',r'\\linespread[^\n]*',r'\\setstretch[^\n]*']
        check(view+'_inherited_font_margin_spacing_controls',all(re.findall(pattern,text)==re.findall(pattern,previous) for pattern in design_patterns))
        check(view+'_imports_shared_authority',r'\input{SCI-MAP-v0.2_SHARED_AUTHORITY_r0.4.tex}' in text)
        check(view+'_entry_source_hash',sha(root/'src'/(entry+'.tex'))==build['entry_sources'][view]['sha256'])
        check(view+'_cover_fragment_hash',sha(root/'src/identity'/(view+'.tex'))==build['generated_cover_sources'][view]['sha256'])
        pdf=root/'pdf'/(filename+'.pdf')
        data=inspection['views'][view]
        check(view+'_built_pdf_hash',sha(pdf)==data['pdf_sha256'])
        check(view+'_compiler_source_dependencies',bool(data['dependency_sources']) and all(sha(root/d['source_path'])==d['sha256'] for d in data['dependency_sources']))
        check(view+'_clean_document_build',not data['tex_errors_or_overfull'] and data['nonempty_pages'] and data['pdfinfo_exit_code']==0 and data['render_exit_code']==0 and data['pages']==data['rendered_pages'])
        reader=PdfReader(pdf,strict=True)
        meta={str(k):str(v) for k,v in reader.metadata.items()}
        check(view+'_metadata_owner_revision_status', 'Grant Wilson' in meta.get('/Author','') and all(s in meta.get('/Title','') for s in ['v0.2','r0.4']) and 'candidate' in (meta.get('/Title','')+' '+meta.get('/Subject','')).lower() and meta.get('/CreationDate','').startswith('D:20260908'))
        extracted='\n'.join(page.extract_text() or '' for page in reader.pages)
        compact=' '.join(extracted.split())
        check(view+'_no_malformed_expanded_empty_support_sentences',not any(bad in compact for bad in ['uses The 2026-09-08','follows The 2026-09-08','authorized. and is not','like-named cause']))
        if view!='rationale':
            rendered_req=re.findall(r'SCI-MAP-REQ-(\d{3})\s*[–—-]',extracted)
            rendered_pred=re.findall(r'SCI-MAP-PRED-(\d{3})\s*[–—-]',extracted)
            check(view+'_rendered_52_25_identities',sorted(rendered_req)==[f'{n:03d}' for n in range(1,53)] and sorted(rendered_pred)==[f'{n:03d}' for n in range(1,26)])
        if view=='engineering':
            check('52_ECS_results_remain_not_assessed',''.join(extracted.split()).count('Result:not_assessed')==52)
        first=''.join((reader.pages[0].extract_text() or '').split())
        hashes=[build['shared_authority']['sha256'],build['original_author_manifest']['sha256'],build['author_packet_inventory']['sha256'],build['owner_directive']['sha256'],build['entry_sources'][view]['sha256']]
        check(view+'_cover_exact_hashes',all(h in first for h in hashes))
        check(view+'_cover_shared_identity_and_build',build['identity'] in first and build['shared_authority']['identity'] in first)
        check(view+'_cover_revision_owner', all(s in first for s in ['GrantWilson','v0.2','r0.4','2026-09-08']))
    return {'status':'pass' if not failures else 'fail','checks':checks,'failed_checks':failures,'scope':'Document and source consistency only; requirement evidence remains prospective.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--package',required=True); p.add_argument('--report')
    a=p.parse_args(); result=verify(Path(a.package).resolve()); output=json.dumps(result,indent=2)+'\n'
    if a.report: Path(a.report).write_text(output)
    print(output)
    raise SystemExit(bool(result['failed_checks']))
