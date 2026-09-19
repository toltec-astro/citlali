"""Compare exact arrays and parsed YAML with bounded parser memory.

Map ordering and flow/block formatting are immaterial. Only the enumerated
runtime/provenance differences are normalized; scientific durations remain.
"""
from pathlib import Path
import argparse, collections, hashlib, json, math, re
import yaml
from yaml.events import *
from yaml.nodes import ScalarNode

TIMERS = set('''ingress_seconds existing_learn_seconds total_seconds Learn_seconds
Consider_seconds Apply_VAL_seconds Consider_evidence_seconds transient_seconds
spectral_seconds Consider_connection_seconds Consider_plan_seconds Apply_seconds
prepare_and_inspect_seconds matched_outcome_seconds Consider_decision_seconds
advance_seconds conditioned_relearning_seconds finalize_publish_bind_seconds
wall_seconds preparation_seconds fit_seconds application_seconds publication_seconds
apply_seconds basis_seconds check_seconds coefficient_seconds covariance_seconds
decomposition_seconds initialization_seconds output_seconds'''.split())
METRICS = {'process_peak_rss_bytes', 'peak_scratch_samples', 'descriptor_bytes', 'logical_owned_bytes'}
PROVENANCE = {'source_revision', 'learning_binding', 'build_identity'}
IGNORED = TIMERS | METRICS | PROVENANCE

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class Parser:
    def __init__(self, path, root):
        self.path=path; self.root=root; self.ignored=[]; self.summaries={}; self.scalars=0
        self.loader=yaml.SafeLoader(''); self.entropies={}
    def scalar(self,e):
        tag=e.tag or self.loader.resolve(ScalarNode,e.value,e.implicit)
        node=ScalarNode(tag,e.value)
        value=self.loader.yaml_constructors[tag](self.loader,node)
        if isinstance(value,str):
            value=value.replace(str(self.root),'<OUTPUT>')
            # APT allocates an occurrence identity per load. Keep exact semantic
            # and envelope hashes and row mapping; normalize only that entropy.
            def entropy(m):
                key=m.group(0)
                return self.entropies.setdefault(key,'apt-v2-occurrence:entropy/<'+str(len(self.entropies))+'>')
            value=re.sub(r'apt-v2-occurrence:entropy/[0-9a-f]{64}',entropy,value)
        elif isinstance(value,float):
            value=value.hex() if math.isfinite(value) else repr(value)
        self.scalars+=1
        return tag,value
    def parse(self):
        with self.path.open() as f:
            self.events=iter(yaml.parse(f,Loader=yaml.CSafeLoader))
            assert isinstance(next(self.events),StreamStartEvent)
            assert isinstance(next(self.events),DocumentStartEvent)
            digest=self.node(next(self.events),())
            assert isinstance(next(self.events),DocumentEndEvent)
            assert isinstance(next(self.events),StreamEndEvent)
        self.loader.dispose()
        return {'semantic_sha256':digest.hex(),'scalar_count':self.scalars,
                'normalized_fields':self.ignored,'entropy_identity_count':len(self.entropies),
                'subtrees':self.summaries}
    def node(self,e,path):
        if isinstance(e,ScalarEvent):
            tag,value=self.scalar(e)
            result=hashlib.sha256(json.dumps([tag,value],ensure_ascii=False,separators=(',',':')).encode()).digest()
        elif isinstance(e,SequenceStartEvent):
            h=hashlib.sha256(b'S'); n=0
            while not isinstance(e:=next(self.events),SequenceEndEvent):
                h.update(self.node(e,path+(n,)));n+=1
            h.update(str(n).encode());result=h.digest()
        elif isinstance(e,MappingStartEvent):
            values={}
            while not isinstance(e:=next(self.events),MappingEndEvent):
                assert isinstance(e,ScalarEvent),('non-scalar key',path)
                kt,key=self.scalar(e); k=json.dumps([kt,key],separators=(',',':'))
                assert k not in values,('duplicate key',path,key)
                v=next(self.events)
                if key == 'source_CAL_receipt_sha256':
                    assert isinstance(v,ScalarEvent)
                    expected=sha(self.root/'donor-continuity/cal/receipt.yaml')
                    assert v.value == expected,('CAL receipt digest binding failed',self.path)
                    values[k]=hashlib.sha256(b'VERIFIED-CAL-RECEIPT').digest()
                elif key in IGNORED:
                    assert isinstance(v,ScalarEvent),('non-scalar normalization',path,key)
                    self.ignored.append({'path':list(path+(key,)),'value':self.scalar(v)[1]})
                    values[k]=hashlib.sha256(b'NORMALIZED').digest()
                else:values[k]=self.node(v,path+(key,))
            h=hashlib.sha256(b'M')
            for key,value in sorted(values.items()):
                h.update(hashlib.sha256(key.encode()).digest());h.update(value)
            result=h.digest()
        else:raise ValueError(('unsupported YAML event',e,path))
        # Enough structure to locate differences without retaining every window.
        if len(path)<=3 or (len(path)<=5 and 'windows' not in path):
            self.summaries[json.dumps(path,separators=(',',':'))]=result.hex()
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True)
    extensions={'.f64','.i64','.u8','.u16','.yaml'}
    inventory=lambda root:{str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix in extensions}
    assert inventory(a.baseline)==inventory(a.candidate),'scientific file inventory differs'
    result={'baseline':str(a.baseline),'candidate':str(a.candidate),'allowed_scalar_normalizations':sorted(IGNORED),
            'binary':[],'yaml':[]}
    for path in sorted(a.baseline.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(a.baseline);new=a.candidate/rel
        assert new.is_file(),('missing',rel)
        if path.suffix in {'.f64','.i64','.u8','.u16'}:
            before=sha(path);after=sha(new)
            result['binary'].append({'path':str(rel),'bytes':path.stat().st_size,'sha256':before,'candidate_sha256':after,'equal':before==after})
    print('binary',len(result['binary']),'different',sum(not x['equal'] for x in result['binary']),flush=True)
    (a.output/'binary-comparison.json').write_text(json.dumps(result['binary'],indent=2)+'\n')
    for path in sorted(a.baseline.rglob('*.yaml')):
        rel=path.relative_to(a.baseline);new=a.candidate/rel
        before=Parser(path,a.baseline).parse();after=Parser(new,a.candidate).parse()
        name=str(rel).replace('/','__')+'.json'
        (a.output/('baseline__'+name)).write_text(json.dumps(before,indent=2)+'\n')
        (a.output/('candidate__'+name)).write_text(json.dumps(after,indent=2)+'\n')
        changes=[key for key in set(before['subtrees'])|set(after['subtrees']) if before['subtrees'].get(key)!=after['subtrees'].get(key)]
        equal=before['semantic_sha256']==after['semantic_sha256']
        row={'path':str(rel),'equal':equal,'baseline':before['semantic_sha256'],'candidate':after['semantic_sha256'],
             'scalar_count_before':before['scalar_count'],'scalar_count_after':after['scalar_count'],'changed_subtrees':sorted(changes)}
        result['yaml'].append(row);print('yaml',rel,equal,'changes',len(changes),flush=True)
        (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    result['pass']=all(x['equal'] for x in result['binary']+result['yaml'])
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
