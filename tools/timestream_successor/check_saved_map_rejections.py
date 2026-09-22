"""Negative native-CLI checks using read-only links to a real saved PTC parent."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import yaml
from prepare_forward_map import digest

def selected(path,section,keys):
    text=[];inside=not section;keep=False;indent=2 if section else 0
    with Path(path).open() as f:
        for line in f:
            if section and line.rstrip()==section+':':inside=True;continue
            if not inside:continue
            start=len(line)-len(line.lstrip(' '))
            if start<indent:break
            if start==indent:keep=line[indent:].split(':')[0] in keys
            if keep:text.append(line[indent:])
    return yaml.load(''.join(text),Loader=yaml.CSafeLoader)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cli',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();request=json.loads(a.request.read_text());source=request['networks'][0];original=Path(source['root']);results=[]
    with tempfile.TemporaryDirectory(prefix='map-binding-negative-') as temp:
        base=Path(temp)
        for case in ('learning_time','apply_schedule'):
            root=base/case;root.mkdir()
            for child in original.iterdir():
                if child.name not in ('learning-receipt.yaml','donor-continuity'):(root/child.name).symlink_to(child,target_is_directory=child.is_dir())
            if case=='learning_time':
                learn=selected(original/'learning-receipt.yaml','',{'configuration_sha256','original_parent','native_time_sha256','rows','network','observation','detectors','native_runs'})
                learn['native_time_sha256']='changed-native-time'
                (root/'learning-receipt.yaml').write_text(yaml.safe_dump(learn,sort_keys=False))
                (root/'donor-continuity').symlink_to(original/'donor-continuity',target_is_directory=True)
                expected='MAP learning/final identity mismatch: native_time_sha256'
            else:
                (root/'learning-receipt.yaml').symlink_to(original/'learning-receipt.yaml')
                donor=root/'donor-continuity';donor.mkdir()
                for child in (original/'donor-continuity').iterdir():
                    if child.name!='apply-receipt.yaml':(donor/child.name).symlink_to(child,target_is_directory=child.is_dir())
                apply=selected(original/'donor-continuity/apply-receipt.yaml','',{'attempt','original_parent','realized_detectors','output_grid'})
                apply['realized_detectors'][0]['phase_native_rows']+=1
                (donor/'apply-receipt.yaml').write_text(yaml.safe_dump(apply,sort_keys=False))
                expected='MAP apply/final identity mismatch: realized_detectors'
            item=dict(request);item['networks']=[dict(source,root=str(root))]
            path=root/'request.json';path.write_text(json.dumps(item))
            config=dict(schema='citlali-development-v1',terminal='map',input=dict(path=str(path),sha256=digest(path)),output=str(root/'output'))
            cfg=root/'development.yaml';cfg.write_text(yaml.safe_dump(config));r=subprocess.run([str(a.cli.resolve()),str(cfg)],capture_output=True,text=True,timeout=90)
            ok=r.returncode!=0 and expected in r.stdout+r.stderr and not (root/'output').exists()
            results.append(dict(case=case,passed=ok,exit_code=r.returncode,expected=expected,output=r.stdout+r.stderr))
    a.report.write_text(json.dumps(dict(tests=results,passed=all(r['passed'] for r in results)),indent=2)+'\n')
    if not all(r['passed'] for r in results):raise SystemExit(1)
    print('2 real-parent identity mutation checks passed; original source files untouched')
if __name__=='__main__':main()
