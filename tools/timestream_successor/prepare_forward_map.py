"""Bind existing PTC publications for the native successor saved-MAP entry."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',nargs=2,action='append',required=True,metavar=('PRODUCT_ROOT','ORIGINAL_INPUT'))
    p.add_argument('--owner-binding',type=Path,required=True)
    p.add_argument('--pixel-size-arcsec',type=float,required=True)
    p.add_argument('--coverage-cut',type=float,required=True)
    p.add_argument('--output',type=Path,required=True,help='new native MAP product directory')
    p.add_argument('--request-dir',type=Path,required=True)
    a=p.parse_args();a.request_dir.mkdir(parents=True,exist_ok=False)
    owner=a.owner_binding.resolve()
    request={'schema':'citlali-map-saved-ptc-v1','owner_binding':{'path':str(owner),'sha256':digest(owner)},
             'pixel_size_arcsec':a.pixel_size_arcsec,'coverage_cut':a.coverage_cut,'networks':[]}
    for root,input_path in a.source:
        root=Path(root).resolve();input_path=Path(input_path).resolve()
        request['networks'].append({'root':str(root),'input':{'path':str(input_path),'sha256':digest(input_path)},
          'receipt_sha256':digest(root/'receipt.yaml'),
          'ptc_receipt_sha256':digest(root/'donor-continuity/ptc/receipt.yaml')})
    target=a.request_dir.resolve()/'map-input.json';target.write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
    dev={'schema':'citlali-development-v1','terminal':'map','input':{'path':str(target),'sha256':digest(target)},'output':str(a.output.resolve())}
    config=a.request_dir.resolve()/'development.yaml';config.write_text(yaml.safe_dump(dev,sort_keys=False))
    print(config)
if __name__=='__main__':main()
