#!/usr/bin/env python3
"""Rebuild POINT cycle diagnostics from immutable products, never rerun FRUIT."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from inspect_forward_map import digest,require
from point_additive_diagnostics import save_json
from point_fruit_diagnostics import diagnose,load


def run(cycle,output):
    started=time.perf_counter();require(not output.exists() and not output.resolve().is_relative_to(cycle.resolve()),'fresh separate diagnostic output required')
    manifest=cycle/'FILES_SHA256SUMS'
    for line in manifest.read_text().splitlines():
        sha,name=line.split('  ',1);p=(cycle/name).resolve()
        require(p.is_relative_to(cycle.resolve()) and digest(p)==sha,'cycle evidence hash mismatch')
    provenance=json.loads((cycle/'provenance.json').read_text());args=provenance['request']
    output.mkdir(parents=True);reports={}
    for a in ['a1100','a1400','a2000']:
        reports[a]=diagnose(a,Path(args['maps'])/a,Path(args['admission'])/a,Path(args['additive'])/a,cycle/'zero'/a/'products',cycle/'reference'/a/'products',output/a)
        previous=cycle/f'diagnostics-{a}'/'m1.fits';current=output/a/'m1.fits'
        if previous.exists():
            require(current.exists(),'report regeneration changed m1 availability')
            before,_=load(previous);after,_=load(current)
            require(all(np.array_equal(before[k],after[k],equal_nan=True) for k in before),'report regeneration changed m1')
    save_json(output/'summary.json',reports)
    save_json(output/'RUN.json',dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        cycle_source_revision=provenance['source_revision'],cycle_manifest_sha256=digest(manifest),cycle_path=str(cycle.resolve()),
        wall_seconds=time.perf_counter()-started,scope='diagnostic-only saved-basis layout repair; no projection, PTC learning, feedback cycle or m1 application',m1_identical=True))
    (output/'FILES_SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='FILES_SHA256SUMS'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cycle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.cycle,a.output)
