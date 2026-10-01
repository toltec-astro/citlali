"""Acquisition recipe admission. No numerical or morphology policy inference."""
from inspect_forward_map import require

def validate_recipe(r):
    oof=r.get('schema')=='citlali-oof-cold-m0-v1'
    if oof:
        require(r.get('mode')=='OOF' and r.get('observation') in (152385,152386,152387)
                and r.get('subobservation')==0 and r.get('scan')==1, 'OOF acquisition identity differs')
        require(r.get('cycles')==0 and r.get('source_gate')=='none' and r.get('feedback') is False
                and 'threshold' not in r and 'source_radius' not in r, 'OOF requires cold M0 without POINT policy')
        require(r.get('array')=='a1100' and r.get('networks')==[1,2,3,4]
                and r.get('ptc')==dict(method='observed-als',rank=3), 'OOF declared population/rank3 control differs')
    else:
        require(r['schema']=='citlali-bright-point-reference-v1' and r['mode']=='POINT', 'unsupported acquisition recipe')
        require(r['subobservation']==0 and r['scan']==2, 'bright POINT requires subobservation0/scan2')
        require(r['cycles']==10 and r['threshold']==3 and r['coverage_policy']=='high-coverage-informational', 'frozen downstream recipe differs')
    return oof
