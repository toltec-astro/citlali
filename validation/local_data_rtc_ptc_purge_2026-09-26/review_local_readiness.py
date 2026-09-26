import json,gzip,re,collections,pathlib,yaml,csv
R=pathlib.Path('/Users/gwilson/work_toltec/local_data'); W=pathlib.Path('/private/tmp/citlali-point-additive-20260923')
x=json.load(gzip.open(W/'validation/local_data_rtc_ptc_purge_2026-09-26/INVENTORY_BEFORE.json.gz','rt')); bybase=collections.defaultdict(list)
for r in x:bybase[pathlib.Path(r['path']).name].append(r)
paths=[
'citlali-validation/v1/point/pointings/reduced/citlali_o152389_0_2_c1.yaml',
'citlali-validation/v1/oof/wilson/1146+399/reduced/citlali_o152385_0_1_c3.yaml',
'citlali-validation/v1/beammaps/3c273/reduced/citlali_o148670_0_2_c1.yaml',
'citlali-validation/v1/beammaps/pointings/reduced/citlali_o148669_0_2_c2.yaml',
'citlali-validation/v1/science/reduction/NGC4449/reduced/citlali_o152390_0_2_c2.yaml',
'2025-ENG-pointing-152388-152389/pointings/reduced/citlali_o152388_0_2_c2.yaml',
'2025-C1-COM-01/pointings-refactor/reduced/citlali_o152389_0_2_c8.yaml',
'2025-C1-COM-01/NGC4449/reduced/citlali_o152390_0_2_c5.yaml',
'beammaps/3c273/152307_redu50/citlali_o152307_0_2_c1.yaml',
'beammaps/pointings/reduced/redu01/citlali_o123424_0_2_c2.yaml']
out=[]
for p in paths:
 if not (R/p).exists():print('absentconfig',p);continue
 d=yaml.safe_load((R/p).read_text()); ins=d.get('inputs',[])
 for i in ins:
  paths2=[(s,'data') for s in i.get('data_items',[])]+[(s,'cal') for s in i.get('cal_items',[]) if s.get('filepath')]
  row=dict(config=p,identity=i.get('meta'),dependencies=[])
  for it,role in paths2:
   old=it.get('filepath'); b=pathlib.Path(old).name; candidates=bybase.get(b,[])
   row['dependencies'].append(dict(role=role,interface=it.get('meta',{}).get('interface'),recorded_path=old,local_candidates=[{'path':r['path'],'bytes':r['bytes']} for r in candidates]))
  out.append(row)
json.dump(out,open('/private/tmp/local-readiness-configs.json','w'),indent=2)
for q in out:
 miss=[i['recorded_path'] for i in q['dependencies'] if not i['local_candidates']];print(q['identity'],q['config'],len(q['dependencies']),'missing',miss)
