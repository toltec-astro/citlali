"""Bind existing POINT acquisitions, dependencies, execution and cost records."""
from pathlib import Path
import sys,json
D=Path(__file__).resolve().parent;W=D.parents[1];sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from inspect_forward_map import digest
from netCDF4 import Dataset
L=Path('/Users/gwilson/work_toltec/local_data/citlali-validation');records=[]
for obs,root in [(152420,L/'successor-rtc-point-transfer-completion-20260930/B-REF'),(152430,L/'successor-rtc-two-stage-20260928/B-REF-rep1')]:
 p=json.loads((root/'freeze/prospective.json').read_text());networks=[]
 for nw in range(1,5):
  cfg=json.loads((root/f'inputs/network{nw}-input.json').read_text())
  raw=Path(cfg['raw']['path']);tune=Path(cfg['tune']['path']);assert raw.is_file() and digest(tune)==cfg['tune']['sha256']
  with Dataset(raw) as nc:
   identity=[int(nc['Header.Toltec.'+k][:]) for k in ('ObsNum','SubObsNum','ScanNum')];tid=[int(nc['Header.Toltec.TargSweep'+k][:]) for k in ('ObsNum','SubObsNum','ScanNum')];assert identity==[obs,0,2];assert tid==[obs,0,1]
  networks.append(dict(network=nw,identity=identity,tune_identity=tid,raw=cfg['raw'],raw_bytes=raw.stat().st_size,raw_hash_status='Preserved existing execution binding; 152420 originals rehashed at fresh preparation; metadata-only inventory does not assert new152430rawhash',processed_tune=cfg['tune'],network_input=bound(root/f'inputs/network{nw}-input.json'),channels=len(cfg['detectors']),canonical_manifest=cfg['manifest']))
 launch=json.loads((root/'LAUNCH.json').read_text());cfg=L/'v2/science/pointings/reduced/redu00/citlali_o152389_0_2_c8.yaml'
 records.append(dict(observation=obs,executed_source=p['source_revision'],prospective=bound(root/'freeze/prospective.json'),dependencies=bound(root/'freeze/dependencies.json'),networks=networks,execution=bound(root/'EXECUTION.json'),launch=bound(root/'LAUNCH.json'),cost={k:launch.get(k) for k in ('exit_seconds','analysis_complete_seconds','aggregate_tree_sampled_RSS','cpu_seconds')},R_effective_config=bound(cfg)))
save_json(D/'results/point-inventory.json',dict(records=records,new_observation_starts=0))
