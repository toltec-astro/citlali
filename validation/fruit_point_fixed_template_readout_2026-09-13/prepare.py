"""Bind approved sources and saved products without evaluating a fit."""
import subprocess
import datetime
from common import HERE,DIAG,PRIOR,OUT,SEEDS,write,digest,preserve
assert not (HERE/'FREEZE.json').exists()
write(HERE/'PRESERVATION_START.json',preserve())
paths=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'PRESERVATION_START.json',DIAG/'NEXT_EXPERIMENT.md',DIAG/'diagnose.py',DIAG/'RESULT_MANIFEST.json',PRIOR/'CASES.json',OUT/'RECEIPTS.json',OUT/'geometry.npz',OUT/'H_truth.npz',OUT/'D_truth.npz']
paths += [OUT/f'{name}_{seed}'/arm/'pass06_maps.npz' for name in ['H','D'] for seed in SEEDS for arm in ['P','C']]
write(HERE/'FREEZE.json',dict(identity='SCI-FRUIT-POINT-FIXED-TEMPLATE-READOUT@r0.1',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    maximum_linear_fits=48,files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]))
(HERE/'FREEZE.json.sha256').write_text(digest(HERE/'FREEZE.json')+'  FREEZE.json\n')
print('Frozen',len(set(paths)),'files; preserved prior evidence.')
