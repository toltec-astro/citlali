"""Advisory Welch SPOD on immutable, published CAL products.

No application import, flags, subtraction, rank selection or missing-data model.
Q is detector x realization, CSD = Q Q* / K, in (mJy/beam)^2/Hz.
Identity detector metric; arithmetic segment-mean removal; periodic Hann;
one-sided density with interior bins doubled. Actual gaps remain gaps.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
import numpy as np
from scipy import linalg
import yaml


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Inputs:
    """Per-run immutable file ledger; verify again after analysis."""
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        p = Path(path).resolve()
        actual = digest(p)
        require(expected is None or expected == actual, f'checksum mismatch: {p}')
        require(str(p) not in self.files or self.files[str(p)] == actual,
                f'input changed: {p}')
        self.files[str(p)] = actual
        return p

    def binary(self, path, dtype, expected=None):
        return np.fromfile(self.bind(path, expected), dtype)

    def yaml(self, path, expected=None, stop=None):
        p = self.bind(path, expected)
        if stop is None:
            text = p.read_text()
        else:
            # Large receipts contain unrelated repeated spectral records. Only
            # parse the top-level prefix, but bind the complete original bytes.
            lines = []
            with p.open() as f:
                for line in f:
                    if line.startswith(stop + ':'):
                        break
                    lines.append(line)
            text = ''.join(lines)
        return yaml.load(text, Loader=yaml.CSafeLoader)

    def unchanged(self):
        for p, h in self.files.items():
            require(digest(p) == h, f'input changed during diagnostic: {p}')
        return True


@dataclass
class CalInput:
    root: Path
    config: dict
    learning: dict
    cal: dict
    values: np.ndarray
    valid: np.ndarray
    times: np.ndarray
    native_rows: np.ndarray
    runs: list
    dt: float
    identities: list
    ptc: dict | None


def read_cal(root, config_path, inputs):
    root = Path(root)
    cfg = json.loads(inputs.bind(config_path).read_text())
    learn = inputs.yaml(root/'learning-receipt.yaml', stop='spectra')
    require(digest(config_path) == learn['configuration_sha256'], 'learning/config mismatch')
    cal = inputs.yaml(root/'donor-continuity/cal/receipt.yaml')
    require(cal['schema'] == 'citlali-cal-output-v1' and cal['unit'] == 'mJy/nominal-beam', 'CAL schema/unit')
    require(learn['network'] == cfg['network'] and learn['observation'] == cfg['observation'], 'scope mismatch')
    profile = json.loads(inputs.bind(cfg['filter_plan']['path'], cfg['filter_plan']['sha256']).read_text())
    factor = profile['factor']
    require(factor == learn['explicit_array_lowpass']['factor'], 'factor mismatch')
    native = inputs.binary(root/'native-time.f64', '<f8', learn['native_time_sha256'])
    require(len(native) == learn['rows'] and np.isfinite(native).all(), 'native time shape/finiteness')
    q = np.arange(0, len(native), factor)
    dt = learn['native_integration_seconds'] * factor
    runs = []
    last = 0
    for a, b in learn['native_runs']:
        require(a == last and a < b <= len(native), 'physical partition invalid')
        require(np.all(np.abs(np.diff(native[a:b])-dt/factor) < 1e-4*dt/factor), 'cadence outside bound')
        runs.append((int(np.searchsorted(q,a)), int(np.searchsorted(q,b))))
        last = b
    require(last == len(native), 'incomplete physical partition')
    ds = learn['detectors']
    require([d['channel'] for d in ds] == [d['channel'] for d in cal['detectors']] ==
            [d['channel'] for d in cfg['detectors']], 'detector ordering mismatch')
    require(len({d['occurrence'] for d in ds}) == len(ds), 'duplicate occurrence')
    values = np.full((len(q), len(ds)), np.nan)
    valid = np.zeros_like(values, dtype=bool)
    folder = root/'donor-continuity/cal'
    for j, d in enumerate(cal['detectors']):
        ch = d['channel']; prefix = f'channel-{ch}'
        require(d['selected_row'].endswith(':row='+ds[j]['occurrence'].rsplit('row=',1)[1]), 'APT occurrence mismatch')
        files = {name: inputs.binary(folder/name, '<f8' if name.endswith('f64') else '<i8' if name.endswith('i64') else '<u2', h)
                 for name,h in d['files'].items()}
        v, slots, causes = (files[prefix+s] for s in ('-value.f64','-slot.i64','-causes.u16'))
        require(len(causes)==len(q) and len(v)==len(slots)==d['available'], 'CAL shape')
        require(np.array_equal(slots, np.flatnonzero(causes==0)), 'CAL eligibility/slot mismatch')
        require(np.isfinite(v).all(), 'unexpected nonfinite in admitted CAL support')
        # Current producer phase-zero schedule is verified against published
        # finite output row identities, never inferred from compressed values.
        rows = inputs.binary(root/f'donor-continuity/{ch}-rows.i64', '<i8')
        require(np.isin(q[slots],rows).all(), 'CAL slot/native-output relation mismatch')
        values[slots,j] = v; valid[slots,j] = True
    ptcpath = root/'donor-continuity/ptc/receipt.yaml'
    ptc = inputs.yaml(ptcpath) if ptcpath.exists() else None
    if ptc:
        require(ptc['schema']=='citlali-ptc-output-v2' and ptc['source_CAL_receipt_sha256']==digest(folder/'receipt.yaml'), 'PTC/CAL parent mismatch')
        require(ptc['input_VAL_generation']==cal['output_VAL_generation'], 'PTC VAL mismatch')
    return CalInput(root,cfg,learn,cal,values,valid,native[q],q,runs,dt,ds,ptc)


def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def supported_windows(valid, runs, nfft, hop):
    """Integer half-open grid slots. Never shorten, pad or bridge a window."""
    require(valid.ndim==2 and valid.shape[1]>0 and nfft>=4 and hop>0, 'window shape/settings')
    common = valid.all(axis=1)
    windows = []
    stretches = []
    for a,b in runs:
        require(0<=a<b<=len(common), 'physical run bounds')
        for lo,hi in intervals(common[a:b]):
            lo+=a;hi+=a
            stretches.append((lo,hi))
            windows.extend((start,start+nfft) for start in range(lo,hi-nfft+1,hop))
    return np.asarray(windows,dtype=np.int64).reshape(-1,2), stretches


def fourier(values, windows, dt):
    require(len(windows)>0, 'insufficient support: no complete Fourier windows')
    n = int(windows[0,1]-windows[0,0])
    require(np.all(windows[:,1]-windows[:,0]==n), 'mixed Fourier lengths')
    w = .5-.5*np.cos(2*np.pi*np.arange(n)/n)
    density = np.sqrt(dt/np.dot(w,w))*np.ones(n//2+1)
    density[1: -1 if n%2==0 else None] *= np.sqrt(2)
    q = np.empty((len(windows),n//2+1,values.shape[1]),dtype=np.complex128)
    for i,(a,b) in enumerate(windows):
        y = values[a:b]
        require(np.isfinite(y).all(), 'unexpected nonfinite in admitted Fourier window')
        q[i] = np.fft.rfft((y-y.mean(axis=0))*w[:,None],axis=0)*density[:,None]
    return np.fft.rfftfreq(n,dt),q


def decompose(q, keep=10):
    """Snapshots x detectors at ONE frequency; positive Hermitian CSD."""
    require(q.ndim==2 and q.shape[0]>=2 and np.isfinite(q).all(), 'at least two finite realizations required')
    k,d = q.shape
    # q.T has physical detector coefficients; conjugation order matters.
    a = q.T/np.sqrt(k)
    small = k<d
    c = linalg.blas.zgemm(1, a, a, trans_a=2) if small else linalg.blas.zgemm(1, a, a, trans_b=2)
    ev,u = linalg.eigh(c,check_finite=False,driver='evr')
    ev=ev[::-1].real;u=u[:,::-1]
    require(ev[-1]>=-max(ev[0],1)*1e-10,'non-PSD CSD')
    ev=np.maximum(ev,0)
    rank=int(np.count_nonzero(ev>ev[0]*1e-12)) if ev[0]>0 else 0
    take=min(keep,rank)
    modes=linalg.blas.zgemm(1, a, u[:,:take])/np.sqrt(ev[:take]) if small else u[:,:take]
    return ev,modes


def overlap(a,b,rank):
    """Mean squared cosines + all principal angles, invariant to phase/rotation."""
    if rank<1 or a.shape[1]<rank or b.shape[1]<rank:
        return None
    s=linalg.svdvals(linalg.blas.zgemm(1, a[:,:rank], b[:,:rank], trans_a=2))
    s=np.clip(s,0,1)
    return dict(mean_cos2=float(np.mean(s*s)),angles_deg=np.degrees(np.arccos(s)).tolist())


def disjoint_halves(windows, ids):
    """No shared samples, with one full-window gap between chosen realizations.

    Alternating separated windows form interleaved halves, not a claim of
    statistical independence from slow processes. Return exact indices.
    """
    chosen=[];after=-1
    for i in ids:
        a,b=windows[i]
        if a>=after:
            chosen.append(int(i));after=b+(b-a)
    return np.array(chosen[::2],int),np.array(chosen[1::2],int)


def orthonormal_span(basis):
    u,s,_=linalg.svd(basis,full_matrices=False,check_finite=False)
    return u[:,s>s[0]*1e-10] if len(s) and s[0]>0 else u[:,:0]
