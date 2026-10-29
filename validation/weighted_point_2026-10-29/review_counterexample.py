"""Synthetic exact-helper numerical reproduction; no observations or source edits.
Candidate: 4f7d31957d2135601145c63387e310622fe16d64
Mirrors Stats, fit, Generator, quantile, and interval block filtering in
src/citlali/core/pipeline/point_optical_response.cpp. Uses Python double
arithmetic for the Apple arm64 native double/long-double realization.
Generator below is the actual C++ SplitMix64/rejection-index algorithm,
with a fixed test seed of 42 (not NumPy RNG and not a detector seed claim).
Normalization is 1, which is the normal detector-response scale.
"""
import math
import sys
import numpy as np

EPS = sys.float_info.epsilon
TINY = sys.float_info.min
MASK = (1 << 64) - 1

class Stats:
    def __init__(self):
        self.w = self.x = self.y = self.xx = self.xy = 0.
        self.n = 0
    def add(self, a, b, weight):
        self.w += weight; self.x += weight*a; self.y += weight*b
        self.xx += weight*a*a; self.xy += weight*a*b; self.n += 1
    def merge(self, b):
        self.w += b.w; self.x += b.x; self.y += b.y
        self.xx += b.xx; self.xy += b.xy; self.n += b.n
    def solve(self):
        d = self.w*self.xx - self.x*self.x
        if self.n < 3 or not self.w > 0 or not d > 32*EPS*max(self.w*self.xx, self.x*self.x):
            return None
        beta = (self.w*self.xy - self.x*self.y)/d
        intercept = (self.y - beta*self.x)/self.w
        return (beta, intercept) if math.isfinite(beta) and math.isfinite(intercept) else None

class Generator:
    def __init__(self, state): self.state = state
    def next(self):
        self.state = (self.state + 0x9e3779b97f4a7c15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
        return z ^ (z >> 31)
    def index(self, n):
        threshold = ((-n) & MASK) % n
        while True:
            x = self.next()
            if x >= threshold: return x % n

def fit(x, y):
    s = Stats()
    for a, b in zip(x, y): s.add(a, b, 1.)
    beta, intercept = s.solve()
    weights = [1.]*len(x)
    for _ in range(20):
        residual = [b-intercept-beta*a for a,b in zip(x,y)]
        center = float(np.median(residual))
        sigma = 1.4826*float(np.median([abs(r-center) for r in residual]))
        if not math.isfinite(sigma) or sigma <= 0: break
        nxt = Stats()
        for j,(a,b,r) in enumerate(zip(x,y,residual)):
            weights[j] = min(1.,1.345*sigma/max(abs(r),TINY))
            nxt.add(a,b,weights[j])
        new_beta,new_intercept = nxt.solve()
        done = abs(new_beta-beta) <= 1e-10*(1+abs(beta)) and abs(new_intercept-intercept) <= 1e-10*(1+abs(intercept))
        beta,intercept = new_beta,new_intercept
        if done: break
    return beta,intercept,weights

def quantile(v, q):
    x=q*(len(v)-1);i=int(math.floor(x))
    return v[i]+(x-i)*(v[min(i+1,len(v)-1)]-v[i])

xs,ys,bs=[],[],[]
for block in range(12):
    if block in (0,1,6,7):
        x=np.linspace(-1,1,10).tolist()
        y=[(.03+.01*(block%6))*a+.0001*math.sin(j) for j,a in enumerate(x)]
    else:
        x=[{2:-100.,3:-70.,4:70.,5:100.,8:-90.,9:-60.,10:60.,11:90.}[block]]
        y=x.copy()
    xs.extend(x);ys.extend(y);bs.extend([block]*len(x))
print('Candidate: 4f7d31957d2135601145c63387e310622fe16d64')
print('RNG: exact native SplitMix64 and rejection-index; fixed synthetic seed 42')
print('Block domain: 12; paired bins: 48; normal-response normalization: 1')
for label,selection in [('full',range(12)),('early',range(6)),('late',range(6,12))]:
    rows=[i for i,b in enumerate(bs) if b in selection]
    x=[xs[i] for i in rows];y=[ys[i] for i in rows];b=[bs[i] for i in rows]
    beta,intercept,weights=fit(x,y)
    grouped={}
    for a,c,block,w in zip(x,y,b,weights): grouped.setdefault(block,Stats()).add(a,c,w)
    kept={block:s for block,s in grouped.items() if s.solve() is not None}
    blocks=list(kept.values());slopes=[];rng=Generator(42)
    for _ in range(2000):
        s=Stats()
        for unused in blocks: s.merge(blocks[rng.index(len(blocks))])
        result=s.solve()
        if result is None: raise RuntimeError('resample_rank_unavailable')
        slopes.append(result[0])
    slopes.sort()
    lo,hi=quantile(slopes,.005),quantile(slopes,.995)
    decision='weak_response' if lo>-.1 and hi<.1 else 'not_weak'
    print(f'{label}: full-domain slope={beta:.15g}; retained blocks={list(kept)}; interval=[{lo:.15g}, {hi:.15g}]; f=0.1 decision={decision}')
