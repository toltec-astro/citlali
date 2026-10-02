"""Finite selected JINC cache enclosure, Decimal directed operations and series.

Validation-only, no sample deposition or alternate science reduction. Exact
parameter decimals, exact saved WCS degree values, and Machin pi define input.
See JINC_NUMERICAL_ARGUMENT.md for remainder, root and rounding arguments.
"""
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING
from functools import lru_cache
import math
import numpy as np

PREC=64
LO=Context(prec=PREC,rounding=ROUND_FLOOR)
HI=Context(prec=PREC,rounding=ROUND_CEILING)
class I:
    def __init__(self,a,b=None):
        self.lo=D(a);self.hi=self.lo if b is None else D(b)
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo>self.hi:raise ValueError('invalid enclosure')
    def __add__(self,o):
        o=as_i(o);return I(LO.add(self.lo,o.lo),HI.add(self.hi,o.hi))
    __radd__=__add__
    def __neg__(self):return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,o):return self+-as_i(o)
    def __rsub__(self,o):return as_i(o)+-self
    def __mul__(self,o):
        o=as_i(o);pairs=[(a,b) for a in (self.lo,self.hi) for b in (o.lo,o.hi)]
        return I(min(LO.multiply(a,b) for a,b in pairs),max(HI.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def __truediv__(self,o):
        o=as_i(o)
        if o.lo<=0<=o.hi:raise ValueError('interval division across zero')
        return self*I(LO.divide(D(1),o.hi),HI.divide(D(1),o.lo))
    def sqrt(self):
        if self.lo<0:raise ValueError('negative square root')
        # Decimal sqrt/exp are correctly rounded (half-even); widen one ulp.
        return I(max(D(0),LO.next_minus(LO.sqrt(self.lo))),HI.next_plus(HI.sqrt(self.hi)))
    def exp(self):return I(LO.next_minus(LO.exp(self.lo)),HI.next_plus(HI.exp(self.hi)))
    def absmax(self):return max(self.lo.copy_abs(),self.hi.copy_abs())

def as_i(x):return x if isinstance(x,I) else I(x)

def atan_reciprocal(n):
    x=I(1)/n;q=x*x;term=x;total=term
    for k in range(1,512):
        term=-(term*q);next_term=term/(2*k+1);total+=next_term
        rem=(term*q/(2*k+3)).absmax()
        if rem<D('1e-58'):return total+I(rem.copy_negate(),rem)
    raise ValueError('atan remainder not established')

@lru_cache(None)
def pi():return 16*atan_reciprocal(5)-4*atan_reciprocal(239)

def jinc(x):
    x=as_i(x);q=x*x/4;term=I(1);total=term
    for k in range(1,512):
        term=-(term*q/(k*(k+1)));total+=term
        ratio=q/((k+1)*(k+2));rem=(term*ratio).absmax()
        # Beyond the decreasing alternating tail, first omitted term bounds
        # every real argument in this interval; symmetric enclosure is safe.
        if ratio.hi<1 and rem<D('1e-54'):return total+I(rem.copy_negate(),rem)
    raise ValueError('Bessel remainder not established')

@lru_cache(None)
def first_root():
    lo,hi=D(3),D(4)
    assert jinc(I(lo)).lo>0 and jinc(I(hi)).hi<0
    while HI.subtract(hi,lo)>D('1e-48'):
        mid=LO.divide(LO.add(lo,hi),D(2));v=jinc(I(mid))
        if v.lo>0:lo=mid
        elif v.hi<0:hi=mid
        else:raise ValueError('root sign enclosure insufficient')
    return I(lo,hi)

def kernel_radius(radius,a,b,rmax=3):
    a,b=I(str(a)),I(str(b));r=as_i(radius)
    return jinc(2*pi()*r/a)*(-(2*r/b)*(2*r/b)).exp()*jinc(first_root()*r/rmax)

def enclose_cache(production,half,cdelt_deg,scale_numerator,b,*,phases=16,rmax="3"):
    """Current original diagonal square WCS only; each binary64 CDELT is exact.

    c=2,a=1.1 are fixed; phases and rmax are explicit selected constants.
    Defaults preserve the original POINT certificate exactly.
    Distinct radii are memoized; cache symmetries change no coefficients.
    """
    if abs(cdelt_deg[0])!=abs(cdelt_deg[1]):raise ValueError('selected diagonal square WCS required')
    step=I(D.from_float(abs(float(cdelt_deg[0]))))*pi()/180
    scale=I(str(scale_numerator))/1000/45
    if phases not in (1,16) or str(rmax) not in ("1.5","3"):raise ValueError("unreviewed enclosure realization")
    side=2*half+1;expected=(phases,phases,side,side)
    production=np.asarray(production).reshape(expected)
    errors=np.empty(expected);lower=np.empty(expected);upper=np.empty(expected);cache={}
    for qy in range(phases):
        for qx in range(phases):
            for iy,dy in enumerate(range(-half,half+1)):
                for ix,dx in enumerate(range(-half,half+1)):
                    # Exact midpoint phase displacement in units of 1/(2*phases).
                    n=(2*phases*dx+phases-1-2*qx)**2+(2*phases*dy+phases-1-2*qy)**2
                    if n not in cache:cache[n]=kernel_radius(I(n).sqrt()*step/(2*phases)/scale,'1.1',str(b),rmax=I(str(rmax)))
                    v=cache[n];index=(qy,qx,iy,ix);actual=D.from_float(float(production[index]))
                    e=max(HI.subtract(actual,v.lo).copy_abs(),HI.subtract(v.hi,actual).copy_abs())
                    # abs difference needs upper rounding regardless of sign.
                    e=max((I(actual)-v).absmax(),e)
                    errors[index]=math.nextafter(float(e),math.inf)
                    lower[index]=math.nextafter(float(v.lo),-math.inf);upper[index]=math.nextafter(float(v.hi),math.inf)
    return dict(errors=errors,lower=lower,upper=upper,unique_radii=len(cache),precision=PREC,
                pi=[str(pi().lo),str(pi().hi)],first_root=[str(first_root().lo),str(first_root().hi)])
