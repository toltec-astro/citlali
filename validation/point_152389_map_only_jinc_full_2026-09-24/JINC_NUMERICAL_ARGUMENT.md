# POINT-152389-JINC-NUMERICAL-ADEQUACY-v1 implementation argument

Owner scientific inequality and ancillary requirements are section4 of the
archived FULL_RUN directive. This document describes engineering proof, not
another scientific criterion. Final certificate binds actual cache and output.

## Finite reference

The validation-only Decimal helper uses64 significant decimal digits with
separate ROUND_FLOOR/ROUND_CEILING contexts for each elementary interval
operation. Decimal sqrt/exp are correctly rounded half-even; each endpoint is
widened one representable decimal step, enclosing the corresponding exact real
function. Decimal negation/absolute use copy_negate/copy_abs to avoid ambient
context rounding. Float inputs use Decimal.from_float, hence exact saved values.
Exact finite decimal shapes and rational scale formulas are not binary64
approximations in the reference. Original diagonal square WCS CDELT degrees
are fixed saved binary64 values, multiplied by enclosed pi/180; production
conversion to radians is therefore covered by the cache coefficient difference.

Pi uses Machin16atan(1/5)-4atan(1/239). Alternating atan series remainder is at
most the first omitted term. J(x)=sum_k (-x²/4)^k/[k!(k+1)!], starting at1;
once subsequent terms decrease, its alternating remainder is bounded by the
first omitted term, uniformly over the input interval. Directed interval
polynomial evaluation plus a symmetric first-omitted-term bound encloses J.
Truncation stops only after a proved bound, never after agreement of precisions.

The first positive J1 root is enclosed by sign-certified bisection on[3,4].
Uniqueness/firstness: d(2J1(x)/x)/dx=-2J2(x)/x. For0<x<=4,
8J2(x)/x² >= 1-x²/12+x^4/384-x^6/23040 >=7/45>0.
The first inequality is the decreasing alternating-series lower truncation;
the cubic in x² decreases on[0,16] and is7/45 at16. Thus J decreases strictly
from1 until4, and interval signs at3 and4 enclose its unique first zero.
Reference precision refinement changes no selected kernel parameter.

With16 phases, offsets are odd integers/32; for the actual diagonal square WCS
radius depends exactly on (32dx+15-2qx)²+(32dy+15-2qy)². Identical radii reuse
one enclosure. This is algebraic symmetry, not removal of any square member.
For each actual production binary64 cache value k_hat, store e_k rounded upward
such that |k_hat-k_star|<=e_k, plus outward binary64 endpoints of k_star.
All cache entries, including square corners and negative lobes, are covered.

## Actual sums (implementation specification)

Assume IEEE754 binary64 round-to-nearest,53 significand bits, no fast-math or
contraction, fixed documented worker/block order. Check actual significand
widths rather than assuming long double is wider. Reject overflow and observed
subnormal/underflow paths from certification; exact zeros remain permitted.
No such exclusion is silently mapped to zero sky. Bound operations use explicit
nextafter enclosures, including lower denominator and tolerance bounds.

For actual per-pixel member count n and w worker blocks, let K=n+w+6, covering
at most n+w additions along any term path and at most6 term operations below.
u=2^-53, gamma=K*u/(1-K*u), evaluated upward. No selected term may have a longer
path. The accompanying absolute-term sums L_hat and coefficient-error sums
D_hat use the same tree. Thus L_star<=L_hat/(1-gamma), D_star<=D_hat/(1-gamma).
For each accumulator E=(gamma*L_hat+D_hat)/(1-gamma) is an upper bound,
with every operation rounded upward. The four coefficient-error terms are:

- N: omega*abs(t)*e_k
- C: omega*e_k
- Q: omega*e_k*(2abs(k_hat)+e_k)
- T: e_k*(2abs(k_hat)+e_k)/fs

The corresponding L terms use absolute computed contribution products.
Gamma's operation path includes contribution products and their absolute-sum
approximation; division is included for T. Positive sums and error sums are
checked for range failures. Realized counts/tree must be retained. Exact-Q
positivity can be proved by Q_hat-E_Q>0; exact-zero/no-member domains are explicit.
Q/T ancillary pass requires E<=1e-6*(hat-E) with a positive lower reference.

For normal binary64 output m_hat=fl(N_hat/C_hat), E_R=gamma_1*abs(N_hat)
encloses the exact residual abs(m_hat*C_hat-N_hat). Evaluate the owner's full
quotient bound with upward numerator/division and downward positive denominator.
Both exact decimal tolerance constants are bounded downward; threshold
arithmetic rounds downward. A passing conservative comparison proves the
required exact inequality, including negative C. Algebraic support alone is
insufficient. Binary64 serialization must preserve actual hatted values bitwise.

Selected input coordinates are authenticated native AST binary64 payloads.
Their exact center/phase membership is audited separately. For positive in-map
centers, split floor(x) and exact fractional remainder rather than forming
x+1/2 near a rounding boundary;16-bin scaling is exact. Out-of-map center
admission is tested before depositing cropped square members. A wrong join or
membership is an operator failure, not an error absorbed by these bounds.
