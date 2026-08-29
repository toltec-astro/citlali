# SCI-JINC v0.1 — Discrete Geometry And WCS-Metric Decision Record r0.2

Status: implementation-blind Stage B author-draft convention; uniquely
specified for review, not implementation-assessed, numerically certified, or
scientific-owner-approved as a completed Stage B contract

Prepared: `2026-08-29`

Scientific owner: Grant Wilson

## Purpose And Scope

The targeted repair requires one exact, replayable discrete JINC operator.
This record fixes center rounding, half-pixel ties, residual phase, phase-bin
membership and representatives, cache keys and extents, point-evaluation
geometry, target-WCS metric, and finite-map membership. These choices refine
the already approved point-phase, fully populated square-support, and
center-required finite-map semantics. They do not select TolTEC numerical
parameters, inspect an implementation, or certify numerical adequacy.

## Coordinate And Target-WCS Contract

The exact AST input is the continuous FITS axis-1/axis-2 pixel coordinate

```text
u_i = (x_i, y_i)
```

from `SCI-AST:rtc_output_grid_coordinates@1`, associated with the same
processed RTC sample `n` and the exact PTC occurrence entering JINC. FITS
pixel coordinates are one-based with integer pixel centers. The finite target
pixel-center domain is

```text
D = {1,...,N_x} x {1,...,N_y}.
```

It is a finite rectangle with no circular axis, wrap, reflection, or other
topological identification.

Base v0.1 accepts only an affine tangent-plane target WCS whose pixel matrix
has the form

```text
M = Delta R,
Delta > 0,
R in O(2),
M^T M = Delta^2 I.
```

`Delta` is the positive angular scale per pixel. `R` carries the exact axis
order, signs, rotation, and handedness, including a possible reflection. The
WCS identity additionally binds projection, coordinate frame, frame
parameters, CRVAL, one-based CRPIX, units, finite bounds, plan, version, and
lifecycle generation. The affine tangent-plane coordinate of pixel center
`q` is obtained from that exact WCS; for separations within the plane, the
reference point cancels and `M` supplies the metric.

An anisotropic or skewed matrix, for which no positive `Delta` satisfies
`M^T M=Delta^2 I`, is incompatible with base v0.1. It is typed incompatible
rather than approximated by a scalar pixel size. A general-WCS successor would
need separately authorized row/column extents and an exact metric.

## Exact Center And Phase Convention

For each axis component `k` in `{x,y}`, define

```text
c_i,k   = floor(u_i,k + 1/2),
phi_i,k = u_i,k - c_i,k.
```

Therefore `phi_i,k` lies in `[-1/2,1/2)`. An exact half-pixel tie
`u_i,k=m+1/2` maps to center `m+1`: ties are toward the positive coordinate
axis, equivalently toward positive infinity.

Let `n_sub` be the explicitly selected positive integer phase resolution. No
odd-parity restriction and no hidden default is authorized. Divide the phase
domain into equal left-closed, right-open intervals. The exact bin and its
midpoint representative are

```text
q_i,k       = floor[n_sub (phi_i,k + 1/2)],
phi_hat_i,k = -1/2 + (q_i,k + 1/2)/n_sub,
q_i,k in {0,...,n_sub-1}.
```

The exact half-open phase domain guarantees the bin index is in range without
clamping or wrapping. The normative phase key is the ordered pair
`(q_i,x,q_i,y)` in FITS axis order. Its exact one-dimensional form is

```text
beta_i = q_i,x + n_sub q_i,y,
```

with axis 1 varying fastest. The complete logical coefficient-cache key is
`(beta_i,d_x,d_y)`, equivalently `(q_i,x,q_i,y,d_x,d_y)`. Storage layout may
differ only if it preserves this exact logical key through a value-preserving
bijection.

For odd `n_sub`, exact zero phase has representative zero. For even `n_sub`,
zero phase lies in the positive-side central bin and has representative
`+1/(2 n_sub)` on that axis. This axis-oriented asymmetry is an observable
consequence of the normative half-open convention. It shall be disclosed and
tested; it shall not be silently symmetrized, shifted, or used to forbid even
`n_sub`.

## Exact Square Extent And Point Evaluation

For array `a`, define the angular radius of the second-factor first zero and
the integer cache half-width by

```text
R_a = s_a (r_max)_a,
h_a = ceil(R_a / Delta).
```

Both `s_a` and `(r_max)_a` belong to the exact authorized array-associated
parameter-set identity. The fully populated square offset domain is

```text
S_a = {-h_a,...,+h_a}^2.
```

No predicate `r <= R_a`, `r' <= (r_max)_a`, or equivalent circular cutoff is
part of membership. Every square offset, including a corner beyond `R_a`, is
point-evaluated.

For offset `d=(d_x,d_y)` in `S_a`, the candidate destination pixel and the
quantized point-phase angular separation are

```text
p_i,d = c_i + d,
r_i,d = || M (d - phi_hat_i) ||_2,
r'_i,d = r_i,d / s_a.
```

Thus the destination is evaluated at the center of pixel `p_i,d`, while the
quantized source position is `c_i+phi_hat_i`. Axis signs, rotation, and
handedness are retained by `M`; the Euclidean norm makes the radial kernel
invariant under the allowed orthogonal `R`.

Define the first positive Bessel root exactly as

```text
j_1,1 = min{x > 0 : J_1(x) = 0}.
```

The point coefficient is

```text
kappa_i,d = Jcal(2 pi r'_i,d/a_a)
            exp[-(2 r'_i,d/b_a)^(c_a)]
            Jcal(j_1,1 r'_i,d/(r_max)_a),
```

where `Jcal(x)=2 J_1(x)/x` for nonzero `x` and `Jcal(0)=1`. The decimal
`3.831706` is only a nonnormative approximation to `j_1,1`. With the exact
root, the second factor has its analytic first zero at `r_i,d=R_a`.

The phase-quantized operator above is the normative base-v0.1 operator. A
continuous-phase operator is not a second normative branch. It may be used
only as an oracle defined by a future exact numerical-adequacy profile.

## Finite-Map Membership And Edge Rule

Membership is resolved in this order:

1. Resolve the upstream JINC sample-admission decision and exact AST/PTC join.
2. Compute the exact rounded center `c_i`.
3. If `c_i` is outside `D`, set `I_i,p=0` for every destination pixel and stop
   this occurrence before footprint evaluation.
4. If `c_i` is inside `D`, enumerate every `d` in `S_a`.
5. Retain the candidate only when `p_i,d=c_i+d` is inside `D` and every other
   local input/coefficient gate passes.

An outside center contributes nowhere even if its square overlaps the map.
An in-map center uses the ordinary destination crop. There is no wrap,
reflection, footprint completion, interior renormalization, edge correction,
replacement contribution, or JINC-then-crop equivalence requirement.

## Plan, Compatibility, And Replay Binding

Before numerical mutation, the resolved JINC plan and compact generative
record shall bind:

- exact AST coordinate role, occurrence, sample `n`, and parent join;
- complete target-WCS identity and proof of `M^T M=Delta^2 I`;
- dimensions and finite bounds;
- `n_sub`, center/tie convention, phase domain, interval polarity,
  bin-index formula, representative formula, and logical key order;
- exact array parameter-set identity and units;
- `R_a`, `h_a`, square offset domain, destination-point convention, and edge
  rule;
- exact analytic-family identity and exact `j_1,1` definition;
- requested, effective, observation-resolved, and realized identities; and
- completion, failure, and supersession state.

A changed center rule, tie direction, phase interval polarity, representative,
key mapping, WCS class/metric, extent formula, evaluation point, square
membership, or edge rule is a changed operator and requires a versioned
successor. Numerical agreement by coincidence is not an alias.

## Engineering Choices Left Open

After the mathematical operator is fixed, these remain engineering choices:

- the Bessel-function evaluation algorithm, subject to a future exact
  numerical-adequacy profile;
- cache storage and physical memory layout that preserve the logical key;
- summation algorithm and reduction organization;
- thread order and parallel execution; and
- internal zero-based indices, provided the exact one-based FITS convention
  is preserved at the scientific boundary.

No such choice may change membership or the coefficient selected by the
mathematical operator.

## Required Draft Predictions

Future conformance work shall, without claiming present execution, cover:

- values immediately below, at, and above positive and negative half-pixel
  ties on both axes;
- values immediately below, at, and above every phase-bin edge;
- `n_sub=1`, representative odd values, and representative even values,
  including the exact even-`n_sub` zero-phase asymmetry;
- cache-extent boundaries and square corners beyond `R_a`;
- analytic zeros and finite negative lobes;
- centers immediately inside and outside every finite-map edge and corner,
  including outside centers whose squares overlap the map;
- allowed rotations and both handedness choices; and
- rejection of skewed and anisotropic target metrics.

Listing these predictions does not establish implementation conformity,
numerical adequacy, validation, achieved performance, readiness, or
production authorization.
