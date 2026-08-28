# SCI-JINC v0.1 — Analytic Identity And TolTEC Parameter Semantics

Status: ODQ-102B semantic/no-numerical-route disposition incorporated; exact
TolTEC numerical realization unavailable; Stage B blocked

Scientific owner: Grant Wilson

Prepared: `2026-08-28`

## Exact Scientific Sources

1. Independent signed-estimator core:
   `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`,
   SHA-256
   `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24`.
2. F. Peter Schloerb, *Spectral Line Data Reduction at the Large Millimeter
   Telescope*, `2019-07-23`, exact PDF SHA-256
   `835fb02e842c9109c2c7ad3f03288882dfac283e63bfcd0f818c7d5379e7e5cd`.
3. Sanitized page-exact method excerpt, original PDF pages 15--19, SHA-256
   `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9`,
   controlled by
   [`AUTHOR_LMT_JINC_REFERENCE_COVER.md`](AUTHOR_LMT_JINC_REFERENCE_COVER.md).
4. The eight exact owner decisions and their digests in
   [`AUTHOR_DECISIONS_AND_OWNERSHIP.md`](AUTHOR_DECISIONS_AND_OWNERSHIP.md).

The memo is owner-designated authoritative for the generic LMT OTF/JINC method
and physical motivation. It is a 3-mm spectroscopic-receiver reference, not
TolTEC array-parameter, optimization, validation or performance authority.

## Collision-Free Peak-Normalized Convention

Memo Equations 6--7 use `jinc_memo(x)=J_1(x)/x` and write the normalized
function as `2 jinc_memo(x)`. SCI-JINC instead defines one unambiguous
peak-normalized function:

```text
J(x) = 2 J_1(x)/x,  x != 0
J(0) = 1,
```

where `J_1` is the first-order Bessel function of the first kind. This is an
exact notation normalization, not a change of function or amplitude.

For physical angular separation `r` and an explicit array-associated angular
scale `s_a`,
define

```text
r'_a = r/s_a >= 0.
```

The memo uses `s=lambda/D`. That is scientific precedent for a diffraction-
scale realization, not authorization of any TolTEC realization. Its Equation
9 then content-binds the generic
dimensionless analytic coefficient family as

```text
kappa_a(r'_a; a_a,b_a,c_a,RMAX_a)
  = J(2 pi r'_a/a_a)
    exp[-(2 r'_a/b_a)^c_a]
    J(j_1,1 r'_a/RMAX_a),

j_1,1 = 3.831706  (the memo's first-positive-zero constant for J_1).
```

The ordered parameter tuple is `(a,b,c,RMAX)`. All four parameters and `r'`
are dimensionless in the memo convention. `a` scales the first JINC argument;
`b` scales the generalized-exponential envelope; `c` is its exponent; and
`RMAX` places the first zero of the second JINC factor. The amplitude is fixed:
all three factors approach one at zero, so `kappa_a(0)=1`.

Analytic zeros of either JINC factor make the product exactly zero. Between
zeros, finite positive and negative lobes retain the ordinary sign of the
complete product; they are not clipped, absolutized or confused with outside
support.

## Owner-Approved SCI-JINC Supersessions

The memo describes radial truncation at `RMAX`. SCI-JINC supersedes that
branch:

- `r_max` places the first zero of the second JINC factor **and** fixes the
  fully populated square-cache half-width;
- no radial membership predicate remains;
- square corners at radii greater than `r_max` are evaluated with the same
  analytic continuation and remain part of the fixed accumulator membership;
- finite-map crop alone removes outside-map square pixels; and
- the coefficient is point-evaluated at an exact quantized sample phase, not
  pixel-area integrated.

Thus `r_max` is not a strict maximum evaluated radius. The memo's continuous
formula supplies the coefficient at every resolved square-cache point,
including points beyond the second factor's first zero.

## Owner-Approved Parameter Semantics And Unavailability

The memo closes the generic formula, Bessel convention, zero limit, amplitude,
envelope, parameter ordering/roles/units and second-factor relation to
`r_max`. `SCI-JINC-ODQ-102B` preserves `r'_a=r/s_a` and establishes that:

1. `s_a` is an angular quantity associated with stable array `a`;
2. `a_a`, `b_a`, `c_a`, and `(r_max)_a` are dimensionless kernel parameters
   whose array association is explicit rather than inferred from a shared
   implementation field;
3. every numerical realization binds the parameter-set owner, source,
   version, units, array association, admissibility, and requested/effective/
   observation-resolved/realized identities; and
4. absent a scientifically authorized parameter set, the affected numerical
   route is unavailable, with no hidden default or inherited fallback.

The memo's FCRAO values `a=1.1`, `b=4.75`, `c=2.0`, `RMAX=3` and Appendix C
86-GHz/3.4-mm simulations are explicitly unavailable as TolTEC values or
performance evidence.

Inherited implementation scales and shape/extent values are not scientific
authority and are not author inputs. No physical interpretation or normative
rule may be inferred from them.

## Deferred TolTEC Numerical Realization

Selecting or optimizing TolTEC numerical values is a separate downstream
scientific exercise. It must name an objective and use appropriate TolTEC/LMT
beam, response, and/or noise evidence. SCI-JINC v0.1 neither performs that
study nor prejudges whether it will retain, approximate, or replace inherited
implementation values.

The generic analytic identity and parameter semantics are recoverable without
implementation. Stage B remains blocked by the other Stage A dispatch gates
and exact packet approval; it may describe typed numerical unavailability but
may not select TolTEC values.
