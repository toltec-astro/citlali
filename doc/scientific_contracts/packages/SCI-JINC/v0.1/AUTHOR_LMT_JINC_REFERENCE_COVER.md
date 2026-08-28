# SCI-JINC v0.1 — LMT OTF/JINC Method Reference Cover

Status: proposed sanitized author input; awaiting scientific-owner approval

Prepared: `2026-08-28`

This cover controls use of
`references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf`,
SHA-256
`a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9`.
The excerpt is derived page-for-page from the immutable 42-page source,
SHA-256
`835fb02e842c9109c2c7ad3f03288882dfac283e63bfcd0f818c7d5379e7e5cd`.

## Admitted Scientific Role

The excerpt is an owner-designated authoritative LMT reference for the generic
OTF/JINC analytic method and its aperture/spatial-frequency motivation. It is
not a TolTEC instrument-parameter, optimization, validation or performance
reference.

In memo Equations 6--7, the memo's lowercase notation satisfies
`jinc_memo(x)=J_1(x)/x`; therefore its written factor `2 jinc_memo(x)` is the
peak-normalized function used by SCI-JINC:

```text
J(x) = 2 J_1(x)/x for x != 0;  J(0)=1.
```

This notation translation changes no analytic function. It prevents collision
with package notation and fixes the removable limit.

With `r' = r/(lambda/D) >= 0`, memo Equation 9 becomes

```text
kappa(r'; a,b,c,RMAX)
  = J(2 pi r'/a)
    exp[-(2 r'/b)^c]
    J(j_1,1 r'/RMAX),

j_1,1 = 3.831706  (memo value for the first positive zero of J_1).
```

Consequently:

- `kappa` is dimensionless and `kappa(0)=1`;
- the ordered tuple is `(a,b,c,RMAX)`;
- `a`, `b`, `c`, `RMAX` and `r'` are dimensionless in the memo convention;
- `a` scales the first JINC argument;
- `b` scales the generalized-exponential envelope;
- `c` is its exponent;
- `RMAX` places the first zero of the second JINC factor; and
- analytic zeros and finite signed lobes remain real kernel values.

## Binding SCI-JINC Supersessions

The following later owner-approved SCI-JINC decisions override the memo where
their scopes intersect:

1. **Square support:** `RMAX`/`r_max` also fixes the fully populated square
   cache half-width. It is not a radial cutoff. Coefficients at square corners,
   including radii greater than `RMAX`, remain in the operator.
2. **Point phase:** the kernel is point-evaluated at the selected quantized
   sample phase. It is not pixel-area integrated.
3. **Signed estimator:** the coefficient is `kappa_ip` and enters the retained
   `N_p/C_p` estimator, `Q_p`, response, covariance and coefficient-squared
   time under the package's collision-free equations.
4. **Conditioning and products:** the memo supplies no SCI-JINC cancellation,
   covariance, formal-support, atomic-product or provenance policy; the
   sanitized inherited-decision table controls those subjects.

The memo's statements that convolution ends at `RMAX` and its figure language
describing radial truncation are unavailable for SCI-JINC support.

## TolTEC-Specific Exclusions

The memo is oriented to LMT 3-mm spectroscopic mapping. The future author must
not import:

- `a=1.1`, `b=4.75`, `c=2.0`, `RMAX=3`, or any other memo value as a TolTEC
  value, default, recommendation or optimum;
- the 86-GHz/3.4-mm SEQUOIA beam, sampling, cell scale or Monte Carlo results;
- a single monochromatic `lambda/D` value as the effective scale for any
  TolTEC array without separate TolTEC authority;
- spectral-line cube, reference-spectrum, system-temperature, RMS-weight or
  focal-plane geometry semantics; or
- any achieved response, noise, flux, resolution, SNR, validation, readiness
  or production conclusion.

SCI-JINC preserves `r'_a=r/s_a`, with `s_a` an explicit array-associated
angular scale, but the memo's `s=lambda/D` does not authorize a TolTEC
realization. No TolTEC numerical parameter set is scientifically authorized
for v0.1; the affected numerical route is typed unavailable without a hidden
default. A future parameter optimization is a separate scientific tranche
with an explicit TolTEC objective and appropriate beam, response, and/or noise
evidence.

## Firewall

Only the exact five-page method excerpt and this cover are proposed author
inputs. The complete memo, intake record, Appendix C simulations, raw owner
feedback, implementation, schemas, audits, tests, reductions, Unity and
production evidence remain prohibited. If the excerpt and cover are
insufficient, the author returns a precise question rather than opening the
full memo or searching for a substitute.
