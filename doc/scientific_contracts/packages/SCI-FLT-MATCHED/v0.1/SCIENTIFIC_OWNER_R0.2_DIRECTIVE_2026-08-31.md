# SCI-FLT-MATCHED v0.1 — Scientific-Owner r0.2 Directive

Directive date: `2026-08-31`

Scientific owner: Grant Wilson

Status: binding review direction for the Stage B r0.2 draft; it selects no
`AO-001` through `AO-006` alternative and makes no implementation,
conformity, validation, performance, readiness, production, freeze, or Unity
claim

The exact owner-supplied directive was received as a 670-line plain-text
object with SHA-256
`0d68ec82837dcffed629bf266d313ef3cca0dcea150692ae9bcd62c1f57a33a1`.
This record preserves its complete normative disposition in a package-local,
reviewable form.

## Preserved r0.1 scientific architecture

The r0.2 draft shall preserve the local normalized matched-template amplitude
estimator, exact matching-template unit response, inverse-covariance GLS as
the sole minimum-conditional-variance route, weaker weights as nonoptimal,
separate observation/coadd applications, complete support, full response,
deterministic covariance propagation distinct from NOI, frozen-state NOI
parity, separate mismatch and learned-state uncertainty, atomic products, the
one-way FLT producer envelope for future FRUIT, and every explicit exclusion.

All 39 r0.1 requirement IDs and 18 r0.1 prediction IDs remain stable. Any
semantic repair is explicit in `SEMANTIC_CHANGE_MAP_R0.2.md`. No r0.1 option
or numerical value is owner-selected.

## Mathematical closure

1. Replace the colliding map-location notation `x/z/Q` with output anchor
   `p`, parent row `q`, parent signal `m_q`, template `t_{pq}`, complete
   bilinear weight `W_p`, numerator `n_p`, denominator `d_p`, estimate
   `\widehat a_p`, and response `R(p,r)`. Retain `x/r` for paired KID readout
   and bare `Q_p` for MAP/JINC authority.
2. Use one coordinate-basis convention: `W_p` contains every quadrature or
   pixel-measure factor. Define covariance matrices, action, units, adjoint,
   self-adjoint metric, and coordinate transformation consistently.
3. Define extraction `E_p`, local parent/template/covariance restrictions, and
   inversion after restriction. A restricted global precision is not the
   inverse of a restricted covariance and would be a different conditional
   method.
4. State the constrained singular GLS theorem with an exact estimable-subspace
   projector, positive-definite restricted covariance, identifiable projected
   template, and a fully specified competitor class. Distinguish covariance
   zero-variance, excluded/infinite-variance, weighting-null, and
   template-unidentifiable modes.
5. Require every admitted scientific weight to be exactly self-adjoint and
   positive semidefinite, with `d_p` exactly real and positive. Numerical
   tolerances belong only to engineering numerical conformance.

## Owner disposition: output-anchor lattice

For base v0.1, evaluate exactly one matched-template anchor at every exact
parent ordinary-MAP pixel center. Apart from exact support restriction, the
amplitude field has the identical parent WCS, frame, grid, shape, indexing,
ordering, and pixel-center convention. Template translation and phase are
defined at those anchors; no interpolation is admitted. Even/odd sampling and
ties follow the declared template representation and must be deterministic.
Subpixel, oversampled, or independently sampled output grids are separately
named successors.

## Estimand, support, state, and response

- Add the general relation `m = sum_r A_r t_r + b + n` and
  `E[\widehat a_p|g] = sum_r R(p,r)A_r + L_p b`. The method performs
  fixed-template, fixed-anchor, one-parameter linear amplitude estimation; it
  does not select or attribute sources, optimize position or morphology, fit
  background, deblend, infer existence, or build a catalog.
- Separate application support, template-query support, state-learning
  influence, response-query support, and storage footprint. Exact-zero
  coefficients do not dereference parent payload.
- Replace opaque learn-once language with `Learn -> Resolve -> Apply`.
  Externally declared state enters `Resolve`; `Apply` never updates from a
  target, candidate, or NOI member.
- Separate fixed-state response from full-procedure response. When state is
  learned from the parent, full-procedure response requires authorized exact
  rerun or declared finite difference. Until then it is unavailable, and a
  consumer may not strengthen the fixed-state result.

## Exact science, realized operators, and option refactor

The scientific operator is the exact normalized operator associated with the
selected `W_p`. Ordinary floating-point behavior belongs to one preregistered
engineering numerical-conformance profile. No `10^-3`, `10^-2`, `99%`, or
similar number is a privileged scientific threshold absent an owner-approved
scientific error budget. A deliberately different scientific operator is a
separate method or realization with its discrepancy retained in its estimand.

For a realized operator `\widetilde L`, attach actual
`R_realized=\widetilde L T` and
`C_realized=\widetilde L C_parent \widetilde L^T`; retain reference response
and covariance separately.

- `AO-003` separates covariance scope (complete, named projected,
  unavailable) from lossless representation (explicit, exact structured,
  exact lineage/on-demand).
- `AO-004` makes exact immutable-state reproducibility and the supported query
  vocabulary invariant; materialized, compact exact, and lineage forms are
  representations only.
- `AO-005` defines response domain, query vocabulary, validity, and consumer
  scope before its representation, and does not guess future FRUIT science.
- `AO-006` makes role-separated SCI-VAL semantics and the dependency graph
  normative. Profile layout is a lossless representation choice. A separate
  response-companion publication/use verdict is mandatory.

## Lifecycle, prediction, radial alternative, and boundaries

The draft distinguishes `not_requested`, `requested`, `effective`,
`disabled`, `unavailable`, `resolved`, `applied`,
`complete_publication_candidate`, `realized`, `failed`, `not_produced`, and
`superseded`. SCI-VAL evaluates an owner-approved named-use profile on the
separate axes request, applicability, eligibility, and realization; it does
not perform filtering or publication and is not observational validation.

`PRED-008` is repaired so convergence applies to draws from the exact declared
parent stochastic law with conditional covariance `C_parent`. An NOI ensemble
supports the prediction only under separate authority establishing that same
target covariance.

`AO-001-C` is renamed
`radially_symmetrized_field_power_spectral_weighting`. It receives exact
half-open bins, ties, finite nonnegative weights, a positive denominator,
conjugate/multiplicity accounting, unit-consistent nonnegative regularization,
separate excluded-null modes, no implicit sparse-bin borrowing, and truthful
source/residual imprint. Radialization itself implies no noise, covariance,
stationarity, isotropy, or optimality claim.

The template-response scientific object may be exactly materialized,
structured, or reconstructed from exact lineage without changing the
estimand.

Exact r0.2 boundary drafts are required for MAP input, template input, NOI
output, and the one-way FLT-to-FRUIT producer envelope. A route-status table
must keep generic estimator definition separate from parent, template,
weighting, numerical conformance, response, covariance, NOI, FRUIT-envelope,
SCI-VAL profile, and implementation-assessment availability. No option may
manufacture an unavailable parent.

## Delivery and nonclaims

The r0.2 return includes both revised views, the exact shared core, notation,
algebra/theorem, anchor, sky/response, state, approximation, representation,
lifecycle, prediction, radial-option, boundary, route, source-byte,
traceability, parity, build, PDF-QA, and manifest records. The exact approved
eight-object author packet remains byte-identical and its manifest SHA-256 is
reproduced as
`255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.

No returned artifact may claim a numerical route, implementation conformity,
covariance or response fidelity, observational validation, source-detection
performance, readiness, scientific freeze, production suitability,
production authorization, or Unity activity.
