# SCI-JINC — Signed-Coefficient JINC Observation Mapmaker Scope Brief

Status: ODQ-101/102B/103 Stage A successor candidate; predecessor Stage A bytes
owner-approved; successor requires renewed exact-byte approval; Stage B blocked

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-28`

Starting authority:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

Approved predecessor identifier:
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f`

Controlled ODQ-101 successor source:
`54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md`,
SHA-256
`4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
Scientific authorship begins after the
[frozen SCI-MAP v0.1/r0.7.1 authority](../../SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md),
which remains unchanged.

- Completed recovery: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Adopted: the signed `N_p/C_p` estimator; distinct `N_p`, `C_p`, `Q_p`;
  positive, zero and negative JINC lobes; eight D003 owner decisions; atomic
  destination identity; and owner acceptance of the retained estimator
- Cited: the implementation-independent core at
  `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`,
  SHA-256
  `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24`
- Abstracted: exact frozen PTC and AST ownership, quantity, identity,
  coordinate, response/covariance, lifecycle and cause boundaries; no source
  or schema mechanics
- Superseded: the independent core's circular radial-cutoff branch and
  pixel-area-integrated branch; both are unavailable
- Deferred/excluded: implementation, schemas, tests, products, audits,
  repairs, validation, reductions, Unity, integration, achieved performance,
  readiness, production behavior, historical tuning, the full 42-page memo
  and its 3-mm/FCRAO numerical examples and simulations
- Cited as a sanitized reusable scientific reference: F. Peter Schloerb's
  `2019-07-23` LMT OTF/JINC method, exact original SHA-256
  `835fb02e842c9109c2c7ad3f03288882dfac283e63bfcd0f818c7d5379e7e5cd`,
  through a page-exact method excerpt and explicit TolTEC-exclusion cover
- Adopted as a controlled successor decision: one PTC-owned versioned
  positive analysis/gridding coefficient registry with explicit per-family
  `SCI-MAP`/`SCI-JINC` consumer permissions, staged selection identity and
  fail-closed no-fallback semantics, using the exact post-freeze source only
  through
  [`AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md`](AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md)
- Genuinely new work: define exact parameter semantics and typed numerical
  unavailability without selecting TolTEC values, register and realize at
  least one exact JINC-permitted PTC family before any numerical route, close
  the listed geometry/numerical owner choices, then render the recovered
  science in the program's two-view form
- Proposed sanitized packet: the exact items in
  [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md)

The raw recovery record, owner feedback, decision conversations and all
implementation/evidence material remain outside the implementation-blind
author channel. The predecessor Stage A packet was owner-approved at
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f`; this ODQ-101/102B/103 successor changes
allowed input bytes and has **not** received renewed exact-byte approval. No
Stage B author is commissioned.

## 1. Purpose And Relation To SCI-MAP

SCI-JINC defines a signed spatial-kernel estimator that transforms exact
admitted PTC occurrences and their exact AST coordinates into one atomic
observation-level JINC map bundle per TolTEC array.

SCI-JINC is a sibling alternative observation mapmaker, not a downstream
stage of SCI-MAP. It consumes exact PTC and AST parents directly and consumes
no SCI-MAP product. It inherits no ordinary positive-coefficient SCI-MAP
estimator, normalization, projection, support, exposure, validity, response,
covariance, product-availability or coaddition rule by analogy.

The central physical distinction is signed cancellation. Positive lobes,
analytic zeros and negative lobes are scientific kernel values. Their signs
must remain visible in normalization, conditioning, response and covariance.

## 2. Observation-Level Scientific Boundary

The operation begins with one exact PTC occurrence `i` and the AST RTC-output-
grid continuous coordinate associated with the same processed sample
realization, under:

- [`SCI-PTC_TO_SCI-JINC v0.1/r0.3`](SCI-PTC_TO_SCI-JINC_BOUNDARY.md);
- [`SCI-AST_TO_SCI-JINC v0.1/r0.2`](SCI-AST_TO_SCI-JINC_BOUNDARY.md); and
- proposed profile
  [`SCI-JINC:jinc_map_contribution@1`](SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md).

It ends after one requested array observation bundle has either been
published atomically with every required product, identity, join and cause or
has been declared unavailable/failed. Required failure suppresses realized
success.

Base v0.1 defines no JINC coadd. A future JINC coadd requires a separately
authorized boundary over complete JINC bundles. Ordinary MAP coadd is not a
candidate default.

## 3. Legitimate Inputs

For occurrence `i` and target pixel `p`, the contract may use only:

- exact transformed signal `z_i=Z_i^PTC`, quantity/unit identity, immutable
  PTC application generation, retention and causes;
- one exact positive analysis/gridding coefficient `omega_i` produced by a
  family/version in the single PTC-owned registry, with explicit `SCI-JINC`
  permission, requested/effective/observation-resolved/realized selection
  identity, user selection or exact versioned mode-policy default,
  index/broadcast, unit, statistic, factors, normalization, population,
  support, generation, separately typed availability/QC, covariance meaning,
  uncertainty and prohibited meanings;
- stable observation, detector occurrence/UID, RTC sample `n`, PTC segment,
  stable array (`a1100`, `a1400`, or `a2000`) and exact group parents;
- frozen AST role `SCI-AST:rtc_output_grid_coordinates@1` associated with the
  same processed sample realization entering JINC, exact target JINC WCS,
  validity, bounds, uncertainty, causes and provenance;
- exact upstream response and covariance roles, each with domain, codomain,
  parents, support, approximation, omitted terms, lifecycle and causes;
- an exact scientifically authorized, array-associated parameter-set identity
  containing angular `s_a` and dimensionless `a_a`, `b_a`, `c_a`, and
  `(r_max)_a`, plus finite positive pixel size and effective integer
  `subpixel_n>=1`, all under one complete analytic identity;
- an exact processed source-template companion when response is requested;
  and
- finite positive processed sample frequency `f_s,i` for
  `jinc_coefficient_squared_time`.

The registry/permission architecture is owner-approved, but no numerical JINC
route exists until an exact registered family permits `SCI-JINC`, is selected
by the user or an authorized versioned mode default, and its compatible
payload and QC are realized. Missing selection without a default, an
unregistered family, absent JINC permission or unavailable/mismatched payload
makes the route unavailable. SCI-JINC must not infer unity, a MAP-permitted
family, `sens`, loading, scatter, inverse variance, precision, significance or
an alternate-family fallback.

## 4. Collision-Free Estimator Algebra

The canonical symbols and both unit cases are defined in
[`NOTATION_AND_UNITS.md`](NOTATION_AND_UNITS.md):

```text
kappa_ip = signed dimensionless point-phase JINC coefficient
omega_i  = positive upstream analysis coefficient
w_ip     = kappa_ip omega_i

N_p = sum_i I_ip omega_i kappa_ip z_i
C_p = sum_i I_ip omega_i kappa_ip
Q_p = sum_i I_ip omega_i kappa_ip^2
m_p = N_p / C_p
A_pi = I_ip omega_i kappa_ip / C_p.
```

If and only if `omega_i=Var(z_i)^-1` for mutually independent admitted
occurrences,

```text
Var_formal(m_p) = Q_p/C_p^2
W_formal,p      = C_p^2/Q_p
Cov(m_p,m_p')   =
  sum_i I_ip I_ip' omega_i kappa_ip kappa_ip'/(C_p C_p').
```

Generally, `C_JINC=A_JINC C_PTC A_JINC^T`. Inverse-square units do not by
themselves establish inverse variance or precision. For dimensionless
`omega_i`, `Q_p/C_p^2` is dimensionless and is not signal variance.

## 5. Analytic Kernel Authority

The owner-supplied Schloerb LMT memo closes the generic two-JINC analytic
family. Define the collision-free peak-normalized function

```text
J(x)=2 J_1(x)/x for x!=0;  J(0)=1,
```

and `r'_a=r/s_a`. Memo Equation 9, normalized without changing its function,
is

```text
kappa_a(r'_a; a_a,b_a,c_a,RMAX_a)
  = J(2 pi r'_a/a_a)
    exp[-(2 r'_a/b_a)^c_a]
    J(3.831706 r'_a/RMAX_a).
```

Thus `(a,b,c,RMAX)` is the ordered dimensionless tuple; `a` scales the first
JINC argument, `b` the envelope, `c` is the envelope exponent, and `RMAX`
places the second factor's first zero. The peak is one. Analytic zeros and
finite signed lobes are retained.

The memo is geared to 3-mm spectroscopic receivers. Its FCRAO values and
86-GHz simulations are not TolTEC authority. `SCI-JINC-ODQ-102B` preserves
`r'_a=r/s_a`, where `s_a` is an explicit array-associated angular scale, and
permits every kernel parameter to be array-associated where scientifically
appropriate. The memo's `s=lambda/D` is precedent, not TolTEC authorization.

No TolTEC numerical parameter set is scientifically authorized for v0.1.
Without one, the affected numerical route is unavailable; no inherited value,
hidden default, or fallback may be used. Selecting or optimizing TolTEC values
is explicitly deferred to a separate scientific exercise with a stated
objective and appropriate TolTEC/LMT beam, response, and/or noise evidence. See
[`ANALYTIC_JINC_IDENTITY.md`](ANALYTIC_JINC_IDENTITY.md) and
[`AUTHOR_LMT_JINC_REFERENCE_COVER.md`](AUTHOR_LMT_JINC_REFERENCE_COVER.md).
Stage B may define parameter meaning, units, association, admissibility,
identity, provenance, and unavailable-state behavior; it may not infer or
optimize TolTEC values from the memo or software.

## 6. Square Cache, Point Phase And Edges

The binding geometry is:

- AST supplies the authoritative coordinate realization, parent-sample
  identity, validity/support facts, causes, and WCS for the same processed
  sample realization entering JINC;
- SCI-JINC rounds the sample center, bins residual phase and point-evaluates
  one phase-quantized kernel matrix;
- `r_max` fixes the second-factor first zero and square-cache half-width;
- every resolved square pixel is evaluated, including corners beyond radial
  `r_max`; no circular cutoff survives;
- no pixel-area-integration branch survives; and
- finite-map crop removes outside pixels without wrap, reflection or an
  interior normalizer. Response and covariance use the actual retained
  membership.

JINC owns sample admission for `SCI-JINC:jinc_map_contribution@1` and, for
each considered destination pixel, owns local offset/radial geometry,
dimensionless radius, finite support and signed `kappa_ip`. It never associates
signal and coordinate by row/order/time/tolerance/detector fallback. Missing,
duplicate, or ambiguous association makes the coordinate unavailable. Sample
admission is distinct from sample-pixel support: outside support and a
contract-defined zero are ordinary no-contribution results, while a negative
coefficient is normal. Every coupled JINC accumulator uses the same admitted
sample-pixel pair and coefficient identity.

The exact tie rule, phase-bin edges/representatives, cache-index rounding,
effective `subpixel_n` realization and convergence/error bound remain open
under `SCI-JINC-ODQ-109`. The case where the rounded center is outside but the
square overlaps the map remains open under `SCI-JINC-ODQ-110`; the owner must
select center-required, overlap-admitted, or another exact rule. See
[`GEOMETRY_DECISION_TABLE.md`](GEOMETRY_DECISION_TABLE.md).

## 7. Signed Normalization And Support

Finite negative `C_p` is admissible. Exact `C_p=0` is unavailable, not zero
sky. Define the dimensionless cancellation statistic

```text
rho_p = abs(C_p) / sum_i I_ip abs(omega_i kappa_ip).
```

Admission requires finite contributors, finite `C_p` and `Q_p`, `Q_p>0`,
`C_p!=0`, and `rho_p` not below a documented floating-point error bound
derived from the realized summation method and contributor count. The bound,
method and policy identities are provenance. The exact bound identity remains
open under `SCI-JINC-ODQ-109`; no unit-bearing `C` or `Q` floor is permitted.

Algebraic membership, resolved numerical conditioning, formal JINC support,
final product validity and downstream eligibility are separate. An empirical
policy may narrow formal support but never promote it. Rejected pixels carry
typed unavailable signal/response/covariance roles with causes, never finite
substitute values.

Required predictions include constant input, one contributor, equal
coefficients, analytic zero, negative lobe, finite negative normalization,
exact cancellation, unresolved and resolved near cancellation, signal-unit
rescaling, common coefficient rescaling and edge truncation.

## 8. Coefficient-Squared Time

The normative product name is `jinc_coefficient_squared_time`:

```text
T_p^(kappa^2) = sum_i I_ip kappa_ip^2/f_s,i.
```

The squared object is `kappa_ip`, not `omega_i` or `w_ip`. The declared unit is
seconds. This is method-specific accounting, not physical acquired exposure,
valid-original exposure, complete temporal support, normalized influence,
white-noise-equivalent time, formal precision, validity or significance.

No physical-exposure product is required in base v0.1. A future need must
import exact original-occurrence exposure lineage as a distinct product; it
must not distribute one physical integration through every JINC lobe.

## 9. Response And Covariance

[`RESPONSE_AND_COVARIANCE_FAMILIES.md`](RESPONSE_AND_COVARIANCE_FAMILIES.md)
separates:

1. fixed-state JINC response;
2. PTC full-procedure finite difference with JINC fixed;
3. JINC re-resolved procedure response; and
4. separately authorized whole-chain RTC-to-CAL-to-PTC-to-JINC response.

The fixed-state response uses exact signal membership, coefficient, phase,
square placement, edge crop, `C_p` and output rows. A realized PTC-grid
response companion begins at JINC and receives the JINC operator once. No
hidden subset or double application is permitted.

Every covariance/weight publication states its exact coefficient meaning,
correlation assumptions, domain, overlap-induced off-diagonal covariance,
edge truncation, unavailable upstream blocks and omitted calibration,
response, selection, nuisance and parameter terms. `C_p`, `Q_p`, time and
counts are not automatically precision. Empirical noise/significance remain
SCI-NOI authority.

## 10. Grouping, Destination And Products

Base product identity is

```text
observation x TolTEC array x JINC plan x target WCS
x product role x lifecycle generation.
```

Base v0.1 produces independent `a1100`, `a1400` and `a2000` observation
bundles. Cross-array/network-combined products are unavailable. Workers,
threads, processes, containers and filenames are not identity. Complete
destination identity is resolved before mutation or the bundle fails
atomically.

[`GROUPING_AND_PRODUCT_ROLES.md`](GROUPING_AND_PRODUCT_ROLES.md) defines exact
population and required, conditional-required, optional and outside roles.
Required roles include signal, `N/C/Q`, formal support, WCS/operator/parameter
identity, upstream causes/lineage, coefficient-squared time and atomic
provenance. Response and covariance are conditional-required under an exact
request/consumer role. Optional absence is allowed only when the role says so;
a requested-required failure suppresses success.

## 11. Producer, Transformer And Consumer Ownership

- ALIGN/AST own occurrence/time identity, RTC-grid coordinates, frame/WCS,
  exact parent-sample association, coordinate validity/support facts,
  coordinate uncertainty and producer causes. They do not decide JINC support,
  coefficient, admission, or general JINC validity.
- RTC owns conditioned signal-grid meaning, response, influence, validity and
  lineage.
- CAL owns calibrated quantity/unit, response/uncertainty, quality and
  lineage.
- PTC owns transformed signal, retention, the one versioned positive
  analysis/gridding coefficient registry, each family's named-consumer
  permissions, family definition/payload, availability/QC, normalization,
  support, provenance/covariance meaning, selection stages, cleaning
  realization, response/covariance state and application generation.
- SCI-JINC consumes `omega_i` only from an exact JINC-permitted realized
  family and owns signed deposition through `kappa_ip`,
  `w_ip=kappa_ip omega_i`, parameter selection, point phase,
  square support, edge policy, normalization, conditioning, JINC response/
  covariance, formal support, grouping, destination and atomic products.
- VAL Registry binds the owner-authored profile and VAL Core evaluates it; VAL
  does not author JINC policy. Frozen VAL is unchanged, so registration of
  `SCI-JINC:jinc_map_contribution@1` requires a versioned successor.
- NOI owns empirical noise/covariance/weight/significance; FLT owns filtering
  and filtered response; BEAM/SRC/MODE own science interpretation; FRUIT owns
  recurrence/feedback/iteration.

No cause or flag prescribes one universal action. Each named use owns its
policy and preserves direct and transitive causes.

## 12. Scientific Numerical Adequacy

The scientific authority may require exact discrete identity/membership,
explicit conditioning, a preregistered accumulation-error bound, no silent
coefficient thresholding and operation-count/conditioning-aware sequential/
parallel agreement. It does not prescribe a summation algorithm, cache
layout, thread order or optimization. Candidate procedures and pass/fail
evidence belong in the later Engineering Conformance Specification and
subsequent assessment.

## 13. Non-Goals And Claim Boundary

SCI-JINC v0.1 does not:

- alter frozen SCI-MAP, PTC, RTC, CAL, ALIGN, AST or VAL;
- tune or optimize `a`, `b`, `c`, `r_max`, `subpixel_n`, thresholds or defaults;
- derive or claim optimum JINC parameters for the three TolTEC bands; that is
  a separate scientific tranche requiring an explicit objective and evidence;
- define JINC coadd, empirical noise/significance, filtering, source fitting,
  Beammap, Pointing/OOF or fruit-loop science;
- inspect, audit, repair, optimize or refactor Citlali;
- inspect schemas, tests, reductions, Unity, generated products or production
  behavior; or
- assert implementation conformity, validation, achieved response/covariance
  fidelity, photometric accuracy, numerical reproducibility, performance,
  readiness or production authorization.

## 14. Remaining Owner Decisions And Stop Gate

The exact ledger is
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
`SCI-JINC-ODQ-101` is resolved for registry ownership, named-consumer
permission, selection lifecycle and fail-closed behavior. It does not register
or realize a family. `SCI-JINC-ODQ-102B` is resolved for generic parameter
semantics and typed numerical unavailability, not numerical values. Numerical
production remains unavailable until both a compatible PTC family and a
separately authorized TolTEC parameter set exist.
`SCI-JINC-ODQ-103` is resolved for exact scientific sample-coordinate
association, JINC-owned map-contribution admission, AST/JINC geometry
ownership, sample-admission/support separation, coupled-accumulator identity,
and cause policy. It prescribes no data-model join mechanism.

Before author dispatch, the owner must:

1. decide the coefficient-squared-time-only base-v0.1 disposition and defer or
   authorize a distinct physical-exposure role (`SCI-JINC-ODQ-104`);
2. decide the center/tie/phase/cache/error-bound policy
   (`SCI-JINC-ODQ-109`);
3. decide the outside-center overlapping-square edge rule
   (`SCI-JINC-ODQ-110`);
4. approve the exact Schloerb method excerpt and cover, controlled PTC
   coefficient-registry source/cover, PTC/AST boundaries,
   admission profile, grouping/product table, response/covariance table and
   inherited-decision table;
5. authorize a versioned VAL registry binding; and
6. approve every exact successor author-packet byte and SHA-256 value.

The requested Stage A repairs and exact proposed packet are recorded in
[`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md) and
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). Stage B has **not**
been launched. If allowed material remains insufficient, a future author must
return one precise scientific question rather than search excluded sources or
infer from software/model memory.
