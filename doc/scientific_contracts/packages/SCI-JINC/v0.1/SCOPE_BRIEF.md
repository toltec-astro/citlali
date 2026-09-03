# SCI-JINC — Signed-Coefficient JINC Observation Mapmaker Scope Brief

Status: ODQ-101/102B/103/104/105/106/107/109/110 Stage A successor candidate; predecessor Stage A bytes
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
  coordinate, lifecycle and cause boundaries; no source or schema mechanics
- Superseded: the independent core's circular radial-cutoff branch and
  pixel-area-integrated branch; both are unavailable
- Deferred/excluded: response/covariance/formal-weight products, standalone
  support/availability roles, generalized provenance, diagnostics,
  implementation, schemas, tests, products, audits,
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
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f`; this ODQ-101/102B/103/104/105/106/107/109/110 successor changes
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
must remain visible in `N_p`, `C_p`, `Q_p`, conditioning and any later
separately authorized response or covariance treatment.

## 2. Observation-Level Scientific Boundary

The operation begins with one exact PTC occurrence `i` and the AST RTC-output-
grid continuous coordinate associated with the same processed sample
realization, under:

- [`SCI-PTC_TO_SCI-JINC v0.1/r0.3`](SCI-PTC_TO_SCI-JINC_BOUNDARY.md);
- [`SCI-AST_TO_SCI-JINC v0.1/r0.2`](SCI-AST_TO_SCI-JINC_BOUNDARY.md); and
- proposed profile
  [`SCI-JINC:jinc_map_contribution@1`](SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md).

It ends after one requested array observation bundle has either published the
fixed five-role schema atomically or has failed closed without a partial or
placeholder bundle. Pixel-local invalid support is ordinary bundle content,
not whole-product unavailability.

Base v0.1 defines the estimator and complete product bundle for one
observation and authorizes no cross-observation JINC combination semantics. A
future JINC coadd requires a separately authorized boundary whose inputs are
complete observation-level JINC bundles. That boundary must define
compatibility, the object combined, exact algebra, response, covariance,
support, validity, provenance and failure semantics. It may not import
ordinary MAP coadd or infer that accumulator-plane addition or normalized-map
combination is authorized.

Observation is the scientific grouping boundary, not a streaming, chunking,
process or memory boundary. Samples or processing chunks from the same
observation may accumulate incrementally into its one complete array bundle
only under the same exact observation, array, JINC plan/realization, target
WCS, admission/parameter/coefficient state and lifecycle generation. A chunk
does not create a product or coadd identity.

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
- an exact scientifically authorized, array-associated parameter-set identity
  containing angular `s_a` and dimensionless `a_a`, `b_a`, `c_a`, and
  `(r_max)_a`, plus finite positive pixel size and effective integer
  `subpixel_n>=1`, all under one complete analytic identity;
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

The following conditional equations preserve recovered mathematics but do not
create base-v0.1 products under ODQ-107. If and only if
`omega_i=Var(z_i)^-1` for mutually independent admitted
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
- the resolved rounded center must first lie in the finite destination domain;
  an outside center contributes nowhere. For an admitted in-map center,
  finite-map crop removes outside pixels without wrap, reflection, completion,
  interior normalization or edge correction.

JINC owns sample admission for `SCI-JINC:jinc_map_contribution@1` and, for
each considered destination pixel, owns local offset/radial geometry,
dimensionless radius, finite support and signed `kappa_ip`. It never associates
signal and coordinate by row/order/time/tolerance/detector fallback. Missing,
duplicate, or ambiguous association makes the coordinate unavailable. Sample
admission is distinct from sample-pixel support: outside support and a
contract-defined zero are ordinary no-contribution results, while a negative
coefficient is normal. Every coupled JINC accumulator uses the same admitted
sample-pixel pair and coefficient identity.

ODQ-109 requires every center/phase/cache realization to be single-valued,
preserve the accepted point-phase and square-support operator, and keep its
total numerical error negligible compared with the approximately `10^-3`
relative fidelity relevant to the instrument. Exact adequate tie, bin,
representative, cache-rounding and accumulation choices are engineering
realizations rather than separate scientific-owner decisions. ODQ-110 applies
the finite-map boundary at occurrence admission: the resolved rounded center
used for cache placement must lie within the finite destination domain before
any footprint evaluation. An outside center makes `I_ip=0` for every `p`, even
when its square would overlap the map. For an admitted in-map center, ordinary
finite-map crop removes outside square pixels without completion,
renormalization or edge correction. See
[`GEOMETRY_DECISION_TABLE.md`](GEOMETRY_DECISION_TABLE.md).

## 7. Signed Normalization And Support

Finite negative `C_p` is admissible. Exact `C_p=0` is unavailable, not zero
sky. Define the dimensionless cancellation statistic

```text
rho_p = abs(C_p) / sum_i I_ip abs(omega_i kappa_ip).
```

Admission requires finite contributors, finite `C_p` and `Q_p`, `Q_p>0`, and
`C_p!=0`. `rho_p` remains a dimensionless conditioning indicator. A finite
nonzero result is usable only when total numerical error is demonstrably
negligible compared with the approximately `10^-3` relative fidelity relevant
to the instrument. If near-cancellation prevents that showing, the pixel is
locally invalid. ODQ-109 prescribes no universal `rho_p` cutoff, contributor-
count/machine-epsilon formula, exact reduction order or unit-bearing `C`/`Q`
floor.

Algebraic membership, resolved numerical conditioning, formal JINC support,
final product validity and downstream eligibility are separate. An empirical
policy may narrow formal support but never promote it. Rejected pixels carry
the `jinc_map` role's local invalid-support state, never a finite substitute
value. They do not make any required whole-product role unavailable and do not
create a role-availability record.

Required predictions include constant input, one contributor, equal
coefficients, analytic zero, negative lobe, finite negative normalization,
exact cancellation, unresolved and resolved near cancellation, signal-unit
rescaling, common coefficient rescaling and edge truncation.

## 8. Coefficient-Squared Time

The owner-authorized sole base-v0.1 time-support product is
`jinc_coefficient_squared_time`:

```text
T_p^(kappa^2) = sum_i I_ip kappa_ip^2/f_s,i.
```

The squared object is `kappa_ip`, not `omega_i` or `w_ip`. The declared unit is
seconds. This is method-specific accounting, not physical acquired exposure,
valid-original exposure, complete temporal support, normalized influence,
white-noise-equivalent time, formal precision, validity or significance.

A separate physical-exposure product is deferred until an identified
scientific use requires it. That future authority must define a distinct
product with exact original-occurrence exposure lineage and semantics; it must
not reinterpret coefficient-squared time or distribute one physical
integration through every JINC lobe.

## 9. Deferred Response, Covariance And Companion Products

ODQ-107 authorizes no response, covariance, formal-weight, standalone support/
availability, diagnostic or generalized provenance product in base v0.1.
The recovered response and covariance mathematics remains preserved in
[`RESPONSE_AND_COVARIANCE_FAMILIES.md`](RESPONSE_AND_COVARIANCE_FAMILIES.md)
for a future concrete scientific use, but that file is excluded from the
implementation-blind base-v0.1 author packet. `C_p`, `Q_p`, time and counts are
not automatically precision, and empirical noise/significance remains
SCI-NOI authority.

## 10. Grouping, Destination And Products

Base product identity is

```text
observation x TolTEC array x JINC plan x target WCS
x product role x lifecycle generation.
```

For one observation, base v0.1 may produce at most one independent bundle for
each stable array admitted and requested under the exact JINC realization.
The produced cardinality is zero through three over `a1100`, `a1400` and
`a2000`. Missing, unavailable, or unrequested arrays produce no placeholder or
empty-array product and do not invalidate a different produced bundle. Cross-
array/network-combined or shared-destination products are unavailable.
Workers, threads, processes, containers and filenames are not identity.
Complete destination identity is resolved before mutation or the bundle fails
atomically.

Same-observation streaming or chunk accumulation may realize that one bundle
when the complete scientific identity and realized JINC state remain the
same. Cross-observation combination is outside base v0.1 and no coadd
arithmetic is implied by the observation accumulators.

Each produced bundle is bound independently to its exact observation, stable
array, JINC realization, destination map geometry and lifecycle generation.
Contributions with different array or destination identities must not be
merged. The ODQ-106 bundle identity is sufficient; no additional per-
contribution provenance, detailed operational-reason archive or synthetic
empty-array product is required.

[`GROUPING_AND_PRODUCT_ROLES.md`](GROUPING_AND_PRODUCT_ROLES.md) defines the
fixed closed schema: required `jinc_signal_numerator` (`N_p`),
`jinc_signed_normalization` (`C_p`), `jinc_quadratic_accumulator` (`Q_p`),
derived `jinc_map` (`m_p`) with local support/validity state, and
`jinc_coefficient_squared_time`. Failure to form any whole-product role
suppresses the bundle; no generic optional/conditional role, placeholder,
per-role availability object or generalized provenance product exists.

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
  square support, edge policy, normalization, conditioning, `N_p`, `C_p`,
  `Q_p`, derived `m_p`, coefficient-squared time, local formal support,
  grouping, destination and atomic fixed-bundle publication. Recovered JINC
  response/covariance semantics are deferred future science, not base roles.
- VAL Registry binds the owner-authored profile and VAL Core evaluates it; VAL
  does not author JINC policy. Frozen VAL is unchanged, so registration of
  `SCI-JINC:jinc_map_contribution@1` requires a versioned successor.
- NOI owns empirical noise/covariance/weight/significance; FLT owns filtering
  and filtered response; BEAM/SRC/MODE own science interpretation; FRUIT owns
  recurrence/feedback/iteration.

No cause or flag prescribes one universal action. Each named use owns its
policy and preserves direct and transitive causes.

## 12. Scientific Numerical Adequacy

Numerical error from finite arithmetic, accumulation/reduction order,
analytic-function evaluation, phase quantization and cache/index realization
must remain negligible compared with the approximately `10^-3` relative
fidelity relevant to the instrument. This is the complete scientific accuracy
principle for base v0.1: no stronger precision, bitwise reproducibility, fixed
reduction order, prescribed summation algorithm, contributor-count floating-
point formula or exact sequential/parallel identity is required.

The realization must still be single-valued, preserve the scientifically
specified membership semantics and accepted point-phase/square-support
operator, avoid silent coefficient thresholding, and establish adequate
conditioning before a local map value is supported. Candidate algorithms,
test constructions, and handling of comparisons near zero belong in the later
Engineering Conformance Specification and subsequent assessment. This Stage A
decision establishes no conformity or achieved-fidelity claim.

## 13. Non-Goals And Claim Boundary

SCI-JINC v0.1 does not:

- alter frozen SCI-MAP, PTC, RTC, CAL, ALIGN, AST or VAL;
- tune or optimize `a`, `b`, `c`, `r_max`, `subpixel_n`, thresholds or defaults;
- derive or claim optimum JINC parameters for the three TolTEC bands; that is
  a separate scientific tranche requiring an explicit objective and evidence;
- define JINC coadd, empirical noise/significance, filtering, source fitting,
  Beammap, Pointing/OOF or fruit-loop science;
- define response, covariance/formal-weight, standalone support/availability,
  diagnostics, generalized provenance or generic optional/conditional product
  machinery;
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
`SCI-JINC-ODQ-104` is resolved: `jinc_coefficient_squared_time` is the sole
base-v0.1 time-support product, and a separate physical-exposure product is
deferred until a scientific use requires and authorizes it.
`SCI-JINC-ODQ-105` is resolved: base v0.1 is observation-only, same-observation
incremental accumulation is permitted under one exact JINC realization, and
any cross-observation combination requires a separately authorized boundary
over complete observation bundles.
`SCI-JINC-ODQ-106` is resolved: an observation may produce at most one
independent bundle for each admitted/requested stable array, absent arrays
produce no placeholders and do not invalidate produced bundles, and
contributions with different array or destination identities must not merge.
`SCI-JINC-ODQ-107` is resolved: every produced bundle contains only required
`N_p`, `C_p`, `Q_p`, derived `m_p` with local support/validity, and
`jinc_coefficient_squared_time`; whole-product failure suppresses the bundle,
while local invalid support does not. No general availability, detailed-cause,
optional/conditional-role or provenance framework is authorized.
`SCI-JINC-ODQ-108` response/covariance products are deferred by ODQ-107 until
a concrete scientific use is separately authorized.
`SCI-JINC-ODQ-109` is resolved: scientific conditioning retains finite
`C_p`/`Q_p`, `Q_p>0`, `C_p!=0`, exact-cancellation rejection, finite-negative
normalization and dimensionless `rho_p`, while numerical error must be
negligible compared with the approximately `10^-3` relative instrument-
fidelity scale. It requires no machine-specific error formula, exact
summation/tie/bin/cache choice, bitwise reproducibility or stronger precision.
`SCI-JINC-ODQ-110` is resolved: an occurrence contributes only when its
resolved rounded cache center lies in the finite destination domain. An
outside center contributes zero to every fixed accumulator for every pixel;
overlapping-footprint admission is prohibited. In-map centers retain ordinary
edge crop, and JINC-then-crop equivalence is not required. No edge correction,
provenance or diagnostic product follows.

No unresolved numbered scientific-scope ODQ remains. Before author dispatch,
the owner or governing registry authority must:

1. approve the exact Schloerb method excerpt and cover, controlled PTC
   coefficient-registry source/cover, PTC/AST boundaries,
   admission profile, fixed grouping/product table and inherited-decision
   table;
2. authorize a versioned VAL registry binding; and
3. approve every exact successor author-packet byte and SHA-256 value under
   `SCI-JINC-STAGE-A-Q002`.

The requested Stage A repairs and exact proposed packet are recorded in
[`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md) and
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). Stage B has **not**
been launched. If allowed material remains insufficient, a future author must
return one precise scientific question rather than search excluded sources or
infer from software/model memory.
