# SCI-JINC v0.1 — Proposed Sanitized Conventions And Ownership

Status: ODQ-101/102B/103 sanitized successor candidate; renewed exact-byte
approval
required

Prepared: `2026-08-28`

This document contains only stable scientific conventions and abstract
producer/transformer/consumer boundaries relevant to signed-coefficient JINC
gridding. It contains no Citlali source behavior, audit finding, repair, test,
validation result, achieved-performance statement, or production-status claim.

Exact source set at
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`:

- SCI-ALIGN v0.1/r0.3 source manifest, SHA-256
  `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac`;
- SCI-AST v0.1/r0.3 source manifest, SHA-256
  `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`;
- SCI-RTC v0.1/r0.12 scientific-owner freeze record, SHA-256
  `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf`;
- SCI-CAL v0.1/r0.5 scientific-owner freeze record, SHA-256
  `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22`;
- SCI-PTC v0.1/r0.5 scientific-owner freeze record, SHA-256
  `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`;
- SCI-VAL v0.1/r0.3 scientific-owner freeze record, SHA-256
  `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6`;
- SCI-MAP v0.1/r0.7.1 scientific-owner freeze record, SHA-256
  `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1`;
  and source manifest, SHA-256
  `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a`;
- exact ordinary `SCI-PTC_TO_SCI-MAP v0.1/r0.1` boundary, SHA-256
  `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7`,
  used only to identify what cannot be silently inherited by JINC;
- controlled post-freeze PTC coefficient-registry predecessor at
  `54475956f6aefb839d43b2f0fb019a142cb64310`, SHA-256
  `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`,
  admitted only under
  [`AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md`](AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md);
  and
- the approved JINC decisions consolidated in the proposed
  [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md).

## Quantity And Capability Boundary

- SCI-JINC v0.1 is nonpolarimetric. A `STOKES` token or index does not by
  itself establish physical Stokes I.
- The input signal is the exact transformed detector-time quantity supplied by
  the approved upstream chain, with its physical role, unit, fixed nominal
  beam/template origin, spectral/calibration lineage, and availability.
- The active frozen chain supports a top-of-atmosphere, point-source-equivalent
  `mJy/beam` boundary for the named nonpolarimetric total-intensity-equivalent
  quantity. A different unit or quantity role requires separate authority.
- JINC transforms the admitted upstream quantity spatially. It does not create
  calibration, timestream conditioning, astrometry, empirical noise,
  filtering, source inference, Beammap inference, or feedback science.
- The JINC observation estimator is distinct from ordinary positive-
  coefficient SCI-MAP and from matched-amplitude, maximum-likelihood,
  filtering, source-fitting, and OOF-transfer estimators.

## Identity

- Array identity is one of `a1100`, `a1400`, and `a2000`. Array name, numeric
  array ID, array index, network, detector occurrence/UID, map group, map
  index, component label, and container position are distinct.
- One logical input occurrence binds observation, detector occurrence and UID,
  stable RTC output sample, exact PTC product/application generation, PTC
  segment, and array/network/group. Sample time is an attribute, not the
  identity by itself.
- Artifact-local detector UID or row number is not a persistent universal
  detector namespace. A cross-product association requires an explicit
  occurrence- and generation-scoped scientific relation; its serialization or
  join mechanism is engineering choice.
- A JINC map identity includes observation, array/group, quantity/component,
  estimator version, WCS/grid, support/phase/conditioning convention, response
  parent, product role, and lifecycle generation.
- A destination index is a local locator. Unique destination ownership is
  resolved from the full product identity before numerical mutation.
- In-memory indices are zero-based. Persisted FITS/WCS pixel coordinates are
  one-based.

## Shape, Ordering, Coordinates, And Frames

- The admitted primary timestream shape is samples by detectors under an exact
  parent product identity.
- JINC outputs are collections of two-dimensional spatial planes with explicit
  group and quantity/component identity.
- Science-coordinate products use a declared equatorial J2000 TAN WCS with
  degree-valued spatial axes when that is the approved target. Pointing/OOF and
  Beammap products may use a declared AltAz tangent plane in arcseconds under
  their own authority.
- FITS/WCS, when used, is the persisted pixel-to-coordinate authority. Memory
  order does not determine axis sign, handedness, orientation, or wrapping.
- The JINC signal coordinate must be the AST-owned coordinate realization
  associated with the same processed sample realization entering JINC and the
  exact target JINC WCS. That association remains exact across alignment,
  filtering, decimation, or other realization changes. Row order, nearest-
  time/tolerance matching, detector ordering, shape, or numerical equality is
  not a substitute. Its data-model mechanism is an engineering choice.
- The ordinary MAP one-hot coordinate/exposure boundary is not a JINC default.
  JINC owns the single `SCI-JINC:jinc_map_contribution@1` admission profile.
- AST supplies coordinate facts and their producer causes. JINC owns local
  offset/radial geometry, dimensionless radius, finite support, signed
  coefficient, and their use relative to destination pixels. AST does not
  author JINC support, coefficient, admission, or general JINC validity.

## Units And Statistical Labels

- The JINC spatial coefficient `kappa_ip` is dimensionless under its declared
  analytic and pixel/array scale convention.
- The radial coordinate is `r'_a=r/s_a`, where `s_a` is an explicit angular
  scale associated with stable array `a`. The generic `s=lambda/D` precedent
  does not select a TolTEC realization.
- `a_a`, `b_a`, `c_a`, and `(r_max)_a` are dimensionless and may be array-
  associated where scientifically appropriate. Their requested, effective,
  observation-resolved, and realized parameter-set identities are distinct.
- Signal and a unit-source response companion carry the admitted signal unit
  when the response is a unit-bearing transformed template.
- The upstream coefficient `omega_i` carries its producer-declared unit and
  normalization. An inverse-square signal unit does not prove inverse variance,
  independence, or full covariance authority.
- `omega_i` comes from one exact family/version in the PTC-owned positive
  analysis/gridding registry. The family must explicitly permit `SCI-JINC`;
  permission for another named consumer does not transfer.
- `N/C` carries the signal unit when numerator and denominator are constructed
  from the same admitted membership and signed coefficient convention.
- Conditional covariance carries the square of the signal unit. `C^2/Q` is a
  formal diagonal inverse variance only under the exact stated upstream
  coefficient and independence/covariance assumptions.
- `jinc_coefficient_squared_time=sum(kappa^2/f_s)` carries seconds. It squares
  only the dimensionless analytic JINC coefficient and is method-specific
  accounting. It is not physical acquired exposure, valid-original exposure,
  complete temporal support, normalized influence, white-noise-equivalent
  time, hits, precision, confidence, validity, or significance.
- Support and validity states are dimensionless logical facts.
- Signal multiplied by the square root of a formal weight is a formal
  standardized signal. `sig2noise` is reserved for an empirically calibrated
  estimator owned by SCI-NOI or another approved authority.

## Signed Coefficients, Support, And Conditioning

- Finite positive, exact-zero, and finite negative JINC coefficients have
  distinct meanings. Negative lobes retain their sign; analytic zeros are not
  outside-footprint sentinels.
- The approved footprint is a fully populated square cache. The parameter
  `r_max` also places the first zero of the second JINC factor; it is not a
  strict radial support maximum.
- The approved subpixel convention is phase-quantized point evaluation after
  rounded-center placement. It is not a pixel-area average.
- Geometric square support, nonzero coefficient support, numerical/formal
  support, temporal support, empirical policy support, and final product
  validity are distinct.
- Exact cancellation is invalid. Near-cancellation is judged by the
  dimensionless ratio of absolute signed sum to absolute-term sum against a
  documented realized summation-error bound, not a unit-bearing threshold.
- A finite value is not proof of validity. Invalid inputs are rejected before
  numerical payload use; multiplying a non-finite value by zero is not masking.

## Availability, Missing Data, Failure, And Lifecycle

- Missing, disabled, automatic, known-but-not-supplied, unavailable,
  unsupported, invalid, and failed are semantic states, not undocumented
  numerical sentinels.
- Every unavailable required input, response, covariance term, or product has
  an owning authority and preserved cause. Unknown is not zero.
- An unavailable scientifically authorized JINC parameter set makes the
  affected numerical route unavailable. No inherited value, shared-field
  assumption, or hidden default supplies it.
- Upstream retention/eligibility, JINC admission, JINC numerical support,
  empirical consumer policy, and final product validity are separate decisions.
- JINC sample admission and sample-pixel support are separate. Outside finite
  support and a contract-defined zero coefficient are ordinary no-contribution
  results, not causes. A negative coefficient is normal. An unavailable or
  ambiguous AST coordinate prevents geometry evaluation and is not outside
  support.
- All accumulators derived from one contribution use the same admitted sample-
  pixel pair and signed-coefficient identity, even when their contract-defined
  algebraic functions differ.
- A required product, companion, join, or write failure propagates. A
  completion marker cannot override missing required output.
- Configuration/provenance flows one way: requested to effective to resolved
  to realized. Execution cannot rewrite the accepted request or an earlier
  product generation.
- New response, covariance, coefficient, or policy generations produce new
  immutable derived products and do not retroactively change an earlier map's
  claims.

## Producer–Transformer–Consumer Responsibilities

### Upstream producers

- **SCI-ALIGN** owns stable occurrence alignment, physical/valid-original
  exposure facts, and their causes.
- **SCI-AST** owns coordinate roles, frame/WCS transformations, parentage,
  exact sample association, coordinate validity/support facts, producer
  causes, and astrometric uncertainty.
- **SCI-RTC** owns raw-timestream conditioning, response, causal influence,
  masks/flags within its scope, and immutable lineage.
- **SCI-CAL** owns calibrated quantity/unit meaning, calibration transfer,
  response basis, quality/validity, uncertainty, and lineage.
- **SCI-PTC** owns transformed signal, retention, cleaning/subspace
  realization, the one versioned positive analysis/gridding coefficient
  registry, each family's definition and named-consumer permissions, user or
  versioned mode-policy selection, requested/effective/observation-resolved/
  realized family identities, coefficient payload and separate availability/
  QC, normalization, support, provenance/covariance meaning, response/
  covariance state, and application generation.
- **SCI-VAL** owns representation-independent evaluation and exact profile
  registration. Named-use policy remains with the scientific producer or
  consumer that owns the use.

### SCI-JINC transformer

SCI-JINC consumes the same positive PTC-produced `omega_i` only when its exact
family/version permits `SCI-JINC`. It owns the signed `kappa_ip` and
`w_ip=kappa_ip omega_i`, spatial support/phase convention, normalization,
cancellation conditioning, conditional covariance, transformed response,
coefficient-squared temporal support, local formal support, destination/
product identity, and JINC provenance. It applies upstream facts without
redefining their meaning.

For JINC map contribution, producer facts and causes cross the boundary but a
producer-owned JINC-usability decision does not. JINC adds local causes only
for genuine JINC failures and uses established cause/support mechanisms at the
existing product granularity; no per-contribution provenance system is added.

### Adjacent and downstream packages

- **SCI-MAP** owns only ordinary positive-coefficient mapmaking/coaddition and
  its exact profiles. Its frozen authority is unchanged.
- **SCI-NOI** owns empirical noise ensembles, empirical covariance/weight, and
  significance calibration.
- **SCI-FLT** owns filter transfer, filtered response/covariance/support/
  validity, and immutable raw-parent binding.
- **SCI-BEAM**, **SCI-SRC**, and **SCI-MODE** own Beammap, source-fit,
  Pointing, and OOF interpretations and their consumer-specific qualification.
- **SCI-FRUIT** owns feedback, recurrence, learning, iteration, convergence,
  and restart.

Consumers may not reconstruct a JINC response from defaults, reinterpret
coefficient-squared time, promote a formally invalid pixel, erase signed-
coefficient lineage, or convert conditional formal precision into empirical
significance.

## Explicitly Unavailable Boundary Facts

The following are supplied by separate proposed sanitized packet items rather
than by this extract: the generic analytic family, exact PTC/AST boundary
candidates, admission profile candidate, grouping/product roles and response/
covariance family table. The following remain unavailable:

- a registered and realized JINC-permitted PTC coefficient family/payload;
- any scientifically authorized TolTEC `a1100`/`a1400`/`a2000` numerical
  parameter-set realization; v0.1 supplies semantics and a typed unavailable
  state only;
- a frozen SCI-VAL registration of `SCI-JINC:jinc_map_contribution@1`;
- a JINC observation-coadd contract;
- numerical parameter values, parameter optimization, summation/phase
  tolerances, or production
  thresholds; and
- implementation conformity, representation fidelity, validation, achieved
  response/performance, readiness, or production evidence.

This proposed extract is content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md) and may enter Stage B
only after scientific-owner approval.
