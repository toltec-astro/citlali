# Citlali Scientific Conventions

## Status And Scope

This is the canonical human-readable statement of scientific identity, units,
coordinate frames, indexing, validity, provenance state, and validation policy
for the current Citlali refactor tree. It describes validated behavior and
explicitly labeled accepted successor contracts; a successor that is pending
independent re-audit does not replace an active validated snapshot. This
document does not redesign the underlying algorithms or promise that historical
file formats already encode every convention completely.

The executable authorities are:

- `validation/product_contracts.json` for versioned product families and
  structural requirements;
- `tools/config/config_leaf_contract_resolved.json` for low-level configuration
  ownership, units, domains, modes, and resolution stages;
- `validation/validation_profiles.json` for active numerical acceptance policy;
- `validation/accepted_runs.json` for accepted evidence; and
- `validation/intended_science_changes.json` for intentional post-baseline
  scientific changes.

This document explains those contracts and records conventions that span more
than one file or subsystem. A disagreement between this document, executable
metadata, and actual writer behavior is a defect to resolve. It is not
permission to choose whichever representation is convenient.

## Capability Boundary

| Reduction intent | Current execution status | Validated contract |
| --- | --- | --- |
| Pointing | Supported | Point profile and product contract |
| Out-of-focus holography (OOF) | Supported as a distinct validation intent using the pointing execution path | OOF profile and product contract |
| Science | Supported | Science profile and product contract |
| Beammap | Supported | Beammap profile and product contract |
| Enabled polarimetry | Planned but unavailable; rejected before execution | No enabled reference contract |
| Auxiliary measured R-channel analysis | Structure only; execution deferred | No execution or product contract |

The KIDs input selector accepts `xs`, `rs`, `is`, and `qs` as primary stream
types. `xs` is the ordinary science-stream default. Support for selecting all
four types does not mean that the active Phase 4 fixtures prove their physical
equivalence or establish identical calibration meaning for every type.

## Scientific Identity

Raw integers are not interchangeable scientific identities. Convert and
validate identity at subsystem boundaries, then use dense indices only inside
bounded numerical loops.

### Arrays

The current TolTEC array identity mapping is:

| Array ID | Array name | Nominal wavelength |
| ---: | --- | ---: |
| 0 | `a1100` | 1.1 mm |
| 1 | `a1400` | 1.4 mm |
| 2 | `a2000` | 2.0 mm |

An **array ID** is the value carried by calibration/APT data. An **array name**
is the stable string used by configuration and product naming. An **array
index** is a zero-based position in a run-local container such as
`calib.arrays`. The three values currently often coincide numerically, but code
must not rely on that coincidence. Per-array configuration is keyed by array
name or validated array ID. A map index or detector index is never an array ID.

### Networks

Network ID is the `nw` identity carried by detector calibration data. The
current instrument mapping is networks 0-6 to `a1100`, 7-10 to `a1400`, and
11-12 to `a2000`. A network index is a dense position in the networks present
for one observation or intermediate object. Products carry the network ID when
the identity must survive ordering or subset changes. New code must not infer a
network ID from a container position.

### Detectors

`apt_uid` or `uid` is the detector identity used to join detector-resolved
products. `ptc_diag_uid` is the corresponding PTC diagnostic identity. A
detector index is a zero-based row or column position in the current APT,
timestream matrix, map group, or fit workspace. Row order is not an external
detector identity.

Use UID fields for joins across products. The current contract does not claim
that an arbitrary row number is stable across observations or calibration
tables, nor does it establish a stronger long-term UID lifetime than the
upstream APT provides.

Beammap `det_N` FITS extension labels identify detector-map slots and are linked
to detector UID fields in the Beammap APT/QC products. `N` is not itself a
detector UID.

### Maps And Stokes

A map index is a zero-based position in a reduction-local map collection. Its
scientific identity is resolved through explicit map-to-array and map-to-Stokes
mappings plus the configured grouping:

- array grouping: one map group per present array;
- network grouping: one map group per present network;
- frequency-group grouping: frequency group within array;
- detector grouping: one map group per detector row.

The current validated capability has one Stokes component: index 0, label `I`.
Indices 1 and 2 are reserved by the legacy adapter for `Q` and `U`, but enabled
polarimetry is rejected and those products are not scientifically validated.

### Observations, Scans, Iterations, And Pixels

- `obsnum` is the external observation identity. An observation index is a
  zero-based position within one reduction request.
- Internal scan indices are zero-based.
- NetCDF `output_scan_index` is the one-based original scan number from the
  full observation. The variable metadata states this explicitly.
- Learning CSV `scan` fields are zero-based. Log messages commonly present
  scans as one-based human-facing numbers; consumers must use the named field,
  not infer its base from a log line.
- Sample bounds, detector rows, Beammap source-crossing slots, and diagnostic
  row/column indices are zero-based unless a product contract says otherwise.
- Fruit-loop iteration identifiers and learning-diagnostic filename iteration
  values are zero-based and absolute across an exact restart. A checkpoint
  completed at iteration `N` resumes at `N + 1`; the new job does not relabel
  that iteration as zero.
- Learning-housekeeping QA `scan_zero_based` and filename iteration values are
  zero-based. `event_time_unix_sec` is the midpoint of the first and last
  finite PTC `TelTime` values in the processed chunk. Housekeeping matches use
  the nearest recorded Unix timestamp and publish both signed
  `sample_offset_sec` (`sample - event`) and absolute `sample_age_sec`; no
  interpolation is implied.
- FITS/WCS pixel coordinates are one-based. In-memory map rows/columns and
  diagnostic peak row/column values are zero-based.

## Shape And Ordering

The primary RTC/PTC detector timestream is a matrix with samples on rows and
detectors on columns. Sample flags have the corresponding sample-by-detector
shape. PTC detector weights and detector metadata are joined by detector row
within the object and by UID across persisted products.

Telescope streams align with the timestream sample axis after the configured
interface synchronization, gap handling, filtering, edge-guard, and
downsampling policy has been resolved. Any auxiliary measured stream must state
its alignment and must not be treated as a synthetic kernel.

The `learning_housekeeping_iter_*.csv` sidecar is correlation evidence only.
It records kelvin-valued TolTEC thermometry and dilution-fridge channels near
busy-network pathology events, including explicit missing/invalid status and
neighbor differences. It neither changes sample flags nor establishes causal
timing. Current housekeeping cadence can be approximately 60 seconds, so the
published sample age is part of the scientific interpretation and must not be
discarded by downstream QA.

Raw TolTEC KIDs files carry the ADC snapshot variable
`Header.Toltec.AdcSnapData` with shape `[2, 4096]`. The first axis is the
producer-confirmed file-boundary axis: index `0` is the beginning of the data
file and index `1` is the end. The second axis contains 4096 raw ADC samples.
It is not an ADC-channel axis. Values are signed 12-bit ADC counts stored in an
`int16`/NetCDF `short` container, with valid count domain `[-2048, 2047]`;
no division by 16 is part of this input contract. The schema constants and
boundary enum live in
`include/citlali/core/pipeline/rawobs_adc_snap.h`.

The fruit-loop restart checkpoint stores operational state, not QA history.
Its sample masks are the canonical disjoint interval union keyed by
observation, zero-based scan, application stage, and detector UID. Its detector
penalties retain their scientific identity and effective value. Schema v2 also
stores accumulated/finalized PTC weight-validation sums, counts, detector
factors, and validity flags, and rejects a changed processed-timestream policy.
Bounded event vectors, housekeeping matches, summaries, and
dropped-diagnostic counters do not affect later flags or weights and are
intentionally not restored. Exact continuation requires unchanged inputs and
science configuration; schema v1 is rejected because it omitted retained
weight-validation state.

Map products are collections of two-dimensional spatial planes. Array,
frequency, and Stokes identity are represented by product grouping, FITS
extensions, WCS spectral/Stokes axes, and the explicit map-index mappings; they
must not be reconstructed from vector position alone. Exact FITS extension and
NetCDF dimension requirements live in `validation/product_contracts.json`.

### Science-Map Bundle Identity And Coaddition

The accepted `SCI-MAP-001` F009/F010 successor contract applies to ordinary
naive, array-grouped Stokes-I observation and coadd maps. Its implementation is
pending independent re-audit; this section states the accepted meaning and does
not declare the repair conformant.

An observation enters a coadd as one immutable ordered bundle. Admission
includes grouping and slot identities, array/network/detector or group identity,
Stokes and applicable frequency identity, signal unit, estimator, response and
required companions, full-precision WCS, shapes, and the versioned coefficient,
contribution, support, validity, and non-finite policies. Identity is formed
from authoritative full-precision inputs. The legacy float-valued map WCS is a
one-way output/compatibility projection and cannot establish equality.

For coadd shape `(R_c, C_c)` and observation shape `(R_o, C_o)`, the only
permitted placement is centered integer common-grid embedding:

```text
R_c >= R_o, C_c >= C_o
(R_c - R_o) and (C_c - C_o) are even
delta_row = (R_c - R_o) / 2
delta_col = (C_c - C_o) / 2
```

The full-precision observation WCS must identify the same world coordinate
after this offset. Shape and the corresponding reference-pixel offset are the
only permitted WCS differences. Any identity, unit, response, frame,
projection, center, scale, orientation, map-order, shape, or policy mismatch
rejects the complete observation before any coadd numerical, identity,
membership, exposure/count, inventory, or provenance state changes. General
reprojection, interpolation, fractional shifts, and implicit recentering are
not part of this contract. The signal-centering operator is `L = I`; coaddition
does not subtract a mean or null mode.

The existing ordinary arithmetic remains `Q += u`, `N += u * signal`, and
`K += u * kernel`, followed by division on finite positive `Q`. The coefficient
`u` is the realized `weight_I` after observation normalization and any existing
optional global empirical rescaling. It is a nonprecision gridding and
normalization coefficient by default. Its inverse-squared signal unit does not
make it inverse variance. Precision requires `SCI-PTC-001` evidence for the
applicable marginal-precision and independence/covariance assumptions; no GLS,
covariance regularization, coadd uncertainty, or standardized-significance
claim follows from this contract.

Sequential and OpenMP execution must apply a declared deterministic or bounded
equivalence policy without unsynchronized shared-pixel mutation. For the fully
compatible, authoritative-valid control, observation order, centered offsets,
and the existing arithmetic operation order remain unchanged.

The declared policy is
`within-scan-exact-scan-farm-2gamma-n-sumabs-v1`. Sequential and
requested-parallel accumulation within one scan share the same
detector/sample-ordered primitive and are exact. Mutex-protected scan-farm
commits may arrive in different orders; each binary64 plane is bounded against
the long-double sum of per-scan planes by
`2 * gamma_n * sum(abs(scan_value))`, with
`gamma_n = n * epsilon / (1 - n * epsilon)`. Integer fact planes remain exact.

An explicitly invalid contribution is skipped before its numerical payload is
evaluated. A declared ordinary contribution requires finite signal, finite
positive coefficient, and finite declared companions; an unexpected violation
is a required pre-mutation failure. Signal, kernel, noise realizations, retained
exposure, and coadd-observation count share the admitted membership and integer
embedding.

The version-one F010 product hierarchy is:

| Product | Storage and unit | Distinct meaning |
| --- | --- | --- |
| `geometric_hits_I` | `int64`, count | Finite in-bounds sample/detector projections before upstream eligibility and estimator selection |
| `contributing_hits_I` | `int64`, count | Terms admitted by the named estimator contribution predicate |
| `coadd_observation_count_I` | `int64`, count | Admitted observation maps contributing to each coadd pixel |
| `upstream_eligible_exposure_I` | `float64`, detector s | Projected detector-seconds eligible under the upstream validity contract before contribution and normalization retention |
| `retained_exposure_I` | `float64`, detector s | Detector-seconds retained after contribution and normalization-support decisions |
| `normalization_support_I` | `uint8`, dimensionless | Numerical division/population support under the normalization rule |
| `science_policy_support_I` | `uint8`, dimensionless | Separate full-cut science-policy support |
| `science_valid_I` | `uint8`, dimensionless | The only authoritative raw science-validity mask |

`coadd_observation_count_I` is not applicable to observation maps. The v1
contract makes the complete F010 bundle explicitly unavailable for JINC and
detector-grouped products. No ordinary positive-coefficient rule is inferred
for JINC; a signed contribution predicate and any corresponding product
availability remain owned by `SCI-MAP-002`.

`coverage_I` is retained only as a bitwise compatibility alias of
`retained_exposure_I`, with detector-seconds meaning. `coverage_bool_I` is a
deprecated bitwise compatibility alias of `science_policy_support_I`. Neither
alias is a science-validity authority.

Normalization and science-policy support use separate versioned rules. Both
select finite strictly positive coefficient values. If `N` values remain, the
zero-based ascending order-statistic index is
`k = floor((floor(0.75 * N) + N) / 2)`. The realized threshold is the selected
coefficient at `k` multiplied by the applicable cut; empty input has threshold
zero. Ordinary normalization uses the `coverage_cut / 10` cut and
science-policy support uses the full `coverage_cut`. Both predicates require a
finite positive coefficient and `coefficient >= realized_threshold`; IEEE
`!(coefficient < threshold)` is not equivalent.

The one-way lifecycle is requested to effective to observation-resolved to
realized state; later stages do not rewrite earlier authorities. Realized
provenance preserves both algorithm/version identities, coefficient product
and lifecycle stage, lossless requested/realized cuts and thresholds,
positive-value count and selected order-statistic index, finite, positivity,
and comparison conventions, counts for each fact and state, required
companions, admitted bundle and observation membership/offsets, and exact
`raw-parent/product` digests. Downstream operators preserve raw
`science_valid_I` and raw-parent identity separately from local numerical
support, response, covariance, and output validity. A finite downstream value
cannot promote a raw-invalid input.

Coefficient stages use a closed vocabulary. Threshold selection records
`pre-observation-normalization-accumulated-coefficient` or
`pre-coadd-normalization-sum-of-admitted-observation-coefficients`. Published
state records
`post-observation-normalization-no-empirical-rescale`,
`post-observation-normalization-global-empirical-rescale-applied`,
`post-coadd-normalization-no-empirical-rescale`, or
`post-coadd-normalization-global-empirical-rescale-applied`. Empirical refresh
does not rewrite admitted observation state.

Filtering first freezes the validated raw F010 bundle. Filtered signal,
coefficient, F010, and compatibility-alias HDUs identify that immutable input
with `RAWSTATE=immutable_input` and one identical lossless `RAWPDGST`; filtered
empirical calculations cannot mutate the raw snapshot or digest. Profiles for
which the complete v1 bundle is unavailable retain their established legacy
coadd arithmetic and carry explicit absence reasons, without claiming F009 or
F010 successor coverage.

F009 and F010 remain `addressed_pending_reaudit`. The human-run
exact-repair-SHA `SCI-MAP-001-UNITY-001` gate is still required.
Calibration/unit/response, projection/WCS, coefficient/covariance, and upstream
eligibility conclusions remain conditioned on `SCI-CAL-001`, `SCI-AST-001`,
`SCI-PTC-001`, and `SCI-VAL-001`, respectively. Historical accepted map
products retain their original product-contract identities and are not
retroactively relabeled as carrying this successor bundle. See
[ADR 0009](adr/0009-science-map-bundle-admission-and-validity.md).

## Coordinate Frames And Astrometry

### Map Frames

| Product family | Frame | Spatial axis units |
| --- | --- | --- |
| Point and OOF maps | AltAz tangent-plane azimuth/elevation offsets around the configured map center | arcsec |
| Beammap detector maps | AltAz tangent-plane azimuth/elevation offsets around the Beammap source | arcsec |
| Science maps | Equatorial J2000 TAN projection centered on the configured field | deg |

The validated AltAz FITS axes are `AZOFFSET` and `ELOFFSET`. Science products
use their recorded equatorial WCS. FITS WCS is authoritative for persisted map
pixel-to-coordinate conversion; callers must not infer axis sign, handedness,
or wrapping from array memory order.

Telescope and detector angular variables in NetCDF use each variable's units.
Current TOD detector longitude/latitude and RA/Dec variables are recorded in
radians, while pointing-offset variables are recorded in arcseconds.

### Pointing Corrections

TolTECA owns selection of pointing-support calibration records. Citlali owns
validation and application of the supplied azimuth/altitude offsets:

- one finite az/alt pair is constant throughout the observation;
- two finite pairs with two positive, increasing MJD support values are
  linearly interpolated in MJD, require observation bracketing, and are not
  extrapolated;
- two finite pairs without a positive MJD support pair use the established
  linear interpolation across the observation span; and
- when no pointing observations are selected upstream, the finite offset
  values supplied in the low-level configuration are the correction request.

Offsets are in arcseconds and MJD support values are in days. Requested,
effective, and realized astrometry are recorded per observation. A later
observation cannot inherit a prior observation's calibration availability or
offset state.

### Beammap Photometry

TolProj determines the Beammap calibrator and estimates its per-array flux;
TolTECA supplies that calibration to Citlali. Citlali does not select the
calibrator or estimate its catalog flux. Each configured array flux is finite
and positive in mJy, with finite nonnegative uncertainty. Observation setup
replaces the complete per-array flux state rather than merging with state from
a previous observation.

## Units

Units belong to values and products, not to variable names alone.

| Quantity | Current convention |
| --- | --- |
| Accepted map signal and kernel | `mJy/beam` |
| Map gridding/normalization coefficient (`weight_I`) | recorded as inverse square of the associated signal unit; nonprecision by default, with precision conditional on `SCI-PTC-001` and applicable covariance evidence |
| Map noise variance | square of the associated signal unit |
| Upstream-eligible and retained exposure | detector-seconds; not unique wall-clock integration time |
| Hit/count products | integer counts with the product-specific sample/detector or admitted-observation meaning |
| Support/validity masks, standardized signal, and signal-to-noise | dimensionless; only `science_valid_I` is authoritative raw map validity under the successor contract |
| TOD signal | the recorded `signal_unit`/`BUNIT` |
| Raw ADC snapshots | signed 12-bit ADC counts, `[-2048, 2047]` |
| PTC weights | inverse square of the recorded signal unit as a dimensional statement; precision/covariance meaning remains conditional on `SCI-PTC-001` |
| Flags, IDs, counts, categories | dimensionless or `N/A` metadata |
| Frequencies | Hz |
| Pointing and Beammap fitted offsets | arcsec |
| MJD support | days |
| HWPR/telescope angular streams where emitted | rad unless the variable states otherwise |

The mapmaking boundary accepts the established unit tokens `mJy/beam`,
`MJy/sr`, `uK`, and `Jy/pixel`, and records conversion metadata. The active
Phase 4 product snapshots use `mJy/beam`; accepting the other tokens does not
erase the need to validate their conversion factors and downstream products.

For configuration, the `unit` field in
`tools/config/config_leaf_contract_resolved.json` is authoritative. A bare
number must not acquire an inferred unit in processor code. Unit conversions
belong at checked cold boundaries and their output metadata must name the
resulting unit.

### Standardized Map Products

The name `sig2noise` is reserved for an estimator intended to have an empirical
noise calibration. For the existing map products its arithmetic is the
jackknife-calibrated pixel quantity `signal * sqrt(empirical_weight)`; the
filtered point-source form is point-source flux divided by point-source
uncertainty. Its FITS metadata names the estimator. The
`SCI-MAP-001` successor does not by itself establish statistical significance:
that interpretation remains conditional on the applicable `SCI-PTC-001` and
`SCI-NOI-001/002` covariance, realization, and calibration evidence.

When empirical noise products are unavailable, Citlali may still publish
`signal * sqrt(formal_mapmaker_weight)`, but it is named
`formal_standardized_signal`, carries estimator type
`formal_weight_standardized`, and explicitly states that it is not a
statistical-significance map. The phase-4 v1 product snapshots predate this
truthfulness rule and are retained as historical contract debt rather than
silently reinterpreted.

### Pointing-Fit Significance And Dynamic Range

Pointing-table schema `citlali-pointing-fit-v2` makes three formerly conflated
quantities explicit:

- `sig2noise` is retained only for backward compatibility. Its historical
  pointing-table definition is fitted amplitude divided by the standard
  deviation of the complete map. Because a bright recovered source contributes
  to that denominator, this is a dynamic-range diagnostic and is not
  statistical significance.
- `peak_over_full_map_rms` carries that same historical value under its
  truthful name.
- `fit_sig2noise` is the fitted amplitude divided by the fitted
  amplitude uncertainty. It is a formal fit-significance diagnostic and does
  not by itself establish an empirical uncertainty in the presence of
  correlated map noise.

Population fruit-loop analysis calibrates a separate empirical point-source
significance by applying one fixed-PSF amplitude estimator to the source and
source-free blank-sky positions. The normal-scaled MAD of blank fitted
amplitudes, standardized by their formal weight uncertainties, calibrates the
source amplitude uncertainty. The estimator version, source-free region,
minimum blank-fit count, units, and validity are recorded in the analysis
manifest. Neither the legacy dynamic range nor a point-source S/N value is a
fruit-loop convergence criterion: convergence keeps amplitude, shape,
centroid, map change, and noise health as separate facts.

### Source Morphology In Pointing Convergence

An unresolved calibrator is assessed against the realized point-source kernel
for its array and iteration. A resolved planetary calibrator is not assigned
that point-source identity. Population convergence analysis convolves the
realized kernel with an observation-epoch, observer-specific uniform disk and
fits the resulting template after recentering its measured kernel centroid.
The disk angular diameter and ephemeris provenance are retained with the
analysis. This construction is a shape and within-observation convergence
model, not an absolute planetary brightness or flux-calibration model.

Planet and unresolved-source yields remain separate. A fitted width censored
at the pointing fitter's upper bound is not made interpretable by stable
iteration-to-iteration motion or by disk normalization; it remains a
measurement-limited PSF result.

## Validity, Missing Data, And Non-Finite Values

Required configuration scalars and vector elements are finite unless the
typed domain explicitly models another state. Missing, disabled, automatic,
and unavailable are semantic states; new code must not invent NaN, zero, or a
negative value as an undocumented replacement for one of them.

Persisted scientific products use these current validity rules:

- Timestream flags identify invalid samples. Detector/APT flags identify
  unusable detectors.
- Map signal-like pixels may be non-finite outside valid support. For the
  `SCI-MAP-001` successor ordinary-naive bundle, `science_valid_I` is the only
  authoritative raw validity mask: it requires normalization support,
  science-policy support, finite signal and every declared companion, and
  admitted identity. `weight_I`, exposure, finite population,
  `coverage_I`, and `coverage_bool_I` are not substitutes. Historical products
  without this successor bundle retain their versioned historical contract and
  cannot be retroactively promoted to the successor validity state.
- Fit-table numerical values may be non-finite when the corresponding fit is
  invalid. `flag`, `flag2`, `good_fit`, `converged`, and fit-quality fields are
  the validity authority for their respective tables.
- Beammap detector-TOD slots are padded; `detector_tod_n_samples` bounds valid
  samples and flags retain per-sample validity.
- Integer identities and required cardinalities must remain present even when
  a related floating-point diagnostic is unavailable.

Several diagnostic NetCDF families do not yet have complete `_FillValue` or
`missing_value` attributes. ECSV units also live in table metadata rather than
column unit fields. These are recorded schema debts. Until a successor schema
is approved, current flags, finite-value rules, and existing metadata remain
authoritative; callers must not silently assign a new sentinel interpretation.

## Configuration And Provenance States

TolTECA owns discovery, ordering, and merge behavior for numbered `NN*.yaml`
authoring files. The generated low-level Citlali YAML is Citlali's immutable
input boundary. Citlali records exact source identity supplied at that boundary
and does not reconstruct missing upstream merge history.

Scientific configuration proceeds in one direction:

1. **Requested:** what the accepted low-level YAML asks for, including expert
   values under disabled sections.
2. **Effective:** context-free normalization and activation decisions, such as
   whether a requested filter is active.
3. **Observation-resolved:** choices requiring sample rate, calibration,
   observation identity, MJD coverage, source context, or hardware
   availability.
4. **Realized:** what execution actually applied or produced, including counts,
   selected fallbacks, and product cardinality.

Disabling a feature does not rewrite its requested expert values. Effective
state records that it is inactive. Observation availability and processor
results do not mutate the request. A one-way adapter may populate a legacy
processor from typed state; no processor may synchronize changes back into the
request.

Provenance labels must identify which state they contain. A configured value
must not be called effective merely because it was copied into an output
header, and an effective intent must not be called realized before execution.

## Product And Failure Policy

The generated low-level YAML determines which configuration-controlled product
families were requested. The product contract then applies both directions:
requested products must exist, and explicitly disabled products must not be
emitted. Required companion diagnostics remain required even when they have no
separate output switch.

A required or enabled output write failure fails the reduction. Optional
diagnostics may be absent only when their contract classifies them as optional;
their absence must not masquerade as a complete requested product. Completion
markers do not override missing products or error-level log records.

## Determinism And Numerical Acceptance

Every accepted candidate must match the exact low-level configuration of its
active profile, produce the required product inventory, report zero unexpected
errors, and compare all required records without silent skips.

The active structural-closeout policies are:

| Profile | Numerical policy |
| --- | --- |
| Point | Zero tolerance across complete products and RTC/PTC timestreams, excluding only the volatile profiling sidecar |
| OOF | Zero tolerance across complete products and requested timestreams, excluding only the volatile profiling sidecar |
| Beammap | Zero tolerance across complete products and detector TOD, excluding only the volatile profiling sidecar |
| Science | Exact product sets and integer diagnostics; map RMS relative difference at most `1e-8`, PTC-weight RMS relative difference at most `1e-9`, detector-median absolute/fractional bounds `5e-5`/`1e-3`, and other diagnostic RMS relative difference at most `1e-7` |

These are behavior-preserving successor-run gates against the accepted refactor
snapshots. Historical OG/refactor comparisons may have separately approved
scientific tolerances, including the 1.5% filtered-science-map allowance; those
historical allowances do not widen the active successor profile.

Parallel execution is expected to be reproducible under the validated
comparison policy after the imported diagnostic-snapshot determinism fix. This
is not a universal promise of bitwise identity for every unvalidated mode,
thread count, library, or future algorithm. Determinism claims must name the
runtime policy and evidence.

An intentional algorithm, default, schema, or product change requires an entry
in `validation/intended_science_changes.json`, comparison with the predecessor,
scientific-owner approval, and a successor validation epoch when the expected
products or numerics change. Existing profiles are not loosened to make a new
result pass.

## Deferred Channels

Only Stokes I is validated. Enabled polarimetry remains a future capability,
not a permanently closed design direction. Its exit condition is an approved
polarimetry/HWPR contract plus an enabled reference dataset and product gate.
Until then, an enabled request fails before scientific execution.

The R/quadrature stream is measured detector data, not a synthetic kernel. Its
execution remains deferred. Before activation, its channel identity, sample
alignment, units, calibration and extinction policy, transfer of linear
operations, flags, learning use, map role, and provenance must be approved.
Plain R-derived modes must not clean the primary science stream by convenience.

## Known Debt And Decisions Still Requiring Scientific Ownership

The following are not silently resolved by this document:

- the guaranteed lifetime of detector UID identity across future APT versions
  and observing campaigns;
- whether future instrument changes preserve the current network-ID mapping and
  ordering;
- complete units and standardized missing-value metadata for diagnostic NetCDF
  and ECSV products;
- an explicit canonical RA wrapping/sign/epoch policy beyond the current
  recorded J2000 WCS behavior;
- scientifically acceptable fallback policy for future missing calibration,
  Beammap prior/flux, or source-protection inputs;
- the enabled polarimetry and HWPR scientific contract;
- the measured R-channel contract; and
- the action policy for raw ADC saturation or low headroom (detection,
  severity, persistence, network exclusion, and reduction failure), tracked as
  retained debt D17; and
- whether OOF should eventually become a distinct public execution type rather
  than a distinct intent routed through pointing execution.

Each decision must name its owner, accepted behavior, provenance, failure
policy, and validation dataset before implementation relies on it.

## Validation Routing For Changes

| Touched behavior | Minimum validation |
| --- | --- |
| RTC/PTC values, flags, detector identity, or TOD schema | Complete point TOD and metadata comparison; add affected long-mode runs when shared behavior changes |
| Pointing fit or AltAz map behavior | Point profile |
| OOF-specific products or multi-observation pointing state | OOF profile |
| Coadds, fruit loops, celestial WCS, or science post-processing | Science profile |
| Beammap calibration, detector maps/TOD, fits, QC, or split outputs | Beammap profile |
| Product inventory, units, frames, indexing, or missing-data semantics | Product contract plus mode numerical profile; intentional changes require a successor contract |
| Provenance state | Full relevant provenance audit and exact low-level config comparison |
| Enabled polarimetry or R execution | No ordinary refactor gate is sufficient; approve the scientific contract and reference dataset first |

All accepted reductions must finish cleanly with zero unexpected error-level
messages. Validation follows the behavior touched; a fast point run does not
substitute for a long-mode gate when the changed behavior is specific to that
mode.
