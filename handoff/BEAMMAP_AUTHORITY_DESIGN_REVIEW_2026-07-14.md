# Beammap Configuration Authority Design Review

**Review date:** 2026-07-14

**Reviewed repository:** `/Users/gwilson/GitHub/citlali-refactor`

**Reviewed source snapshot:** `cf59ec036`

**Purpose:** Preserve the completed read-only review of the Beammap
configuration-authority design for a subsequent Codex task. This handoff does
not introduce an implementation or revise the governing roadmap.

## Review Scope

The review covered:

- `AGENTS.md`
- `doc/REFACTOR_STATUS.md`
- `doc/BEAMMAP_CONFIG_AUTHORITY.md`
- `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md`
- `tools/config/audit_beammap_boundary.py` and its focused tests
- `tools/config/beammap_legacy_paths.json`
- the typed Beammap config model, validation, readers, and fitter adapter
- Beammap iteration, prior, fitting, masking, reference-selection, detector
  TOD, map-output, detector-table, and finalization paths
- existing mapmaking, post-processing, and timestream provenance boundaries
- the adjacent observation-scoped `beammap_source.*` photometry/flux path

No source or documentation was changed as part of the review. The Beammap
boundary audit and its six focused tests passed at the reviewed snapshot:

```text
beammap config boundary: paths=74 defaults=True typed_model=True
literal_boundary=True authority=True provenance=missing drift=False

Ran 6 tests
OK
```

## Executive Conclusion

The 74-leaf boundary is correctly scoped as the frozen `beammap.*` requested
policy domain in the current low-level configuration. All 74 current leaves
have typed destinations, and Beammap execution generally reads the typed
object. The boundary should not absorb configuration owned by mapmaking,
RTC/PTC, noise products, post-processing, or observation photometry.

The boundary is not, however, the complete set of scientific inputs used by a
Beammap reduction. In particular, observation-scoped `beammap_source.*`
identity and flux data materially affect Beammap calibration but belong to an
adjacent calibration/photometry domain. This dependency and exclusion should
be stated explicitly so that “74-leaf Beammap policy boundary” is not read as
“all Beammap scientific configuration.”

The target contract in `doc/BEAMMAP_CONFIG_AUTHORITY.md` is sound:

```text
merged YAML -> immutable Beammap request -> effective Beammap plan
            -> narrow numerical adapters -> realized iteration/output record
```

The current implementation does not yet satisfy the “immutable request” part.
The installed `BeammapConfig` is a mixed requested, effective, and
observation-resolved carrier: loaders normalize phase, prior, and split
policy; disabled mapmaking overwrites the iteration count; and observation
setup mutates prior enablement and the prior filepath. The next implementation
must separate these states without changing numerical algorithms.

## Assessment of the 74-Leaf Surface

### What is correctly included

The manifest correctly freezes policy under `beammap.*` for:

- iteration and convergence
- phase strategy
- reference-detector subtraction and derotation
- detector weighting
- Gaussian fit support radius
- RFI and scan-band masking
- split-by-flag map output
- soft-prior configuration and alignment policy
- detector-specific PTC TOD selection
- array/network fit-quality and sensitivity flagging
- sensitivity PSD limits

No typed leaf gap was found in this surface during the code trace.

### What must remain outside

The following are adjacent authorities and should be referenced or
cross-checked, not copied into the Beammap request:

- mapmaking grouping, method, geometry, map counts, and logical map writes
- raw and processed timestream policy and required write counts
- post-processing fit attempt/valid-fit cardinality
- noise-product policy and realization cardinality
- `beammap_source.*` observation identity and source flux calibration
- realized Gaussian parameters, detector flags, prior matches, masks, maps,
  and detector tables

### Qualification on the number 74

The number 74 is a frozen snapshot count, not a semantic schema size. It
counts individual indices of fixed vectors in `data/config.yaml`, while the
readers consume those values as whole vectors and validate their size against
the runtime array map. The manifest is therefore appropriate for drift
characterization, but a future schema change in array cardinality should not
be interpreted as an automatic scientific-policy expansion.

### Limits of the current audit

The green static audit proves:

- the manifest schema, count, ordering, and digest
- exact agreement with the current default `beammap` tree
- confinement of detected literal `beammap` paths to an expected file set
- presence of expected typed structs
- expected boundary reader and adapter call counts/order
- the declared authority-inventory state
- the deliberate absence of dedicated Beammap provenance

It does not yet prove:

- 74/74 per-leaf reader coverage
- 74/74 requested/effective serialization coverage
- validation of every scalar and vector element
- absence of dynamically constructed raw-YAML paths
- absence of later mutations to the supposed request
- a complete execution-read census
- lifecycle or output-cardinality completeness

The audit is therefore a valid characterization gate, not yet the Beammap
authority completion gate.

## Current Requested/Effective Conflation

The review found the following present mutations or normalizations:

1. `normalize_beammap_phase_strategy` forces `locator_iter` to zero and moves
   `measurement_start_iter` after it before the typed object is installed.
2. Prior phase-specific gates and score penalties inherit their base values
   when the phase-specific keys are absent.
3. A null/missing prior path disables `priors.enabled` during loading.
4. Split flag values are sorted and deduplicated; an empty value retains the
   `[0, 1]` defaults.
5. Disabling mapmaking overwrites `beammap.iteration.max_iterations` with one.
6. Observation setup disables priors for non-detector grouping or failed
   loading.
7. Prior loading replaces a requested relative path with a resolved path.

These behaviors should be preserved initially, but represented as explicit
effective or observation-resolution records with reasons.

## Effective Policy That Provenance Must Record

### Context-free effective resolution

Record both the requested value and the effective result for:

- locator iteration, including the forced-zero correction
- measurement-start iteration, including ordering correction
- whether the iteration schedule contains any measurement pass
- legacy phase behavior when `phase_strategy.enabled=false` (locator at zero,
  measurement beginning at one)
- requested and effective maximum iterations, including the no-map forced-one
  behavior
- whether convergence is active, based on mapmaking, tolerance, and phase
  eligibility
- inheritance of `max_d2_iter0`, `max_d2_after_iter0`,
  `score_lambda_iter0`, and `score_lambda_after_iter0`
- prior enablement suppressed by an empty/null requested path
- split flag defaulting, sorting, and deduplication
- positional flagging-vector resolution to named TolTEC arrays
- per-phase detector-weighting behavior
- whether fit-radius support is active (`fit_radius_fwhm > 0`)
- zero-sentinel interpretations for disabled gates, upper limits, rejection
  caps, and sensitivity policy

The `ptc_after_iter0` name deserves special attention: the current execution
uses PTC weights in a Beammap measurement phase, not simply for every numeric
iteration greater than zero. Provenance must state the actual effective rule.

### Observation-resolved policy

Record:

- observation identity, effective grouping, detector/map/scan counts, and
  whether detector-grouped Beammap policy is active
- effective activation of priors, RFI masking, scan-band masking,
  detector-specific TOD, split output, and detector fitting
- prior activation result and reason: not requested, null path,
  non-detector grouping, unresolved file, invalid schema, empty/no-valid-row
  table, or loaded
- requested prior path, resolved path, content identity/hash, source schema,
  input/kept/dropped row counts, `(array, network)` group count, and slot count
- prior-frame interpretation (`centered`, `derotated`) and the fixed
  `1e-3` arcsec sigma floor/absolute-value repair currently applied to rows
- reference-detector resolution method: configured detector, automatic
  `nw=3`, fallback `nw=2,3,4`, all-unflagged fallback, or disabled
- reference candidate count, resolved detector, applied reference offsets,
  and derotation elevation/frame
- detector-TOD output path, requested uniform/dense slot counts, realized slot
  count, unique selected scan count, and maximum PTC sample count
- split-output realization: effective activation, maps per requested flag,
  empty partitions, and any fallback to standard unsplit output

## Realized Cardinalities and Ownership

### Beammap provenance should own

Per observation:

- observation index/identity
- detector, map, and scan counts
- lifecycle started/completed state

Per Beammap iteration:

- iteration index and phase
- iteration attempted/completed state
- active/unconverged map count
- mapmaking pass count, including a second pass after successful scan-band
  masking
- whether the source-aware RTC rerun occurred
- whether the fitting context completed
- newly converged and total converged map counts

At the terminal state:

- requested/effective maximum iterations
- iterations attempted and completed
- terminal iteration
- termination reason: maximum reached, all maps converged, or failure
- final detector-result row count
- good/bad fit and converged/non-converged counts
- final flagged/unflagged detector counts

Beammap-specific required outputs:

- expected/completed APT table writes and row counts
- expected/completed fit-QC table writes and row counts
- expected/completed detector-specific TOD file count and its
  detector/slot/sample shape
- realized split partitions and maps written per partition
- the selected final PTC-output iteration
- overall `outputs_completed` and `reduction_completed` state

### Existing domains should remain authoritative

Do not duplicate these values in an independently mutable Beammap record:

- Mapmaking provenance owns observation map counts and required logical map
  write counts.
- Post-processing provenance owns Beammap fit-context count and aggregate
  attempted/valid fit counts.
- Raw/timestream-output provenance owns RTC/PTC chunk and required-write
  cardinalities.
- Noise provenance owns realization and empirical-product cardinality.

Beammap completion should cross-check those sidecars/plans. It may record an
expected cross-domain context count or a stable reference, but it should not
create a second authority for the same aggregate.

The existing post-processing Beammap check only requires at least one fit
context. Dedicated Beammap lifecycle validation must require the exact number
of completed fit contexts implied by completed Beammap iterations and
observations.

## Adjacent Source-Flux Safety Issue

`beammap_source.*` should remain outside the 74-leaf manifest, but its current
lifecycle is a prerequisite for trustworthy Beammap provenance:

- `BeammapConfig::source` is reset before reading each observation, but the
  legacy `source_flux_mJy_beam` map is populated without first being cleared.
  A later observation can therefore inherit an array flux omitted from that
  observation.
- Invalid or missing required source flux calls `std::exit` from
  `Engine::get_photometry_config`, contrary to the project library-failure
  contract.
- Observation-scoped source identity/flux values are not constructed and
  validated as one atomic value before processor state is mutated.

The smallest safe treatment is not to expand the 74-leaf domain. Instead,
create or complete the adjacent observation source/flux value, replace prior
state rather than merge it, validate it before Beammap execution, propagate
failure normally, and let Beammap provenance reference its resolved identity
and calibration source.

## Smallest Safe Implementation Sequence

Implementation remains gated on successful science and Beammap validation of
the active post-processing domain. After those gates pass:

1. **Record the prerequisite gate.** Update the living status and validation
   ledger for the accepted post-processing science and Beammap runs before
   beginning Beammap authority work.

2. **Strengthen characterization without changing runtime behavior.** Declare
   adjacent domains and exclusions; add mechanical 74/74 reader and requested
   serializer coverage; add vector-element, shape, and named-array mapping
   validation; and make the audit capable of detecting request mutation and
   the eventual provenance checkpoint.

3. **Introduce a pure, unwired execution plan.** Add an immutable request,
   effective snapshot, explicit resolution records, resettable
   observation/realized state, and pure YAML serializers. Test every current
   normalization, disabled sentinel, and repeated-observation reset. Do not
   publish a sidecar or claim execution authority at this checkpoint.

4. **Perform the documented fitting-radius cut first.** Supply the effective
   `fit_radius_fwhm` through a narrow fitting input and retire the config-load
   synchronization into shared `map_fitter` policy. Leave Gaussian fitting,
   workspaces, and realized results unchanged.

5. **Move existing policy mutations into resolution.** Phase/split/prior
   inheritance and no-map iteration behavior belong in effective state.
   Prior loading, grouping activation, reference selection, and source-flux
   availability belong in observation state. Execution may temporarily
   consume a one-way effective compatibility snapshot while consumers are
   switched.

6. **Add lifecycle and required provenance.** Hook observation and Beammap
   iteration begin/complete events, Beammap-specific output completion, and
   exact cross-domain cardinality checks. Publish one required atomic,
   versioned Beammap sidecar only after successful pipeline output and
   lifecycle completion. Publication failure must fail the reduction.

7. **Retire compatibility state only after validation.** Run focused C++
   tests, the strengthened boundary audit, all config profiles, the local CLI
   build, full CTest, and full config preflight. The user then runs the matched
   Unity Beammap gate. Require identical merged config, zero serious log
   records, exact detector identities/flags and accepted products, valid
   provenance, no skipped required records, and no unexplained performance
   change before removing the compatibility snapshot and updating the
   inventory/status.

This sequence deliberately avoids a field-by-field re-migration of policy
that is already typed-authoritative.

## Completion Criteria

The Beammap authority domain is complete only when all of the following hold:

- The post-processing science and Beammap prerequisite gates are accepted and
  recorded.
- The frozen `beammap.*` surface is mechanically covered 74/74 by direct
  typed readers and requested/effective serializers.
- Every scalar and vector element has finite/range/shape validation appropriate
  to its semantics, with named array identity made explicit.
- The accepted Beammap request is immutable after parsing.
- Every normalization or fallback has an effective or observation-resolution
  result and reason.
- Beammap execution consumes the effective plan or one narrow one-way adapter;
  no execution path falls back to raw YAML.
- The shared fitter no longer owns requested fit-radius policy.
- Prior loading and reference selection do not mutate the request.
- Observation-scoped source flux cannot leak across observations and invalid
  flux does not terminate the process from library code.
- Exact observation and Beammap iteration lifecycle/cardinality is enforced,
  including early convergence.
- Beammap-specific required outputs have expected/completed cardinality and
  failures propagate.
- Cross-domain map, fit, and timestream cardinalities agree without duplicate
  authorities.
- A versioned Beammap sidecar is written atomically only after complete
  success; writer or lifecycle failure fails the reduction.
- Focused tests, build, CTest, config profiles, full preflight, boundary audit,
  and the matched Unity Beammap scientific/product gate pass.
- The accepted Beammap run has zero unexpected error-level messages and no
  required product is skipped.
- No Gaussian-fit, prior-match, detector-flagging, detector-map, RTC/PTC, or
  mapmaking numerical algorithm changed as part of this authority migration.

## Decisions Requiring the Project Owner

The implementation must not silently choose among these:

1. Should invalid phase settings be coerced as today or rejected? Is a
   locator-only run, where no measurement pass fits inside `iter_max`, valid?
2. Does `ptc_after_iter0` mean numeric iteration index greater than zero, or
   the current measurement-phase behavior?
3. When priors are requested, which failures are acceptable fallbacks and
   which are fatal: null path, missing file, malformed schema, no valid rows,
   dropped non-finite rows, or sigma repair?
4. Is taking the absolute value of prior sigma and flooring it at
   `1e-3` arcsec an approved scientific normalization?
5. Is a requested split-flag partition with no matching detectors a valid
   zero-cardinality output, a warning, or a required-output failure?
6. Is the current “no requested flags matched, therefore write unsplit maps”
   fallback scientifically and operationally acceptable?
7. Are out-of-range configured reference detectors allowed to fall back to
   automatic selection? Are the `nw=3`, `nw=2,3,4`, and all-unflagged
   fallbacks approved?
8. What is the canonical array ordering/identity contract for all positional
   flagging vectors? May the runtime array map differ between observations?
9. What do zeros mean for sensitivity PSD limits, upper flag limits,
   prior-distance gates, RFI/scan-band rejection caps, and fit-radius support?
10. What is the required source-flux behavior for a missing array: fatal,
    named-catalog fallback, or explicitly uncalibrated processing? Incidental
    inheritance from prior observation state is not acceptable.
11. What are the authoritative source RA/Dec frame, range, wrap, and missing
    value rules?
12. Are duplicate detector-TOD scan selections acceptable when requested
    uniform/dense slot counts exceed available scans, or should counts be
    capped/rejected?
13. Which Beammap products are mode-required even without a dedicated enable
    leaf, particularly the APT table and fit-QC table?
14. Should Beammap provenance retain only the prior file path/hash/schema, or
    must the exact prior source bytes be archived through the broader
    calibration-source provenance mechanism?

## Over-Engineering Risks

Explicitly avoid:

- Expanding this authority domain to absorb mapmaking, post-processing,
  RTC/PTC, noise, or observation photometry.
- Creating a monolithic Beammap plan that stores detector maps, full fit
  arrays, masks, prior slots, or detector tables.
- Serializing per-detector scientific results into the provenance sidecar;
  those belong in existing APT, fit-QC, FITS, and NetCDF products.
- Duplicating fit, map, noise, or timestream cardinality in multiple sidecars.
- Introducing an event bus, generic workflow DAG, polymorphic execution-plan
  framework, dependency-injection container, or service registry for this
  bounded migration.
- Adding new cross-cutting public mutable state to `Engine`.
- Reworking Gaussian fitting, Ceres behavior, prior matching/alignment,
  detector flagging, detector-TOD selection, RTC/PTC cleaning, or mapmaking.
- Adding allocations, virtual dispatch, string-map lookups, filesystem access,
  or provenance logging to detector/sample/pixel hot loops.
- Wrapping every hot-loop integer in a new identity type when a checked cold
  boundary mapping is sufficient.
- Publishing a schema or sidecar before the lifecycle and required-output
  completion rules are executable and tested.

## Recommended Handoff Position

The next Codex task should begin only after the living status confirms that
post-processing science coadd routing and Beammap iterative-fit validation have
passed. Its first Beammap change should be a bounded characterization/plan
checkpoint, followed by the narrow fitter-radius cut. It should not redesign
scientific algorithms, broaden the domain, or claim provenance completion
until the matched Beammap validation gate accepts the exact implemented
checkpoint.
