# SCI-AST-001 bounded repair and re-audit handoff — 2026-08-01

Status: prepared and held for the approved `SCI-ALIGN-001` application
interface; no AST repair launch or application edit is authorized yet.

## Authority and disposition

The project owner approved `SCI-AST-001-D001` through `D008` in
`SCI-AST-001_COORDINATOR_DECISION_2026-08-01.md`. This handoff translates
those decisions into a bounded future implementation lane. It does not approve
the assessed implementation, close a finding, select a presently usable repair
SHA, launch a task, authorize Unity work, launch re-audit, integrate code, or
expand production use.

- Governing implementation assessed by the audit:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Application authority ref at audit dispatch: `codex/refactor-mainline`.
- Audit branch: `codex/audit-sci-ast-001`.
- Audit content commit: `429e1b5361683ba15c8d897ba22bdc4c4d03bf91`.
- Audit identity commit: `e3553bc0fcaa158ed4d986f59e9f25e5e2eeac7a`.
- Final audit artifact SHA-256:
  `0be6771bbe5653bd42e90bc9a8cec1cd69ad84af971e6e7bca3d2fc21ed4bd98`.
- Frozen independent core: `SCI-AST-001_INDEPENDENT_CORE.tex`, commit
  `17d683ada3856ecb5f0a5c42eed744cb219a3586`, SHA-256
  `ed1fe3bf68ed53974b8c910bd3824717eb0cf5ff11d0b27c0fdf27aa6e606276`.
- Owner-decision content commit:
  `8ee5b9fc192877106c4d5d747dfef75646edcfd9`.
- Owner-decision identity commit:
  `55bbdd4e0146cbf6d97dc39da232dd54868af9a8`.
- Owner-decision artifact SHA-256:
  `c2e8f2c18f2d96a27afd6b6a083408b17562e596ee1e2851a00de3d9c7e5e9d9`.
- CAL identity amendment: `CAL-D002-IDENTITY-001` at
  `0a17d088aded7fc6c18a59522f5b2b2fce9749ad`, artifact SHA-256
  `d40def5bb1450446abc858d2acfc1924c2f65276f8f48eec89cf05a171cdcf6a`.
- Canonical incoming ALIGN handoff `SCI-AST-001-XAUD-001` SHA-256:
  `20494a19a26273aca21a968d5ebe521dd444675d50763e2ef67ca01022fe9ccd`.
- ALIGN phase-zero evidence commit:
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`, exact parent
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Required future repair branch: `codex/repair-sci-ast-001`.
- Required future worktree: a fresh Codex app worktree from the exact
  coordinator-selected application SHA, never an audit or coordination branch.

Contract is `approved`; assessed implementation is `nonconformant`;
validation is `in_progress`; production is `existing_use_only`; verdict is
`amend`; and re-audit is `required` until all closure gates succeed.

## Repair-base decision and dependency hold

There is no admissible exact AST implementation base at this review.

The audit baseline
`9aae0e669384c5c0c0dda93debc194d6b8dac787` is retained as the exact
comparison authority, but it lacks the typed ALIGN field/grid interface that
`SCI-AST-001-F004`, `F011`, and `F012` require. Starting AST application edits
directly from it would either duplicate ALIGN ownership or force a later
unreviewed interface transplant.

The current ALIGN repair-line commit
`53c7154a3633dfe19dc036cfb5a6250f729a897d` is also not admissible. Its exact
parent is the audit baseline, and its delta contains only phase-zero diagnostic
and evidence artifacts. It contains no ALIGN application-source change.
`SCI-ALIGN-001` is explicitly paused for owner authority at
`ALIGN-P0-D001` through `D005` before phase-one fixtures or application edits.

The active MAP, CAL, convolve/noise, AST audit, ALIGN audit, and coordination
branches are not repair bases for AST and must not be inherited or
cherry-picked merely because they are newer.

The future exact AST base must be one of the following, selected and recorded
by the coordinator after ALIGN's exact-SHA evidence and fresh re-audit are
accepted:

1. the exact accepted `codex/repair-sci-align-001` successor application
   commit that implements the approved ALIGN-to-AST interface; or
2. an exact clean integration commit containing that reviewed ALIGN interface
   and no unreviewed scientific lane.

The governing AST audit sequence requires completion and re-audit of ALIGN
before AST application repair. This handoff does not amend that sequence.
Coordination documents and AST fixture design may advance while ALIGN is in
repair, but no AST application branch or edit begins from an unaccepted ALIGN
candidate.

## Mandatory prelaunch gates

Before creating `codex/repair-sci-ast-001` or editing application code, the
coordinator must record all of the following:

1. `ALIGN-P0-D001` through `D005` are resolved and the active-field registry is
   owner-reviewed.
2. ALIGN phase one has implemented the ordered detector-reference grid and
   every AST-consumed field's stable identity, native-source mapping, units,
   frame, topology, validity/support, timing residual, origin, interpolation
   method, and original-versus-synthesized eligibility.
3. The exact ALIGN candidate has passed its required focused, compatibility,
   product, lifecycle, sequential/compiled-path, config, and performance gates
   without required-data skips.
4. The human-run exact-ALIGN-repair-SHA evidence has been returned and a fresh
   ALIGN re-audit has accepted the interface consumed by AST.
5. The coordinator has reviewed the ALIGN-to-AST interface and its new or
   superseding final return handoff against the final AST decisions; the
   pre-repair `SCI-AST-001-XAUD-001` alone is not sufficient.
6. The exact AST base SHA and ref are recorded, the future worktree is clean,
   and no `codex/repair-sci-ast-001` branch or other AST repair worktree already
   exists.
7. The selected SHA is proven independent of the AST audit and coordination
   branches and does not silently absorb unfinished MAP, CAL, convolve/noise,
   or unrelated implementation work.
8. A fresh repair-dispatch manifest is frozen with the selected application
   SHA, accepted final ALIGN return handoff, final AST owner-decision digest,
   CAL identity amendment, and every still-applicable incoming handoff. The
   accepted ALIGN return must supersede or explicitly disposition the
   pre-repair `SCI-AST-001-XAUD-001`; the audit inbox manifest is not reused as
   a repair manifest.

Any failed or ambiguous gate keeps launch status `held_for_ALIGN_interface`.

## Phase 0 — fixtures, bounds, and compatibility evidence

After the prelaunch gates pass, but before changing AST implementation, add or
freeze focused fixtures and offline measurements for the approved contract.
Commit that evidence separately if any owner stop condition is reached.

Phase zero must:

1. bind the exact consumed ALIGN grid/registry/version and reconstruct its
   ordered sample identity without repeated per-sample IDs;
2. trace the realized sign, basis, rotation, handedness, composition order,
   frame, epoch, longitude topology, correction support, detector binding,
   WCS, validity, and product identity through the current operator;
3. identify exact approximately one-square-degree Science, Point, OOF, and
   Beammap compatibility fixtures, including center, edge, corner,
   source-crossing, centroid, and PSF-width evidence;
4. preregister astrometric compatibility tolerances from existing evidence,
   not from the repair candidate;
5. measure the pointing-support time-quantization error and drift-rate bound,
   deciding whether integer seconds remain adequate under `D004`;
6. compare the current small-angle operator offline with an independent exact
   spherical reference over the supported footprint, focal-plane offset, and
   correction envelope, without adding exact spherical work to production;
7. measure full-TOD coordinate quantization over the supported domain and
   decide whether its existing large-array representation passes `D008`;
8. freeze the map-center inverse-TAN response fixtures at the equator and
   representative declinations, including high declination and wrap; and
9. inventory full/mini TOD coordinate fields, WCS/catalog precision, per-field
   units, factorized validity, product availability, and current provenance
   cardinality/size/runtime.

Stop for owner direction if the established transform moves ordinary sources,
the time or large-array precision bounds fail, the small-angle discrepancy is
not negligible relative to accepted astrometric performance, factorized
validity is impossible for an enabled path, the ALIGN interface is incomplete,
or the bounded product changes are materially burdensome.

## Bounded implementation work packages

The following is the maximum future repair surface.

### F004, F011, and F012 — ALIGN admission and coordinate validity

- Consume the approved ALIGN ordered grid and per-field registry; do not
  reconstruct clocks, interpolate telescope fields, or redefine gap/origin
  eligibility in AST.
- Admit stable field identity, source mapping, units, frame, topology,
  support/validity, timing residual, origin, interpolation method, and
  original/synthesized eligibility before coordinate use.
- Give each persisted field its actual registry metadata and remove the
  generic all-radians telescope attribute.
- Keep ALIGN eligibility, projection validity, product support, and signal
  flags distinct. Block invalid or unsupported coordinates before geometry,
  integer conversion, mapmaking, fitting, feedback, or persistence.
- Persist dedicated AST validity through the approved factorization: one
  packed aligned-sample status, detector admission once per detector, and
  product-level counts. Do not emit a routine detector-by-sample validity
  matrix. A nonfactorable enabled case fails the affected coordinate product
  and returns for owner review.
- Limit downstream work to AST input admission, product-boundary checks, and
  required-failure propagation. Do not change a downstream estimator in this
  lane; its owning package retains the recipient re-audit.

### F001 and F002 — TAN domain and longitude topology

- Require finite projection inputs, finite `D > 0`, and finite continuous
  output. Every other case is explicit coordinate invalidity; never map a
  singular or back-hemisphere direction to center, clamp it, or reuse stale
  coordinates.
- Use one canonical shortest-signed longitude difference in `[-pi, pi)` in
  both wrap directions and normalize persisted longitudes to the approved
  interval. Apply the same topology after inverse TAN and at adapters.
- Keep projection-invalid, valid-but-outside-product, and upstream-ineligible
  states distinct and fail required products under the approved policy.

### F003 — truthful automatic-only WCS configuration

- Preserve exact numeric zero in `crpix1`, `crpix2`, `crval1_J2000`,
  `crval2_J2000`, `tan_ra`, and `tan_dec` as the legacy `automatic` sentinel.
- Preserve current centered/source-derived WCS for admitted automatic
  requests.
- Reject every nonzero or non-finite request at configuration admission with
  a field-specific unsupported error. No admitted field may be ignored,
  overwritten, or silently defaulted.
- Do not implement explicit nondefault WCS control in this repair.

### F005 — pointing-correction support

- Implement explicit constant, two-present finite strictly increasing
  bracketed-MJD, and both-absent legacy observation-span modes.
- Reject mixed, equal, reversed, non-finite, zero-span, unbracketed, clamped,
  nearest, extrapolated, or stale support.
- Retain integer-second representation only if the preregistered drift-rate
  and quantization bound is astrometrically negligible. Otherwise improve only
  to the demonstrated need and remain compatible with the admitted ALIGN time
  identity; do not wholesale retime data.

### F006 and F013 — transform identity and detector binding

- Preserve the demonstrated end-to-end signs, tangent basis, focal-plane
  rotation, handedness, and composition order. Document each stage and use a
  single explicit boundary adapter where needed; do not introduce a global
  sign flip, axis swap, or rotation change.
- Admit either a proven observation-local/target-row-order binding or an
  explicit keyed mapping. Verified-row mode requires exact artifact and
  observation provenance, network order, per-network count/tone order, and
  unique acquisition keys; unkeyed reorder fails. Explicit-key mode is
  permutation invariant.
- Require exact matching x/y counts, finite measured positions, and unique
  admitted acquisition keys or UIDs when that identity is present. Duplicate,
  missing, conflicting, non-finite, or out-of-lifecycle binding fails before
  coordinate use.
- Keep target acquisition identity, selected measured Beammap row and matcher
  edge, and optional design identity distinct. Perfect design matching is not
  required or claimed; design identity must not change measured-geometry
  coordinates.
- Retain the small-angle production hot path inside the phase-zero validated
  envelope. Test just inside, exactly at, and just outside its declared
  boundary; fail outside it and do not add a production exact-spherical
  fallback.

### F007 and F008 — map-center uncertainty only

- When a product reports equatorial positional uncertainty, evaluate one local
  inverse-TAN 2x2 response at the realized map/WCS tangent center and apply it
  to the available map-center covariance, including the cross term.
- Preserve native tangent-plane uncertainty without unnecessary conversion.
- Missing or unmodeled terms are explicitly unavailable, never zero. Do not
  fabricate a total or a new precision claim.
- Do not calculate or store per-sample, per-time, per-detector, per-pixel,
  response-grid, dense, or broadly composed correction/ALIGN/APT/frame/
  systematic covariance.

### F009 and F010 — compact lifecycle, WCS, and precision

- Persist requested, effective, observation-resolved, and realized AST state
  atomically and one-way. Include the ALIGN grid/registry identity, support,
  frame/WCS, detector binding, approximation envelope, validity/uncertainty
  availability, product links, algorithms/contracts, counts, and digests.
- Reuse the ALIGN grid identity and admitted detector mapping; do not repeat
  routine per-sample identity arrays.
- Use double precision for compact product-level WCS authorities and
  fitted/catalog sky coordinates, preserving FITS index base, scale sign,
  handedness, frame, epoch, and longitude topology.
- Preserve native AltAz tangent coordinates for Point, OOF, and Beammap, and
  explicit equatorial J2000 TAN coordinates for Science. Publish
  `FK5`/`EQUINOX=2000` only when headers prove that identity; otherwise
  preserve ICRS or apply one named approved transform. Never default a missing
  epoch. Mark ambiguous retained products `legacy_unverified`; they cannot
  support a new precision claim.
- Feedback and fruit-loop boundary admission preserves signed `CDELT`,
  handedness, and frame identity. Never erase a scale sign with an absolute-
  value comparison.
- Retain the existing full-TOD coordinate representation only if the frozen
  quantization gate passes. Do not widen the large arrays without a new owner
  disposition.

### F011 — full and mini TOD products

- Full TOD retains its coordinate arrays with the compact state required to
  interpret their units, frame, identity, and validity.
- Mini TOD gains no coordinate or validity arrays. It explicitly marks those
  products unavailable and carries only its approved compact identity,
  availability, and summary counts.
- Product and writer failures for required outputs propagate to the CLI.

### F014 — simulation parity

- Supported AltAz and RA/Dec simulations reuse the applicable real-data
  coordinate preparation, topology, frame identity, support state, and AST
  operator.
- Reject Galactic simulation and every other unimplemented frame at
  configuration admission. Do not implement Galactic simulation in this
  repair.
- Simulation fixtures add no work or allocation to ordinary real-data
  production paths.

### F015 — evidence

- Add focused equation, boundary, identity, validity, WCS, product,
  provenance, simulation, sequential/parallel, compatibility, storage, and
  performance gates tied to the exact repair SHA.
- Prepare but do not execute a successor `SCI-AST-001-UNITY-002` request after
  all local gates pass. The governing-SHA `SCI-AST-001-UNITY-001` request is an
  unsupplied immutable audit artifact and must not be reused.

## Prohibited scope

Do not:

- modify or repair ALIGN clocks, interpolation, gaps, scans, origin,
  eligibility, exposure, or field topology in the AST lane;
- change CAL responsivity/extinction, APT generation, TolAPT matching,
  Beammap fitting, MAP estimators, RTC/PTC filters, source estimators, OOF,
  convolve/noise, or fruit-loop behavior;
- replace the established coordinate hot path with per-sample exact spherical
  operations or impose an arbitrary timing/precision threshold;
- add per-sample/per-detector/per-pixel Jacobians, response grids, dense/full
  covariance, broadly composed uncertainty, or a fabricated total;
- widen every detector-coordinate timestream to double, add repeated
  per-sample identity, add detector-by-sample validity, or expand mini TOD;
- implement explicit nondefault WCS controls, Galactic simulation, new frame
  transformations, EOP/refraction processing, polarimetry, or a perfect design
  identity claim;
- amend `FRAMEWORK-COMP-D005` or `D006`, or launch the held closure pilot;
- change a demonstrated global sign, handedness, rotation, source location,
  map pixel value, science weight, or production profile except for a
  separately approved and preregistered defect effect;
- inherit or cherry-pick unfinished MAP, CAL, convolve/noise, audit, or
  coordination branches without a separate integration decision; or
- contact Unity, push, merge, rebase, launch re-audit, or authorize production.

## Required local repair gates

At one exact repair SHA, without required-data skips or unexpected error-level
records, return at least:

1. TAN center/domain/boundary fixtures covering both sides of `D=0`, exact and
   adjacent values, back hemisphere, non-finite input/output, round trips, and
   valid-but-outside-product separation.
2. AltAz and equatorial wrap fixtures in both directions, exact `pi` tie,
   inverse-TAN normalization, horizon/pole cases, and ordinary controls.
3. Automatic-zero WCS success plus per-field positive, negative, smallest
   nonzero, NaN, infinity, and combined rejection before partial output.
4. Constant, bracketed-MJD, and legacy-span support fixtures at endpoints and
   midpoint, plus mixed/equal/reversed/non-finite/unbracketed/no-extrapolation
   failures and the frozen time-adequacy bound.
5. Stage-by-stage sign/basis/rotation/handedness/composition fixtures and
   approximately one-square-degree center/edge/corner source recovery.
6. Fixtures for each detector-binding mode the candidate actually supports;
   require the selected verified-row or explicit-key mode, and test both only
   if both are implemented. Cover x/y count and finiteness, key/UID uniqueness,
   reorder or permutation as applicable, duplicate, missing, conflicting,
   network subset, measured-versus-design identity, and lifecycle reset.
7. Offline small-angle versus exact-spherical results just inside, exactly at,
   and just outside the declared envelope, plus explicit proof that ordinary
   production executes no new exact spherical work.
8. Map-center inverse-TAN covariance fixtures at representative declinations
   and wrap, with cross term, unit, symmetry, availability, no fabricated
   total, and no off-center/dense allocation.
9. Double WCS/catalog writer-reader round trips and the full-TOD quantization
   bound, including scale sign, CRPIX indexing, handedness, frame, and epoch.
10. ALIGN field-registry admission and exact per-field unit/topology/frame/
    validity attributes; false generic radians must be absent.
11. Factorized validity and every named AST admission/product boundary,
    including projection-invalid, product-unsupported, upstream-ineligible,
    detector-invalid, and required-product failure; no matrix product or
    downstream estimator change is introduced.
12. Full/mini TOD schema and reader round trips proving present versus
    unavailable coordinate products and no mini-TOD size expansion.
13. Atomic four-stage provenance and tamper/write-failure tests for every
    identity, product link, version, count, cardinality, and digest.
14. Supported AltAz and RA/Dec real/simulation parity plus configuration-time
    rejection of Galactic and other unsupported frames.
15. Sequential and every compiled alternate-path equivalence for coordinates,
    validity, counts, products, and required-failure state.
16. Focused CTest, baseline/product-contract/provenance validators, full config
    preflight, relevant sanitizers, and zero unexpected serious records.
17. Runtime, allocation, full/mini storage, and unchanged ordinary map/source/
    centroid/PSF compatibility evidence against preregistered gates.

The standard build and configuration gates include:

```text
cmake --build build --target citlali_cli -j 8
ctest --test-dir build
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
```

Interpret the audit's frozen `A01`--`A23` suite through the later owner
decisions: derivative and Monte Carlo checks apply only to the approved
map-center response and available map-center covariance; unavailable support
covariance is not manufactured; exact spherical geometry is an offline oracle,
not a production requirement; and no fixed positive `D_min`, universal UID,
dense provenance, or wholesale full-precision timestream follows from the
independent-core proposal when the owner decision supersedes it.

## External evidence and fresh re-audit

After all local gates pass, prepare a new human-run
`SCI-AST-001-UNITY-002` request against the exact repair SHA. It must use the
SSH alias `unity_toltec`, bind build/dependency/raw/config/product identities,
exercise Point, OOF, Science, and Beammap compatibility plus relevant invalid
and product cases, and return exact artifacts and timings. Codex does not
contact Unity. The coordinator reviews the request before the user runs it.
The operational cohort must include Point observation 152389, OOF observations
152385--152387, Science observations 152390--152392, and Beammap observation
148670 unless the coordinator records an exact replacement with equivalent
coverage.

A fresh re-auditor in a fresh worktree and suggested branch
`codex/reaudit-sci-ast-001` must assess the exact repair SHA and complete
returned bundle. It must verify `F001` through `F015`, all owner decisions,
the accepted ALIGN interface, downstream handoffs, numerical/product/runtime
compatibility, and the continued absence of prohibited scope. No finding
closes and production remains `existing_use_only` merely because the local
tests pass or outputs resemble historical maps.

## Repair return requirements

Return one coherent commit or a clearly ordered minimal series on
`codex/repair-sci-ast-001`, with a clean worktree and:

- exact selected base and consumed ALIGN interface identities;
- phase-zero bounds, fixtures, preregistered tolerances, and owner stops;
- changed-file and source/equation/decision trace;
- finding-by-finding disposition for `F001` through `F015`;
- all local commands, results, skips, timings, sizes, and artifact digests;
- explicit proof of the owner's no-overkill constraints;
- disposition of all incoming and outgoing handoffs and integration conflicts;
- the proposed human-run exact-SHA Unity request, held for coordinator review;
  and
- an explicit statement that implementation remains nonconformant and
  production remains `existing_use_only` until dependency acceptance,
  integration, returned external evidence, and fresh re-audit succeed.
