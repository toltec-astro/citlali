# SCI-ALIGN-001 bounded repair and re-audit handoff — 2026-08-01

## Authority and disposition

The project owner approved `ALIGN-OD1`--`ALIGN-OD8` and the cross-cutting
`ALIGN-C001` compactness constraint. This handoff authorizes one bounded
repair lane. It does not approve the assessed implementation, authorize
production or polarization science, close a finding, request or perform
Unity work, or launch the required fresh re-audit.

- Governing implementation assessed by the audit and selected repair base:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Application authority ref at the decision: `codex/refactor-mainline`.
- Audit branch and final audit commit: `codex/audit-sci-align-001` at
  `aeeac7f36e1ab0ab17bfbf3f603364faff02d715`.
- Final audit artifact SHA-256:
  `6aaed0e6e16e4c37cd24d15b98346f84024ffd7920bd0524e7a170dbc728a393`.
- Frozen independent core: `SCI-ALIGN-001_INDEPENDENT_CORE.tex`, SHA-256
  `4ee7b7e9cbe883ea626afe2e3d22756b20f556a2e06115d4a2832f2e78469785`.
- Owner decision: `SCI-ALIGN-001_COORDINATOR_DECISION_2026-08-01.md` at
  `4f905f4f39461c8f9a86b0bf589880362d0a49f7`.
- Required repair branch: `codex/repair-sci-align-001`.
- Required worktree: a fresh Codex app worktree from the exact repair-base
  SHA, never the audit or coordination branch.

The selected base is the exact source assessed by the audit and still named
by `codex/refactor-mainline`. The unfinished MAP, CAL, and convolve/noise
lanes do not land first and must not be inherited by the ALIGN repair. They
remain independent branches from the same or separately governed bases.
Later integration must review shared scientific conventions, product
contracts, provenance, test registration, and validation files rather than
assuming conflict-free landing.

Contract remains `approved`; implementation remains `nonconformant`;
validation remains `in_progress`; production remains `existing_use_only`;
verdict remains `amend`; and re-audit remains `required` until the complete
closure sequence succeeds.

## Mandatory phase 0 — raw timing authority and field registry

Before any application-code edit, inspect the exact repair-base source and
available representative raw metadata, then produce a deterministic timing
authority report and complete generated field registry. Commit those
artifacts alone and return to the coordinator and project owner. Application
implementation must not begin until that review is recorded.

The phase-0 report must:

1. inventory every detector, telescope, and optional HWPR time/header field
   used or available at the application boundary, including exact raw name,
   interface identity, units, epoch, counter width, rollover, cadence,
   acquisition bounds, duration, missing/non-finite state, and source
   authority;
2. trace every configured interface offset through requested, effective,
   observation-resolved, and realized state, with its sign, unit, reference
   interface, and exactly-once application stage;
3. derive detector reference cadence, phase, slot tolerance, and observed
   jitter bounds from authoritative headers plus measured representative
   data rather than silently infer them from row count or current arithmetic;
4. generate the full telescope/HWPR variable registry with stable field ID,
   scientific identity, units, frame, topology, validity policy, maximum
   support span, permitted operator, and output identity for every field;
5. classify each variable as continuous scalar, circular angle, declared
   half-open step/state, or exact-only; counters, flags, timestamps, and
   categorical state must never default to ordinary linear interpolation;
6. compare the proposed shared slot operator and offset convention against
   the current realized row/slot mapping on representative ordinary data,
   reporting every changed row, residual distribution, pointing/time
   residual, source-crossing time, centroid, and PSF-width impact available;
   and
7. preserve machine-readable inventories and comparison tables plus a
   concise human report, all with exact source/input identities and SHA-256
   digests.

Use local or owner-supplied raw files only; do not contact Unity. If an epoch,
sign, reference, rollover, cadence, field meaning, or authoritative source
cannot be proven; if registries conflict; or if ordinary accepted samples
would move materially relative to current telescope/detector timing
performance, stop and return the evidence for an owner decision. The owner
must review the complete generated registry even when no conflict is found.

## Phase 1 — contract fixtures before implementation

After the phase-0 review is approved, add focused failing fixtures expressing
the successor contract before changing its implementation. Implement the
frozen `T01`--`T18` corpus without required skips, including:

- positive-add fractional offsets, exact coincidences, endpoints, constant
  and affine signals, sinusoidal response, circular wrap and ambiguity, and
  declared step/exact-only fields (`T01`--`T06`);
- bounded internal gaps, edge/over-limit gaps, invalid endpoints, jitter,
  exact half-slot cases, duplicates, collisions, and nonmonotonic timestamps
  (`T07`--`T09`);
- half-open `N=7, M=3` windows, final partial support, exact boundaries,
  short/empty/subset identities, and overlap rejection (`T10`--`T11`);
- conditional shared-endpoint covariance, timing/model bounds, and distinct
  acquired exposure for original, synthesized, unavailable, and sliced
  support (`T12`--`T14`), without making unavailable full-covariance or
  timing-response claims;
- missing telescope support, absent optional HWPR, two-observation lifecycle,
  sequential/compiled-path equivalence, and required-product failure
  propagation (`T15`--`T18`); and
- source-crossing, centroid, pointing-residual, and PSF-width compatibility
  checks showing no material degradation of the already accepted timing
  performance.

Each fixture must name its finding and decision IDs and distinguish exact
mathematical assertions, compatibility comparisons, and source regressions.

## Phase 2 — bounded implementation

The following is the maximum authorized scientific repair surface.

### F002 and F007: reference grid, clocks, offsets, and admission

- Preserve a detector/KIDs-led common grid and detector support. Represent
  the reference clock, cadence, phase, tolerance, native time residual, and
  stable slot identity explicitly.
- Apply one observation-constant, positive-add seconds offset to each native
  interface coordinate exactly once before all alignment logic. Do not round
  a fractional offset to an integer sample. Omitted authoring may resolve to
  a typed zero only as `schema_default_zero`; a nonzero value requires proved
  authority. A time-varying offset requires a separately versioned model and
  is outside this repair.
- Use one shared round-half-up slot operator for masks and value placement,
  with residual magnitude strictly below half a sample. Reject duplicate,
  nonmonotonic, colliding, incomparable, and out-of-tolerance rows before
  conditioning.
- Do not wholesale retime accepted data. Derive the frozen cadence, phase,
  and tolerance from phase 0 and preserve current source-crossing and PSF
  performance within the reviewed compatibility bounds.

### F001, F004, and F006: typed field alignment, HWPR, and lifecycle

- Implement the owner-reviewed field registry as the single topology and
  validity authority. Continuous scalar, shortest-arc circular, declared
  half-open step/state, and exact-only fields use only their approved bounded
  operators. Do not extrapolate beyond valid support.
- Preserve native time coordinates and field-specific units, frames, and
  state. Do not interpolate flags, counters, timestamps, or categorical
  values as scalars. Ambiguous circular or state support becomes unavailable
  with an explicit reason.
- Initialize, reset, and observation-own all detector/telescope/HWPR alignment
  state. Apply an authoritative HWPR offset once. Absent optional HWPR is
  explicit and nonfatal for intensity; required invalid HWPR fails. This
  repair must not implement demodulation or authorize polarization science.
- Make real, simulation, direct, and gap-population paths construct the same
  complete state without cross-observation leakage or process-lifetime
  mutable ownership.

### F003: typed gap policy and continuity-only synthesis

- Detect gaps observation-wide before slicing and preserve their native
  interface, signal domain, network/detector scope, start/end, sample count,
  and duration independently of `xs`, `rs`, `is`, or `qs` selection.
- For each realized half-open time chunk or physical scan, evaluate both the
  longest run and cumulative missing support by count and duration. Exactly
  25 percent is permitted; strictly greater than 25 percent flags the entire
  affected network for that chunk, not unrelated networks. Preserve acquired
  rows and their origin.
- A bounded internal detector-network gap at or below the approved limit may
  receive only the approved signal-domain continuity surrogate, with exact
  source IDs/weights, gap flags for every detector in that network, and a
  separate downstream filter guard. Never extrapolate at an edge or partly
  fill an over-limit run.
- Treat pointing, HWPR, detector, and state gaps according to their distinct
  registry contracts. Ordinary bounded telescope resampling is not a native
  gap; missing required pointing invalidates mapped science; ambiguous Hold
  state may invalidate a scan even when short.
- Synthesized detector values provide continuity only: zero direct hits,
  acquired exposure, independent statistical weight, degrees of freedom,
  and significance. Their guard value remains the original acquired sample,
  and any broader consumer use requires its own approved contract.
- Retain the approximately one-second historical UDP-gap scale as an
  atypical-gap warning context, not as the universal scientific admission
  rule.

### F005: scan and processing-window identity

- Represent physical scans, processing chunks, science windows, context
  windows, and output windows as distinct half-open objects.
- Preserve raster legs, including the first post-Hold sample, and use the
  full continuous observation unless chunking is requested. Duration
  partitioning uses the approved round-half-up rule; count partitioning
  distributes all samples; final partial support is retained.
- Assign stable zero-based IDs and use an explicit one-based adapter only at
  a declared compatibility boundary. Short, empty, rejected, and unusable
  identities remain recorded without padding, deletion, or renumbering; no
  universal two-second minimum is authorized.
- Keep the science window immutable, store context separately, and reject
  overlapping or inconsistent cardinality rather than silently trim or
  clamp it.

### F008--F011: compact realized state, eligibility, response, and exposure

- Persist a compact generative representation of the common grid, offsets,
  field-registry version, scan windows, exception runs, aggregate support,
  and availability manifest. Normal mapping is implicit; standard operation
  must not emit dense per-sample/per-detector provenance.
- Keep origin, validity, interpolation method/reason, source identity and
  weights, continuity permission, science eligibility, and filter guard as
  distinct typed facts. Persist exact compact exception intervals or RLE;
  expanded endpoint maps and detailed IDs are `as_requested`.
- Record each field's true units, frame, topology, native/output identity,
  stable scan ID, requested/effective/resolved/realized plan, and required
  output-write disposition. Required product failure propagates to the CLI.
- Store the realized sparse/generative mapping and availability of response,
  covariance, timing/model uncertainty, and selection uncertainty. Provide
  expanded mappings, transfer/covariance fixtures, endpoint identities, or
  digests `as_requested`. Do not invent independence, zero missing terms, or
  a full covariance product.
- Propagate covariance only when a valid input covariance and requesting
  consumer contract exist. ALIGN owns only its temporal mapping response;
  downstream filters own their additional response.
- Separate nominal support span, retained scan duration, valid original
  acquired exposure, synthesized support, and unavailable support. Never
  coadd endpoint-center spans as acquired exposure.
- Measure runtime, I/O, and storage costs. If even compact exception catalogs
  are burdensome, their identification may become `as_requested` after
  coordinator review without changing the approved scientific semantics.

### F012: local evidence

Run focused tests for every touched behavior; sanitizers for lifecycle and
index safety; CTest for affected targets; baseline and product-contract tests
implicated by changed outputs; sequential and every compiled alternate-path
comparison; and the full config preflight. A successful run has zero
unexpected error-level messages. Record exact commands, source SHA,
configurations, fixtures, results, skips, timings, storage effects, and
artifact digests. A required fixture skipped for missing data is not a pass.

Do not contact Unity. Once all local gates pass, prepare—but do not execute—an
updated `SCI-ALIGN-001-UNITY-001` human request against the exact repair SHA.
It must use the SSH alias `unity_toltec`, include the audit's real and
synthetic cases, record the exact build/dependency/raw/config identities, and
return timing, mapping, scan, flag, exposure, product, and compiled-path
evidence. The coordinator will review the request before the user runs it.

## F014 — downstream handoff and parallel-work rule

The four integrated ALIGN handoffs remain constraints, not authorization to
repair their target estimators:

- `SCI-CAL-001-XAUD-001`;
- `SCI-AST-001-XAUD-001`;
- `SCI-RTC-001-XAUD-002`; and
- `SCI-VAL-001-XAUD-004`.

The ALIGN repair must add boundary/interface fixtures proving that compact
origin, support, timing, gap, scan, and availability state can be consumed
without treating synthesized samples as acquired or independent. It must not
modify CAL extinction/calibration, AST coordinate estimation, mature RTC/PTC
filters, or VAL/MAP eligibility algorithms. Target-package owners and their
fresh audits retain authority over estimator behavior.

MAP and CAL repair work may continue independently from their selected bases.
Before any lane is integrated or sent to exact-SHA external evidence, the
coordinator must compare the candidate interfaces and explicitly disposition
all applicable handoffs, overlapping product contracts, provenance schemas,
test registration, and restrictions. Do not silently absorb another lane's
unreviewed implementation.

## Exclusions and stop rules

Do not:

- change scientific algorithms owned by CAL, AST, RTC, PTC, MAP, VAL, FLT,
  MODE, BEAM, NOI, SRC, JINC, or fruit-loop packages;
- broaden mature RTC, PTC, JINC, or Wiener-filter behavior while repairing
  their ALIGN input contract;
- implement polarimetry, full covariance, dense standard provenance,
  time-varying clock-drift correction, or a new significance estimator;
- repair in the audit or coordination worktree, or inherit/cherry-pick the
  active MAP, CAL, convolve/noise, audit, or coordination branches;
- contact Unity, use the network, install software, push, launch an audit or
  re-audit, merge, rebase, or authorize production; or
- combine unrelated cleanup, performance work, build integration, or an
  estimator redesign with this repair.

Stop for coordinator/owner direction if phase 0 lacks authority, the complete
generated registry has not been reviewed, ordinary accepted samples move
materially, current source-crossing/centroid/PSF timing performance degrades,
the 25-percent or typed-gap rule cannot be represented without a downstream
algorithm change, compact products exceed the reviewed burden, a required
cross-package authority is missing, or the necessary change would cross an
exclusion above.

## Repair handoff requirements

Return one coherent repair commit or a clearly ordered minimal series on
`codex/repair-sci-align-001`, with a clean worktree and:

- phase-0 raw-field, timing-authority, generated-registry, jitter, and
  compatibility artifacts plus the recorded owner disposition;
- exact changed-file inventory and source/equation/decision trace;
- finding-by-finding disposition for `F001`--`F014`;
- `T01`--`T18`, compatibility, performance, and all local gate commands and
  results;
- compactness measurements and any approved `as_requested` downgrade;
- target-interface tests and the disposition state of all four handoffs;
- unresolved assumptions and integration conflicts with MAP, CAL, and
  convolve/noise candidates;
- exact proposed human-run Unity request, held for coordinator review; and
- an explicit statement that implementation remains nonconformant and
  production remains `existing_use_only` until integration, returned
  exact-repair-SHA evidence, and fresh re-audit succeed.
