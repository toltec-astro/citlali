# Single-observation SCIENCE with iterative feedback

Owner: Grant Wilson. Execution owner: Grant for every build, test, probe,
benchmark, reduction and validation runner. Codex prepares and statically reviews
the candidate and owner commands. All runtime gates are pending. No push, Unity
access, canonical integration, runtime replacement or cleanup is authorized.

Base: `8738316bb6091706849f493b46d1f657b03f79cf`, clean local mainline.
One spine increment: `codex/science-single-20261106`, isolated checkout under
the external `citlali-science-single-20261106/source` packet. Other worktrees,
POINT policy design, original inputs and prior evidence remain untouched.
Tier 2: cross-stage frame, application route and publication changes.

## Scope and recovered authority

Grant requested the single-observation SCIENCE route for 152389, carrying over
POINT performance improvements, and explicitly selected iterative feedback in
this increment. The [planning scope](../../doc/SCIENCE_SCALING_SCOPE_2026-11-06.md)
supplies the one-node Unity target and later scaling envelope. This increment
uses the existing small POINT acquisition in celestial SCIENCE processing;
it does not qualify the extended-field or multi-observation policies.

Read AGENTS and engineering, successor and review/conformance governance;
accepted incorporation and digests are in the integration ledger. Current state,
architecture, scientific conventions and application baselines distinguish
accepted implementation, scientific contracts and historical validation. The
WP-7.1 authority router and ADRs remain unchanged. Spack is the build authority;
Conan is historical. No new scientific-contract package or derivation is begun.

Recover and reuse:
- AST's existing `bounded-v2-radec-gnomonic-detector-offset-v1` at exact RTC
  schedule times, with fixed FK5/J2000 header source center and existing offset
  signs. SCIENCE FITS uses degree-valued RA---TAN/DEC--TAN WCS.
- Accepted calibrated RTC/CAL domain, source-offset health/noise exclusions,
  empirical inverse-noise PTC/MAP coefficients, and the unchanged compact-source
  POINT-EXPECTED/BASE selector. The explicit SCIENCE compact-source profile
  records this reuse; it does not claim an extended-source protection policy.
- Original CAL is immutable across all cycles. AST source-exclusion coordinates
  remain AltAz; MAP and feedback operands use celestial tangent coordinates.
  The initial-map covariance/BASE selector stays frozen, later models replace
  predecessors, and exact sampled operands are restored by the existing owner.
- In-memory producer/CAL/APT/health handoffs, deterministic segment workers,
  reduction-owned array pool, requested/effective resource budgets, detective
  exports and final original-input verification from the accepted POINT repairs.

## Owned boundaries and exclusions

The configured application owns requested/effective mode and frame, native stage
sequencing and final publication. AST owns coordinates; CAL/health/PTC/FRUIT/MAP
retain their current scientific owners. FITS and companion records must bind
the same full-precision grid, units, observation, array, cycle and frozen health.
The shared entry remains strict: POINT cannot silently run SCIENCE or vice versa.
The SCIENCE product carries no POINT assessment, pointing table or telescope
correction. A missing required map or I/O failure prevents completed publication.

No coadds, reprojection, new calibration, thresholds, filters, cleaning model,
source-selection model, dependency upgrade, general extended-field admission,
multi-node execution or production activation. Final scientific use remains
pending owner-run evidence. Shared POINT code must retain existing behavior.

## Gates and reassessment

Focused owner-run gates cover strict mode/scope/configuration, celestial AST and
WCS metadata, feedback frame/center/units and generation mismatch, replacement
lifecycle, publisher failures, packaging, worker budgets and POINT regressions.
Broader gates: build/version identity, enabled CTest, baseline-tool suite and
full required configuration preflight. Use retained Mac Spack software checks
first; representative Unity performance/scale evidence remains separate.

The later owner-run 152389 packet binds the exact candidate, executable,
environment, original inputs, requested feedback budget and fresh outputs.
Review every all-array cycle, health and detector reports, celestial positions,
feedback outcome, required products, time/memory and transport I/O. AltAz and
RA/Dec maps do not imply pixelwise equality. No numerical threshold is invented.
Prepare the reduction command after software-gate review.

Before handoff, commit a coherent candidate and request fresh-context read-only
independent review of its exact SHA. Grant owns all execution, including reviews'
reproduction requests. Stop and reassess missing scientific authority, frame or
calibration ambiguity, broad field/coadd requests, ownership violations or scope
expansion. Runtime results and scientific/behavioral disposition stay pending;
architecture and repository dispositions follow static exact-source review.

## Prepared implementation and test map

- Explicit SCIENCE entry/compact overlay, strict acquisition/mode/frame/coadd
  admission and requested/effective semantics; no automatic POINT conversion.
- Shared configured observation/session and weighted-cycle implementation.
  AST evaluates both health-mask AltAz and map/feedback celestial coordinates
  once during prepared CAL admission. Later cycles reuse the prepared domain.
- Celestial FITS/plan/bundle serialization and feedback provenance validate
  center, frame, axis units, array, health generation and previous cycle.
- SCIENCE publication is hidden until all maps and final input checks succeed.
  Final publication has no pointing-table, assessment or final POINT fit.
- New Python SCIENCE tests exercise synthetic known tangent offsets, frozen
  selector/replacement, all-array publication, malformed output, in-memory
  iteration and final publication ordering. Native tests cover scope, grid,
  budget and feedback provenance. Existing POINT regressions remain gates.
- Runtime packaging includes the new entry/modules/overlay. The installed
  package's historical POINT name stays unchanged; no dependency changes.

Static review and Git whitespace inspection are the only performed checks.
Build, focused Python/native cases, full enabled CTest, baseline tests and
configuration preflight remain pending Grant's execution. Synthetic tests do
not establish celestial accuracy, source recovery, scaling or time/memory
performance; those require the subsequent bound observational packet.
The default overlay requests five feedback applications (M0 through M5),
2 arcsec pixels, 0.15 coverage and ranks 3/4/6. Failure to select a required
source model stops feedback truthfully at the last complete all-array maps.
This is a fixed budget, not a convergence claim.

Disposition before runtime evidence: scientific/behavioral conformance pending;
architecture/ownership prepared for independent exact-source review; repository
scope isolated from unchanged mainline. No integration, deployment or push.

## Independent-review repair within scope

Read-only review of initial candidate 38923418d33df6c4f9aa699f944ff98cc4fc4a38
identified degree-valued synthetic FITS headers using Astropy's ordinary
formatting before exact bundle comparison. Both fixture paths now use the
existing full-precision header serializer; exact equality checks remain.
The shared FRUIT tangent projector's interface comments, parameter names and
error wording now explicitly cover both caller-bound AltAz and celestial
tangent arcseconds. Projection arithmetic, domain and selection policy are
unchanged. Final exact-source review follows this bounded correction; all
runtime gates remain pending Grant.
