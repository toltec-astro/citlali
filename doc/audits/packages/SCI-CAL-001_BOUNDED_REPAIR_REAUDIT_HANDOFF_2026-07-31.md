# SCI-CAL-001 bounded repair and re-audit handoff — 2026-07-31

## Authority and disposition

The project owner approved `CAL-D001`--`CAL-D005` and subsequently supplied
the exact low-opacity operator and q-model continuity gate. This handoff
authorizes one bounded repair lane. It does not approve the assessed
implementation, authorize production, close a finding, request or perform
Unity work, or launch the required fresh re-audit.

- Governing implementation assessed by the audit and selected repair base:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Application authority ref at dispatch: `codex/refactor-mainline`.
- Audit branch and final audit commit: `codex/audit-sci-cal-001` at
  `27b0916e725696597c3ba84fb6a82bf6cf0ea356`.
- Frozen independent core: `SCI-CAL-001_INDEPENDENT_CORE.tex`, SHA-256
  `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`.
- Owner decision: `SCI-CAL-001_COORDINATOR_DECISION_2026-07-31.md` at
  `e8bd929008140e2ea8b44bfdc80b0a531b488765`.
- Opacity amendment:
  `SCI-CAL-001_OPACITY_DECISION_AMENDMENT_2026-07-31.md` at
  `4dabc750de69d53c852317ab77c735febd00a1b5`.
- Required repair branch: `codex/repair-sci-cal-001`.
- Required worktree: a fresh Codex app worktree from the exact repair-base
  SHA, never the audit or coordination branch.

The selected base is the exact source assessed by the audit and still named
by `codex/refactor-mainline` at dispatch. The active MAP repair is a separate
branch from the same base. CAL may proceed independently, but later
integration must review shared documentation, test registration, product
contracts, and provenance files rather than assuming conflict-free landing.

Contract remains `approved`; implementation remains `nonconformant`;
validation remains `in_progress`; production remains `fail_closed`; verdict
remains `amend`; and re-audit remains `required` until the complete closure
sequence succeeds.

## Mandatory phase 0 — q-model continuity preflight

Before any application-code edit, evaluate the governing repair-base source
at every atmospheric q-model selection boundary under
`CAL-D001-OPACITY-001`.

The preflight must:

1. derive the exact full-precision selection thresholds from the governing
   source rather than use rounded audit prose;
2. identify the model selected on each side of every boundary;
3. evaluate every TolTEC band across the declared valid airmass domain,
   including its endpoints and representative interior values;
4. report left and right transmission, line-of-sight optical depth, absolute
   jump, relative jump, source coefficients, evaluation precision, and a
   documented floating-point roundoff bound; and
5. preserve a deterministic machine-readable table plus a concise human
   report with their SHA-256 digests.

The new low-opacity interpolation explicitly repairs zero through q25 and is
constructed to equal the q25 anchor exactly. If any boundary strictly above
q25 is analytically unequal or has a numerical jump exceeding the documented
roundoff bound, stop before changing application code. Commit only the
preflight artifacts and return to the coordinator for an owner scope
decision. Do not broaden the repair to modify any above-q25 model.

Only when every above-q25 boundary passes may the task continue to phase 1.

## Phase 1 — contract fixtures before implementation

Add focused failing fixtures that express the approved repair contract before
changing the implementation. At minimum cover:

- exact low-opacity endpoints and geometric interior interpolation for all
  arrays and representative valid airmasses;
- full sample-airmass application with top-of-atmosphere `X_ref = 0`;
- continuity, monotonicity, zero-opacity identity, model-boundary behavior,
  and finite positive transmission/correction;
- rejection of negative/non-finite opacity, non-finite or out-of-domain
  elevation/airmass, invalid logarithm/transmission, missing bracket/support,
  and unsupported target-unit state;
- UID-permutation invariance and fail-closed missing, duplicate, conflicting,
  or unproven raw-column/APT joins;
- exact factor decomposition and target-unit restrictions;
- conditional variance/weight propagation under a multiplicative factor;
- named calibration-nuisance availability and correlation scope without
  mislabeling conditional weight as total precision; and
- realized-provenance round trips at full precision.

Fixtures must distinguish mathematical assertions from current-source
regression and must name their finding/decision IDs.

## Phase 2 — bounded implementation

The following is the maximum authorized scientific repair surface.

### F001--F003: extinction and fail-closed validity

- Replace the finite-positive q0 plateau with the exact geometric
  transmission interpolation in `CAL-D001-OPACITY-001`.
- Apply the correction with the full sample airmass and top-of-atmosphere
  pivot. Do not convert a line-of-sight result back to zenith opacity and then
  apply it without sample airmass.
- Preserve exact zero and q25 endpoint equality and the unmodified above-q25
  q-models, subject to the phase-0 stop rule.
- Resolve requested, effective, observation-resolved, and realized opacity,
  model, airmass, and factor state without processor-to-request backflow.
- Validate all domains before mutating TOD or publishing required products;
  required invalid state propagates as failure rather than NaN, unity, zero,
  or a partial calibrated result.

### F004: detector/APT identity

- Bind every raw TOD column to the selected APT row by an explicit stable
  detector identity and the approved common/design UID chain, never by row
  position.
- Retain and validate observation-local detector identity, array, network,
  selected APT digest, parent artifact digests, matching lineage, and
  flux-calibration lineage where supplied by the approved boundary.
- Reject missing, duplicate, conflicting, non-finite, or unproven mappings
  before calibration mutates data. Reordering valid APT rows must leave
  detector-resolved results invariant.
- Do not implement or redesign Beammap, `toltec_beammap`, TolAPT, TolProj, or
  TolTECA in this branch. Citlali consumes and verifies their declared
  artifact identities; absent required upstream identity fails closed.

### F005, F006, and F008: factors, units, response, and uncertainty

- Keep `flxscale`, `responsivity`, Beammap `sens`, extinction, target-unit
  transfer, and any compatibility `fcf` as separately named factors with the
  exact meanings approved in `CAL-D002`.
- Initially admit only top-of-atmosphere point-source-peak `mJy/beam`.
  Unsupported units and extended/integrated interpretations fail closed.
- Preserve the originating calibration beam/template separately from the
  realized map/filter response; do not claim response preservation without a
  named normalization contract.
- Propagate conditional variance as `a^2 v` and conditional inverse variance
  as `w / a^2` over valid support.
- Represent calibration systematics through named nuisance values,
  uncertainties, provenance, validity, and correlation scopes as approved in
  `CAL-D004`. Missing nuisance uncertainty is unavailable, never zero.
- Do not introduce a dense sample covariance by default, redefine mature RTC
  or PTC estimators, or promote conditional weights to total uncertainty or
  significance.

### F007: realized provenance and products

Persist enough lossless realized state for a reader to reconstruct or verify:

- exact detector/APT identity and artifact lineage;
- `tau225` source, support/validity, selected model, exact thresholds and
  anchor, sample airmass domain, pivot, per-sample factor range, and
  low-opacity operator/version;
- each applied factor, its units, reciprocal convention, exclusions, and
  composition into the total multiplier;
- target-unit and calibration-beam/template identity;
- conditional statistical uncertainty semantics and each calibration
  nuisance/correlation scope; and
- validity/failure disposition and exact source/config/input identities.

Prefer typed lower-level contracts and one-way legacy adapters. Do not add
new cross-cutting public state to `Engine`, bidirectional typed/legacy
synchronization, or a second provenance authority.

### F009: local evidence

Run focused tests for all touched behavior, CTest for affected targets, the
baseline/product-contract tests implicated by changed products, and the full
config preflight. A successful run has zero unexpected error-level messages.
Record exact commands, source SHA, configurations, fixtures, results, skips,
and artifact digests. A required fixture skipped for missing data is not a
pass.

Do not contact Unity. Once every applicable local gate passes, prepare—but do
not execute—the exact-repair-SHA `SCI-CAL-001-UNITY-001` human request from the
audit. The coordinator will review it before the user runs anything.

## Open ALIGN dependency and parallel-work rule

`SCI-CAL-001-F010` remains open while `SCI-ALIGN-001` is active. The repair may
consume only the already approved abstract input contract: ordered detector
and telescope identity, common time axis, timestamps, aligned elevation,
sample duration where applicable, gap/interpolation state, and exact
original/synthesized eligibility.

Do not inspect the active ALIGN audit worktree, anticipate its findings, or
invent an ALIGN implementation contract. CAL fixtures may use explicit
synthetic aligned inputs. Any conclusion about real per-sample identity,
elevation, duration, interpolation, or eligibility remains conditioned.

The CAL repair may complete local work in parallel, but Unity evidence and
fresh CAL re-audit must wait until the coordinator integrates and disposes any
applicable ALIGN result or handoff. If ALIGN later changes the approved CAL
input boundary, amend the repair and evidence plan explicitly; do not absorb
the change silently.

## Exclusions and stop rules

Do not:

- change scientific algorithms owned by ALIGN, AST, RTC, PTC, MAP, FLT,
  MODE, BEAM, NOI, SRC, JINC, or fruit-loop packages;
- broaden support beyond the initial approved `mJy/beam` point-source path;
- alter q-models above q25 without a successor owner decision;
- repair in the audit or coordination worktree;
- merge or cherry-pick audit/coordination branches into the repair branch;
- contact Unity, use the network, install software, push, launch an audit or
  re-audit, or authorize production; or
- combine unrelated cleanup, performance work, build integration, or a
  numerical redesign with this repair.

Stop for coordinator/owner direction if the phase-0 gate fails, the required
identity cannot be proven from the application boundary, the bounded nuisance
representation requires a new cross-package product authority, an approved
unit/beam meaning conflicts with source reality, or the necessary change
would cross an exclusion above.

## Repair handoff requirements

Return one coherent repair commit or a clearly ordered minimal series on
`codex/repair-sci-cal-001`, with a clean worktree and:

- phase-0 continuity artifact identities and disposition;
- exact changed-file inventory and source/equation/decision trace;
- finding-by-finding disposition for F001--F010;
- local test/gate commands and results;
- unresolved assumptions, especially the active ALIGN dependency;
- product/provenance compatibility and migration notes;
- exact proposed human-run Unity request, held for coordinator review; and
- an explicit statement that implementation remains nonconformant and
  production remains fail-closed until integration, returned exact-SHA
  evidence, and fresh re-audit succeed.

