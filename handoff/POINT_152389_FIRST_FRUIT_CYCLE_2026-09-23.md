# POINT 152389 first FRUIT cycle — completed bounded experiment

## Subsequent bounded continuation

The owner subsequently authorized cycles2–10, now completed under
`handoff/POINT_152389_FRUIT_TEN_CYCLES_2026-09-23.md`. The first-cycle products
and the historical single-cycle stop below remain preserved. The continuation
applied saved m1 through m9 and stops with m10 unapplied.

## Completion disposition — 2026-09-23

The historical preflight below is superseded by owner approval and successful
execution. Exactly one nonzero cycle per array is complete; all fresh zero
controls reproduce their parents exactly. Stop reason: authorized single cycle
complete, not convergence. m1 is saved and has not been projected or applied.

Owner-facing [ASSESSMENT.md](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-fruit-one-cycle-20260923/ASSESSMENT.md) includes all five map/model
measurements, figures, total/residual/restored FITS, zero controls, m1, full
support comparisons, PTC convergence/state changes, exact provenance and runtime.

Compact repository evidence is
`validation/point_152389_fruit_first_cycle_2026-09-23/completion.json`.
The original 1,015-file cycle and corrected 23-file diagnostic manifests verify;
their hashes and assessment hash are in that record. Original parent products,
this branch's unrelated main-checkout changes and all original receipts remain
untouched. No integration, push, production activation or Unity access occurred.

Runtime implementation: `dfc10ef5fd8ad880958c57278b6b07d36d841242`, tree
`902c48d507f46b3d0d52d66fcab241289ca024b6`. Diagnostic repair:
`46bbeb182bb5e973c8356b51b2c8b509f7186c1b`, tree
`45934f7c42ea997a0c60a989f4dbb2a301a3df50`. Exact runtime binary and build
cache/source snapshots/logs are preserved in the product evidence directory.
Current repair binary builds successfully and identifies 46bbeb182. Local cached
AppleClang evidence remains supplemental, not a Spack V2 reproduction.

| Array | M0 amplitude | T1 amplitude | Change | m1 admitted pixels |
|---|---:|---:|---:|---:|
| a1100 |816.311|919.123|+12.59%|32 (+4,−0)|
| a1400 |554.151|643.449|+16.11%|44 (+6,−0)|
| a2000 |593.401|761.964|+28.41%|59 (+2,−0)|

Amplitudes are mJy/nominal beam above the fitted constant. All five-product
fits succeed without active constraints; all 40″ fit/noise domains remain
complete. Coverage, normalization and input/output causes remain unchanged.
Inherited network 3/scan 5 nonconvergence remains visible; no new failure or
network rescue. Maximum decomposition error 1.81e-11 mJy/nominal beam.
Shallower negative bowls and increased amplitudes are descriptive processing
changes, not independent proof of recovered flux. The frozen original selection
scale is not post-FRUIT uncertainty; a2000 spatial-model caveat persists.

One diagnostic defect was corrected after the authorized run: saved row-major
basis bytes had been read column-major in the projector-distance comparison.
That comparison never feeds scientific PTC learning or application. Original
receipts remain preserved; diagnostics-r2 is authoritative for the corrected
ranges 0–0.424,0–1.423,0.058–1.410. Report-only regeneration verifies immutable
cycle hashes and identical m1 arrays; no second scientific cycle was run.

Validation: three native focused tests cover numerical source projection/PTC/
rejoin/MAP, signs/missing support/stale-state rejection and pixel association.
Three Python tests cover replacement/removal and saved-basis layout. After the
repair 232 successor tests/111 subtests and 28 selected CTests pass. Before the
repair 207 baseline tests/137 subtests and full required config preflight passed;
the diagnostic-only change does not invalidate those unchanged-scope gates.
Six native execution logs have no unexpected error-level messages. All nine
comparison/profile/footprint figures were inspected. Independent exact-SHA
implementation and repair reviews pass scientific/behavioral,
architecture/ownership and repository/evidence with recorded limitations.
Final completion-record review at `dc4eab09e772f9ab9ae95d0ecd2f6d44f2ebf7c6`
(tree `9b294da44b42fd85feab06cbe64b5f7c438ca3a5`) passes all three categories
with recorded limitations and no remaining findings. The reviewer independently
verified the durable manifests, assessment hash, numerical completion fields,
artifact links, repair binary identity, archived build/test logs and clean tree.
See `validation/point_152389_fruit_first_cycle_2026-09-23/review-closure.md`.

Recommendation: one further separately authorized bounded cycle with m1 under
the same controls, then compare change magnitudes, widths and support. Do not
automatically select the latest/highest-amplitude result as scientifically best.

## Owner continuation and implementation

The owner subsequently answered **“I agree with these choices”**, explicitly
accepting both proposed outside-disk and direct identical-operand conventions.
They are recorded in the owner-binding.json beside the preflight evidence.
The prerequisite questions below are historical and resolved; the authorized
implementation and one-cycle execution completed on the same branch.
No additional approval is required for those choices.

## Historical preflight disposition

**Authorized work is incomplete. No zero-feedback replay or nonzero feedback
cycle has run.** The September 23 owner directive authorizes exactly one cycle
per array at unity gain, then replacement-candidate inference and stop. It
supersedes the admission task's no-FRUIT stop. It explicitly retains scientific
removal/rejoin compatibility as a prerequisite. Two numerical method details
remain unresolved in the recovered authority; specific questions are pending
with the owner. This is not a claim that the requested experiment is forbidden
by the older generic-core freeze, nor a finding that the proposed route is
numerically incompatible.

The complete current directive and reproducible read-only evidence are in
`validation/point_152389_fruit_first_cycle_2026-09-23/`. The verification script
is an archived diagnostic, not the intended FRUIT entry point. It reads the
explicit saved development worktree and original input paths; it writes only
its own output directory. No implemented FRUIT route is claimed by this record.

## Work-order preflight

- Owner/purpose: Grant Wilson; first bounded feedback cycle for POINT 152389,
  separately a1100/a1400/a2000, using the reviewed threshold-3 candidates.
- Risk tier/slot: Tier 2 scientific-module probe continuing the existing lane.
- Exact base: `b91730e753b2ff57ae6cdda4928b8e779567e7de`.
- Branch/worktree: `codex/map-successor-first-forward-001`,
  `/private/tmp/citlali-point-additive-20260923`; initially clean. The unrelated
  main checkout remains untouched; no second implementation branch.
- Governance read: ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE,
  REVIEW_AND_CONFORMANCE. Their exact digests in `source-identities.json` match
  accepted source `06a3ade51c1b3f38887295433d913811bf25cd14`, effective through
  `77507836325eff9f469062d5884481ea37599594` and the integration-ledger record.
- Sequencing: latest owner directive, recorded verbatim as OWNER_DIRECTIVE.txt,
  extends the completed admission task. Existing status/handoff governs inputs.
- Scientific authority: SCI-FRUIT core v0.1/r0.4, reviewed delivery
  `5ce01b62ddfb6bb57120e872a02da0fd8cfae5ed`, canonical inventory
  `3dc912d9f6a58d9f92561ad10b9442485c5b47232e5327389d2fb3b89c296e92`;
  September 18 PTC observed-entry owner binding; current AST/MAP and uniform
  coefficient boundary identities verified by the existing MAP inspector;
  accepted admission inputs and unchanged descriptive POINT measurement.
- Architecture: AST owns exact occurrence coordinates, PTC owns preparation,
  learning/consideration/application and coefficients, VAL owns named uses,
  MAP owns deposition/normalization, FRUIT owns selected/applied models and
  removal/rejoin/transition records. No Engine growth or legacy integration.
- Included: original pre-PTC parent reuse, zero control, one nonzero cycle,
  decomposition, replacement inference, focused tests and independent review.
- Excluded: RTC/CAL changes, network rescue, rank/threshold tuning, feedback in
  r, noise relearning, a second cycle, production integration/activation/push.
- Expected implementation areas: typed FRUIT owner, saved-product successor
  adapter, bounded numerical tests, existing POINT diagnostic tools and records.
- Gates after boundary resolution: zero-control values/support/normalization/
  source fits; bypass arithmetic/missingness; actual projection/remapping;
  residual-bound fresh PTC state and stale-state rejection; replacement mask;
  numerical synthetic chain; focused native tests, applicable CTest/baseline/
  successor checks, full config preflight; fresh-context exact-SHA review.
- Environment: local Python `/Users/gwilson/tolteca/bin/python`; one numerical
  thread. Existing local AppleClang/cached-dependency native build, if used,
  is supplemental and not Spack-backed V2 reproduction. Unity is human-only.
- Stop/review triggers: unresolved scientific operand/support choice; material
  unexplained zero-control mismatch; changed parent identities; scope expansion.
  Successful execution would stop with m1 unapplied, not with convergence.
- Integration/push/activation/cleanup: not authorized.

## Verified inputs and what they establish

The admission ASSESSMENT.md hash is
`b3a779e1ee31ee92079fd293a9ab516b38e6a0e2c4d03089fa5c05cd917f6a47`.
The final admission companion-r2 manifest is
`42e656d6fedefc1a47599333983bc5a7db31acb42e78be0ec3449a80fe577f71`;
the inherited additive companion manifest is
`d650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8`.
All manifest payloads verify. The existing inspector verifies all three MAP
bundles, their exact PTC/CAL handoffs and numerical replay of the already-cleaned
data. Saved source measurements reproduce under the unchanged fitter. This
**does not constitute the required fresh-PTC zero-feedback control**.

All seven frozen SCI-FRUIT core source hashes verify. Exact source/input hashes,
parent measurements and per-network counts are retained in the JSON evidence.

| Array | Candidate valid pixels | Admitted pixels | Intentional zero pixels | Existing MAP contributions outside candidate disk / total |
|---|---:|---:|---:|---:|
| a1100 | 1,257 | 28 | 1,229 | 14,200,303 / 16,389,673 (86.642%) |
| a1400 | 1,257 | 38 | 1,219 | 5,799,172 / 6,719,154 (86.308%) |
| a2000 | 1,257 | 57 | 1,200 | 2,430,694 / 2,781,761 (87.380%) |

These counts use the original MAP operator's containing-pixel association, not
a proposed new reverse projection. They cover existing admitted MAP members,
not all pre-PTC learning entries. Each stored 41×41 candidate also contains 424
unavailable pixels outside the circular domain; positions beyond the square
are not defined by that file. Every pixel inside the 40″ disk is currently valid.

The verified contributing networks are 0–5, 7–9, 11–12. Their 66 saved segments
all bind the observed-entry ALS recipe with configured rank 10 and per-detector
eligible mean removal without restoration. Existing processing scope is per
network/processing segment, not one observation-wide fit. Saved `input.f64`,
eligible masks, detector relations and slot arrays provide exact restart inputs
for the admitted fit columns. Their component hashes verify. New eligibility
still needs the full saved producer facts; old retained MAP membership alone
must not define the new PTC or MAP domain.

Read-only verification cost: 42.654 s wall, 11.262 s CPU, 437,780,480 bytes peak
RSS. This is input checking, not companion execution or reduction runtime.

## Specific unresolved method choices

1. **Outside-domain behavior.** The candidate authority defines zero only on
   valid deliberately excluded pixels. The 40″ disk is a computational boundary,
   explicitly not a physical source boundary. The saved candidate records NaN
   elsewhere. Neither threshold selection nor the absence of boundary contact
   defines the projected operand outside that disk. Treating those NaNs as zero
   or dropping all associated samples would silently choose different methods.
   Proposed bounded choice awaiting owner disposition: no feedback contribution
   outside the analysis disk, explicitly recorded as an application-domain
   choice, retaining original data/eligibility there; unsupported pixels inside
   the disk remain unavailable. No noise extrapolation or assertion of measured
   zero sky follows.

2. **Removal/rejoin mapping and reference.** AST supplies coordinates at assigned
   RTC times; forward MAP supplies one-hot containing-pixel deposition. The
   pre-PTC CAL and PTC output unit is mJy/nominal-beam, but matching units and
   coordinates do not by themselves define sky-model input response or additive
   reference compatibility. The recovered source records explicitly leave the
   complete RTC/CAL astronomical response unavailable. This alone need not block
   an explicitly defined empirical direct route, but it cannot be silently
   replaced by an identity response claim. Proposed bounded choice awaiting
   owner disposition: sample the exact candidate at each AST containing pixel
   on the saved post-RTC/CAL grid; subtract before PTC; rejoin that identical
   operand after PTC, without extra filtering, beam convolution, calibration
   conversion, background restoration or claims of recovered upstream response.
   The source-relative reference and model-supported mean modes must be recorded
   explicitly. If response-aware removal is intended instead, its actual
   transformation must be bound before execution.

These are governed by SCI-FRUIT requirements 004/005/008/009, `core:equations`
and `core:support`: exact compatible operands and separately defined supports;
unknown contributions are not zero; narrower residual-only composition requires
explicit permission. The current owner's directive preserves these conditions.
The scientific/engineering code is not being used to select a missing method.

## Compact implementation plan after those choices are resolved

1. Bind m0, original y, AST grid and frozen selection scale by verified hashes;
   keep candidate, owner-selected model and applied operand separate. Record
   precise direct/projection response convention and support classifications.
2. Implement typed immutable FRUIT input/operand/residual state. Reuse PTC
   numerical preparation/learning/application with unchanged saved masks,
   grouping and solver controls. Identity binding rejects stale learned state.
   Recompute input-dependent eligibility normally; retain paired r as inert.
3. Reuse AST evaluation for the complete required pre-PTC domain, not merely
   old MAP members. Rejoin the exact operand on contracted PTC output support;
   resolve one new uniform-coefficient MAP operator for total and both components.
4. Run bounded fixtures and fresh zero replay first. On a passing control,
   execute one nonzero cycle per array through the intended successor entry
   point. Save R1, A1, T1 and common-operator accounting, learned-state/support
   diagnostics and separately timed stages; no residual-image shortcut.
5. Apply the unchanged 40″ descriptive fit to unmasked T1. Infer m1 with its own
   centroid/background, threshold 3, original noise field/validity intersected
   with new support, complete replacement semantics. Save and do not apply m1.
6. Inspect comparable-range figures/profiles and report M0/zero/T1/m0/m1
   measurements, template/aperture definitions, bowls and support changes.
   Complete changed-scope gates and independent exact-SHA review; recommend
   but do not execute a next action.

## Reassessment and pending completion

Trigger: missing explicit projection/removal/rejoin and unequal-support method
choices. Scope remains bounded; existing inputs are verified and unchanged.
Disposition: request the two specific owner decisions; preserve the verified
input evidence and implementation plan. Dependent numerical work has not begun.
No astronomical result, passed cycle gate, or completed milestone is claimed.

This evidence-only checkpoint changes no application, test, configuration or
scientific authority. Native CTest/baseline/config gates are not triggered by
these records; they remain required as applicable for the future implementation.
Fresh-context read-only review of checkpoint
`bd4563f4ac9506a861bea8377dd2ba9c5b0620e9`, tree
`c3a8b22544d56dacfee29a8b3b0db68f9d9388aa`, passes with the recorded limitations:
scientific/behavioral pass for the evidence-only scope; architecture/ownership
pass; repository/evidence pass. No repair-required findings. The reviewer
independently confirmed that both questions are substantive under the current
directive, not a blanket historical prohibition. All five archive checksums,
fourteen source identities and seven core-module hashes verified. Read-only
reproduction matched verification.json exactly excluding runtime (39.70 s wall,
10.09 s CPU, 472,006,656 bytes peak RSS). The inherited network-3/scan-5
nonconvergence remains recorded. No native gate or cycle was performed. This
closure changes review documentation only and does not extend that verdict to
an unimplemented route. Owner-required decisions remain pending regardless of
elapsed time. The authorized final stop condition has not yet been reached.
