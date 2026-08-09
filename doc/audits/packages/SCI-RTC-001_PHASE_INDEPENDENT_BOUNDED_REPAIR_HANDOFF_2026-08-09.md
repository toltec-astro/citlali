# SCI-RTC-001 phase-independent bounded repair handoff

Date: 2026-08-09

Handoff ID: `SCI-RTC-001-PHASE-INDEPENDENT-REPAIR-READY-001`

Status: prepared for coordinator verification and separate owner launch;
repair not authorized or launched

## Exact authority and proposed repair identity

- Governing application base:
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20` (parent
  `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, tree
  `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`).
- Proposed repair branch: `codex/repair-sci-rtc-001-phase-independent`.
- Proposed fresh worktree:
  `/private/tmp/citlali-repair-sci-rtc-001-phase-independent`.
- Frozen audit: `3319d7424c732c1c9fc300c336e4d428e6f91068`
  (parent/core `3620434eb988662210b2466ee357ffc8f891aa58`, tree
  `8b01ee4d8117816904d7f078682c0c62a2ea88ac`).
- Owner D001--D004 authority:
  `doc/audits/packages/SCI-RTC-001_OWNER_DECISION_2026-08-08.md`, SHA-256
  `5f5ebb52735a70510d75f5a6954ef825fc425c00e43ab0aadf2363f2e0609723`.
- ALIGN-deferred compatibility authority:
  `doc/audits/packages/ALIGN_DEFERRED_COMPATIBILITY_BOUNDARY_OWNER_POLICY_2026-08-09.md`,
  SHA-256
  `15e27271f049b7522029240b6f1cc5ff86a8bfb323244663a5dcb2575d771123`.

The proposed base is the canonical application snapshot assessed by the RTC
audit. Selecting it does not accept the implementation or change the axes:
contract `proposed`, implementation `nonconformant`, validation `in_progress`,
production `existing_use_only`, verdict `amend`.

## Compatibility input boundary

The repair may consume the existing assigned-time grid only through exact
identity `ALIGN-ASSIGNED-TIME-COMPAT-001`. Every affected state and product
must bind the realized ALIGN/assigned-grid identity and carry
`physical_event_semantics: unavailable` or equivalent.

This handoff authorizes no choice of physical sample start/end/centroid,
half-/whole-sample correction, timing prior, detector-time absolute oracle, or
final AST placement. Absolute phase, sub-sample astrometry, timing-sensitive
source-mask accuracy, and corrections remain fail closed.

## Mandatory initial scope checkpoint

If separately launched, the repairer must verify the exact base/ref/parent/
tree, fresh worktree, proposed branch, and clean state, then return before
editing with:

- exact paths proposed from the allowlist below;
- finding/decision-to-file and finding/decision-to-test mapping;
- exact existing identity/provenance owner for each added field;
- confirmation that no DSP, calibration, ALIGN/AST, PTC, map, or product
  algorithm redesign is proposed;
- local Citlali science reduction and Unity: prohibited;
- delegation/external review/contact: prohibited;
- first viable artifact: deterministic replacement/influence eligibility and
  complete signal/response-parity fixtures on the assigned grid; and
- next return: after those fixtures and before writer/provenance expansion or
  broad test execution.

Silence prohibits a capability. A needed new path, schema, generalized
framework, timing interpretation, or cross-package change requires a stop and
separate coordinator/owner decision.

## Initial implementation path allowlist

RTC operator and response:

- `include/citlali/core/timestream/rtc/despike.h`
- `include/citlali/core/timestream/rtc/despike2.h`
- `include/citlali/core/timestream/rtc/downsample.h`
- `include/citlali/core/timestream/rtc/filter.h`
- `include/citlali/core/timestream/rtc/kernel.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`

Existing requested/effective/realized state and provenance boundaries:

- `include/citlali/core/pipeline/raw_timestream_authority.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_resolution.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/pipeline/raw_timestream_config_serialization.h`
- `include/citlali/core/pipeline/timestream_output_provenance.h`
- `include/citlali/core/pipeline/timestream_scan_context.h`

Direct existing RTC callers/configuration boundaries:

- `include/citlali/core/engine/detail/rtc_config_impl.h`
- `include/citlali/core/engine/detail/kidsproc_direct_rtc_impl.h`
- `include/citlali/core/engine/detail/beammap_source_aware_rtc_impl.h`
- `include/citlali/core/engine/detail/beammap_timestream_pipeline_impl.h`
- `include/citlali/core/engine/detail/lali_timestream_output_impl.h`
- `include/citlali/core/engine/detail/rtcdiag_output_impl.h`

Existing test/build registration may be updated only for focused local RTC
fixtures:

- `tests/CMakeLists.txt`

At the initial checkpoint the repairer must name each exact new focused test
path before creating it. No other application, test, config, validation, or
output path is authorized by this readiness record.

## Bounded change and test traceability

| Authority/finding | Bounded change | Required focused local evidence | Explicit exclusion |
| --- | --- | --- | --- |
| D001 / F003 | Mark every replaced/synthesized signal and every downstream influenced signal scientifically ineligible; preserve compact typed cause/support through filters and decimation. | Donor/replacement, synthesis, non-finite, chained FIR/IIR/notch, full decimation-support, endpoint, repeat, and cause-retention fixtures. | Bulky per-sample provenance products; CAL factor-ratio repair; physical timing interpretation. |
| D002 / F003/F004 | Apply every enabled response-changing stage to both signal and complete kernel/realized local response on the exact assigned grid, or mark complete response unavailable. | Constant/impulse/ramp/sinusoid/notch, donor influence, configured projection, sequential/OpenMP, and signal/response parity fixtures with explicit unavailable controls. | Absolute phase or physical centroid claim; empirical beam/astrometric fidelity; response algorithm redesign. |
| D002 / F005 | Fail closed when source-mask geometry, frame, identity, or validity is unavailable; ensure signal and response consume the same admitted mask state. | Valid/invalid/missing coordinate, detector permutation, shape/radius, mask-validity, source-crossing-on-assigned-grid, and unavailable-response fixtures. | Timing-sensitive mask-accuracy claim, correction, or final AST placement. |
| D004 / F006 | Make FIR/IIR/notch, edge, representative assigned time, rate, phase label, support, state, and downsampling semantics exact and consistent on the existing lattice. | First/last/short scan, missing context, odd/even phase-zero downsample, 0.5x/2x/4x rate, chained state, reset, and edge/support round trips. | DSP redesign, new cadence family, physical event phase, costly response computation. |
| D003/D004 / F007 | Give outer, inner, full, mini, diagnostic, simulated, and processed outputs immutable stage identities and parent/process links; serialize exact coefficients/state/full-precision multirate semantics once per coherent segment. | Product identity collision, parent/link, requested/effective/realized state, coefficient/state, assigned-grid identity, unavailable-event state, writer/reopen, incomplete/duplicate/stale bundle fixtures. | Duplicate computation, per-sample state/provenance arrays, generalized lifecycle redesign. |
| F011 | Exercise the actual local production call paths for the bounded changes. | Focused exact-candidate fixtures, direct callers, writer/reopen, sequential/OpenMP, real/simulation parity, configuration preflight, full CTest, applicable baseline/product-contract gates, and zero unexpected error-level output. | Local science reduction, Unity, astronomical validation, broad/costly campaign. |
| F001/F002/F008/F009/F010 | Preserve as conditioned or outside this repair except for the phase-independent interface guards above. | Record exact dependency state and fail-closed controls. | CAL/ALIGN/AST repair, calibration algebra, timing correction, absolute placement, downstream acceptance. |

## Required realized contract

For each coherent RTC stage/segment, persist or resolve:

- immutable stage and parent/processing identity;
- exact assigned-grid identity and `physical_event_semantics: unavailable`;
- detector/scan/sample ordering, rate/lattice label, representative assigned
  time, edge rule, and causal support;
- exact replacement/synthesis causes and compact downstream influence state;
- exact filter/notch/multirate coefficients, normalization, state/reset,
  ordering, support, and complete response availability;
- source-mask identity and validity, with timing-sensitive accuracy unavailable;
- requested/effective/observation-resolved/realized RTC state; and
- exact links required by PTC/VAL products without accepting those consumers.

Never present a partial kernel as the complete conditioned response. Never
equate finite/unflagged with original-valid, independent, complete support, or
scientific eligibility.

## Local validation and handback

After the two required scope returns, run only focused local implementation
gates plus the repository-required deterministic suite:

1. focused successor RTC success/failure and production-call-path fixtures;
2. actual product writer/reopen and provenance/link checks;
3. sequential/OpenMP and real/simulation interface parity;
4. full CTest with disabled/skipped tests reported;
5. `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`;
6. applicable baseline, authority-sync, and product-contract gates; and
7. `git diff --check`, exact path inventory, artifact digests, and clean state.

Return one exact repair commit/parent/tree, changed paths, test commands and
results, artifact/evidence digests, traceability, axes unchanged, and every
conditioned dependency. Stop for coordinator review before re-audit,
integration, merge, push, Unity, or any downstream action.

## Explicit exclusions and non-authorization

- no calibration/opacity/responsivity or donor-target factor repair;
- no ALIGN resampling/event-semantics or AST coordinate/placement repair;
- no guessed timing correction, offset prior, or physical phase/centroid claim;
- no timing-sensitive source-mask accuracy or empirical response claim;
- no mature DSP redesign, new estimator, new cadence family, or expensive
  response computation;
- no PTC, VAL, MAP, BEAM, covariance, or scientific-product expansion;
- no local science reduction, Unity request/access, external contact, broad or
  costly execution;
- no production change, repair launch, re-audit, downstream launch, merge, or
  push.

Only the project owner may launch this repair against the exact proposed base,
branch, and digest-bound handoff after independent coordinator verification.
