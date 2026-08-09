# SCI-RTC-001 learned sampling Stage A handoff

Date: 2026-08-09

Handoff ID: `SCI-RTC-001-LEARNED-SAMPLING-STAGE-A-READY-001`

Status: design prepared; implementation not launched; exact application base
and numerical-tolerance authority unresolved

## Authority

- Coordination parent:
  `c078179df5916c54b6ab0ee3789fcde925b43d87`.
- Durable decision:
  `doc/adr/0009-learned-rtc-sampling-plan.md`.
- Detailed plan:
  `doc/RTC_LEARNED_SAMPLING_PLAN_2026-08-09.md`.
- Existing phase-independent RTC repair candidate:
  `24f28ea9de6b4a1a3ff81d07944fa5fc2565f240`.
- Existing phase-independent RTC re-audit:
  `e3acba9b2154234042778adc9737d08e30652ec5`.

The repair candidate remains nonconformant under its accepted re-audit; this
handoff does not select it or any other application commit as the Stage A
implementation base. The coordinator must select and verify an exact
application base before launch.

## Bounded Stage A Result

Stage A may add an observe-only learned-sampling planner that:

1. accepts a typed `fixed` or `learned` request while hard-gating learned
   execution to `advisory_only`;
2. resolves a conservative metadata bootstrap;
3. calculates per-scan/per-array candidate plans analytically;
4. reports the complete parameterized candidate metrics and resolves one
   common observation recommendation only when exact owner-approved tolerances
   are supplied;
5. records exact requested/bootstrap/learned/resolved identities and reasons;
6. compares that recommendation with the unchanged fixed execution; and
7. emits deterministic diagnostics without changing any science sample.

The existing factor, FIR, phase-zero decimator, timestamps, flags, detector
data, RTC/PTC/map inputs, maps, weights, and persisted science-product cadence
must remain byte-for-byte governed by the fixed execution path.

## Initial Technical Checkpoint

Before editing, a launched task must return `READY` with:

- exact base/ref/parent/tree and clean fresh worktree;
- proposed branch `codex/rtc-learned-sampling-stage-a`;
- exact approved numerical-tolerance authority, or an explicit metrics-only
  checkpoint that cannot select a recommendation;
- exact requested/effective/observation-resolved/realized owners it will reuse;
- exact telescope-speed, beam, cadence, FIR-response, diagnostics, and test
  paths proposed;
- proof that no established DSP or science-output path will consume the
  recommendation;
- finding of every existing positional sample-index learning interface that
  would need native-row identity before Stage B; and
- confirmation of all prohibitions below.

A new broad framework, public `Engine` state, generalized schema migration, or
need to modify execution before Stage A evidence is a stop for owner review.

## Required Stage A Evidence

- exact valid-interval maximum speed with invalid/gap/turnaround and boundary
  fixtures; retain p50/p95/p99.5 diagnostics;
- circular and elliptical beam-projection calculations;
- exact FIR frequency response and phase-zero alias calculation;
- candidate enumeration with deterministic rejection reasons;
- common-plan resolution invariant to detector order and thread count;
- missing/invalid beam, cadence, telescope, scan, and factor-policy failures;
- native-cadence conservative fallback only where requested;
- oversampling advisory without failure or execution change;
- requested/bootstrap/learned/resolved serialization and digest round trip;
- repeated/sequential observation reset and restart compatibility analysis;
- focused tests, full applicable deterministic repository gates, and zero
  unexpected error-level output.

No source injection is required as scientific evidence. Deterministic vectors
may be used to verify the implementation of the analytical operator.

## Stop Conditions And Non-Authorization

Stage A must stop for:

- inventing a numerical scientific tolerance or changing an approved
  tolerance;
- execution of a learned factor or filter;
- per-array or per-scan applied cadence;
- noise-dependent plan selection;
- a timing/event-semantics or sky-placement inference;
- alteration of RTC/PTC/MAP/VAL/BEAM science, flags, weights, products, or
  production status;
- local science reduction, Unity access/request, external contact, costly
  campaign, repair/re-audit creation, merge, or push.

Completion returns one exact application commit, path inventory, tests and
digests, unresolved decisions, and an unexecuted Stage B proposal. It does not
authorize Stage B.
