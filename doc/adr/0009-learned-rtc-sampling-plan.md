# ADR 0009: Learned RTC Sampling Plan

- **Status:** Accepted
- **Recorded:** 2026-08-09
- **Decision owners:** Citlali project owner and RTC scientific owner

## Context

RTC currently receives a fixed requested low-pass and downsampling policy. A
single fixed policy is easy to reproduce, but it cannot use the substantial
variation in beam size and telescope scan speed to choose an efficient cadence.
Conversely, continuously adapting the filter or cadence during science
execution would make the downstream transfer, time grid, and provenance hard
to reason about.

Citlali already has an explicit reduction-learning lifecycle with learn,
learn-with-model, and apply phases. The project owner selected that lifecycle
for a future optional RTC sampling mode: a conservative metadata-derived plan
is used while learning, the learner resolves a complete sampling plan, and the
apply phase executes the frozen plan without further adaptation.

## Decision

RTC sampling has two requested modes: `fixed` and `learned`. Existing fixed
behavior remains the compatibility default. Learned mode has distinct
requested, bootstrap, learned, resolved, and applied states.

The bootstrap plan is derived before detector science processing from stable
metadata and preflight facts:

- measured native detector cadence;
- approved per-array beam identity and size;
- maximum valid telescope speed within each science-valid scan;
- scan identity, boundaries, and available support;
- permitted integer downsampling factors; and
- owner-approved astronomical-transfer, alias, and sampling tolerances.

The bootstrap is deliberately conservative and initially common across all
arrays and scans. It uses the smallest admitted beam, the largest valid scan
speed, and an explicit safety margin. If required metadata are unavailable or
no decimated candidate is safely admissible, the implementation uses native
cadence when that fallback is explicitly supported; otherwise it fails before
RTC science execution.

The learner may evaluate preferred plans per scan and array, but resolution is
constrained by the downstream time-grid and transfer contract. The first
executable version resolves one common observation cadence and filter. Later
per-array or per-scan execution requires a separate downstream compatibility
decision.

“Optimal” means **maximum safe reduction**: select the largest integer
downsampling factor, and an analytically validated realizable low-pass filter,
subject first to the approved astronomical-response, alias-rejection,
sampling, and downstream-compatibility constraints. No plan is selected by
silently weakening a constraint.

The apply phase consumes only the immutable resolved plan. It does not
relearn, retune, or silently fall back. A mismatch in observation, scan, beam,
cadence, plan identity, or required support fails before apply.

## Scientific And Lifecycle Rules

- Sampling sufficiency uses the maximum valid telescope speed in a
  science-valid scan, not a percentile. Median, p95, and p99.5 remain
  diagnostics.
- The bounded software transfer is calculated analytically from the admitted
  scan-projected beam response, realized FIR coefficients, and exact
  phase-zero decimator. Synthetic source injection is not a scientific
  acceptance requirement.
- Physical detector integration-event semantics remain unavailable. A
  constant telescope-to-detector timing offset does not change speed derived
  internally from telescope positions and times, but this decision authorizes
  no astrometric timing correction.
- Learned sample masks and intervals that cross the cadence transition use
  native row identity or native-time support, never bootstrap-downsampled
  positions as their durable identity.
- A learn-to-apply product transition has intentionally changed transfer and
  cannot count as convergence evidence. Convergence assessment begins only
  after the apply plan is frozen.
- Final science products bind the resolved-plan digest and material realized
  state. A detailed learned-plan artifact is diagnostic unless a declared
  consumer gives it a stronger role.
- Manual oversampling remains valid and is advisory only. Learned mode never
  changes a fixed request.

## Consequences

- The sampling decision becomes deterministic, inspectable, and replayable
  without continuous runtime adaptation.
- A metadata-only bootstrap can be implemented before optional noise-aware
  learning. Any future noise-aware objective is a separate scientific
  expansion because it makes plan selection data-dependent.
- RTC-only learned output requires either pre-RTC metadata resolution or a
  bounded learn/rewind/apply pass; it cannot claim an apply product when no
  resolved plan was used.
- Per-array and per-scan efficiency remain available future extensions, but
  they must account for heterogeneous time axes, flag support, PSD identity,
  weights, map transfer, and product schemas.

## Rejected Alternatives

- **Continuously adaptive runtime filtering:** rejected because it changes the
  transfer and cadence during accumulation without an immutable plan boundary.
- **Immediate per-scan/per-array execution:** deferred because current
  downstream compatibility has not been established.
- **Use p99.5 speed as the safety authority:** rejected in favor of the maximum
  valid in-scan interval speed; percentiles remain diagnostics.
- **Treat oversampling as invalid:** rejected because it is an efficiency and
  noise-bandwidth concern, not by itself a scientific failure.
- **Require source injection to validate a linear operator:** rejected; exact
  analytical response is the scientific authority, while implementation
  fixtures may still use deterministic test vectors.

## Supersession

A future ADR may permit per-array or per-scan applied cadences, a noise-aware
objective, or a different sampling authority only after the affected RTC/PTC,
mapmaking, provenance, restart, and product contracts are approved and
validated.

## Evidence And Plan

- [`../RTC_LEARNED_SAMPLING_PLAN_2026-08-09.md`](../RTC_LEARNED_SAMPLING_PLAN_2026-08-09.md)
- [`../REDUCTION_LEARNING_REFACTOR_PLAN.md`](../REDUCTION_LEARNING_REFACTOR_PLAN.md)
- [`../audits/packages/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_HANDOFF_2026-08-09.md`](../audits/packages/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_HANDOFF_2026-08-09.md)
