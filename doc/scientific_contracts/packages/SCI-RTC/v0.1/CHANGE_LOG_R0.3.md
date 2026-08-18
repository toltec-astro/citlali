# SCI-RTC v0.1 change log: r0.2 -> r0.3

Date: `2026-08-18`

## Targeted rationale changes

- Section 2 now shows a finite outer sequence of immutable
  learn--resolve--apply cycles and the default final replay on original input.
- Section 5 now covers intended-consequence assessment, predicted versus
  measured conditioned PSD, hidden-line discovery, artifact discrimination,
  cumulative budgets, finite stopping, and nonconvergence.
- Added the required two-line example and notch-edge-artifact counterexample.
- Preserved all other r0.2 sections and scientific organization.

## Appended formal authority

- Definitions `SCI-RTC-DEF-027`--`029`: bounded iterative refinement,
  complete cumulative successor plan, and final accepted plan/termination.
- Equation `SCI-RTC-EQ-029`: finite cycle, within-cycle immutability,
  original-input evaluation, and final replay.
- Requirements `SCI-RTC-REQ-071`--`082`: cycle bound, immutable apply,
  cumulative plans, replay/cascade, evaluation, candidate discrimination,
  budgets, stability, stopping, provenance, and restart.
- Predictions `SCI-RTC-PRED-039`--`046`: the eight required cycle, artifact,
  replay, rejection, nonconvergence, maximum-cycle, and restart fixtures.

## Decision and claim state

- Added `SCI-RTC-OWNER-037`--`050` for every numerical stopping, stability,
  candidate, budget, data-splitting, parent-rule, and fallback choice.
- Online adaptation remains deferred under `SCI-RTC-OWNER-027`.
- No existing identifier was renumbered and no production number was chosen.
- No implementation evidence was inspected; all stronger claim layers remain
  unassessed.
