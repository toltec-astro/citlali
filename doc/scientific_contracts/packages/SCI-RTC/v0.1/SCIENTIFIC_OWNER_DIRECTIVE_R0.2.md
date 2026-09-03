# SCI-RTC v0.1 scientific-owner revision directive for r0.2

Status: binding scientific-owner authorship direction received `2026-08-18`.

The complete source directive was supplied by Grant Wilson in the review
packet titled **SCI-RTC v0.1 Revision Directive: Make Learn–Apply the
Scientific Organizing Principle**. This repository record preserves its
binding scope and dispositions without importing implementation evidence.

## Binding method

- Revise the science-team rationale from r0.1 to r0.2.
- Remain implementation-blind: do not inspect implementation, audits, tests,
  reductions, or production behavior.
- Make learn → resolve → immutable apply the top-level RTC scientific method.
- Treat online adaptation as a separate time-varying estimator.
- Use one ten-part scientific template for every operation: purpose, signal
  model, operator, design parameters, learn, resolve/apply, astronomical
  effects, calibration effects, validation, and unavailable states.
- Explain temporal frequency through realized scan speed/direction, projected
  beam, source class, cadence, and prior response.
- Define notches, low/high/band-pass filters, FIR order/taps, despiking,
  donors, masks, state, boundaries, non-finite handling, and decimation by
  scientific meaning rather than configuration labels.
- Retain raw donor scaling `flxscale_q/flxscale_d` for valid compatible exact
  occurrences, while stating that it does not prove physical donor
  equivalence and may not circularly use a new factor from the same Beammap.
- Preserve complete response, support, covariance, timing/coordinate, and
  calibration-plan consequences.
- Do not invent production values. Unresolved choices remain owner decisions.
- Do not claim implementation conformity, validation, science qualification,
  or production readiness.

## Required deliverables

The directive requires revised rationale and formal PDFs, updated definitions,
requirements and predictions, rationale-to-contract crosswalk, owner ledger,
change log, consistency report, and cross-package follow-ups for CAL, BEAM,
AST, PTC, VAL, MAP, and downstream filtering. Those artifacts are indexed by
`README.md`.

## Author disposition

No requested scientific point conflicted with the approved v0.1 boundary.
The conservative existing rule that all replacement- or synthesis-influenced
outputs are scientifically ineligible is retained as an already selected
v0.1 policy; r0.2 now explains its information-loss consequence. No numerical
owner choice was inferred.
