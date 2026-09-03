# SCI-RTC v0.1/r0.10 candidate change log

Date: 2026-08-21

Status: candidate scientific authority reopened from the unchanged frozen
v0.1/r0.9 baseline; owner review pending. No implementation, validation,
performance, science-qualification, or production claim is made.

## Authority and scope

Grant Wilson authorized the bounded reopening recorded in
`SCIENTIFIC_OWNER_REOPENING_DIRECTIVE_R0.10.md`. The reopening uses RTC's
existing conditioned-$r$ extension point and changes no conditioned-$x$
algorithm, numerical equation, CAL boundary, unrelated operation policy, PTC
policy, or SCI-VAL policy.

## Normative additions

- Added `SCI-RTC-DEF-039`, defining conditioned $r$ as optional paired content
  under the same context, plan, realized record, and exact conditioned-$x$
  grid.
- Added `SCI-RTC-EQ-036`, defining the coordinate-diagonal paired affine
  response and conditional joint-covariance propagation while retaining zero
  cross-coordinate numerical branches.
- Added `SCI-RTC-REQ-109`--`114`, one requirement for each reopening decision:
  role/optionality; pair-coherent artifacts; exact grid/failure isolation;
  response/covariance; leakage/source protection; and downstream handoff.
- Added `SCI-RTC-PRED-072`--`077`, one falsifier for each decision.
- Added resolved owner-ledger entries `SCI-RTC-OWNER-084`--`089`.

## Surgical supersessions and clarifications

- Superseded only the r0.9 statement that conditioned $r$ remained unavailable
  pending a separate channel-specific pipeline. It is now the bounded paired
  companion, not an independent lifecycle.
- Extended existing boundary, artifact, sampling, response, covariance,
  validity, output, leakage, source-protection, and handoff text to the approved
  conditioned-$r$ semantics.
- Preserved the complete existing conditioned-$x$ numerical operator in
  `SCI-RTC-EQ-005` and its $x\leftarrow r=0$ identity in `SCI-RTC-EQ-006`.
- Preserved immutable raw-$r$ parentage, x-only SCI-CAL, diagnostic-only RTC
  atmosphere use, no numerical $x\leftrightarrow r$ mixing, and downstream
  ownership of PTC/SCI-VAL policy.

## Inventory change

| Authority class | r0.9 | Candidate r0.10 |
| --- | ---: | ---: |
| Definitions | 38 | 39 |
| Displayed equation tags | 37 | 38 |
| Assumptions | 12 | 12 |
| Requirements | 108 | 114 |
| Predictions | 71 | 77 |
| Owner entries | 83 | 89 |
| Owner states | 63 open, 1 conditional, 14 resolved, 5 deferred | 63 open, 1 conditional, 20 resolved, 5 deferred |

## Explicit non-changes

- No new package or sibling-product architecture.
- No default joint-$r$ activation.
- No change to despiking, filter, notch, level-shift, sampling, or CAL numerical
  policy beyond the pair semantics explicitly approved.
- No RTC-owned downstream $x\leftarrow r$ correction or nuisance predictor.
- No RTC aggregation of cross-package validity or usage.
- No implementation inspection, test execution, configuration inspection,
  generated-science-product inspection, external literature, or web input.
