# SCI-RTC v0.1 — Internal Stage A Dossier

Status: internal, implementation-informed, quarantined from scientific authors

Date: `2026-08-17`

This dossier may inform the Scope Brief but must never enter an
implementation-blind author packet. It records what the current repository
appears to assign to RTC, not what the scientifically correct implementation
must be.

## Discovery Identity

Implementation-informed discovery used
`origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20`.
Later topic branches were inspected only for documented owner decisions and
status. No Unity system, reduction, or test execution was used.

## Apparent Current Surface

The current tree places mature raw-timestream numerical work under:

- `include/citlali/core/timestream/rtc/rtcproc.h`;
- `include/citlali/core/timestream/rtc/calibrate.h`;
- `include/citlali/core/timestream/rtc/despike.h` and `despike2.h`;
- `include/citlali/core/timestream/rtc/filter.h`;
- `include/citlali/core/timestream/rtc/downsample.h`;
- `include/citlali/core/timestream/rtc/kernel.h`; and
- mode/orchestration details under `include/citlali/core/engine/detail/`.

Typed request, resolution, lifecycle, and provenance responsibilities appear
under `include/citlali/core/config/` and
`include/citlali/core/pipeline/raw_timestream_*`. The current architecture
describes the intended one-way path as requested configuration to effective
plan, observation-resolved plan, realized execution, and provenance, with a
one-way adapter into established processors.

The apparent configured and diagnostic surface is substantially broader than
the minimal scientific core. It includes calibration/extinction switches,
despiking and source protection, FIR/IIR/fixed and dynamic notch filtering,
filter-edge guards, downsampling, kernel generation, line diagnostics,
network/coincidence masks, and mode-specific output roles. Their existence
does not establish that every feature belongs in v0.1 or that any default,
threshold, ordering, estimator, or numerical method is correct.

## Apparent Inputs

- detector-resolved KIDs signal selected by a channel policy;
- aligned sample/scan identities, timestamps, telescope coordinates, gaps,
  and synthesized/support state;
- observation-local detector/APT binding and calibration inputs;
- source and coordinate context for source-aware masks or kernels;
- requested filtering, despike, downsampling, kernel, and diagnostic policy;
- initial/state/context information at scan, chunk, iteration, observation,
  and restart boundaries; and
- product-role and downstream execution state.

## Apparent Outputs And Consumers

- conditioned full/mini or otherwise role-specific detector timestreams;
- sample flags and RTC diagnostic products;
- a kernel/response companion for downstream PTC/map/Beammap use;
- sample-rate, filter, downsampling, edge, and execution provenance;
- diagnostic event, line, notch, detector, and network summaries; and
- conditioned input to optional PTC, MAP, BEAM, VAL, NOI, and MAP-003 response
  consumers.

This inventory is deliberately descriptive. The scientific contract must
decide the minimal atomic output and honest availability states independently
of existing filenames and schemas.

## Mode And Product Tension

Current shared code serves Pointing, OOF, Science, and Beammap. The frozen
SCI-BEAM contract now requires the primary standardized detector Beammap in
raw `Delta f/f`, while the reusable RTC core originally framed the primary
path as calibrated top-of-atmosphere `mJy/beam`. This is not resolved by
choosing the current code path. The Scope Brief instead requires an explicit
product-role signal domain and forbids silent conversion or substitution.

Similarly, calibration science belongs to SCI-CAL even where its application
occurs inside RTC orchestration. RTC may own the ordered application and
response consequences of an admitted CAL operator; it may not derive or
silently repair that operator.

## Mature Numerical Boundary

Repository guidance classifies RTC numerical algorithms as mature and
performance-sensitive. Stage A therefore uses implementation only to identify
the scientific questions and surrounding interface. The eventual contract
may specify response, normalization, state, support, validity, and failure
obligations, but this task does not redesign existing despike, FIR, IIR,
notch, or downsampling code and does not infer correctness from it.

## Historical Evidence Kept Quarantined

The following were searched and classified but are excluded from authorship:

- `SCI-RTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` and its findings/evidence;
- CAL, ALIGN, AST, and MAP-003 handoffs into RTC;
- phase-independent RTC repair and re-audit records;
- learned-sampling Stage A implementation, repair, successor, re-audit, and
  owner-acceptance records;
- `RTC_FLAGGING_AUDIT_2026-03-16.md` and RTC diagnostic handoffs;
- current tests, config defaults, validation profiles, accepted runs, and
  product inventory; and
- all production and Unity status.

These records may later support conformity review. They cannot prescribe the
implementation-independent contract.

## Scope Questions Extracted Without Answers

1. What exact primary signal quantity and unit applies to each product role?
2. Which upstream operators are applied within RTC, and what order is
   scientifically material?
3. How do despike detection, donor selection, replacement, masks, filter state,
   and edges enter the realized operator and response?
4. What support/influence causes make an output ineligible even when finite?
5. When is a scalar temporal transfer adequate, and when is a factorized local
   detector-time Jacobian required?
6. What conditional covariance can be propagated, and which stronger
   uncertainty claims must remain unavailable?
7. What exactly constitutes the phase-zero output time, support, and alias
   identity?
8. How do fixed and learned sampling share one contract without allowing
   unapproved adaptive execution?
9. Which product companions are mandatory for PTC, VAL, MAP, BEAM, NOI, and
   response-tracer consumers?
10. Which diagnostics are scientifically inert and which, if any, may alter a
    selected plan or validity decision?

## Firewall Attestation

The sanitized Scope Brief states the problem, approved prior decisions, and
conditional interfaces without exposing current source behavior, audit
findings, repairs, tests, validation, or production state. This dossier and
its source paths are prohibited author inputs.
