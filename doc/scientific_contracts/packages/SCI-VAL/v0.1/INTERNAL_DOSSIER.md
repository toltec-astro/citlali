# SCI-VAL v0.1 — Internal Stage A Dossier

Status: implementation-informed scope evidence; prohibited from the Stage B
author packet

Date: `2026-08-20`

## Firewall Notice

This file records why SCI-VAL is needed and where the current implementation
appears to expose validity-related behavior. It is not scientific authority.
No statement here may be used by a Stage B author to select an estimator,
encoding, precedence rule, threshold, or eligibility policy.

## Current Interface Inventory

The historical package inventory named these primary areas:

- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/timestream/ptc/ptcproc.h`
- `include/citlali/core/mapmaking/naive_mm.h`
- `include/citlali/core/mapmaking/jinc_mm.h`

Current inspection also found validity-related state in pipeline flagging,
restart, output, calibration, map bundle, and diagnostics paths. The
implementation currently uses combinations of sample flags, detector flags,
finite checks, positive-weight tests, masks, support checks, product-specific
availability fields, and output-specific fit/QC fields. That diversity is
scope evidence only; none of those encodings is presumed to be the correct
cross-stage scientific model.

## Historical Evidence Inventory

The recovered audit ledger at
`8c581bfb26f01b187f4f1e0565f4457bcc25f099` contains eleven handoffs targeting
SCI-VAL:

- XAUD-001: approved MAP distinction; reusable only through current authority
- XAUD-002: MAP source/audit evidence; later audit only
- XAUD-003: CAL source/audit evidence; later audit only
- XAUD-004: ALIGN origin/synthesis evidence; later audit only
- XAUD-005: superseded AST handoff
- XAUD-006: corrected AST validation evidence; later validation only
- XAUD-007: RTC source/audit evidence; later audit only
- XAUD-008: superseded PTC overbroad invalidation handoff
- XAUD-009: owner-amended PTC distinction; reusable through approved decision
- XAUD-010: TEL-input structural evidence; later producer/consumer audit only
- XAUD-011: MAP-003 dependency evidence; later audit only

The historical `doc/RTC_FLAGGING_AUDIT_2026-03-16.md`, repair records, exact
source traces, and queued ALIGN/PTC scan-metadata defect are likewise excluded.
They may seed post-contract conformity tests but cannot enter independent
derivation.

## Scope Risks Revealed By Inspection

1. A Boolean flag can conflate cause, action, and scope.
2. A finite payload can survive despite incomplete provenance or causal
   influence state.
3. A detector-level decision can be applied differently from a sample-level
   decision.
4. A processing mask can be mistaken for acquisition validity.
5. A late output decision can be misread as retroactive fit invalidity.
6. A consumer can conflate positive weight, numerical support, and science
   eligibility.
7. A downstream finite result can obscure an invalid or unavailable parent.
8. Missing typed state can be mistaken for a false cause or a clean sample.

These risks define questions for the sanitized Scope Brief. They do not
establish that a particular implementation is conformant or nonconformant.

## Quarantine Rules

The Stage B author must not receive this dossier, the XAUD files, audit
reports, source paths, code excerpts, current product schemas, repair records,
tests, validation output, Unity records, production status, or the current
Boolean/bit representation. Later conformity work may reopen those sources
only after the scientific contract is frozen.
