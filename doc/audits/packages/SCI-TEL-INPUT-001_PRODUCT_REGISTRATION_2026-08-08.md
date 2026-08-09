# SCI-TEL-INPUT-001 audit/repair/re-audit product registration

Record ID: `SCI-TEL-INPUT-001-REG-D001`

Date: 2026-08-08

Status: registered; not dispatched

Owner decision: create one durable audit -> bounded repair -> independent
re-audit product for the telescope-file ingress boundary. This record does not
launch any phase.

## Product identity

- Package ID: `SCI-TEL-INPUT-001`
- Name: Telescope-file preparation, row identity, and Citlali ingress
- Tier: B -- interface and response
- Primary boundary: operational TolTECA raw `tel*.nc` selection and
  `*_recomputed.nc` production through Citlali telescope-file admission
- Citlali application authority available at registration:
  `origin/codex/refactor-mainline` commit
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, parent
  `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, tree
  `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`
- TolTECA producer authority available at registration: `origin/main` commit
  `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`, parent
  `4acee1ccb11a13084834149c7d6e6685a87d3d6b`, tree
  `3f3e4b8136bf5528b203cb3bb8b474233bb27a85`
- Registration authority: the documentation-only scientific-audit
  coordination commit containing this record; the coordinator reports its
  exact identity after commit.

This is a cross-repository interface package. Citlali-refactor owns the audit
record and downstream contract. TolTECA remains a read-only upstream authority
until a later owner-approved repair dispatch and explicit TolTECA maintainer
opt-in place it in modification scope.

Tier B is valid only while this package consumes approved AST coordinate-
transform authority and approved ALIGN resampling abstractions. An observed
coherent row displacement, unresolved source association, temporal or
astrometric response question, unit/frame mismatch, or need to choose or
rederive event semantics, coordinate formulas, uncertainty, or response is a
mandatory Tier A promotion trigger before that work continues.

## Included audit scope

The independent audit will derive and assess the contract for:

1. selection and association of the raw telescope file with the intended
   observation, detector inputs, and reduction;
2. immutable raw-source identity and the derivation identity of the generated
   telescope file;
3. telescope dimension, row count, ordering, first and last record, native
   timestamp, nominal 20-ms cadence, gaps, duplicates, and missing/non-finite
   state;
4. one-to-one row pairing between `TelTime`, pointing-state inputs, Hold, and
   the three recomputed outputs `ActParAng`, `SourceRaAct`, and
   `SourceDecAct`;
5. a strict allowlist of modified or added NetCDF variables and bitwise
   preservation of every other variable and dimension;
6. cache reuse, source/version binding, atomic production, partial-file
   failure, retry, and stale-product rejection;
7. the Citlali admission boundary for telescope identity, time coverage,
   cadence, ordering, and coherent row association; and
8. provenance and typed validity sufficient for downstream ALIGN, AST, and
   VAL consumers to distinguish verified original rows, recomputed values,
   gaps, and unresolved producer state.

The audit must test the first and last rows, a Hold transition, a missing
native row, duplicate/non-monotonic timestamps, a stale generated file, a
partial generated file, and deliberate plus/minus one telescope-row
corruption. A coherent equal-length row shift is an explicit falsification
case because ordinary length and monotonicity checks cannot detect it.

## Explicit exclusions and ownership boundaries

- `SCI-ALIGN-001` continues to own native telescope-to-detector-grid mapping,
  interpolation, scan slicing, gap treatment, and the downstream timing and
  support response.
- `SCI-AST-001` continues to own the physical coordinate-frame, epoch, sign,
  handedness, parallactic-angle, and astrometric correctness of the pointing
  transformation. `SCI-TEL-INPUT-001` owns only correct source-row/time pairing
  and faithful publication of the transform result.
- `ENG-STATE-001` owns generalized lifecycle, immutable provenance,
  required-product completion, and failure propagation. This product owns the
  concrete telescope-file instance of those obligations and must hand off any
  generalized defect.
- `SCI-VAL-001` owns downstream eligibility and flag consumption; this product
  must supply the producer/ingress validity facts.
- Acquisition-clock, integration-event, internal-counter, PPS-counter, and
  delivered-timestamp semantics are an orthogonal investigation. Its accepted
  facts may later enter this audit through the frozen handoff protocol, but
  this registration does not identify a hardware cause or authorize a timing
  correction.
- RTC, PTC, mapmaking, Beammap fitting, and scientific-map validation are not
  part of this package except as named downstream consequences.

## Required lifecycle

### Independent audit

The future dispatch must freeze a fresh inbox manifest and an independent
interface core before opening the quarantined TolTECA producer implementation,
Citlali consumer implementation, or post-core ALIGN evidence. The core will
define typed raw and derived file identities, the row-preservation operator,
allowed mutations, time/phase invariants, failure semantics, response to a
row displacement, and downstream restrictions.

The audit may use static source inspection, digest and NetCDF-structure checks,
and existing exact raw/generated fixture pairs. A local Citlali reduction,
Unity access, new data production, or broad/costly execution requires a later
scope checkpoint and separate authorization. No such execution is authorized
here.

### Bounded repair

Repair begins only after coordinator review and owner disposition of the audit
findings. The repair dispatch must name the exact repository, base commit,
files, findings, tests, evidence, and stop rule. Likely ownership routes are:

- TolTECA producer selection, cache, atomicity, row preservation, and generated
  provenance;
- Citlali telescope admission and fail-closed consumer validation;
- AST transformation correctness; or
- ENG/VAL generalized provenance and validity consumption.

The audit may route findings to more than one owner, but no broad cross-
repository repair is implied. TolTECA modification remains prohibited until
the project owner and TolTECA maintainer explicitly approve it.

### Independent re-audit

The re-audit must use a fresh role-separated task and worktree, assess the
exact repair successor or bound multi-repository successor set, repeat the
row-identity and deliberate-corruption fixtures, verify product provenance and
failure behavior, and disposition every accepted handoff. Repair tests and
plausible reductions are evidence, not re-audit independence or production
authorization.

## Sequencing and evidence

The product is registered but not launched. It is scientifically orthogonal
to PTC and MAP repair work and may be prepared while those lanes continue.
Before dispatch, the coordinator must decide its position relative to the
current ALIGN acquisition-event-semantics investigation and freeze any
accepted results as pre-core authority or post-core evidence according to the
handoff rules.

Existing raw/recomputed ALIGN corpus observations and the same-T0 cadence-
lattice study are candidate post-core evidence only. They do not establish a
universal producer contract and are not silently admitted by this record.

## Resource default and stop rules

Under `FRAMEWORK-EFFORT-001`, launch preparation defaults to Terra High and the
bounded independent interface core and implementation/product trace to Sol
High or XHigh. A Tier A promotion may justify Sol Max for one coherent timing
or response contradiction. Ultra is not authorized by this registration and
would require a new written decomposable-workstream trigger.

Stop before each of the following without a fresh authorization: substantive
audit work, application or TolTECA edits, local reduction, Unity evidence,
costly execution, repair, re-audit, downstream audit launch, integration,
production expansion, or push.

## Registration disposition

- contract status: `not_started`
- implementation status: `not_assessed`
- validation status: `not_started`
- production status: `existing_use_only`
- verdict: `pending`
- remediation branch/commit: null; repair not authorized
- re-audit status: `not_started`

Registration preserves existing operational use without approving a new
timing, pointing, provenance, or scientific-validity claim.
