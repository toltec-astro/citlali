# ENG-STATE-001 Tier C lifecycle, provenance, and failure-flow audit

## Audit identity and boundary

- Package: `ENG-STATE-001` (Tier C).
- Governing application SHA: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Evidence basis: static inspection only.  The worktree was clean at that SHA
  before inspection.
- Inbound handoffs: none; the frozen inbox manifest is empty.

This first viable artifact assesses lifecycle ownership, provenance,
required-product publication, reproducibility, and structured failure flow.
It does **not** assess or alter any estimator, calibration, astrometry,
timing, validity semantics, units, response, scientific contract, consumer
authorization, or package status axis.  It does not claim runtime behavior.

## Authorized inspection set and method

Only the following existing paths were inspected:

- `include/citlali/core/session/`
- `include/citlali/core/pipeline/`
- `validation/product_contracts.json`
- `validation/accepted_runs.json`
- `tools/baseline/audit_reduction_run.py`

The review used repository identity checks and static `rg`/header inspection;
no build, test, reduction, Unity contact, external-evidence request, or new
tooling occurred.

## Observed Tier C lifecycle and provenance evidence

1. `ReductionSession` has an explicit `ready -> running ->
   succeeded|failed` lifecycle.  `ReductionResult` carries a structured
   status, diagnostics, product roots, and provenance artifact paths.  Typed
   domain errors map to stable result codes, including `output_failed`.

2. `ReductionConfigState` retains a typed configuration and runtime
   provenance.  `runtime_provenance.yaml` serializes requested and effective
   runtime configuration and realized runtime resources; the existing
   baseline auditor has a corresponding `--require-runtime-provenance` gate.

3. `config_source_manifest.yaml` records copied input paths, sizes, and
   SHA-256 digests, plus a SHA-256 of the merged configuration snapshot.
   Atomic YAML publication removes a partial temporary file and rethrows on a
   failed write.

4. Observation/provenance lifecycle helpers distinguish effective policy,
   observation state, and realized state.  Examples include mapmaking,
   coadd, raw/processed timestream, post-processing, and beammap provenance.
   Their writers reject incomplete lifecycle state where the corresponding
   plan requires it.  The accepted-run ledger retains sidecar identities and,
   for retained raw-timestream records, semantic checks such as completed scan
   and required-write counts.

5. The product registry classifies required enabled products as fatal on
   failure.  `fail_required_output` logs an error and throws a typed output
   error; `ReductionSession` converts that error to a structured failed
   result.  Ordered writers retain and rethrow the first asynchronous failure.

## Findings

### ENG-STATE-001-F001 — required timestream writer completion has no static caller in scope

- Class/severity/basis: `evidence_gap`, P1, `observed`.
- Evidence: `TimestreamOutputWriters` provides `rethrow_if_failed()` and
  `verify_complete(expected)`, but the static caller census in the authorized
  session/pipeline scope found only their definitions.  Conversely,
  `publish_completed_raw_timestream_provenance()` derives an expected write
  count, marks the raw-timestream plan complete, and writes its provenance
  sidecar without a direct call to either check in that helper.
- Consequence: this does not establish an implementation defect—the omitted
  caller may be outside the authorized source set or a failure may terminate
  through another route—but static evidence alone cannot prove that every
  required asynchronous write is checked before successful completion and
  provenance publication.
- Required disposition: retain the existing fatal-output policy and all
  consumer restrictions.  Do not repair from this report.

### ENG-STATE-001-F002 — final CLI failure disposition is unassessed

- Class/severity/basis: `evidence_gap`, P1, `observed`.
- Evidence: session and pipeline headers establish typed failure creation and
  conversion to `ReductionResult`, but the CLI entry boundary is outside the
  approved inspection set.  Therefore this artifact cannot confirm the final
  CLI exit/reporting behavior for `output_failed`, `io_failed`, and
  `unhandled_exception` results.
- Consequence: the required-output-to-CLI contract remains unconfirmed at the
  governing SHA.  This is a bounded engineering evidence gap, not a
  scientific-contract finding.
- Required disposition: no status-axis or consumer-policy change; do not
  expand source scope without coordinator approval.

## Verdict and retained status

Verdict: `pending`.  The audit establishes a static Tier C architecture with
explicit state and structured failure mechanisms, but it does not establish
complete required-output enforcement or final CLI propagation.  The ledger's
existing axes remain unchanged: `contract_status: not_started`,
`implementation_status: not_assessed`, `validation_status: not_started`,
`production_status: existing_use_only`, and `verdict: pending`.

## Compact coordinator brief / next gate

First viable artifact complete.  Before any validation execution or further
artifact, the coordinator must decide whether to authorize a second static
scope checkpoint that names (1) the final CLI boundary and (2) the concrete
writer completion call sites needed to resolve F001 and F002.  If authorized,
that checkpoint must remain Tier C and preserve all scientific-package
contracts, consumer restrictions, and status axes.  Local reductions, Unity
evidence, builds, tests, repairs, and a ledger-update proposal remain
prohibited unless separately authorized.
