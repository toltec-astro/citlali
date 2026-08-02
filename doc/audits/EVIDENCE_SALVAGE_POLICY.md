# Evidence-Salvage Policy

Policy ID: `FRAMEWORK-SALVAGE-001`

Status: active for every costly numerical study governed by
`FRAMEWORK-NUM-001`

Authority: project-owner framework corrective action, 2026-08-02

## Purpose

A failure in an execution harness, parser, evaluator, or decision recorder does
not by itself make completed scientific model output invalid. This policy
requires raw computation and downstream evaluation to carry separate validity
states, defines when raw evidence may be reused, and preserves confirmatory
independence when a successor evaluator is needed.

This policy does not relax model identity, provenance, warning admission,
completeness, parsing, scientific acceptance, or fail-closed domain rules. It
prevents the broader and unjustified inference that every downstream software
defect requires every valid upstream calculation to be repeated.

## Required separation

Every costly study shall give separate, versioned identities and validity
states to at least these layers:

1. frozen scientific protocol and case set;
2. model inputs and execution context;
3. model executable and raw outputs;
4. execution sidecars, warnings, and raw parser;
5. evaluator and derived metrics;
6. decision recorder and scientific verdict.

The minimum raw-output states are:

- `valid_raw`: the artifact bytes have passed all applicable model-input,
  execution, completeness, hash, sidecar, and provenance checks;
- `salvageable_raw_pending_preservation_and_admission`: the forensic record
  supports reuse, but the only available live copy is volatile or writable and
  has not yet been copied, rehashed, and protected at a durable location;
- `valid_raw_pending_successor_admission`: the original record establishes
  those properties, but an independent successor admission has not yet been
  completed;
- `invalid_raw`: a named defect affects the model input, model-affecting
  execution, raw payload, completeness, hash/sidecar binding, or provenance of
  that artifact; and
- `unknown_raw`: the available record cannot establish validity.

The minimum parser/admission states are:

- `valid_admission`: an independently bound parser recovers the required raw
  payload and the return-status/warning policy admits it;
- `invalid_admission`: the parser or warning-admission route is defective or
  rejects the artifact for a named reason;
- `not_admitted`; and
- `superseded_admission`, which retains its historical identity.

A parser defect changes parser/admission validity, not the immutable raw
bytes. It leaves scientific reuse pending until a separately frozen parser
validates the payload. It changes raw validity only when independent evidence
shows that the underlying output itself is malformed, incomplete, or bound to
the wrong model execution.

The minimum evaluator states are:

- `valid_evaluation`;
- `invalid_evaluation`;
- `not_evaluated`; and
- `superseded_evaluation`, which retains its immutable historical identity.

An invalid or absent parser or evaluator result never silently changes a
`valid_raw` artifact to `invalid_raw`.

## Raw-evidence admission test

Raw output may be reused only when an independent review verifies all of the
following for each claimed tuple or an exact digest-bound collection:

- model inputs, their ordering, units, and domain are the frozen values;
- executable/model identity is exact and the model-affecting execution
  conditions match the frozen context;
- the output is complete for the tuple it claims to represent;
- return status, warnings, unresolved records, and admission policy are
  preserved and accepted by the frozen or successor protocol;
- parsing independently recovers the required payload and metadata;
- raw-output and sidecar hashes, tuple bindings, and aggregate provenance
  verify; and
- no defect under review could have affected any of those facts.

A digest match is necessary but is not sufficient when the original parser or
warning-admission logic is itself in question. Conversely, a downstream
decision failure is not evidence of a model-output defect when all admission
facts pass.

## Successor protocol

When raw evidence is salvageable, the successor shall:

1. leave the original protocol, runner, artifacts, and failure decision
   immutable;
2. create new versioned identities for the successor protocol, runner,
   parser, evaluator, schema, result, and execution context;
3. bind every reused artifact to its original tuple and execution context by
   SHA-256;
4. declare a closed reused-artifact set and a closed missing-computation set;
5. independently reparse and admit reused raw artifacts before any scientific
   metric is evaluated;
6. generate missing evidence in a distinct delta cache or release;
7. verify the union of reused and new raw evidence against the complete frozen
   case set;
8. apply one frozen successor evaluator to the entire union; and
9. disclose reuse and both raw/evaluator validity states in the result.

New output must not overwrite or relabel old output. The original evidence
cache remains read-only. A successor may refer to it by a digest-bound manifest
or use read-only links, but its result must distinguish reused from newly
computed members.

A writable or temporary live cache is not a durable reuse authority even when
its recorded digest is valid. Before successor admission, copy it to a named
durable location without transforming payload bytes, independently verify the
complete manifest and aggregate digest, record the copy operation, and protect
the preserved copy from mutation. Until then its state is
`salvageable_raw_pending_preservation_and_admission`, not `valid_raw`.

## Confirmatory independence

Successor reuse remains confirmatory only if no decisive partial scientific
metric was inspected before the successor evaluator and its acceptance logic
were frozen. Prohibited inspection includes partial candidate error, candidate
ranking, maximum-error location, band-integrated acceptance metrics, or any
other quantity that could tune the successor decision rule.

Integrity work is permitted before freezing the successor: file and hash
inventory, tuple completeness, return-status and warning checks, independent
parsing, guard reconstruction, and diagnosis of the harness defect. These do
not select a scientific answer.

If decisive partial metrics were inspected, the successor must declare the
loss of confirmatory independence. It then becomes descriptive unless a new,
independently justified confirmatory design is approved.

## Scope of repetition

The default is missing-only or defect-scoped recomputation, not a fresh full
cache. A completed model calculation must be repeated only when a concrete
scientific, integrity, provenance, completeness, warning-admission, parsing,
or independence failure reaches that calculation.

The salvage record shall map each failure to its smallest affected scope:

- one artifact or tuple;
- one execution shard or condition family;
- one parser/evaluator version; or
- the full study, only when the defect actually spans the full study.

Procedural conservatism, implementation convenience, or a previous request for
a “fresh cache” is not by itself a scientific reason to discard verified raw
evidence. Any full-repeat decision must record the concrete causal reason and
manager approval.

## Authorization boundary

A salvage disposition authorizes neither new computation nor scientific
evaluation. Before any expensive successor execution, the study must pass the
`FRAMEWORK-NUM-001` Tolerance-and-Stop-Condition Register, model-free guard
preflight, independent review, and execution-readiness certificate gates.

The current application of this policy is recorded in
`doc/audits/packages/SCI-CAL-001_EL25_RAW_EVIDENCE_DISPOSITION_2026-08-02.md`
and its machine-readable companion. That disposition does not launch AM,
authorize CAL repair, or launch a re-audit.
