# Numerical Proportionality And Cost-Control Policy

Policy ID: `FRAMEWORK-NUM-001`

Status: active for every newly authorized costly numerical study and for any
successor execution of a stopped or failed costly study

Authority: project-owner framework corrective action, 2026-08-02

The **audit manager** is the scientific-audit coordinator or a named delegate
responsible for execution cost and control readiness. This role does not
acquire authority to change the project owner's scientific contract,
acceptance gate, production scope, or resource decision.

## Purpose

This policy prevents an implementation tolerance, floating-point identity
diagnostic, or defensive engineering assertion from acquiring scientifically
irrelevant veto power over an expensive study. It strengthens, rather than
relaxes, the existing requirements for provenance, preregistration, exact
identity, fail-closed scientific domains, and falsifiable acceptance gates.

The controlling rule is proportionality: every condition that can stop,
invalidate, or scientifically fail a study must state what requirement it
protects, why its comparison is appropriate, how its threshold was derived,
and what effect the admitted error can have on the final scientific metric.
"Very small", extra digits, and implementation convenience are not
derivations.

This policy does not change any approved scientific tolerance, candidate,
passband, domain, application implementation, or production disposition. It
governs how audit studies are designed, authorized, executed, evaluated, and
salvaged.

## Scope

A **costly numerical study** is any study that its audit manager classifies as
costly because rerunning it exposes material wall time, CPU/GPU allocation,
memory, storage, external scheduling, scarce data access, or human operational
cost. There is deliberately no universal CPU-hour or byte threshold. The
manager records the estimated cost and why the classification applies.

Before a new costly execution, or before resuming or replacing a stopped
costly execution, the study must have all of the following:

1. a frozen scientific protocol and preregistered case set;
2. a machine-readable Tolerance-and-Stop-Condition Register;
3. an executable, model-free guard preflight and its machine-readable report;
4. an evidence-salvage plan fixed before execution;
5. independent review of the register, preflight coverage, salvage plan,
   raw/evaluator separation, and proposed certificate attestations; and
6. a machine-readable execution-readiness certificate signed by the audit
   manager and binding that independent review.

The canonical schemas and templates are:

- `doc/audits/schemas/tolerance-and-stop-condition-register-v1.schema.json`;
- `doc/audits/schemas/expensive-study-preflight-report-v1.schema.json`;
- `doc/audits/schemas/expensive-execution-readiness-certificate-v1.schema.json`;
- `doc/audits/templates/TOLERANCE_AND_STOP_CONDITION_REGISTER_TEMPLATE.yaml`;
- `doc/audits/templates/EXPENSIVE_STUDY_PREFLIGHT_REPORT_TEMPLATE.yaml`; and
- `doc/audits/templates/EXPENSIVE_EXECUTION_READINESS_CERTIFICATE_TEMPLATE.yaml`.

The documentary preregistration and review aids are
`doc/audits/templates/COSTLY_NUMERICAL_STUDY_PREREGISTRATION_CHECKLIST.md` and
`doc/audits/templates/EXPENSIVE_STUDY_INDEPENDENT_REVIEW_CHECKLIST_TEMPLATE.md`.

`tools/audits/validate_expensive_study_controls.py` is the mechanical launch
gate. Schema validity alone is not authorization; the command must also pass
with `--launch-gate` against the exact frozen register, preflight report, and
readiness certificate.

## Condition taxonomy

Every registered condition has exactly one class.

### A. Exact identity and integrity

This class protects byte or discrete identity: hashes, source/file identity,
schema, required fields, row counts, cardinality, provenance binding,
unsupported domains, malformed records, and enumerated values. Exactness is
appropriate only when the compared objects actually have an exact identity.
It does not authorize exact equality between mathematically equivalent
floating-point construction paths.

Class A may stop before expensive execution or invalidate precisely scoped
evidence when the identity it protects is necessary. Its register entry still
states the failure scope and salvage consequence.

### B. Derived numerical correctness

This class protects an analytic or algorithmic numerical invariant. An
aborting Class B condition requires either an analytic bound, conditioning
analysis, interval argument, or ULP analysis tied to the operations and
representation actually used, or a quantified propagation from the numerical
discrepancy to every affected final scientific metric. Arbitrary absolute or
relative epsilons are prohibited.

If neither a direct bound nor a final-metric propagation can be established,
the condition is a Class D diagnostic. More digits or a smaller literal do not
make a check more rigorous.

### C. Scientific acceptance

This class tests the owner-approved scientific contract. Its metric, support,
comparison, and threshold must trace directly to a named scientific decision
or approved package contract. A Class C failure is a valid scientific result,
not corrupt evidence and not a reason to repeat a conforming calculation.

Scientific acceptance should normally be evaluated after the raw model stage
is complete. It must not be hidden inside a generator as an early abort when
the raw outputs remain scientifically interpretable.

### D. Engineering diagnostic

This class records implementation consistency, suspicious numerical state,
performance, or defensive observations that do not independently establish
scientific invalidity or evidence corruption. It warns and records by default.

A condition that remains Class D is warning-only. If an explicit impact
analysis establishes that the condition can corrupt raw evidence, violate a
required identity, exceed a derived numerical-correctness bound, or test the
owner-approved scientific contract, the frozen register must reclassify it as
Class A, B, or C respectively and satisfy that class's derivation, action, and
approval rules. This explicit reclassification is how a former diagnostic may
gain non-warning authority; a source-code assertion or an impact claim left in
Class D is never sufficient.

## Required comparison and impact record

The register records every condition capable of stopping, invalidating, or
scientifically failing the study. It also records a diagnostic when its
implemented source route can stop, when it participates in raw-evidence
admission, or when the framework review explicitly demotes it from veto power
to warning. Ordinary informational/logging warnings with no evidence or
control consequence need not be registered.

- stable condition ID and source location(s);
- the action actually implemented by each source route, independently of the
  proposed registered action;
- quantity, units, expected scale, and reference identity;
- exact, absolute, relative, ULP, interval, set, schema, or cardinality
  semantics;
- the canonical threshold literal and its representation (`binary32`,
  `binary64`, decimal, integer ULP, or exact/discrete), without an intervening
  lossy JSON-number conversion;
- condition class and protected scientific or integrity requirement;
- threshold derivation and its authority;
- arithmetic model, precision, rounding, operation order, and conditioning;
- maximum propagated effect on each affected final scientific metric, or an
  exact-integrity or derived-correctness not-applicable rationale;
- failure mode prevented, affected validity layer, precisely invalidated
  artifact/stage scope, effect on already written raw output, and salvage
  consequence;
- whether inputs are data-dependent;
- when data-dependent, why the inputs are unavailable before execution and the
  synthetic or fault-injection test that exercises the route;
- earliest stage at which the check can be exercised;
- preflight test and required tuple/boundary/branch coverage;
- estimated computational cost exposed if the condition fires; and
- required scientific-owner, audit-manager, integrity-owner, or
  numerical-methods-owner approval.

For every aborting numerical tolerance, one of these is mandatory:

1. a derived bound showing why the tolerance is required for numerical
   correctness; or
2. a quantified mapping from that error into the final scientific metric.

If neither exists, the action is `warning`. Do not replace a rejected epsilon
with a new framework-wide epsilon.

## Action semantics

The allowed actions are deliberately distinct:

- `hard_stop`: stop the affected execution stage. It does not retroactively
  invalidate already written raw artifacts unless the register says why those
  artifacts share the failed integrity or numerical defect.
- `invalid_evidence`: mark the precisely identified artifact or stage invalid.
  Other artifacts retain an independent validity state.
- `scientific_failure`: complete, valid evidence does not meet the scientific
  acceptance gate. Preserve it as a scientific result.
- `warning`: continue, record the condition, operands, branch, and impact
  disposition.

A check present only in source code and absent from the frozen register has no
invalidating authority. If such a check fires, execution may halt safely for
triage, but the event is a harness-governance defect. Raw evidence is then
assessed under the salvage policy; it is not automatically discarded.

Approval roles are controlled. Class C requires the scientific owner. A
non-warning Class A condition requires the audit manager or named integrity
owner; a non-warning Class B condition requires the audit manager or named
numerical-methods owner. Warning conditions use `not_required`. The approval
record identifies the responsible person and exact decision; a free-form role
label cannot confer authority.

## Mandatory guard preflight

Every costly-study harness must provide a dry-run or preflight mode that
performs no scientific model calculation. Before execution it must:

1. enumerate every frozen case and tuple;
2. exercise every non-data-dependent condition over all preregistered tuples;
3. exercise every known constant, Decimal/binary conversion, coordinate
   transformation, candidate dispatch, boundary neighbor, and output-format
   path;
4. report stable condition IDs and taken branches;
5. compare every source/preflight-discovered route ID and implemented action
   with the frozen registered action;
6. report complete tuple, boundary, condition, and branch coverage; and
7. fail if any abort-capable guard is unregistered or any deterministic guard
   remains unexercised.

Every harness created or revised after this policy takes effect must route
abort-capable checks through one condition-ID dispatcher so an unknown ID or
action mismatch is mechanically rejected. Unchanged legacy source may use a
complete static guard inventory only through an explicit manager-approved
legacy exception; the inventory includes each route's implemented action and
is bound in the preflight report. Relabeling a source `require` or exception as
a registered warning does not change its implemented action and fails the
gate. No deterministic guard whose inputs are already known may first become
reachable during model execution. A `data_dependent` label must identify the
unavailable input and stage, carry independent-review acknowledgement, and
bind a synthetic or fault-injection test; it is not an exemption from guard
coverage.

For a package audit, the source-level guard census follows the independent-core
freeze and respects the same pre-core/post-core quarantine as the rest of
implementation inspection. Framework planning before that freeze may require
the census as a future deliverable but may not expose quarantined package
source to the independent derivation.

## Execution-readiness certificate

The audit manager owns the cost-proportional certificate. A `ready` decision
attests that:

- all aborting and invalidating conditions are registered and justified;
- all deterministic conditions have complete frozen-tuple and boundary
  coverage;
- no engineering diagnostic has silently acquired scientific veto power;
- raw model execution and downstream evaluation have separate identities and
  validity states;
- remaining data-dependent guards and their exposed cost are understood;
- the salvage policy is bound before execution;
- the exact register, protocol, runner, preflight report, case set, and
  evaluator are digest-bound; and
- an independent reviewer found no unregistered hard guard or unsupported
  scientific threshold.

The manager may deny readiness without changing the scientific protocol. A
`ready` certificate establishes readiness only for the exact execution named
by its digests and scope; it is not itself launch authorization. Any later
authorization can refer only to that exact scope. The certificate does not
authorize application repair, production, Unity access, a different candidate,
or a re-audit.

## Evidence-salvage policy

The standalone reusable policy is
`doc/audits/EVIDENCE_SALVAGE_POLICY.md`; the rules below are its mandatory
framework summary.

Raw scientific calculation and downstream evaluation always have separate
validity states and versioned provenance.

An evaluator or harness defect does not invalidate raw model output unless a
review establishes that it affected at least one of:

- model inputs or their ordering;
- executable/model identity;
- execution conditions that affect the model;
- raw-output completeness for the claimed tuple;
- admitted warning/error policy;
- parsing or numeric payload;
- hashes, sidecar binding, or provenance; or
- independence of the scientific test.

If those properties remain valid, a successor protocol may reuse the raw
output. It must:

1. preserve the failed study and its original decision record unchanged;
2. create a new protocol, evaluator, and result identity;
3. bind every reused raw artifact and its original execution context by
   digest;
4. state exactly which artifacts are reused and which computation is missing;
5. evaluate reused and newly generated raw evidence through one frozen
   successor evaluator;
6. verify that the reuse authority is durable and read-only; when the only
   live cache is temporary or writable, make and independently hash a named
   preserved copy before admission, then place new evidence in a distinct
   delta cache or release;
7. declare the reuse in the final result; and
8. independently verify the union before scientific evaluation.

Preregistration independence is preserved only when no partial decisive
scientific metric, ranking, maximum-error location, or candidate comparison is
examined before the successor evaluator is frozen. Operational inspection of
hashes, completeness, warning admission, and the harness failure itself is
permitted. If decisive partial metrics were inspected, the successor must
state the resulting loss of confirmatory independence and either change to a
descriptive study or obtain a separately justified design.

## Audit-manager workflow

For every proposed costly study the audit manager shall:

1. classify cost and record the estimate;
2. freeze scientific cases, candidates, metrics, and gates;
3. require the complete condition register before model execution;
4. run the model-free preflight across the frozen case set;
5. obtain independent review;
6. sign a readiness certificate binding the accepted independent review;
7. run the mechanical launch gate against the exact final certificate;
8. preserve raw outputs independently of evaluator products; and
9. stop after execution for coordinator review when the governing audit
   protocol requires it.

If a costly study stops, the manager first performs a salvage assessment. A
fresh full cache is not the default. Discarding valid raw evidence requires a
specific scientific, provenance, completeness, warning-admission, parsing, or
independence reason.

## Independent review checklist

The independent reviewer shall verify at minimum:

- source-level inventory of every abort, assertion, exception, and invalidation
  route in the study harness and evaluator;
- one-to-one registration of every route with invalidating authority;
- threshold derivations and impact propagation;
- complete model-free reachability of deterministic guards;
- exact frozen-tuple, boundary, conversion, dispatch, and formatting coverage;
- separation and digest binding of generator, raw output, parser, evaluator,
  and scientific decision;
- action classification under A--D;
- no condition remaining Class D with veto power;
- cost exposed by remaining data-dependent conditions; and
- a concrete salvage path that preserves preregistration independence.

The reusable checklist is
`doc/audits/templates/EXPENSIVE_STUDY_INDEPENDENT_REVIEW_CHECKLIST_TEMPLATE.md`.

## SCI-CAL-001 application

The current incident is dispositioned in
`doc/audits/packages/SCI-CAL-001_EL25_RAW_EVIDENCE_DISPOSITION_2026-08-02.md`.
That record preserves the stopped study, treats the 672 completed grids as
salvageable raw evidence subject to its exact integrity checks, and recommends
the documented successor harness/evaluator change as the only admissible
correction pending separate authorization. It does not authorize the
replacement execution. The one-percent representation-fidelity gate and every
scientific tuple, candidate, passband, and domain remain frozen.

This amendment does not implement or authorize the separately held
composition-closure changes `FRAMEWORK-COMP-D005` or
`FRAMEWORK-COMP-D006`.

## Enforcement and transition

This policy applies immediately to:

- every new costly audit study;
- every next expensive execution of an active audit;
- every replacement or resumed execution after a harness/evaluator failure;
  and
- every external evidence request classified as costly by its manager.

Existing completed evidence is not retroactively invalidated. Existing
unexecuted requests must satisfy this policy before launch. Existing active
executions are not mutated mid-run; their next expensive stage requires a
certificate.

Failure of the framework launch gate is a governance stop, not a scientific
result. It may not be bypassed by editing a source literal, relabeling a study
as cheap, or treating manager confidence as threshold derivation.
