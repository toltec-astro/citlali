# SCI-MAP v0.2/r0.4 ECS result and realization vocabulary

Status: **CANDIDATE prospective-evidence specification**. This record asserts
no conformance result and no evidence-artifact realization.

## Requirement-result axis

Every `SCI-MAP-REQ-NNN` record has exactly one result:

| Exact value | Meaning |
| --- | --- |
| `pass` | Complete positive evidence establishes the requirement for the exact candidate. |
| `fail` | Evidence was evaluated and contradicts the requirement. |
| `blocked` | Evaluation cannot proceed because one exact named authority, dependency, profile, source binding, or prerequisite is unavailable. |
| `not_applicable` | The requirement is outside the exact candidate's declared role, with an owner-approved rationale. |
| `not_assessed` | Applicable evidence has not yet been executed or reviewed. |

Skipped required evidence is `not_assessed`, never `pass`. `incomplete` is not
a requirement result. `UNASSESSED` is not an independent value.

## Evidence-artifact realization axis

Every evidence record separately states the realization of its evidence
artifact:

| Exact value | Meaning |
| --- | --- |
| `complete` | The declared artifact was produced in full and is immutable/addressable. |
| `incomplete` | Some declared artifact content is absent. |
| `failed` | Artifact production was attempted and terminated in failure. |
| `not_produced` | No artifact-production attempt occurred. |

Artifact realization never substitutes for the requirement result. For
example, a complete artifact can establish `pass` or `fail`; an unreviewed
complete artifact remains `not_assessed`; an unavailable exact prerequisite
can yield `blocked` while the evidence artifact is `not_produced`.

## MAP product and attempt axes

These ECS fields also remain separate from both axes above:

| Axis | Exact values or rule |
| --- | --- |
| MAP application-attempt outcome | `succeeded`, `failed`, or `not_produced`; absent before application |
| MAP product realization | Exact immutable product identity only for an actual realized product; otherwise absent |
| Failure evidence | Applied-attempt identity, failure stage, cause, affected scope, and nonmutation evidence for a failed attempt; never a MAP product |
| Companion availability | Exact response/covariance status and cause within a permitted base bundle; `unavailable` is not producer failure |

The owner-directed empty-support attempt uses `not_produced`, has no product
realization, and records cause `no_support_authorized_output_rows`.

## Candidate profile status axes

For the r0.4 MAP/VAL generation, profile semantics are `candidate`; owner
approval, Registry registration, and source binding are each `pending`;
activation/evaluability is `unavailable`; and object-specific decision
realization is `unavailable`. These are not requirement results or artifact
states, and this package asserts no profile evaluation.

## Initial candidate state and verdict

All 52 placeholders in this author package are `not_assessed` with
evidence-artifact realization `not_produced`. A final conformance verdict is
`pass` only when every applicable requirement is `pass`, every
`not_applicable` result has its exact owner-approved rationale, and no result
is `fail`, `blocked`, or `not_assessed`. Any `fail` makes the exact candidate
fail. Any `blocked` or `not_assessed` prevents a pass. This prospective rule
does not evaluate the candidate.
