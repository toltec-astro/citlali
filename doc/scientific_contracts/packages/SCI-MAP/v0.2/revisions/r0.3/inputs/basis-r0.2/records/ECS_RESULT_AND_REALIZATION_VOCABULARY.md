# SCI-MAP v0.2/r0.2 ECS result and realization vocabulary

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

## Initial candidate state and verdict

All 52 placeholders in this author package are `not_assessed` with
evidence-artifact realization `not_produced`. A final conformance verdict is
`pass` only when every applicable requirement is `pass`, every
`not_applicable` result has its exact owner-approved rationale, and no result
is `fail`, `blocked`, or `not_assessed`. Any `fail` makes the exact candidate
fail. Any `blocked` or `not_assessed` prevents a pass. This prospective rule
does not evaluate the candidate.

