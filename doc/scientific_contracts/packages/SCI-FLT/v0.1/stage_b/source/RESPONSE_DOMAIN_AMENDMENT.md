# SCI-FLT-FIXED v0.1 Full-Procedure Response-Domain Amendment

Record identity: `SCI-FLT-FIXED-RESPONSE-DOMAIN-AMENDMENT v0.1/freeze-candidate`

Status: implementation-blind Stage B closure draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## Exact domain requirement

```text
Delta y_parent-FP = A_Theta,J Delta m_parent-FP.
```

Baseline and perturbed parent-procedure products must define one exact
compatible parent-grid difference on the frozen FLT domain. A change in row
membership, availability, WCS, quantity, or support that makes any required
difference undefined has three effects: `J_full` remains frozen, the parent
state-change record remains visible, and affected transformed response rows
are unavailable. Common array shape alone does not establish compatibility.

## Preserved response boundaries

Fixed-state, already-realized parent-grid, parent full-procedure with FLT fixed,
and FLT-re-resolved procedure response remain distinct families. Every admitted
composition applies the identical `A_Theta,J` exactly once.
