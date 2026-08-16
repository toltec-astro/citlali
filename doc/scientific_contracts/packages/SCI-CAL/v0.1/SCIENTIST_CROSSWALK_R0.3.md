# SCI-CAL v0.1 Rationale r0.3 Scientist/Engineering Crosswalk

Status: complete grouped routing of all formal items; the unchanged
engineering contract remains normative

## Assumptions

| IDs | Scientist-facing location | Engineering obligation retained |
| --- | --- | --- |
| ASM-001, ASM-002 | Input and physical-model sections; Q01--Q02 | Input support, ordinary channel, and binding |
| ASM-003 | Calibrator-to-target section; Q03--Q04 | Selected immutable child APT, source ancestry, and association |
| ASM-004 | Input section and formal appendix | Explicit affine convention |
| ASM-005 | Physical model and factor table | Factor direction, plane, and once-only lineage |
| ASM-006 | Atmosphere section; Q06--Q07 | Atmosphere inputs, time support, policy |
| ASM-007 | Uncertainty section | Fixed conditional-state covariance |
| ASM-008 | Response section | Originating/downstream response ownership |
| ASM-009 | Atmosphere and validity sections; Q07 | Coherent segment quality |
| ASM-010 | Atmosphere/response sections; Q05 | Passband limitations |
| ASM-011 | Atmosphere and decision sections; Q06 | Unresolved numeric operator |

## Requirements

| IDs | Scientist-facing location | Engineering obligation retained |
| --- | --- | --- |
| REQ-001--REQ-005 | Input and validity | Scope, typed state, validity, one-way resolution |
| REQ-006--REQ-012 | Calibrator-to-target | Acquisition/APT identity, binding, association, lifetime |
| REQ-013--REQ-016 | Physical model and factor section | Absolute factor, pointing disposition, once-only multiplier |
| REQ-017--REQ-020 | Factor-role table | Relative roles, compatibility decomposition, lineage |
| REQ-021--REQ-026 | Atmosphere and formal appendix | Atmosphere meaning, time/operator support, invariants |
| REQ-027--REQ-031 | Atmosphere and validity; Q05--Q07 | Opacity policy, coherent quality, passband limits |
| REQ-032--REQ-034 | Uncertainty | Conditional covariance, variance, weight, unavailable state |
| REQ-035--REQ-039 | Uncertainty and formal appendix | Nuisance scopes, total covariance, companions |
| REQ-040--REQ-044 | Response | Beam/template basis, realized response, unit exclusions |
| REQ-045--REQ-048 | Validity and machine-state appendix | Exact states, reasons, lineage, product links |
| REQ-049, REQ-050 | Validation; Q09 | Separate claim layers and preregistered criteria |

## Edge Predictions

| IDs | Scientist-facing location | Engineering test family retained |
| --- | --- | --- |
| EDGE-001--EDGE-005 | Atmosphere and formal appendix | Zero, monotonicity, zenith, exact-node, seam checks |
| EDGE-006--EDGE-010 | Atmosphere and validity | Domain, bracketing, engineering/outside states |
| EDGE-011--EDGE-013 | Factor section | Unity, omitted/duplicate/inverted, corrected-APT challenges |
| EDGE-014--EDGE-019 | Factor association | Row/order/network, missing/duplicate, association/design tests |
| EDGE-020--EDGE-024 | Uncertainty | Scalar, dense, unavailable, common, nonlinear uncertainty |
| EDGE-025--EDGE-027 | Response | Unit-peak source, response-changing kernel, beam approximation |
| EDGE-028 | Input and validity | Unsupported stream or target |
| EDGE-029 | Input and formal appendix | Zero signal and affine convention |
| EDGE-030 | Factor association and validity | Observation-order state replay |

The PDF appendix carries the same map. The detailed one-row-per-requirement
draft crosswalk remains in CROSSWALK.md. The r0.3 ownership clarification
changes the scientific producer/transformer/consumer explanation; it does not
delete or weaken any engineering requirement. The PDF's Table 4 decision
snapshot is checked against SCIENTIFIC_OWNER_DECISION_LEDGER.md.
