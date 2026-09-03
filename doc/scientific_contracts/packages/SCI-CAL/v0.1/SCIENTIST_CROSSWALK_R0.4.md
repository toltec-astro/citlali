# SCI-CAL v0.1 Rationale r0.4 Scientist/Engineering Crosswalk

Status: refreshed for engineering conformance r0.3; the engineering contract
remains normative and the science rationale is r0.4

## Assumptions

| IDs | Scientist-facing location | Engineering obligation retained |
| --- | --- | --- |
| ASM-001, ASM-002 | Input and physical-model sections; Q01--Q02 | Approved physical-input authority, support, ordinary channel, and binding |
| ASM-003 | Calibrator-to-target section and Figure 1; Q03--Q04 | Producer-owned source APT, TolProj-selected immutable child APT, immutable TolTECA delivery, and association |
| ASM-004 | Input section and formal appendix | Explicit affine convention |
| ASM-005 | Physical model and factor table; Q03--Q05 | Complete generating record, child-transform/transfer domain, factor direction, plane, and once-only lineage |
| ASM-006 | Atmosphere section; Q06--Q07 | Atmosphere inputs, time support, policy |
| ASM-007 | Uncertainty section | Fixed conditional-state covariance |
| ASM-008 | Response section | Originating/downstream response ownership |
| ASM-009 | Atmosphere and validity sections; Q07 | Coherent segment quality |
| ASM-010 | Atmosphere/response sections; Q05 | Passband limitations and source-factor/atmosphere/output compatibility state |
| ASM-011 | Atmosphere and decision sections; Q06 | Unresolved numeric operator |

## Requirements

| IDs | Scientist-facing location | Engineering obligation retained |
| --- | --- | --- |
| REQ-001--REQ-005 | Input, validity, and Q01--Q09 | Scope, typed state, authority snapshot, claim-specific availability, and one-way resolution |
| REQ-006--REQ-012 | Calibrator-to-target section and Figure 1 | Acquisition/source-APT/child-APT identity, immutable delivery, binding, association, and lifetime |
| REQ-013--REQ-016 | Physical model and factor section; Q03--Q04 | Complete source generating record, approved child transform/transfer domain, pointing disposition, once-only multiplier |
| REQ-017--REQ-020 | Factor-role table and producer--transformer--delivery--consumer explanation | Relative roles, compatibility decomposition, ownership/delivery separation, and package lineage |
| REQ-021--REQ-026 | Atmosphere and formal appendix | Atmosphere meaning, time/operator support, invariants |
| REQ-027--REQ-031 | Atmosphere and validity; Q05--Q07 | Opacity policy, coherent quality, passband limits, spectral compatibility, and Q06-only closure scope |
| REQ-032--REQ-034 | Uncertainty | Conditional covariance, variance, weight, unavailable state |
| REQ-035--REQ-039 | Physical model, uncertainty, and formal appendix; Q02 | Nuisance scopes, total covariance, companions, and local-versus-end-to-end response |
| REQ-040--REQ-044 | Calibrator-to-target and response sections | Producer-owned source/child beam-template basis, realized response, unit exclusions |
| REQ-045--REQ-048 | Calibrator-to-target, validity, and machine-state appendix; Q01--Q09 | Exact claim-specific states, reasons, complete authority/ownership/delivery lineage, and product links |
| REQ-049, REQ-050 | Response and validation; Q01--Q09 | Separate input/factor, transfer, broadband, response, structural, fidelity, and performance claims with preregistered criteria |

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

The r0.4 PDF appendix carries the stable grouped ID map. The detailed
one-row-per-requirement crosswalk remains in CROSSWALK.md. Both views state the
producer--transformer--delivery--consumer boundary, distinguish TolTECA
delivery from scientific ownership, and apply every Q01--Q09 consequence
without resolving it. No stable formal ID is added, removed, or renumbered.
The PDF's Table 4 decision snapshot is checked against
SCIENTIFIC_OWNER_DECISION_LEDGER.md.
