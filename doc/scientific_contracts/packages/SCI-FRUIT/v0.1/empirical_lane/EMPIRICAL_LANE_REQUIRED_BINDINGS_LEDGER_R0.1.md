# SCI-FRUIT v0.1 — Empirical-Lane Required Bindings Ledger r0.1

Status: **owner-review ledger; missing values remain unavailable rather than
defaulted**

## Readiness Against The Accepted Nine Decisions

| ID | Required binding | Present state | `EL-G0` deliverable | Gate before use |
| --- | --- | --- | --- | --- |
| `ELB-001` | lane lifecycle and gate authority | candidate gate architecture exists | exact role, access, review, incident, stop, and supersession plan | G0 approval; every later gate separately |
| `ELB-002` | exact historical-control build, configuration, parent route, grouping, stopping, and paired protocol | recovered recurrence and typed role only; exact executable control profile unavailable | `HISTORICAL_CONTROL_ID` candidate with reproducibility check and known limitations | GD for development use; GF for qualification freeze |
| `ELB-003` | generic profiles, applicability domains, and downstream exclusions | two motivating generic names; no profile approved | exact profile candidates, pre-output selection facts, domains, exclusions, and no-profile behavior | GD; exact claims freeze at GF |
| `ELB-004` | inference target, population construction, strata, and immutable splits | unavailable | lineage-safe inventory and proposed development/qualification/challenge split under a named custodian; no outcome opening | GD for development split; GF/Q for held-out split |
| `ELB-005` | metrics, truth/null strategy, support, pairing, uncertainty, multiplicity, tail/outcome/failure/unavailable/catastrophic rules | candidate-neutral skeleton and safeguards only | exact metric/protocol registration candidates and feasibility plan | estimands before GD; full decision rules at GF |
| `ELB-006` | protected/prioritized dimensions, bands, thresholds, and credibility construction | framework approved; numerical values unavailable | candidate-neutral priorities plus development procedure for values not knowable a priori | priorities before GD; final values at GF |
| `ELB-007` | candidate-family, submission/access/unblinding, specialization, adaptation, stopping, override, and generation policies | boundaries approved; finite family and policies unavailable | bounded hypothesis family, maximum submissions, causal policy families, deadline, and generation rules | GD; exact candidate set at GF |
| `ELB-008` | out-of-domain, fallback/unavailability, no-replacement, narrowing, and evidence-combination policies | typed roles and general rules accepted; exact routes unavailable | exact decision table preserving no-replacement and typed unavailability | GD for development interpretation; GF for claims |
| `ELB-009` | repository/branch, inputs, outputs, provenance, cadence, resources, and stop rule | accepted branch known; lane generation and resources unavailable | exact proposed lane branch/ref, environment capture, product/retention list, compute/storage limits, cadence, and abort conditions | G0 preflight; GD/Q re-bound separately |

## Additional Gate-D Admission Checklist

An `EL-GD` candidate is not reviewable until all of the following are exact:

1. one `LANE_GATE_ID` and responsible scientific, data-custodian, execution,
   and review roles;
2. a reproducible `HISTORICAL_CONTROL_ID`, or a typed no-go if exact control
   construction fails;
3. proposed profile and applicability-domain records;
4. an immutable lineage inventory and exact development population;
5. candidate-neutral metric estimands, signs, units, support, pairing, and
   failure semantics fixed before tuning;
6. a finite candidate-family and hypothesis/resource envelope;
7. an exact development-only access control and contamination procedure;
8. a repository, branch, software/environment, provenance, output, retention,
   checkpoint, compute, storage, and wall-time plan; and
9. an owner-review record that states what development can and cannot decide.

## Qualification Admission Checklist

Before `EL-GQ`, `EL-GF` must additionally freeze exact `METHOD_ID`, `CLAIM_ID`,
candidate and hypothesis set, qualification/challenge populations, all metric
and decision equations, materiality/non-inferiority bands, uncertainty and
multiplicity, support and outcome accounting, terminal policies, resource
caps, label access, unblinding event, monitoring, reuse, replacement-population
rule, and deviation handling.

No item may be supplied after its protected outcome is visible and then
backdated into the frozen registration.
