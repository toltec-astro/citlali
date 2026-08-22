# SCI-AST v0.1 Stage B r0.3 Formal Parity Report

Status: implementation-blind document-coherence check only.

| Required parity item | r0.3 result | Authority traced |
| --- | --- | --- |
| `s/j/n/d/p` meanings | present and non-aliased | notation and role parents; REQ-006, 073-075 |
| `x/r` reservation | paired KID readout only | notation; immutable ALIGN parent; REQ-010-012 |
| circular interval `[-P/2,P/2)` | exact in narrative and formal view | `wrap`; REQ-038; PRED-018 |
| role-factored AST parents | direction/tangent/pixel/nominal/RTC layers preserved | parent equations; REQ-056-060, 073-077 |
| complete spherical oracle | full tangent vector, norm, unit direction, exponential map, and zero branch | `exp`; REQ-035-036; PRED-032, 045 |
| RTC-grid coordinate ownership | `SCI-AST:rtc_output_grid_coordinates@1` covers every ordinary RTC-grid science product | REQ-074-079; PRED-035-038 |
| no angular signal filtering | explicit | `not-angular-filter`; REQ-078-079; PRED-037 |
| no double field rotation | representation and embodied-operation counts required | REQ-024, 034; PRED-014-016 |
| `G_pi` and MAP ownership | MAP deposition authority retained; base AST has no general stencil | REQ-080-083; PRED-038-040 |
| physical acquired vs valid-original exposure | imported without reinterpretation from exact shared boundary | ALIGN EQ-018 and boundary table |
| exact occurrence time vs observing-state fields | imported exact `t_s`; state evaluated/mapped at that time | REQ-021 and shared boundary |
| typed unavailable states | role-local and nonnumeric | notation; REQ-056-060, 074-077 |
| stable identifier preservation | 90 requirements, 50 predictions, and 15 assumptions in exact sequence | durable verifier |

This report does not establish conformity, validation, empirical adequacy,
scientific freeze, readiness, observational performance, or production use.
