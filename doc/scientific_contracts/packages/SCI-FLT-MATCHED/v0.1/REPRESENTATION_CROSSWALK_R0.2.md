# SCI-FLT-MATCHED v0.1 r0.2 Science/Representation Crosswalk

Status: proposed Stage B r0.2 distinction; all affected routes unselected

| Scientific object/invariant | Scientific scope or query decision | Exact representation choices | Identity rule |
| --- | --- | --- | --- |
| Template response per unit amplitude | Source-domain reference, parent response, anchor relation, sampling/phase/support, CAL/BEAM lineage, validity, generation, supported queries | materialized; exact structured; exact lineage/query reconstruction | Representation-only change does not alter template or amplitude estimand. |
| Conditional covariance of actual field | `AO-003-A` complete; `AO-003-B` named projected; `AO-003-C` unavailable | `AO-003-D` resident explicit or exact structured; `AO-003-E` exact lineage/on-demand | Equivalent representations retain one covariance identity. Changing complete/projected/unavailable scope changes scientific availability. |
| Resolved immutable state | Exact state identity and required query vocabulary | `AO-004-A` full; `AO-004-B` compact exact; `AO-004-C` lineage reconstruction | Representation-only change does not change estimator or signal generation. |
| Typed response object | Fixed/FP/realized/reference type, anchor domains, query vocabulary, validity, units/calibration, and consumers | `AO-005-A` materialized; `AO-005-B` exact structured; `AO-005-C` lineage/on-demand | Representation-only change preserves response identity and query answers. A different query domain or response type changes science availability. |
| SCI-VAL verdict policy | PA/SA/SP/CU/NU/RU/FH meanings, four axes, dependency graph, actions, versions | `AO-006-A` separate records; `AO-006-B` grouped records; `AO-006-C` seven-role vector | Layout is lossless only. It cannot change policy or collapse roles into an actionable scalar. |
| Numerical realization | Exact `n_p/d_p` science and realized/reference distinction | One preregistered engineering profile for algorithm/environment/precision | Profile or representation pass is not scientific authority. An intentional scientific operator discrepancy gets distinct identity. |

Storage footprint is `S_store` and has no scientific authority. Every exact
representation advertises a finite query vocabulary; an unsupported query is
typed unavailable rather than approximated or returned as zero.
