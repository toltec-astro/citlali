# Semantic change map — r0.5 to r0.6

Status: final frozen micro-repair traceability

| Area | r0.5 defect | r0.6 closure |
| --- | --- | --- |
| parent typing | one symbol could denote stochastic vector and partial observed payload | distinct `D_model/M` and `D_m/m_obs` objects |
| GLS domain | exact-zero observed dependency could obscure model authority | all `D_loc` coordinates require model/covariance authority for A |
| conditioning | parent identity could be misread as fixing the draw | `h_pre` explicitly excludes observed values/digests and draw-dependent facts |
| lifecycle | candidate/decision preceded realization in the listed order | Applied -> Realized -> Complete -> PublicationDecided -> Published/NotProduced |
| AO multiplicity | authorization could read as one global A-versus-C choice | multiple package authorizations allowed; one exact per realization |
| title | recommendation remained implicit in provisional rendering | option 1, “Matched-template map amplitude estimation,” selected and rendered |
| role records | files were called profiles without policy/Registry closure | active r0.6 role meanings are frozen semantics; Registry registration and evaluability remain unavailable |
| bundle | historical/repository links and Stage A verifier were not standalone | authority-only bundle policy B, zero unresolved local links, standalone active verifier |

REQ-001 through REQ-050 and PRED-001 through PRED-025 retain their identities.
No new obligation required REQ-051 or PRED-026.
