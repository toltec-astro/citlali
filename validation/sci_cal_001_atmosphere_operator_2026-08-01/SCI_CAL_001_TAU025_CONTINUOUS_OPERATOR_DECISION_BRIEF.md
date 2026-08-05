# SCI-CAL-001 TAU025 continuous-operator decision brief

Frozen D007 evidence was evaluated read-only from cache manifest `932ab48ec5d79e1455f970b576e177f2a3673f7108a79b1e9fa2f86db9905208`. The held-out table has 18000 rows across 25 profile identities, three TolTECA v1 passbands, and alpha={-1,0,2,4}.

Recommendation for owner decision: select `piecewise_linear_los_tau_v1` only as a **profile-identified evaluation operator** over `0.15 <= tau225 <= 0.25`, `25 <= EL <= 80`. It is fail-closed outside that domain or without one of the frozen AM profile identities. Its maximum held-out fractional correction error is `0.532005%` (p95 `0.369984%`, RMS `0.143272%`); exact-node LOS-tau error is `5.551e-17`.

The requested owner choice is whether this profile-conditioned numerical representation, including its 0 opacity and 0 sampled monotonicity violations, is acceptable for an engineering-only versioned operator. No generic profile selector was tested or inferred. This result is representation fidelity only; it does not establish observational 5--10% absolute flux accuracy or approximately 5% repeatability, and does not authorize implementation or production use.
