# SCI-RTC v0.1/r0.4 consistency report

Status: implementation-blind contract consistency review; not validation.

| Required invariant | Rationale | Formal contract | Result |
| --- | --- | --- | --- |
| Attempt and accepted-plan indices are distinct | §2 | DEF-027--029, EQ-029, REQ-071/073/080--081 | Consistent |
| Initial evaluation is defined for identity or fixed `Pi_0` | §2 | EQ-029 | Consistent |
| Rejection creates no accepted plan or evaluation | §§2/5 | EQ-029, REQ-073/080--081, PRED-039/043 | Consistent |
| Phase-zero selection consumes final pre-decimation stream | §8 | DEF-009, EQ-011, REQ-028--030 | Consistent |
| `x`/`xs` always denotes raw detector `Delta f/f` | §§1/10 | DEF-001--002, REQ-001--004 | Consistent |
| RTC completes before downstream SCI-CAL | §10 | DEF-004, EQ-003--005, REQ-005/013, PRED-005 | Consistent |
| Donor `flxscale` ratio is not absolute calibration | §§4/10 | EQ-001--002/004, REQ-014--015 | Consistent |
| Direct synthesis/replacement is universally excluded | §§4/11 | DEF-013, EQ-020b, REQ-019--020 | Consistent |
| Noncenter influence is cause-preserving consumer input | §§4/11 | EQ-020a--020b, REQ-019--020/052, PRED-018 | Consistent |
| Role paths do not inherit one another's plans | §1 matrix | DEF-002, REQ-002/012 | Consistent |

The r0.3 iterative-refinement architecture, cumulative-plan requirements,
original-input replay, and finite stopping remain unchanged except for the
attempt/accepted-plan correction. Numerical owner choices remain open.
Implementation conformity, representation fidelity, observational
performance, science qualification, validation, and production readiness are
not assessed.
