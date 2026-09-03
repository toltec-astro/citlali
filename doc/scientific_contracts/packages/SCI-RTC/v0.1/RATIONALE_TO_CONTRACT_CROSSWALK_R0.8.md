# SCI-RTC v0.1/r0.8 Rationale-to-Contract Crosswalk

Status: bounded Decision 9 review aid; `CROSSWALK.md` remains the exact
ID-level authority record.

| Decision 9 element | Science-team rationale | Shared formal authority |
| --- | --- | --- |
| Additive baseline only; no gain/responsivity change | §5 | DEF-035, DEF-038; EQ-033; ASM explicit non-goals; REQ-101, REQ-107; PRED-059, PRED-070 |
| Finite transition width in physical time | §5 | DEF-036; EQ-033; REQ-094, REQ-096; PRED-055 |
| Sample support from timing vector, cadence/scan independent | §5 | DEF-036; EQ-033; REQ-096; PRED-055 |
| Transition unmodeled, excluded, and explicitly flagged | §§5, 10 | DEF-036--037; EQ-033--034; REQ-096, REQ-098, REQ-101--102; PRED-059 |
| Physical transition support distinct from propagated influence | §§5, 10 | DEF-036; REQ-096, REQ-102; PRED-055, PRED-057 |
| Optional additive correction on stable plateaus | §5 | DEF-035, DEF-037--038; EQ-033--034; REQ-013, REQ-098, REQ-101; PRED-059 |
| Insufficient support: no invented offset; boundary retained | §5 | EQ-034; REQ-101, REQ-106; PRED-071 |
| Compact normal-run state and population summaries | §§5, 10 | REQ-050--051, REQ-094, REQ-102 |

The rationale contains explanatory prose only. The companion Engineering
Conformance PDF imports the six-file formal authority exactly once.
