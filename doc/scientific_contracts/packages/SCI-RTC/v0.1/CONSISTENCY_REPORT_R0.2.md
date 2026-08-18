# SCI-RTC v0.1/r0.2 consistency report

Status: implementation-blind internal consistency review; no conformity or
validation result.

| Required topic | Rationale | Formal authority | Ledger / unavailable behavior | Result |
| --- | --- | --- | --- | --- |
| Learn–resolve–apply lifecycle | §§1--2 | DEF-021--025, EQ-028, REQ-055--058 | OWNER-029; adaptive branch OWNER-027 deferred | Consistent |
| Operation definitions | §§4--9 | DEF-006, DEF-008, DEF-026, REQ-059 | operation-specific owner entries | Consistent |
| Notch width and adaptation | §5 | EQ-024, REQ-061, PRED-030 | OWNER-030; OWNER-027 deferred | Consistent |
| Low/high/band-pass meaning | §6 | REQ-062, PRED-033 | OWNER-031 | Consistent |
| FIR order and tap count | §7 | EQ-025--026, REQ-063, PRED-032 | OWNER-033 | Consistent |
| Donor definition and limits | §4 | DEF-005, EQ-002/007, REQ-064--065 | OWNER-003--005; circular factor forbidden | Consistent |
| Decimation and aliasing | §8 | EQ-011/014/021/023, REQ-028--036/066 | OWNER-002, OWNER-011--020 | Consistent |
| Phase and coordinate registration | §§7--8 | EQ-026--027, REQ-067, PRED-036 | OWNER-034 and AST follow-up | Consistent |
| Calibration and Beammap transfer | §10 | EQ-001--004, REQ-013--016/065/068 | OWNER-001/024/035 | Consistent |
| Covariance, support, eligibility | §11 | EQ-012/015--020b, REQ-018--020/037--047 | conservative v0.1 ineligibility retained | Consistent |
| Scientific validation | §12 | REQ-069--070, PRED-030--038 | OWNER-036 | Consistent |

## Claim audit

- Scientific contract: draft r0.2 for owner review.
- Numerical owner choices: unresolved where listed.
- Implementation conformity: not assessed.
- Representation fidelity: not assessed.
- Observational performance and science qualification: not assessed.
- Production readiness: not assessed.

No contradiction was found between the r0.2 directive and the approved v0.1
scope. The r0.1 mathematical core was preserved and strengthened; no
production value or current behavior was inferred.
