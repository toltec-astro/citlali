# SCI-BEAM v0.1 — Stage A Crosswalk

Status: traceability scaffold; no normative requirements exist yet

The Stage B author will replace this scaffold with exact requirement-to-source,
equation, assumption, owner-decision, edge-case, and rendered-view coverage.
The present table proves only that every proposed topic has an identified
authority route or an explicit derivation gap.

| Proposed topic | Recovered source or authority route | Stage B treatment |
| --- | --- | --- |
| Program, recovery, firewall, layout, review, stop rule | Library program and pilot review | Binding process |
| AltAz tangent-plane frame, arcsec units, WCS authority | Current `SCIENTIFIC_CONVENTIONS.md` | Sanitized owner-approved extract |
| Stable detector identity and row/slot distinction | Current convention and product contracts | Sanitized boundary; author derives minimum identity |
| Source selection and per-array flux | Current convention and Beammap config authority | Upstream TolProj/TolTECA boundary; conditional input |
| Conditioned signal, causal validity, response, coefficient | RTC/PTC/VAL/CAL boundaries and historical handoffs | Abstract conditional interfaces only |
| Detector maps and map validity | SCI-MAP conditional boundary | Abstract conditional interface only |
| Soft prior identity and reliability | TolAPT reliability contract | Owner-approved sanitized producer extract |
| Downstream APT/calibration/sensitivity use | `toltec_beammap` guide and status | Ownership boundary only |
| Source/beam model and finite-source coupling | No BEAM core recovered; proposed primary references | Genuine independent derivation |
| Likelihood, support, covariance, nuisance propagation | No BEAM core recovered | Genuine independent derivation |
| Prior role, fallback, iteration, convergence | Producer boundary plus open owner decisions | Independent derivation under owner scope |
| QC states and numerical thresholds | Current implementation inventory only | State semantics derived; numerical policies owner-controlled |
| Calibration candidate and promotion | CAL handoff plus open Q007 | Conditional derivation; no automatic promotion |
| Atomic result bundle and consumer compatibility | Current product inventory plus open Q009/Q012 | Independent scientific abstraction |
| Edge and falsifiable predictions | Scope Brief section 8 | Normative Stage B predictions |

No line in this scaffold establishes scientific authority or implementation
conformity.
