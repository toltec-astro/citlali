# SCI-CAL v0.1 — Scientific Decision Log

Status: scope and v0.3 boundary decisions approved; unresolved scientific
substance remains pending

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

This log is intentionally concise. The complete scientific boundary and
author task live in [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md).

| Decision | Approved substance |
| --- | --- |
| `CAL-SCOPE-D001` | SCI-CAL ends at calibrated detector samples plus conditional uncertainty, quality, and canonical lineage. Maps are downstream consumers, not CAL estimators. |
| `CAL-SCOPE-D002` | Carry the retained fixed line-of-sight-optical-depth operator into authorship as structurally adopted but physically and observationally qualified. The contract must state its authorized support and unavailable claims. |
| `CAL-SCOPE-D003` | For `0.15 < tau225 <= 0.25`, no calibrated SCI-CAL output is authorized until a continuous engineering operator is separately adopted. The state must be truthfully unavailable/uncalibrated, with no silent extrapolation. |
| `CAL-SCOPE-D004` | Approve the exact four-item author packet and its sanitized supersession and convention/ownership documents. |
| `CAL-SCOPE-D005` | SCI-CAL v0.1 calibrates only the ordinary `xs` detector stream. No other measured stream inherits that meaning. |
| `CAL-SCI-D006` | For the v0.3 rationale, Beammap calibration/source-APT production owns the calibrator model, source atmosphere, Beammap amplitude and beam/template fit, pointing treatment in source calibration, and source-APT `flxscale` derivation. TolProj owns target-to-source association, the observation-specific child APT, and only approved child transformations. SCI-CAL applies the selected child factor once and target atmosphere. MAP/FLT owns realized downstream response. This owner clarification supersedes any broader earlier statement assigning source-calibration meaning to TolProj. |
| `CAL-SCI-D007` | Adopt producer--transformer--consumer as the package boundary rule: the producer owns meaning, the transformer owns only an explicitly approved transformation and lineage, and the consumer applies without reinterpretation. |
| `CAL-SCI-D008` | Adopt the v0.3 three-artifact science-rationale architecture and the library house standard as the template for subsequent packages. Accuracy, explicit gaps, and traceability govern freeze; completeness by invention does not. Further rationale changes require an owner decision or engineering-contract change. |

Grant also confirmed the previously recovered CAL scientific decisions named
in Scope Brief section 6, including the layered APT identity model, once-only
factor composition, conditional variance/weight transfer, package-level
reconstructibility, and the separation of structural atmosphere correctness
from physical and observational performance.

The original scope decisions authorized implementation-blind scientific
authorship. Decisions D006--D008 approve the v0.3 ownership boundary,
producer--transformer--consumer rule, and reusable rationale architecture.
They do not resolve Q01--Q09, establish implementation conformity, authorize
validation, or change production state.
