# SCI-CAL v0.1 — Scientific Decision Log

Status: Q01--Q09 incorporated into science r0.5 and engineering r0.4;
validation evidence and final owner acceptance remain pending

Scientific owner: Grant Wilson

Initial approval date: `2026-08-16`

Additional bounded-repair authorizations: `2026-08-20`

This log is intentionally concise. The complete scientific boundary and
author task live in [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md).

| Decision | Approved substance |
| --- | --- |
| `CAL-SCOPE-D001` | Historical scope: SCI-CAL ends at calibrated detector samples plus conditional uncertainty, quality, and canonical lineage. Maps are downstream consumers, not CAL estimators. The conditional-uncertainty assignment is superseded by `CAL-SCI-D013`; statistical noise estimation is downstream. |
| `CAL-SCOPE-D002` | Carry the retained fixed line-of-sight-optical-depth operator into authorship as structurally adopted but physically and observationally qualified. The contract must state its authorized support and unavailable claims. |
| `CAL-SCOPE-D003` | Historical engineering-opacity no-output rule, superseded by `CAL-SCI-D013` after adoption of the continuous content-bound operator. No silent extrapolation remains allowed. |
| `CAL-SCOPE-D004` | Approve the exact four-item author packet and its sanitized supersession and convention/ownership documents. |
| `CAL-SCOPE-D005` | SCI-CAL v0.1 calibrates only the ordinary `xs` detector stream. No other measured stream inherits that meaning. |
| `CAL-SCI-D006` | For rationale r0.3, Beammap calibration/source-APT production owns the calibrator model, source atmosphere, Beammap amplitude and beam/template fit, pointing treatment in source calibration, and source-APT `flxscale` derivation. TolProj owns target-to-source association, the observation-specific child APT, and only approved child transformations. SCI-CAL applies the selected child factor once and target atmosphere. MAP/FLT owns realized downstream response. This owner clarification supersedes any broader earlier statement assigning source-calibration meaning to TolProj. |
| `CAL-SCI-D007` | Adopt producer--transformer--consumer as the package boundary rule: the producer owns meaning, the transformer owns only an explicitly approved transformation and lineage, and the consumer applies without reinterpretation. |
| `CAL-SCI-D008` | Adopt the r0.3 three-artifact science-rationale architecture and the library house standard as the template for subsequent packages. Accuracy, explicit gaps, and traceability govern freeze; completeness by invention does not. |
| `CAL-GOV-D009` | The prefix `v` identifies the scientific-contract/package version; `r` identifies the revision of a particular representing document. SCI-CAL remains contract v0.1; at the time of D009 the science rationale was revision r0.3. |
| `CAL-GOV-D010` | After the r0.3 cleanup there is no further stylistic round. A later rationale revision requires formal resolution of Q01--Q09, a normative engineering-contract change, validation evidence that changes an evidentiary status, or a genuine scientific inconsistency. |
| `CAL-GOV-D011` | Repair engineering conformance as document revision r0.2 against science rationale r0.3 while retaining contract v0.1 and all stable inventories. The engineering view must carry Q01--Q09, the producer--transformer--delivery--consumer boundary, source/child APT lineage, broadband convention, pipeline-order constraints, and claim-layer limitations without resolving scientific decisions or asserting implementation conformity. |
| `CAL-GOV-D012` | Correct the implementation-blind consistency-review findings as science rationale r0.4 and engineering conformance r0.3: expand D007's boundary name to producer--transformer--delivery--consumer and expose immutable TolTECA delivery in both views, restore the pointing-disposition semantics, remove the out-of-packet direct citation and undefined `RTC` term, correct crosswalk provenance, and distinguish the document issue dates. Preserve v0.1, all stable inventories, Q01--Q09, and all scientific algebra. |
| `CAL-SCI-D013` | Incorporate the scientific owner's 2026-08-20 dispositions of Q01--Q09 as science rationale r0.5 and engineering conformance r0.4. This decision supersedes D001's assignment of conditional-noise production to CAL and D003's engineering-opacity no-output rule: measurement-noise estimation is downstream, and operator-supported engineering samples use the same calibration law without a science-quality claim. It defines ordinary `xs`, CAL-before-PTC ordering, frozen SCI-BEAM `flxscale`, closest-accepted-APT selection and optional scientist-directed TolProj array rescale, reference frequencies/spectra, the content-bound WVR/AM/passband operator, whole-observation opacity policy, uncertainty ownership/unavailability, and the concrete Beammap closure plus associated-pointing transfer workflow. Preserve contract v0.1 and all 11/50/30 stable IDs. Treat 1%, 5%, and 5--10% as reporting benchmarks; final scientific acceptance remains owner-owned. |

Grant also confirmed the previously recovered CAL scientific decisions named
in Scope Brief section 6, including the layered APT identity model, once-only
factor composition, conditional variance/weight transfer, package-level
reconstructibility, and the separation of structural atmosphere correctness
from physical and observational performance.

The original scope decisions authorized implementation-blind scientific
authorship. Decisions D006--D010 approve the r0.3 ownership boundary,
producer--transformer--consumer rule, reusable rationale architecture,
version axes, and stopping rule. D011 authorizes the bounded engineering
alignment repair, and D012 authorizes the consistency-review correction and
expands D007's boundary terminology without changing its ownership rule. D013
resolves Q01--Q09's scientific policy while retaining explicitly unavailable
numerical uncertainty products and an owner gate for achieved validation. It
does not establish implementation conformity, execute validation, or change
production state.
