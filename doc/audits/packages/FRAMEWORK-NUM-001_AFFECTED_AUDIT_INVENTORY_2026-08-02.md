# FRAMEWORK-NUM-001 affected-audit inventory — 2026-08-02

Record ID: `FRAMEWORK-NUM-001-INVENTORY-001`

Status: screening complete for the coordination-line artifacts named below;
no prior audit or application artifact modified

Review baseline: `codex/scientific-audit-framework` commit
`e7b09e0aff2b47f83da760c13ca4edd9f8e013ea`

## Scope and method

This is a risk inventory, not a retroactive finding against every listed
tolerance. The review searched coordination-line audit packages, handoffs,
contracts, returned-evidence reviews, and locally available SCI-CAL execution
records for:

- absolute, relative, ULP, and exact floating-point comparisons;
- numerical conditions with aborting or invalidating authority;
- engineering diagnostics used as acceptance gates;
- deterministic guards first reached during costly execution; and
- evidence designs that could discard or regenerate valid upstream output.

The inventory classifies what must enter a Tolerance-and-Stop-Condition
Register before the **next costly stage**. It does not invalidate completed
evidence, change an approved scientific contract, or silently amend any frozen
package.

The historical CAL execution examples were read at exact branch
`codex/sci-cal-001-atmosphere-operator`, commit
`5d1597ca2d18f5e35519f6e62b5a014aea736fad`, below repository-relative root
`validation/sci_cal_001_atmosphere_operator_2026-08-01`. Their identities are:

| Artifact | SHA-256 |
|---|---|
| `AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_ERRATUM_2026-08-01.md` | `590f49007065e604aced97fc391067e981a94d7336db1cec81512dd0de893e4e` |
| `AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_RECORD_2026-08-01.md` | `ce20486a7734d6a14781dbbd7f8e45b0c55e385d4c724918497468a0c93d3177` |
| `NATIVE_REGENERATION_REPORT.md` | `a1f370251cd9e4ecb27b717225094b34a9c6fc5067ae7266a62df8d884906b9b` |
| `H2O_SCALE_HYPOTHESIS_REPORT.md` | `1519928944075689f07e3b041fcba35a9f4f2c1042345df06cd14a6d47e2c5b6` |

## Systemic framework exposure

At the review baseline, `doc/audits/README.md` expressly described audit
packages as hand-maintained records without a formal schema/validator, and
`doc/audits/audit-ledger.yaml` recorded
`hand_edited_no_schema_or_validator`. The package prompt, scientific-contract,
and Unity-request templates preregistered scientific tolerances, but did not
require a stable-ID census of every source-level abort or invalidation route,
model-free full-tuple guard coverage, raw/evaluator validity separation, or a
cost-readiness certificate.

That framework gap allowed a check implemented only in a runner to acquire
invalidating authority without protocol traceability or impact propagation.

## Audit inventory

| Inventory ID | Package or lane | Concrete evidence | Risk/disposition before next costly stage |
|---|---|---|---|
| `NUM-INV-001` | `SCI-CAL-001` EL25 confirmation | The frozen runner `validation/sci_cal_001_atmosphere_operator_2026-08-01/run_am12_el25_confirmation_study.py`, SHA-256 `bcc4bc9f59574424e1daab652ab0316f8a694998155d9c3daa246e1e6260fb22`, line 1926 used an unregistered `5.0e-17` absolute guard. It stopped after 12/16 cases even though parsed transmission matched the target. The source contains 162 lines with `require(` and no stable condition-ID dispatcher or model-free full-case preflight. | **Confirmed critical governance failure.** No replacement AM run may launch until every abort/invalidation route has been inventoried and registered, deterministic guards pass all 16 cases without AM, raw and evaluator states are separated, and readiness is certified. |
| `NUM-INV-001A` | `SCI-CAL-001` AM12 successor-adoption v1/v2 | `AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_ERRATUM_2026-08-01.md` records a v1 stop after 1,025 AM requests because a documentary `stage` cache-key field differed although the physical argv did not. The corrected v2 study regenerated all 1,025 requests; its execution record later established identical physical argv and identical normalized numeric output for every pair. The corrected loader could validate all fixed training-grid artifacts without AM. | **Confirmed preflight/salvage failure.** Documentary keys and complete cache lookup must be preflighted. A successor must justify regeneration at the raw-model layer rather than infer it from a changed runner identity. Preserve the immutable v1 failure record. |
| `NUM-INV-001B` | `SCI-CAL-001` native AM regeneration | `NATIVE_REGENERATION_REPORT.md` records a first 180-case attempt whose numeric data lines were exact but whose shared-cache warnings invalidated execution, followed by another numerically exact, warning-free attempt excluded for incomplete execution-context/output binding before the canonical run. | **Confirmed readiness gap, conservative integrity handling.** Cache/warning and provenance gates were legitimate Class A controls, but their deterministic setup should have failed before 180 calls. Keep raw and execution-contract validity distinct so an exact payload can be assessed for salvage without weakening provenance. |
| `NUM-INV-001C` | `SCI-CAL-001` H2O-scale predecessors | `H2O_SCALE_HYPOTHESIS_REPORT.md` records an unlocked/unbound attempt stopped after 1,888/3,100 direct fitted-scale grids and a later context-bound attempt stopped after 1,811 matched outputs because parsed-array retention projected about 7.75 GB. The first exclusion was correctly fail-closed; the second was an execution-engineering failure and its raw subset was not assessed for reuse. | **Confirmed costly-readiness and salvage gap.** Locking, context binding, and memory shape belong in readiness/preflight. Preserve the exclusions, but require a causal raw-validity assessment before future full regeneration. |
| `NUM-INV-002` | `SCI-CAL-001` deterministic/Unity validation | `SCI-CAL-001_INDEPENDENT_CORE.tex` preregisters `64 epsilon`, relative `1e-12`, absolute `1e-14`, `1e-12` unit round trips, and bitwise parallel identity. These may be defensible Class B or exact-integrity checks, but the current package does not provide the audit-wide register and propagated-impact fields now required. | **Register and review; do not presume defect.** Preserve the owner-approved one-percent representation-fidelity gate. Numerical fixture tolerances must state arithmetic/conditioning derivation and action; engineering equivalence checks cannot silently veto costly evidence. |
| `NUM-INV-002A` | `SCI-CAL-001` EL25 registered arithmetic bounds | The EL25 preregistration records Decimal recomparison `5e-78`, target matching `1e-12`, and other `1e-10`/quantization bounds. Unlike the hidden `5e-17`, these are registered, but the record does not uniformly map their arithmetic derivation and maximum propagated effect into the final calibration metric. | **Carry forward unchanged pending register review.** Registration prevents surprise but does not by itself prove proportionality. Do not silently alter any value; classify and justify each before successor execution. |
| `NUM-INV-003` | `SCI-MAP-001` Unity campaign and MAP contract | `SCI-MAP-001_SCIENTIFIC_CONTRACT_AUDIT.tex` contains analytically motivated `gamma_n`/`64 epsilon` bounds, well-conditioned `1e-12` fixture bounds, a `1e-12`-degree WCS bound, and external `atol=2e-8`, `rtol=1e-10` comparison. `SCI-MAP-001_UNITY_ED1_OWNER_DECISION_2026-08-02.md` explicitly labels the latter regression bounds, not scientific truth. | **Mandatory register before human launch.** Retain exact scientific/product gates. Keep the regression comparisons as Class D warnings unless an impact analysis supports explicit reclassification as A, B, or C; bind the WCS and reduction-order derivations. The prior 118.34-GiB ledger review is a positive cost-proportionality precedent, not a defect. |
| `NUM-INV-004` | `SCI-ALIGN-001` timing/alignment validation | `SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md` rejected a data-tuned `4.063 ms` proposal and selected the derived uniqueness condition `abs(residual) < dt/2`, with half ties fail-closed. `SCI-ALIGN-001_INDEPENDENT_CORE.tex` also uses relative `1e-12`, `64 epsilon_mach`, wrapped `1e-12` rad, and covariance `1e-12` fixture comparisons. | **Good proportionality precedent plus register obligation.** The `dt/2` derivation should be recorded as Class B. The remaining fixture tolerances need arithmetic and metric-impact records before any costly exact-SHA evidence campaign; no current owner timing decision is changed. |
| `NUM-INV-005` | `SCI-AST-001` projection validation | `SCI-AST-001_SCIENTIFIC_CONTRACT_AUDIT.tex` records the implementation condition `abs(D) < epsilon` that aliases singular/backside TAN directions to map center. `SCI-AST-001_COORDINATOR_DECISION_2026-08-01.md` already rejects a near-zero epsilon in favor of strict forward-hemisphere domain and fail-closed validity. | **Historical systemic example, already scientifically dispositioned.** Do not edit application code in this framework action. Any future AST evidence harness must register its guards and preflight projection boundaries before costly execution. AST remains held behind ALIGN evidence/re-audit. |
| `NUM-INV-006` | `SCI-MAP-001` stopped evidence-design attempts | The MAP Unity reviews preserve immutable stopped packages and narrowed a proposed 118.34-GiB exhaustive term ledger rather than requiring it merely because a schema existed. The ED1 successor later stopped because required primitive authority was unavailable; it did not fabricate evidence or launch Unity. | **Positive stop/salvage pattern.** Preserve historical returns. Future costly MAP work must declare which raw reductions or captures remain independently valid and repeat only missing/invalid scope. |
| `NUM-INV-007` | `SCI-FLT-001` returned evidence | `SCI-FLT-001_RETURNED_EVIDENCE_COORDINATOR_REVIEW_2026-08-01.md` reports numerical residuals and preserves the result as bounded evidence. The coordination artifact did not expose a confirmed hidden tolerance with invalidating authority. | **No current incident found at package level.** Apply the register prospectively before any new costly FLT execution; do not reopen or alter the returned record solely because this framework changed. |
| `NUM-INV-008` | Active baseline validation profiles | `validation/validation_profiles.json` uses `atol=0` and `rtol=0` for whole floating-product comparisons in point, OOF, and Beammap profiles. Exact hashes, schemas, masks, aliases, and other semantic identities remain appropriate, but whole-product floating exactness is a broad implementation invariant without a per-product arithmetic justification in the profile itself. | **Prospective register review.** Preserve currently accepted baseline behavior. Before a costly recapture or invalidating use, distinguish exact semantic identity from floating regression and state the repeat/salvage consequence. |
| `NUM-INV-009` | `SCI-CAL-001` q-boundary phase-zero preflight | The locally reviewed phase-zero artifact `validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py` at evidence commit `ae99be1cef8c390d0e7490835ffca1f31da7ebc0` derives its roundoff screen from binary64 unit roundoff and conditioning and executes before costly AM work. | **Positive Class B precedent.** Reuse this pattern: derive the arithmetic bound, connect it to the checked operator, and exercise the boundary before expensive execution. |

## Limits of this inventory

- This was not a full static analysis of every historical branch or every
  application test. A frozen convolve/noise audit package was not present in
  `doc/audits/packages` at the review baseline, so no source-only conclusion is
  recorded for it here.
- A text search cannot prove complete guard coverage. Each costly successor
  runner needs a source-level guard census or centralized condition-ID
  dispatcher plus independent review.
- Exact hash, schema, row-count, identity, and unsupported-domain checks remain
  valid Class A controls. Their appearance in a search is not evidence of
  over-tolerance.
- Completed audits are not retroactively invalidated. The policy attaches to
  their next costly execution, replacement execution, or re-audit evidence
  campaign.

## Priority

1. Hold and disposition the stopped `SCI-CAL-001` EL25 evidence.
2. Apply the new launch gate to the next CAL and MAP costly execution.
3. Apply it to ALIGN before exact-SHA external evidence and to AST when its
   ALIGN dependency clears.
4. Screen later audit runners at package freeze; do not reopen completed work
   without a concrete trigger.
