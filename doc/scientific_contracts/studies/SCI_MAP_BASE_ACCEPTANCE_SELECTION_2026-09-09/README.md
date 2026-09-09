# Base observation map: initial acceptance examples and handoff

## Program adherence and prior-work recovery

This bounded selection follows the [contract-library charter](../../README.md)
and [downstream roadmap](../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It is a preparation note, not a new scientific package or engineering work order.
The owner accepted the recommendation to select existing base-map examples,
identify genuine gaps, and then park this task pending a reviewed, conforming
MAP-facing timestream output: “I like this approach.”

Recovery used the [prior-work registry](../../PRIOR_WORK_REGISTRY.md),
[r0.4 recovery](../../packages/SCI-MAP/v0.2/revisions/r0.4/PRIOR_WORK.md),
and later MAP acceptance/activation records on canonical commit
`86c20b31f7300ba4063be044380b61cd0baf25eb`.
[SOURCE_BINDING.json](SOURCE_BINDING.json) binds the selected unchanged sources.

| Recovered material | Disposition here |
| --- | --- |
| Frozen SCI-MAP v0.2/r0.4 requirements, predictions, zero and coverage fixtures | **Adopt** the existing expected behavior; select a small initial subset below. |
| Generic freeze, completed profile admission and contractual activation | **Cite** the completed decisions; historical candidate/pending labels in frozen examples do not reopen them. |
| Existing ECS procedure and prospective record shapes | **Cite** for the future engineering handoff; no result record is populated here. |
| Coaddition, stronger numerical response/covariance products and independent FRUIT | **Defer** from this base-observation selection. |
| Application, configuration, implementation tests, operational evidence | **Exclude** from this contract-preparation task; no implementation inspection or scientific inference from code. |

No new scientific derivation is needed for this selection. Engineering and
Review/Conformance governance were read; the note receives a fresh-context,
read-only exact-commit review for science consistency, ownership/scope, and
repository/evidence integrity. Existing frozen document/PDF evidence is reused.

## Exact scope and authority

The reference is `SCI-MAP:base_observation_map@1`, with
`SCI-MAP:one_hot_containing_pixel@1` and explicitly selected
`SCI-PTC:uniform_constant@draft-0.1`. This specifies the reference case, not a
default, an observation selection or permission to run it. The five exact
registered identities, source digest and active occurrence policy
`SCI-MAP:map_upstream_admission@2` are given by the completed
[activation record](../../acceptance/SCI_MAP_PROFILE_ACTIVATION_2026-09-09/ACTIVATION_RECORD.md).
The [approved base-path reference](../SCI_MAP_PROFILE_ACTIVATION_2026-09-09/README.md#concrete-reference-ordinary-base-observation-map)
lists the exact PTC retention, coefficient Registry/family/handoff, PTC-to-MAP
boundary/composition and same-sample AST/original-footprint boundary identities.
Their literal approved draft spellings remain unchanged.

The [freeze](../../packages/SCI-MAP/v0.2/SCIENTIFIC_OWNER_FREEZE_R0.4.md)
and later activation are authoritative together: scientific content is frozen
and the registered generation is contractually evaluable. Neither supplies
object facts, a passing evaluation, an implementation or execution authority.

## Small initial acceptance set

These six groups are navigation labels, not new contract or fixture identities.
`PRED` and `REQ` refer to the existing
[prediction crosswalk](../../packages/SCI-MAP/v0.2/revisions/r0.4/PREDICTION_CROSSWALK.md)
and [shared requirements](../../packages/SCI-MAP/v0.2/revisions/r0.4/src/common/requirements.tex).
The [canonical predictions](../../packages/SCI-MAP/v0.2/revisions/r0.4/src/common/edge_cases.tex)
and ECS remain the source of every expected result.

| Group and existing source | Selected case and unchanged expectation |
| --- | --- |
| **1. Input admission** — PRED-010/011; REQ-004/006/010/027/034/035; ECS stages A–C | Pair a stipulated complete, explicitly selected uniform handoff with missing selection/publication, wrong parent/generation, and nonpassing typed admission. Registry presence supplies nothing; structural rejection precedes payload retrieval. Safely classify admitted payloads before membership/arithmetic; zero coefficients do not contribute, and negative/non-finite/unrepresentable coefficients retain their exact invalid causes. Invalid examples do not authorize another family. A verified MAP decision needs all four requested/applicable/eligible/realized axes; PTC handoff/QC cannot replace it. |
| **2. Arithmetic and measured zero** — PRED-001/003/025; REQ-010–014/033–035 | Reuse [zero fixtures A and B](../../packages/SCI-MAP/v0.2/revisions/r0.4/records/PRED025_QUANTITY_SPECIFIC_ZERO_FIXTURES.md): admitted signals +3 and −3 mJy/beam with unit coefficients give numerator 0, normalization 2, map value 0; three zero signals give normalization 3 and map value 0. Preserve the general one-pixel sum/quotient identity and constant-input prediction. These are valid zeros only on support-authorized rows with all other gates passing. |
| **3. Pixel placement** — PRED-004/009, index-only part of PRED-025 fixture C; REQ-005/015/026/043/045/047 | Select an interior cell, an internal upper edge, outer upper extent/outside, and in-bounds index zero. Each in-grid point owns exactly one half-open cell; internal upper edges belong to the adjacent cell; the outer upper extent/outside contributes nowhere. No splitting, wrapping or clamping. Zero index is valid under its exact identity/domain; invalid or unrepresentable conversions retain their declared failure scope. Fixture C's coadd offset case is outside this selection. |
| **4. Low-coverage and empty support** — PRED-012/025 fixture D; REQ-009/012/031–033/035/046/047 | Reuse the complete existing [small-N table and cut cases](../../packages/SCI-MAP/v0.2/revisions/r0.4/records/COVERAGE_CUT_EXPERT_OVERRIDE_AND_SMALL_N_FIXTURES.md), including N=0 and the repeated-index transitions. Keep both inclusive thresholds, the predeclared population, and finite-positive normalization. `coverage_cut=0` removes only the relative cut. Above one needs its independently authorized same-value/scope override; invalid requests fail before application. Zero normalization permits no division. Valid applied empty support gives `not_produced`, exact cause `no_support_authorized_output_rows`, and no MAP product, validity state, completion marker or fabricated zero; it is not failure. |
| **5. Exposure lineage** — REQ-007/025/026/029/030; ECS original-footprint cases | Select one original with several descendants at different pixels, mixed estimator admission, missing original coordinate, and boundary loss. Deduplicate exact original identities and place exposure once at the original's own AST coordinate, never at each descendant. Upstream-eligible and retained populations remain separate; missing coordinate/lineage authority makes the affected exposure role unavailable. Distinguish geometry, route candidacy, estimator contribution and exposure; none is precision. These are specified cases, not supplied numerical exposure arrays. |
| **6. Complete base output and honest status** — base-observation parts of PRED-013/017/025; REQ-008/033/046–048 | Select a successful nonempty base bundle with unavailable upstream response/covariance, pre-application rejection, required-publication failure, and the empty-support attempt above. A successful base product retains the complete MAP-local operator and exact signal rows, parents, reached lifecycle and companion status/cause; missing stronger numerical companions alone do not invalidate it. Success has all five lifecycle stages; earlier stops retain only reached stages. Required publication failure cannot claim completion. A permitted in-memory route retains the same complete logical bundle and does not waive PTC publication. |

Numerical fixtures already supplied are the zero examples and small-N table.
The other entries reuse algebraic or qualitative expected cases. No executable
fixture, observed pass or implementation test is supplied by this selection.
No coverage default, tolerance, new profile, numerical parameter or new
scientific decision is introduced.

This is an initial acceptance set, **not a full conformance qualification**.
It does not mark omitted requirements `not_applicable` or waive any ECS gate.
For example, exact representation/WCS checks, the rest of the exposure cases,
supported execution modes and every other applicable base-role obligation
remain due when the engineering owner makes the corresponding claim.
Coadd and stronger-role obligations do not become base-role prerequisites.

## Gaps, ownership and stopping point

No missing scientific decision or expected behavior was found for the selected
cases. The remaining gaps concern engineering realization and evidence:

- **Timestream engineering owner:** provide the first reviewed, conforming
  MAP-facing output, its exact candidate/authority identities and evidence,
  complete positive-rank PTC publication/retention and coefficient handoff,
  same-sample AST/full parents, and original-footprint lineage/coordinate
  facts or their explicitly typed availability. The whole timestream program
  need not be finished before that bounded handoff is useful.
- **Separately authorized MAP engineering owner:** select the candidate and
  build environment; realize immutable synthetic cases and independent
  expected-value artifacts; preregister comparisons and required outputs;
  then test under the existing [ECS procedure](../../packages/SCI-MAP/v0.2/revisions/r0.4/src/engineering-conformance.tex).
  Required facts and active profiles precede VAL evaluation; the resulting
  artifact is verified before admission, and numerical membership follows
  all remaining gates. Document checks here establish none of those results.
- **This contract task:** preserve this selection and stop. Resume for a
  focused read-only handoff comparison when that reviewed output is available
  and the owner brings it here. Record any mismatch, scientific consequence
  and existing acceptance criterion; application repair belongs to an explicit
  engineering increment.

No additional owner science decision is requested. Application work and
numerical execution remain separately gated, as do representative-environment
qualification and production use. Local document checks are not Spack or
deployment evidence. FRUIT remains independent and is not a waiting condition
for this base-map handoff. This preparation does not integrate or push a branch,
send work to another task, or start a monitoring job.
