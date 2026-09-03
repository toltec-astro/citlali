# SCI-ALIGN / SCI-AST Stage B r0.2 Horizontal Coherence Audit

Closure revision prepared: `2026-08-22`

Audit mode: fresh, clean, implementation-blind correction-closure re-audit at
high effort, using the same bounded durable-text evidence policy as the initial
horizontal audit.

## Final disposition

**Exact horizontal coherence is established within the bounded Stage B r0.2
text evidence set.** All three prior findings are closed. No new blocking
scientific conflict, bounded correction, editorial observation, or scientific
owner question was introduced.

Final open-finding counts:

| Classification | Open count |
| --- | ---: |
| Blocking scientific conflict | 0 |
| Bounded correction | 0 |
| Editorial observation | 0 |
| None | 9 audited horizontal topics |

This closure is a horizontal contract-coherence result only. Scientific
approval, implementation conformity, representation fidelity, empirical or
observational validation, freeze, readiness, and production authorization
remain unassessed.

## Evidence boundary

The re-audit read only the previously authorized durable text authorities
within:

- `SCI-ALIGN/v0.1-stage-b-r0.2/`: `src/**/*.tex`, crosswalk, notation/symbol
  and formal-ID change maps, owner and availability registers, source manifest,
  visual-QA report, and `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`;
- `SCI-AST/v0.1-stage-b-r0.2/`: `src/**/*.tex`, crosswalk, slot/direction,
  role-parentage, and requirement/equation/prediction change maps, owner and
  availability registers, source manifest, boundary-identity proof,
  visual-QA report, and `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`.

PDFs, rendered pages, tools, verification intermediates, implementation,
schemas, tests, validation or production evidence, repository guidance/status,
other packages, web sources, and raw thread history were not inspected. Neither
r0.2 package was modified by this audit; only this report was revised.
No separate prior-audit or repair artifact was inspected; only the three prior
finding statements already resident in this required report were carried
forward for explicit closure disposition.

## Prior findings and closure dispositions

| Prior finding | Prior classification | Closure disposition |
| --- | --- | --- |
| `H-BLOCK-001`: SCI-ALIGN reused KID-reserved `x` for generic field, stacked, response, and covariance operands | Blocking scientific conflict | **Closed.** Every inspected generic non-KID operand is now in the neutral `v` value/input family; lowercase `x` and `r` occur in canonical/narrative ALIGN sources only as paired KID coordinates. Stable formal IDs and mathematical relations are unchanged, and both change maps trace the repair. |
| `H-BOUND-001`: ALIGN Figure 2 promoted `interpolated` to an origin state | Bounded correction | **Closed.** Figure 2 now separates the canonical origin, method, and refinement axes exactly. |
| `H-EDIT-001`: AST retained ambiguous `nonpolarimetric Stokes-I use` wording | Editorial observation | **Closed.** AST now uses `ordinary nonpolarimetric coordinate path`, preserves optional HWPR timing only as an ALIGN-parent fact, excludes polarization operations, and explicitly states that raw KID `x` is not Stokes I. |

No prior finding was silently downgraded or reconciled. Each closure is supported
by the exact source and trace references below.

## Closure evidence

### H-BLOCK-001 — closed

SCI-ALIGN now reserves `x` and `r` exclusively for the paired physical KID
readout coordinates throughout the inspected canonical and narrative sources:

- `src/common/notation.tex` lines 34-42 uses `V_D`, `V_Tv`, and `V_Hv` for
  generic/native value containers and `C_v` for input covariance, while the
  paired physical coordinates remain `(x,r)^acq`;
- `SCI-ALIGN-EQ-004` and `005` use generic endpoints `v_a,v_b`
  (`src/common/equations.tex` lines 56-74);
- `SCI-ALIGN-EQ-007` uses stacked `boldsymbol v_D`, `boldsymbol v_T`,
  `boldsymbol v_H`, and `boldsymbol v` (lines 89-106);
- `SCI-ALIGN-EQ-011` uses `E[boldsymbol v]` (lines 148-159);
- `SCI-ALIGN-EQ-013` uses `E[boldsymbol v]` and `C_v` (lines 172-179);
- `SCI-ALIGN-EQ-014` uses `v_a,v_b` in the unchanged scalar variance relation
  (lines 181-188);
- the scientist rationale repeats `v_a,v_b` in its motivating scalar equation
  (`src/scientific-rationale.tex` lines 200-208).

An exhaustive occurrence scan of the permitted ALIGN LaTeX sources found
lowercase `x` only in the paired KID chain, paired mapping equation, paired
validity/provenance requirements and predictions, and text explicitly naming
that KID role. The retired generic forms `x_a`, `x_b`, generic `boldsymbol x`,
`C_x`, and `X_D/X_Tv/X_Hv` do not occur in the inspected canonical or narrative
sources. Within the bounded evidence set they appear only as prior forms in
explicit change maps.

Traceability is complete:

- `NOTATION_AND_SYMBOL_CHANGE_MAP.md` lines 16-18 maps generic endpoints,
  stacked operands/containers, and covariance/expectation inputs to the neutral
  `v` family; lines 22-26 states the complete exclusive reservation.
- `REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md` lines 13 and 16-21 maps
  `EQ-004`, `005`, `007`, `011`, `013`, and `014`, explicitly preserving their
  mathematics and stable identities.
- `CROSSWALK.md` lines 48-55 maps the neutral-operand repair and the Figure 2
  correction to their canonical locations without changing requirement or
  prediction IDs.

Formal inventories remain exact and duplicate-free: SCI-ALIGN has precisely
`REQ-001`-`055`, `PRED-001`-`026`, and `EQ-001`-`020`; SCI-AST has precisely
`REQ-001`-`090` and `PRED-001`-`050`. The inspected equations retain the same
operators, coefficients, conditions, response, and covariance mathematics;
only the generic operand family changed.

### H-BOUND-001 — closed

The ALIGN Figure 2 target-slot fact card now records exactly
(`src/scientific-rationale.tex` lines 282-288):

- origin: `original / synthesized / unavailable`;
- method: `exact / linear / circular / held / surrogate / none`;
- refinement: `original-invalid / guarded`.

This matches the independent origin and method axes in
`src/common/notation.tex` lines 47-61, the distinct tuple members in
`SCI-ALIGN-EQ-019`, and `SCI-ALIGN-REQ-034`. The correction is traced by
`CROSSWALK.md` line 52 and the visual-QA report lines 38-45. It changes no
formal identity or owner question.

### H-EDIT-001 — closed

AST now uses the exact phrase `ordinary nonpolarimetric coordinate path` and
states the complete limitation consistently:

- `SCI-AST-ASM-014`, `src/common/assumptions.tex` lines 89-95;
- the frame/product-family definition, `src/common/definitions.tex` lines
  164-180;
- `SCI-AST-PRED-047`, `src/common/edge_cases.tex` lines 228-232;
- the scientist narrative, `src/scientific-rationale.tex` lines 340-344.

Across these authorities, AST does not interpret polarization; optional HWPR
timing may remain only as an ALIGN-parent fact; it authorizes no demodulation,
polarization calibration or response, or Stokes reconstruction; and raw KID
readout `x` is explicitly not identified or relabeled as Stokes I. The repair
is traced by `REQUIREMENT_EQUATION_PREDICTION_CHANGE_MAP.md` line 14,
`SLOT_DIRECTION_CHANGE_MAP.md` line 17, `CROSSWALK.md` line 68,
`OWNER_DECISION_REGISTER.md` line 100, and `AVAILABILITY_REGISTER.md` line 20.
No requirement ID or content changed; only `PRED-047` wording was clarified.

## Exact shared-boundary identity

The two installed `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` files were compared
byte-for-byte and independently hashed:

| Copy | SHA-256 |
| --- | --- |
| SCI-ALIGN r0.2 | `359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf` |
| SCI-AST r0.2 | `359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf` |

The byte comparison succeeded. Profile identity remains exactly
`SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; the compatibility/supersession rule remains
unchanged; the body contains no SHA-256 value and therefore no self-hash. Both
source manifests and `BOUNDARY_IDENTITY_PROOF.md` record the same required
digest.

The SHA-256 values recomputed for the repaired LaTeX sources and their permitted
maps/crosswalks match the current entries in both source manifests.

## Full horizontal-coherence matrix

| Topic | Final classification | Closure result and exact authorities |
| --- | --- | --- |
| 1. Stable ALIGN slot, local row, RTC sample | None | `s` remains stable with `(o,s)`; `j` is local storage row only; `n` is stable RTC output identity; reconstruction of `s` from `j` remains prohibited. Shared boundary lines 55-70, 113-119, 128, and 169-180; `SCI-ALIGN-REQ-003`; `SCI-AST-REQ-052`, `073`-`077`; `SCI-AST-PRED-035`. |
| 2. Exclusive `x/r` and unambiguous reference/sky symbols | None | H-BLOCK-001 is closed. ALIGN uses `x/r` only for paired KID coordinates; neutral `v` carries generic operands. `i_ref`, `t^ref`, `delta_(i->ref)`, AST `u_sky`, and TAN `zeta_1,zeta_2` remain unambiguous. |
| 3. Circular interval and antipodal availability | None | Both authorities retain exactly `[-P/2,P/2)` and typed unavailability for exactly antipodal interpolation absent explicit unwrap authority. Shared boundary lines 67-70; `SCI-ALIGN-EQ-005`, `REQ-018`, `PRED-005`; AST `wrap`, `REQ-038`, and `PRED-018`. |
| 4. Exact shared profile, compatibility, bytes, and hash | None | Exact profile remains `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; compatibility/supersession is preserved; copies are byte-identical at the required SHA-256 without boundary self-hash. |
| 5. Nonpolarimetric and optional-HWPR scope | None | H-EDIT-001 is closed. Both packages preserve the ordinary nonpolarimetric path, raw KID identity, and optional-HWPR limitations without authorizing polarization operations. |
| 6. Role-factored AST parents and dependency-limited validity | None | Direction, tangent, continuous-pixel, nominal-pixel, and RTC roles still extend exact parents without retroactive WCS/pixel invalidation. AST parent equations; `ROLE_FACTORED_PARENTAGE_MAP.md`; `SCI-AST-REQ-056`-`060`, `073`-`077`; `PRED-025`, `030`, `035`, `036`. |
| 7. Geometry-only incidence/stencil versus MAP-owned `G_pi` | None | `I^geom_pi` remains a non-estimator geometry candidate with no numerical contribution, normalization, conservation, MAP-support/validity, response, or covariance authority; exact `G_pi` remains MAP-owned and requires the complete MAP request. `SCI-AST-REQ-080`-`083`; `PRED-038`-`040`; AST geometry-incidence and `gpi-parentage` equations. |
| 8. Stable formal identities and semantic traceability | None | All formal inventories are exact, sequential, and duplicate-free. The generic-operand, Figure 2, and nonpolarimetric wording repairs are fully traced by the current change maps and crosswalks without renumbering or mathematical change. |
| 9. Narrative consistency and owner questions | None | H-BOUND-001 is closed. Both narratives remain subordinate to the complete engineering authorities, preserve layered availability and ownership, and leave all existing open/deferred questions unresolved. No new owner question appears. |

## Preserved owner questions

The correction closure does not close, reopen, or modify any owner question.

SCI-ALIGN remains open/deferred at:

- `SCI-ALIGN-ODQ-101`: producer event/epoch/time and offset-validity domain;
- `SCI-ALIGN-ODQ-102`: exact observing-state registry and composition semantics;
- `SCI-ALIGN-ODQ-103`: producer `Hold`, transition, and physical-scan semantics;
- `SCI-ALIGN-ODQ-104`: detector continuity-surrogate authority and limits;
- `SCI-ALIGN-ODQ-105`: response/detail/covariance/model/selection-uncertainty tiers;
- `SCI-ALIGN-ODQ-109`: optional-HWPR registry and future polarimetry authority;
- `SCI-ALIGN-ODQ-110`: pointing-correction record family and semantics.

`SCI-ALIGN-ODQ-106` through `108` remain decided and were not reopened.

SCI-AST remains open/deferred at:

- `AST-OWNER-Q001`: aligned observing-state registry and composition history;
- `AST-OWNER-Q002`: named center-selection owner by family;
- `AST-OWNER-Q003`: map-center covariance producers/conditioning/cross terms;
- `AST-OWNER-Q004`: quantitative small-angle/time/precision adequacy authority;
- `AST-OWNER-Q005`: deferred MAP-003 retained-grid disposition;
- `AST-OWNER-Q006`: geometry producer/association/transform authority;
- `AST-OWNER-Q007`: family-specific Jacobian request types;
- `AST-OWNER-Q008`: RTC-grid requesting families and exact RTC profile owner.

New owner questions: **none**.

## Final closure statement

The three prior corrections are closed, full SCI-ALIGN/SCI-AST Stage B r0.2
horizontal coherence is preserved, and no new finding or owner question was
introduced within the bounded evidence set.

This re-audit does not assess or establish scientific approval, implementation
conformity, representation fidelity, empirical or observational validation,
freeze, readiness, or production authorization.
