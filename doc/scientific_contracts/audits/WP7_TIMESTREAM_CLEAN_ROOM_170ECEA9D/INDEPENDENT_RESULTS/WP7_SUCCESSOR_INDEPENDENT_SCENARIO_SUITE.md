# WP-7 Successor Independent Scenario Suite

Audit date: 2026-08-26  
Independent phase: complete  
Source commit: `170ecea9de1ee810da7d7e45a489a4545ccd623d`  
Suite status: locked implementation-blind contract oracles; not executed validation

## Purpose and comparison discipline

This suite was derived from the admitted WP-7 clean-room authorities before receipt of comparison material. It exercises ordinary, RTC-only, unavailable, conflict, response, uncertainty, exposure, lifecycle, and provenance states. It is not copied from a prior scenario set and assigns no legacy finding identifier.

For later execution:

- identities, memberships, parent links, causal roles, source digests, status tokens, application counts, and lifecycle ordering compare exactly;
- numerical comparison uses only the precision/tolerance declared by the exact governing operator; a semantic mismatch is never absorbed into a tolerance;
- `unavailable`, `invalid`, `ineligible`, `decision_unavailable`, `disabled`, `not_requested`, `rejected`, `failed`, and numeric zero remain distinct;
- a correct failure or unavailable state is a successful contract outcome for a negative scenario;
- passing these scenarios would establish only the specifically named conformance or validation layer under a separately authorized execution program.

## Coverage matrix

| Required category | Scenarios |
| --- | --- |
| Ordinary route | `IS-001`--`IS-008` |
| RTC-only terminal route | `IS-009`--`IS-012` |
| Unavailable and fail-closed states | `IS-013`--`IS-020` |
| Conflict and compatibility | `IS-021`--`IS-025` |
| Response | `IS-026`--`IS-030` |
| Uncertainty and covariance | `IS-031`--`IS-034` |
| Exposure and influence | `IS-035`--`IS-038` |
| Lifecycle and provenance | `IS-039`--`IS-043` |

## Ordinary-route scenarios

### `IS-001` — exact identity-path witness

**Setup:** Supply one valid original native paired `x/r` occurrence from an exact approved Tune/readout mapping. Make the detector stream the ALIGN reference interface; use exact slot coincidence and `delta_ref->ref=0`. Select RTC pass-through with no despiking, no level-shift correction, no filters, `M=1`, and phase zero. Supply exact supported AST runtime parents, one unique child-APT factor, exact-time zero WVR opacity, supported elevation and `alpha=0`. Configure one feasible positive-rank PTC group with full time-local application rank and all five PTC named-use decisions eligible.

**Expected:** ALIGN is one-hot with original origin and unchanged paired values. RTC preserves value, order, cardinality, support, raw-`r` parentage, and identity response. AST binds both ALIGN-grid and RTC-grid coordinates to the exact occurrence. CAL emits `flxscale * x` because atmosphere correction is unity and applies every required role once. PTC emits and retains the exact configured-rank transformed signal with the declared centering/null/loss state. The route establishes `TS-S`/`TS-C` structural closure only.

**Prohibited outcome:** hidden default selection, implicit signal synthesis, inferred pointing, added calibration factor, fabricated covariance, or response/qualification promotion.

### `IS-002` — keyed APT permutation invariance

**Setup:** Evaluate the same CAL parent twice. In the second realization, arbitrarily permute selected APT rows while preserving immutable row-occurrence keys, artifact identity, acquisition binding, and values.

**Expected:** After restoring scientific occurrence order, calibrated values, factor-instance identities, validity states, and PTC inputs are identical. Row position never becomes identity.

**Prohibited outcome:** a changed factor caused only by row order or selection of a first/nearest row.

### `IS-003` — once-only CAL factor challenge

**Setup:** Hold all state fixed and choose a nonunity factor `f` in the canonical CAL multiplier. Evaluate correct, omitted, duplicated, and inverted-factor challenges.

**Expected:** Relative to correct output, the ratios are `f^-1`, `f`, and `f^-2`, respectively. The causal lineage still records the exact role and instance. A factor already embodied in the selected child APT has runtime application count zero.

**Prohibited outcome:** value-only deduplication without role identity, or applying parent and embodied child factors together.

### `IS-004` — supported WVR interpolation and observation classification

**Setup:** Use two finite, producer-valid, same-observation WVR records bracketing all detector samples and the complete observation window. Include threshold crossings around `0.15`, with `tau_mean <= 0.15`, `tau_max <= 0.175`, and all values within the numerical operator support.

**Expected:** Sample opacity follows the written binary64 linear interpolation with both endpoint identities. The classifier uses endpoint/source breakpoints, continuous trapezoid duration weighting, analytic crossings, and assigns exactly `science_qualification_eligible`. Supported samples are calibrated. No achieved science-quality claim is made.

**Prohibited outcome:** detector-sample-count weighting, cadence smoothing, a separate excursion-duration threshold, or promotion to `science-qualified`.

### `IS-005` — `engineering_only` ordinary mathematics

**Setup:** Give a completely covered finite observation with `tau_max <= 0.25` that fails the `science_qualification_eligible` mean/peak condition but remains inside sample-local operator support. Supply all exact ordinary PTC prerequisites.

**Expected:** CAL assigns `engineering_only`; sample-local supported values use the same numerical atmosphere operator. All five PTC profiles preserve the class. The class alone does not prohibit basis fit, loading fit, operator application, output retention, or response-companion mathematics, and none of those decisions creates CAL science qualification.

**Prohibited outcome:** a global PTC veto, an alternate engineering calibration law, or a science-quality upgrade.

### `IS-006` — network-group isolation

**Setup:** Select PTC network mode for an array with at least two networks. Keep one network's CAL data and exact group state fixed while changing admitted data/support only in the other network.

**Expected:** The unchanged network's centering, fitted subspace, rank, time-local guard, transformed output, and compatible fixed-state kernel remain identical. The changed network has its own new state.

**Prohibited outcome:** cross-network borrowing or state changes in the untouched network.

### `IS-007` — array-group coupling

**Setup:** Repeat `IS-006` under the explicitly selected single array-wide PTC group.

**Expected:** A change in one network may alter the array-wide subspace and outputs in another network because both are in one declared group. The result remains one array-local operator and is not represented as sequential network-then-array cleaning.

**Prohibited outcome:** claiming isolation in array mode or silently applying both grouping levels.

### `IS-008` — exact time-local PTC full-rank guard

**Setup:** Use a feasible positive configured rank globally. At one group-time, provide a finite application mask whose normal matrix has numerical rank equal to the configured rank under frozen tolerance; at an adjacent group-time, make it deficient.

**Expected:** The first group-time is eligible for exact detector-right mask-aware application. Data and any compatible kernel at the deficient group-time are unavailable with exact cause. Other groups are unchanged.

**Prohibited outcome:** lower-rank pseudoinverse output, rank clipping, interpolated coefficients, admitted masked zeros, or cross-group detector borrowing.

## RTC-only terminal-route scenarios

### `IS-009` — successful logical RTC terminal completion

**Setup:** Explicitly disable PTC and provide a complete required consumer-neutral RTC logical stream and final observation-level RTC facts. Produce stream elements incrementally across multiple engineering chunks without requiring one observation-sized serialized object.

**Expected:** The route completes successfully after RTC. CAL, PTC, and MAP are not entered and no product/response/covariance states for them are fabricated. Temporary cross-chunk state need not be persisted. No unnamed consumer acceptance is required.

**Prohibited outcome:** CAL or PTC execution, a PTC-disabled map, a mandatory observation-sized file, or external-consumer acceptance claim.

### `IS-010` — RTC terminal required-content failure

**Setup:** Use the RTC-only request but omit one required logical member or fail finalization of a required observation-level fact.

**Expected:** The RTC terminal route fails with exact incomplete-content cause. Readable partial content may remain diagnostic but is not labeled a complete RTC publication.

**Prohibited outcome:** successful terminal status based only on partial materialization.

### `IS-011` — chunk-boundary invariance

**Setup:** Apply one RTC operator whose scientific domain is a declared segment. Execute it once as one segment and again as several arbitrary engineering chunks with the required state, overlaps, guards, and finalization.

**Expected:** Both executions produce the same domain-level scientific result within the operator's declared tolerance, with identical support, phase, representative occurrences, and causes. Chunk edges do not become scientific support or resets unless the operator explicitly assigns them that role.

**Prohibited outcome:** chunk-dependent filtering, per-chunk exposure creation, or mandatory per-chunk sidecars inferred from completion language.

### `IS-012` — conditioned `r` local unavailability with valid `x`

**Setup:** Request conditioned `r` on the common grid. Introduce an admitted `x`-only donor repair whose full causal influence overlaps several RTC outputs and supply no justified `r` reconstruction.

**Expected:** Conditioned `x` remains available where its own state permits. Conditioned `r` and its response are unavailable over the full donor influence, without dropped/reindexed grid locations. The common coordinate and raw-`r` parentage remain available.

**Prohibited outcome:** copying the donor repair into `r`, invalidating otherwise valid `x`, or dropping unavailable `r` locations.

## Unavailable and fail-closed scenarios

### `IS-013` — malformed native pair

**Setup:** Supply a native `x` occurrence with no exact `r` partner, or with an ambiguous Tune/mapping revision.

**Expected:** Required native-pair admission fails at the affected scope. ALIGN and RTC do not synthesize a partner or infer the transform.

### `IS-014` — ALIGN assignment collision

**Setup:** Give two original detector rows that both assign to one slot, or a tolerance at/above half a sample.

**Expected:** ALIGN fails before conditioning; no arbitrary row is selected and no downstream route is entered for the failed scope.

### `IS-015` — circular antipode

**Setup:** Request midpoint interpolation of a declared circular field whose valid endpoints are exactly antipodal, with no unwrap authority.

**Expected:** The field is unavailable with an ambiguity cause. Other fields and detector occurrences remain independently evaluable.

### `IS-016` — unbracketed or cross-observation WVR

**Setup:** Place a detector time before the first current-observation WVR record, after the last, across a source-invalid gap, or bracket it only with a record from another observation.

**Expected:** The affected sample state is `outside_supported_calibration` with the exact WVR cause; no calibrated value or multiplier is emitted. Supported samples elsewhere remain independent.

**Prohibited outcome:** nearest/hold value, unity correction, header fallback, climatology, default, or cross-observation opacity.

### `IS-017` — atmosphere numeric-domain failure

**Setup:** Supply finite opacity/elevation outside `0 <= tau225 <= 0.25` or `25 <= elevation_deg <= 80`, or an unsupported/nonfinite spectral-index request.

**Expected:** Sample-local CAL output is unavailable/invalid at exact scope and no clamping or alpha interpolation/extrapolation occurs. Observation classification remains a distinct axis.

### `IS-018` — unrecoverable RTC-grid pointing

**Setup:** Remove the required geometry association, coordinate parent, or rotation state needed to recover the RTC-grid coordinate.

**Expected:** The observation attempt halts. No CAL, PTC, ordinary science, or companion-ML handoff is published. Raw inputs and typed diagnostics may remain.

### `IS-019` — PTC rank zero or unrealizable positive rank

**Setup:** Request `k=0` in one case and a positive rank exceeding exact group-local feasibility in another.

**Expected:** Each request fails closed with exact group and cause. There is no transformed signal, centering-only output, clipped rank, RTC-terminal reinterpretation, or map.

### `IS-020` — PTC required operator input missing

**Setup:** At an otherwise valid fit-excluded/application-requested occurrence, remove one required loading, detector binding, metric term, coordinate transform, boundary state, or coefficient-recomputation input.

**Expected:** The affected exact group-time application is unavailable. A calculated substitute is not ordinary PTC output.

## Conflict and compatibility scenarios

### `IS-021` — conflicting duplicate WVR record

**Setup:** Supply two nonidentical WVR values with the same source time, and make that time an exact match or required bracket endpoint.

**Expected:** Opacity is unavailable at the exact time and for every bracket using it. The sample cause is `wvr_tau225_conflicting_duplicate`; complete-window classification is `opacity_quality_unavailable` unless a higher-precedence invalid-input condition applies.

### `IS-022` — VAL structural conflict

**Setup:** Present conflicting parent/profile/source-binding identities before the named-use decision domain is established.

**Expected:** Applicability is unknown and the decision is `decision_unavailable`; no numerical consumer action is authorized. All conflicting records remain visible.

### `IS-023` — decisive false plus unrelated unknown

**Setup:** After structural applicability is established, make one required PTC restriction decisively false and leave an unrelated advisory or non-gating fact unknown.

**Expected:** The exact named use is `ineligible`; the unrelated unknown remains recorded. No permitting fact rescues the exclusion.

### `IS-024` — quality-token scope collision

**Setup:** Provide only the raw phrase `outside supported calibration` with no owning field/object scope. In separate controls, supply the exact sample-local field and the complete successor observation-class inputs.

**Expected:** The raw unscoped token is unavailable and is not guessed into either a sample state or observation class. The sample-local field canonicalizes to `outside_supported_calibration`. The observation class is recomputed by the successor classifier and is not obtained by spelling conversion.

### `IS-025` — passband alias and digest guard

**Setup:** Use an exact legacy `toltec_v1_a1100` node-table spelling with array `a1100` and the approved member/set digests. Then alter the array, spelling, member digest, or set digest in separate challenges.

**Expected:** The exact legacy spelling canonicalizes only to `tolteca_v1_a1100` while preserving the raw spelling as provenance. Every altered challenge fails closed; no fuzzy name or array inference is accepted.

## Response scenarios

### `IS-026` — RTC identity response

**Setup:** Use the exact `IS-001` RTC identity plan on valid original paired support.

**Expected:** The RTC-local `x` response is identity. If conditioned `r` is requested and available, the canonical pair response is block-diagonal identity with zero cross-coordinate derivatives. Support and representative occurrence remain exact.

### `IS-027` — data/kernel state identity in PTC

**Setup:** Supply a compatible CAL-grid response companion to a resolved PTC group-time that passes the full-rank guard.

**Expected:** The companion is acted on by the same frozen group, mask, metric, rank, subspace, generalized inverse/tolerance, boundary, and support as data. It does not enter learning or alter the science result, and frozen `lambda` is not subtracted from it.

### `IS-028` — no double upstream response

**Setup:** Create two logically equivalent response entries: one companion already on the CAL grid and one source-domain parent with an admitted complete `K_up->CAL` in a future authorized fixture.

**Expected:** The CAL-grid companion enters `J_Theta` directly. The source-domain companion uses `J_Theta o K_up->CAL` exactly once. Applying `K_up->CAL` again is rejected as a domain/identity error.

### `IS-029` — response unavailable without source/beam seed

**Setup:** Request the packet's source/beam-to-PTC conditional response role without adding any authority beyond the locked clean-room packet.

**Expected:** The PTC-local derivative may remain defined, but the complete source-domain role is unavailable because the concrete source/beam `K_up->CAL` seed is absent. No beam, MAP, or implementation default fills it.

**Readiness consequence:** confirms `TS-R` is not ready.

### `IS-030` — response state independent of signal realization

**Setup:** Realize a valid ordinary PTC transformed signal while withholding the complete-chain response seed.

**Expected:** Product realization is `realized` and complete response is `unavailable` or `not_computed_or_not_requested_for_this_product` as appropriate. One state does not rewrite the other.

## Uncertainty and covariance scenarios

### `IS-031` — missing covariance seed

**Setup:** Realize the exact ordinary signal route but supply no admitted CAL-grid statistical covariance.

**Expected:** The calibrated and PTC signals may remain valid; conditional covariance is unavailable. No zero variance, infinite weight, diagonal/white assumption, or independence assertion is created.

**Readiness consequence:** confirms `TS-U` is not ready from this packet alone.

### `IS-032` — conditional covariance propagation with off-diagonal terms

**Setup:** Under a future separately admitted covariance producer, supply a dense positive-definite CAL-grid covariance with cross-time and cross-detector terms and hold the exact PTC state fixed.

**Expected:** PTC conditional covariance equals `J_Theta C_Y J_Theta^T`, retains axes, units, support, group, rank, approximation and omitted-correlation declarations, and preserves nonzero off-diagonal structure. This tests propagation only and does not retroactively supply the missing packet authority.

### `IS-033` — selection term cannot be silently omitted

**Setup:** Compare an ensemble in which fitted rank, support, mask, or group state can change with one in which state is frozen.

**Expected:** The frozen case may report conditional covariance. The varying-state case must retain the between-selection term or explicitly label its omission/selection uncertainty unavailable. The two are not both called total covariance.

### `IS-034` — common nuisance does not average down

**Setup:** Under a future quantified nuisance authority, apply one common fractional calibration-scale term to many samples in the same array.

**Expected:** The nuisance contribution has rank-one form proportional to `mu mu^T` and does not decrease as independent `N^-1/2` noise. Missing cross-array covariance remains unavailable.

## Exposure and influence scenarios

### `IS-035` — original-invalid versus synthesized exposure

**Setup:** Compare an original-valid occurrence, original-invalid occurrence, ALIGN-synthesized occurrence, missing slot, and RTC-replaced representative occurrence.

**Expected:** Original-valid has producer-authoritative `e_acq` and matching `e_vo`; original-invalid may have `e_acq>0` and `e_vo=0`; synthesized/missing adds zero acquisition; replacement creates no new acquisition and the representative is not independent exposure. All causes remain distinct.

### `IS-036` — nonrepresentative influence is not a universal veto

**Setup:** Let an RTC output have an original, unreplaced representative occurrence but depend through a filter on a different synthesized or replaced occurrence.

**Expected:** The representative can satisfy the direct independent-exposure invariant while nonrepresentative influence remains an exact cause for the named use owner. VAL does not turn that influence into a universal exclusion or erase it.

### `IS-037` — overlapping outputs do not duplicate acquisition

**Setup:** Produce several filtered/decimated RTC and PTC outputs whose transitive supports overlap the same original occurrences.

**Expected:** Exposure accounting resolves and deduplicates stable original-occurrence identities. Summing output cadence, filter width, kernel sums, finite cells, weights, or hits is rejected as physical exposure.

### `IS-038` — use-qualified exposure remains separately owned

**Setup:** Request a retained, projected, or generic usable exposure value without a separately named use owner, exact population, deduplication rule, formula, units, and missing behavior.

**Expected:** The requested derived exposure is unavailable. Original `e_acq`/`e_vo` facts and lineage remain available and unchanged.

## Lifecycle and provenance scenarios

### `IS-039` — RTC plan immutability

**Setup:** Learn evidence, resolve one plan, begin apply, and then introduce new candidate evidence mid-apply.

**Expected:** The active plan remains unchanged. The new evidence belongs to a successor learn-resolve-apply attempt with a new identity; it cannot mutate the current result.

### `IS-040` — PTC one-fit ordinary lifecycle

**Setup:** Run the ordinary configured-rank PTC route and change an advisory post-fit diagnostic while keeping CAL parent and resolved operator state fixed.

**Expected:** Exactly one immutable-parent fit and zero support-changing refinements are recorded. The diagnostic record may change; support, rank, subspace, transformed data, and kernel do not.

### `IS-041` — reference-first lineage reconstruction

**Setup:** Materialize compact products containing exact parent references rather than copied upstream histories.

**Expected:** The complete signal, coordinate, response-status, support, exposure, use-decision, and lifecycle chain is resolvable without duplicating the complete APT, WVR, ALIGN, RTC, or CAL histories. Missing a required parent link makes only the dependent role incomplete/unavailable.

### `IS-042` — successor source-binding digest mismatch

**Setup:** Evaluate a registered PTC named-use profile while omitting or altering the approved WP-7 successor compatibility digest, base source-binding digest, or profile-registry digest.

**Expected:** The successor-generation decision is `decision_unavailable`. VAL does not substitute the latest source, alter the immutable profile, or reinterpret an earlier base-generation decision.

### `IS-043` — observation-order and state-leakage challenge

**Setup:** Process two observations with distinct Tune mappings, ALIGN offsets, APT children, WVR records, RTC plans, and PTC groups in both orders and in isolated/parallel schedules.

**Expected:** Each observation reconstructs the same requested, effective, observation-resolved, learned, applied, and published identities in every schedule within its numerical gate. No factor, offset, WVR record, plan, state, or parent from one observation appears in the other.

## Tier implications of the suite

| Tier | Scenario implication |
| --- | --- |
| `TS-A` | The complete suite enforces type, ownership, failure, lifecycle, and no-cycle invariants. |
| `TS-S` | `IS-001` supplies the exact witness; `IS-002`--`IS-008` exercise its identity, CAL, PTC, and use-policy closure. |
| `TS-C` | `IS-001`, `IS-012`, `IS-018`, and `IS-035`--`IS-038` exercise coordinate and exposure-lineage completion. |
| `TS-R` | `IS-026`--`IS-030` distinguish local propagation from the absent source/beam seed; the tier remains unavailable. |
| `TS-U` | `IS-031`--`IS-034` distinguish exact conditional propagation from the absent admitted covariance seed; the tier remains unavailable. |
| `TS-T` | No scenario promotes a lower-tier result. A future stronger-claim suite requires a separately named claim and new owner-bound response/uncertainty authorities. |

## Independent-phase stop

This suite is locked with the independent report. Do not amend it using regression or comparison material. Provide the separate comparison packet for a later mapping-only phase.
