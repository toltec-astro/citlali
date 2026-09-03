# RTC–CAL–PTC Horizontal Contract-Coherence Findings

Audit identity: `v0.1/r0.1`, implementation-blind, branch `codex/scientific-contract-library`, commit `9564bcca0323dacb8bea13a5ec4bbbf3b908de8f`. RTC and PTC are frozen scientific authorities. CAL is draft and not frozen. These findings make no claim about code, data products, conformity, validation, performance, or readiness.

## HCF-001 — Core product roles and once-only operations

- **Classification:** COHERENT
- **Scope:** RTC→CAL→PTC main `x` path
- **Exact producer clauses:** RTC `SCI-RTC-EQ-001–006`, `SCI-RTC-REQ-001–005/013–016/103`; CAL `SCI-CAL-EQ-004–006`, `SCI-CAL-REQ-013–016/020–031/047–048`.
- **Exact consumer clauses:** CAL `SCI-CAL-REQ-001/013–016`; PTC `SCI-PTC-REQ-001/010/046/063/078/087`.
- **Explanation:** RTC owns conditioned raw `x`, not absolute calibration. Its donor `flxscale` ratio is raw-coordinate convention transfer. CAL owns the sole selected absolute factor and target-atmosphere correction, each applied once. PTC begins from the immutable already calibrated CAL parent and does not repeat either operation.
- **Scientific consequence:** The three package-local roles can be composed without double calibration when the exact parents and operators are available.
- **Profile invariant:** Yes: CHAIN-INV-002/004/007–011/013/016/019/021/027–031/036–037/039–040.
- **Possible owner dispositions:** None required for this invariant; retain exact application counts and parent identities.
- **Downstream impact:** Establishes the safe algebraic skeleton, but not the currently unavailable numerical CAL output.

## HCF-002 — Exact RTC product and signal semantics are not yet bound by CAL

- **Classification:** MISSING CONSUMER DISPOSITION
- **Scope:** RTC→CAL
- **Exact producer clauses:** `SCI-RTC-DEF-004/018`, `SCI-RTC-REQ-001–005/103` guarantee an exact conditioned-`x` handoff with bundle lineage.
- **Exact consumer clauses:** `SCI-CAL-REQ-001/003/020`; `CAL-OWNER-Q01` leaves the physical `xs` meaning, unit, sign, preprocessing history, and valid domain unresolved.
- **Explanation:** RTC names and guarantees its exact product. CAL admits ordinary `xs` but does not bind that term to the SCI-RTC conditioned-`x` product or its exact coordinate convention. Similar names or compatible array shapes cannot close an identity relation.
- **Scientific consequence:** The profile cannot presently assert that CAL consumes the exact RTC output or that RTC’s coordinate/unit/sign/reference survive the handoff.
- **Profile invariant:** No for CHAIN-INV-001/003; CHAIN-INV-002/007 are conditional on this binding.
- **Possible owner dispositions:** (a) bind CAL input exactly to the SCI-RTC atomic conditioned-`x` product; (b) authorize a named intervening transformation with exact identity/response/support; or (c) declare the packages noncomposable on this route.
- **Downstream impact:** Blocks an authoritative end-to-end numerical parent chain and all claims depending on exact input semantics.

## HCF-003 — Direct replacement and transitive influence align through PTC

- **Classification:** EXPLICIT NARROWING
- **Scope:** RTC→PTC, with CAL as the required bridge
- **Exact producer clauses:** `SCI-RTC-DEF-011–013`, `SCI-RTC-EQ-020`, `SCI-RTC-REQ-019–020/046–052`.
- **Exact consumer clauses:** `SCI-PTC-REQ-003/011–017/089`.
- **Explanation:** RTC universally excludes an output as an independent measurement when its exact representative occurrence was ALIGN-synthesized or RTC-replaced. It preserves noncenter synthesized/replaced influence as typed cause and dependency. PTC agrees on direct exclusion and narrows the remaining domain for each PTC-owned named use using a conjunctive policy; unknown required predicates yield `decision_unavailable`.
- **Scientific consequence:** PTC does not collapse all influence into a universal rejection and does not rescue an upstream exclusion.
- **Profile invariant:** Yes: CHAIN-INV-018/022/024/038. Passage through CAL remains subject to HCF-006.
- **Possible owner dispositions:** None for RTC/PTC semantics; CAL must still state its carriage/disposition.
- **Downstream impact:** Preserves scientific independence claims and allows use-specific treatment of noncenter influence.

## HCF-004 — Complete upstream response ending on the CAL grid lacks one producer guarantee

- **Classification:** RESPONSE GAP
- **Scope:** RTC→CAL→PTC
- **Exact producer clauses:** RTC `SCI-RTC-DEF-004/010`, `SCI-RTC-EQ-006/012–015`, `SCI-RTC-REQ-037–041`; CAL rationale §2 noncommutation identity, `CAL-OWNER-Q02`, `SCI-CAL-EQ-008–009`, `SCI-CAL-REQ-039–043/047`.
- **Exact consumer clauses:** `SCI-PTC-DEF-041`, `SCI-PTC-REQ-061–066/087`.
- **Explanation:** RTC truthfully publishes its RTC-local response or unavailability. CAL defines its local multiplier and same-support companion propagation. PTC, however, requires one complete admitted upstream response ending on the CAL grid, including source/detector, beam/scan, RTC, and CAL contributions. CAL does not unambiguously promise that cumulative object in its product. A set of local objects is not automatically the exact cumulative object PTC names. Further, CAL’s sample-dependent atmosphere correction does not generally commute with RTC temporal filtering; RTC requires exact composition or an owner-approved noncommutation bound, while CAL `Q02` leaves the cross-stage order unresolved.
- **Scientific consequence:** Complete-chain response-dependent use is unavailable, even though RTC-local, CAL-local, and PTC-local response objects may each be well-defined. Guessing a composition could omit a term or apply one twice.
- **Profile invariant:** No for CHAIN-INV-015; local propagation in CHAIN-INV-010/016 and domain separation in CHAIN-INV-029 remain supported.
- **Possible owner dispositions:** (a) make CAL compose and publish the cumulative response, including the filter/atmosphere relation; (b) define a separate cross-package response assembler with exact domain/parent authority; or (c) make PTC consume an ordered response bundle with an exact composition rule and approved noncommutation bound.
- **Downstream impact:** Blocks complete PTC chain response and any response-dependent amplitude/peak claim.

## HCF-005 — CAL “point-source-peak” and “point-source-equivalent” wording conflict

- **Classification:** NORMATIVE INTERNAL CONFLICT
- **Scope:** CAL internal, with CAL→PTC consequence
- **Exact producer clauses:** CAL formal `SCI-CAL-REQ-002/040–043` and engineering abstract use “point-source-peak”; the active r0.3 rationale §§1–2 and 5 uses “point-source-equivalent, beam-peak-normalized” and makes literal peak conditional on realized downstream response or renormalization.
- **Exact consumer clauses:** `SCI-PTC-REQ-001/010/061/087` consumes point-source-equivalent mJy per fixed nominal beam and explicitly refuses to infer literal peak, unchanged beam, extended-source fidelity, preserved absolute level, or detector-combination fidelity from the unit.
- **Explanation:** The rationale’s narrower interpretation is scientifically compatible with PTC, but the rationale cannot silently override the higher-priority formal wording. “Point-source-peak” is therefore not safely equivalent to “point-source-equivalent” in the current CAL package.
- **Scientific consequence:** Literal peak claims must be withheld; the profile may state only the narrower point-source-equivalent convention as a conservative proposal, not frozen CAL authority.
- **Profile invariant:** CHAIN-INV-012 is unresolved; PTC’s refusal to overinterpret the unit remains supported.
- **Possible owner dispositions:** (a) amend the CAL formal core to point-source-equivalent plus a conditional literal-peak clause; (b) retain literal peak and strengthen the required response/renormalization authority; or (c) define two distinct output roles.
- **Downstream impact:** Blocks unqualified source-peak language and requires an owner resolution before an end-to-end amplitude claim.

## HCF-006 — CAL does not govern all RTC cause and representative-state axes

- **Classification:** MISSING CONSUMER DISPOSITION
- **Scope:** RTC→CAL
- **Exact producer clauses:** `SCI-RTC-DEF-011–013/018`, `SCI-RTC-EQ-020/022`, `SCI-RTC-REQ-019–020/046–052/103–105`.
- **Exact consumer clauses:** `SCI-CAL-REQ-003–004/045–048` require typed input/validity/lineage but do not explicitly preserve or disposition RTC representative synthesis/replacement and transitive influence.
- **Explanation:** RTC publishes material scientific state that later eligibility and response claims can depend on. CAL’s generic validity and lineage clauses do not establish whether every RTC cause axis is carried, narrowed, or ignored. CAL may not silently turn a cause into validity, a mask, zero, or an original-exposure claim.
- **Scientific consequence:** Cause-dependent CAL eligibility and downstream claims cannot be made from the CAL product alone.
- **Profile invariant:** No for CHAIN-INV-006/023; RTC→PTC semantics remain coherent under HCF-003 if the facts are preserved across the bridge.
- **Possible owner dispositions:** (a) require transparent carriage of all RTC causes with CAL-local validity separate; (b) define an explicit CAL narrowing policy; or (c) require PTC to bind the RTC bundle separately and prohibit CAL-dependent claims on those axes.
- **Downstream impact:** Blocks complete support/eligibility reconstruction and contributes to the full-parent gap.

## HCF-007 — PTC’s required complete RTC parent is not fully guaranteed by CAL lineage

- **Classification:** MISSING PRODUCER GUARANTEE
- **Scope:** CAL→PTC with RTC ancestry
- **Exact producer clauses:** RTC `SCI-RTC-DEF-018`, `SCI-RTC-EQ-022`, `SCI-RTC-REQ-103–105`; CAL `SCI-CAL-REQ-047–048` supplies canonical lineage/product links.
- **Exact consumer clauses:** `SCI-PTC-REQ-002` requires the complete RTC parent, including conditioned `x`, raw-`r` parent, exact grid, selectors, segmentation, validity, response, uncertainty, replacement, and influence.
- **Explanation:** RTC has the required atomic bundle, and PTC explicitly requires it. CAL’s lineage clauses do not clearly guarantee that its product binds the entire RTC bundle or a resolvable immutable reference to it. PTC can bind RTC separately, but the exact cross-product relation must be authoritative rather than inferred.
- **Scientific consequence:** The complete parent chain is not reconstructible solely from the current handoff promises.
- **Profile invariant:** No for CHAIN-INV-017.
- **Possible owner dispositions:** (a) require CAL to bind an immutable RTC bundle reference; (b) define a chain manifest binding CAL and RTC parents jointly; or (c) narrow PTC’s requirement through explicit owner action.
- **Downstream impact:** Blocks complete PTC interpretation of RTC-derived support, influence, response, and uncertainty.

## HCF-008 — Uncertainty types are structurally compatible but numerical completeness remains open

- **Classification:** UNCERTAINTY GAP
- **Scope:** chain-wide
- **Exact producer clauses:** RTC `SCI-RTC-EQ-016–019`, `SCI-RTC-REQ-042–045`; CAL `SCI-CAL-EQ-010–013`, `SCI-CAL-REQ-032–039`, `CAL-OWNER-Q08`.
- **Exact consumer clauses:** `SCI-PTC-REQ-057–060/066/088`, `PTC-OWNER-OD-009`.
- **Explanation:** All three packages distinguish conditional measurement covariance from nuisance/systematic, selection, response, model/null-space, cross-coordinate, and cross-observation terms; missing terms are unavailable, not zero. CAL’s exact uncertainty-product set/completeness is still open, and PTC covariance is deferred. Thus the types compose, but no complete numerical covariance or total uncertainty can be promised.
- **Scientific consequence:** A transformed numerical product may be scientifically available with truthfully limited or unavailable covariance, but no total-uncertainty claim is allowed.
- **Profile invariant:** CHAIN-INV-014/025/039 are supported as typed conditional states; CHAIN-INV-026 is an explicit claim narrowing.
- **Possible owner dispositions:** (a) define a minimum conditional covariance handoff only; (b) require selected CAL nuisance/correlation products; or (c) require a complete cross-stage covariance package.
- **Downstream impact:** Limits weighting, significance, and total-error claims; does not necessarily block a signal-only product.

## HCF-009 — Disabled and required-failure semantics agree

- **Classification:** COHERENT
- **Scope:** chain-wide
- **Exact producer clauses:** RTC `SCI-RTC-REQ-046–053`; CAL `SCI-CAL-REQ-004/010–012/021–031/045–048`.
- **Exact consumer clauses:** PTC `SCI-PTC-REQ-003/066/073–077`.
- **Explanation:** Invalid association/factor/atmosphere/support produces no calibrated CAL number; it is not identity calibration. PTC does not repair an unavailable parent. PTC disabled means no PTC product, and direct CAL→MAP is a separate route outside PTC.
- **Scientific consequence:** Product existence, response availability, covariance availability, and evidence status remain separate. Required failure terminates the requested route.
- **Profile invariant:** Yes: CHAIN-INV-010/020/027/036–037/040.
- **Possible owner dispositions:** None required for these semantics.
- **Downstream impact:** Prevents finite defaults, stale products, and identity-cleaned pseudo-products.

## HCF-010 — Optional conditioned-`r` branch is coherent but has no supplied producer

- **Classification:** MISSING PRODUCER GUARANTEE
- **Scope:** optional RTC→PTC diagnostic branch; CAL excluded
- **Exact producer clauses:** RTC `SCI-RTC-REQ-092/103/108`; RTC owner entries `071/074` state that conditioned `r` needs a separate operator and remains unavailable without it.
- **Exact consumer clauses:** `SCI-PTC-REQ-078`, decided `PTC-OWNER-Q001`, and `PTC-OWNER-OD-005`.
- **Explanation:** Raw `r` is retained but is neither calibrated nor processed by the `x` operator. PTC allows only an optional, separately conditioned, diagnostic-only and inert/advisory `r` parent. No producer, unit, response, support, validity, leakage state, uncertainty, provenance, or exact CAL-grid relation is supplied.
- **Scientific consequence:** The optional branch is unavailable; the main calibrated-`x` chain is unaffected. An incompatible conditioned-`r` grid also blocks only that branch.
- **Profile invariant:** CHAIN-INV-031–033 are supported as separation/failure rules; no invariant asserts that a conditioned-`r` product exists.
- **Possible owner dispositions:** author a separate `r` producer; omit the branch; or define a successor PTC mode with different authority.
- **Downstream impact:** No base-v0.1 `r` diagnostic product; no effect on `x` fit membership, subtraction, output, or coefficients.

## HCF-011 — CAL cannot currently produce a numerical calibrated product

- **Classification:** MISSING PRODUCER GUARANTEE
- **Scope:** CAL→PTC numerical chain
- **Exact producer clauses:** `SCI-CAL-ASM-011`, `SCI-CAL-REQ-021–031/045–046`, `CAL-OWNER-Q06` require a content-bound atmosphere operator and otherwise demand explicit unavailable/no calibrated output.
- **Exact consumer clauses:** `SCI-PTC-REQ-001` and `PTC-OWNER-OD-007` require an admitted numerical CAL parent with complete response/uncertainty status.
- **Explanation:** The exact atmosphere nodes, orientation, content identity, and support are not supplied. CAL explicitly forbids numeric atmosphere evaluation or calibrated output until that record exists. PTC therefore lacks its mandatory numerical parent.
- **Scientific consequence:** The present packet supports structural composition only; no numerical CAL or PTC product exists under this chain authority.
- **Profile invariant:** CHAIN-INV-034 is unavailable; all numerical CAL→PTC invariants are conditional.
- **Possible owner dispositions:** supply the exact CAL atmosphere record; authorize a different explicit operator; or leave numerical CAL unavailable.
- **Downstream impact:** This is a direct blocker to an end-to-end numerical implementation audit.

## HCF-012 — CAL has not selected its position relative to PTC

- **Classification:** OWNER DECISION REQUIRED
- **Scope:** chain order
- **Exact producer clauses:** RTC `SCI-RTC-EQ-004`, `SCI-RTC-REQ-013` place CAL after RTC; PTC `SCI-PTC-REQ-001/046/063` begins from immutable CAL output.
- **Exact consumer clauses:** `CAL-OWNER-Q02` and CAL rationale §2 leave baseline/affine ordering relative to common-mode/PCA/filtering/map stages unresolved; PTC `CROSS_PACKAGE_FOLLOWUP.md` routes CAL-before-PTC to the CAL owner.
- **Explanation:** RTC and PTC support RTC→CAL→PTC, but CAL does not yet adopt that complete order. This is not an asserted opposite order, so it is not classified as an order contradiction; more than one sequence remains scientifically possible under CAL’s open authority.
- **Scientific consequence:** The full chain operation order is proposed, not authoritative.
- **Profile invariant:** No for CHAIN-INV-035 until CHAIN-OD-002 is resolved.
- **Possible owner dispositions:** CAL before PTC; PTC before CAL with a complete redefinition of quantities and factors; or separate ordered product roles.
- **Downstream impact:** Blocks a single authoritative numerical chain and response composition.

## HCF-013 — PTC’s fixed-nominal-beam precondition is stronger than CAL’s guarantee

- **Classification:** CONSUMER STRENGTHENING
- **Scope:** CAL→PTC photometric/response boundary
- **Exact producer clauses:** CAL response-basis definition and `SCI-CAL-REQ-040–043` bind the calibrated result to the originating Beammap template/beam occurrence and require the realized downstream response separately; CAL itself does not certify the downstream response.
- **Exact consumer clauses:** `SCI-PTC-ASM-001`, `SCI-PTC-REQ-001/010` require point-source-equivalent mJy per fixed nominal beam.
- **Explanation:** CAL preserves an originating, potentially occurrence-specific Beammap response basis and a conditional peak interpretation. That does not by itself establish equivalence to the single fixed nominal beam identity assumed by PTC. RTC’s description of the downstream CAL role cannot fill a guarantee that CAL itself does not make.
- **Scientific consequence:** PTC’s fixed-nominal-beam input meaning is unavailable unless CAL supplies that identity or an explicit response-basis conversion/renormalization.
- **Profile invariant:** No for CHAIN-INV-041.
- **Possible owner dispositions:** (a) change PTC to preserve the exact CAL response-basis identity; (b) add a CAL-owned conversion to a fixed nominal beam with complete response/uncertainty lineage; or (c) define separate PTC input roles.
- **Downstream impact:** Blocks an unqualified fixed-nominal-beam PTC product and joins the response/peak claim blockers.

## Terminology and state comparison

| Term | RTC meaning | CAL meaning | PTC meaning | Audit relation |
|---|---|---|---|---|
| signal | Conditioned raw detector `x`; calibrated product is distinct | Admitted ordinary `xs`, then calibrated detector quantity | Exact calibrated `x` parent, then transformed detector signal | Conditional continuity; CAL input binding open |
| raw | Admitted aligned `x/r`; `r` retained raw | Pre-CAL `xs` is not fully defined by Q01 | Raw `x/r` excluded from PTC main input | Different stage concepts |
| calibrated | Not an RTC output | Factor + atmosphere applied once | Immutable CAL parent convention preserved | Exactly staged |
| point-source-equivalent | RTC calls downstream CAL product equivalent | Active CAL rationale uses equivalent/beam-peak-normalized | Exact PTC input convention | Compatible, but see formal conflict |
| point-source peak | Not an RTC claim | Formal core uses it; literal peak later made response-conditional | Explicitly not inferred from unit | **Terminology/internal collision** |
| nominal beam | RTC describes the downstream CAL role as fixed-nominal-beam, but does not own CAL | Originating Beammap response basis; downstream realized response separate | Requires fixed nominal beam | **Consumer strengthening unless converted** |
| response | Complete RTC-local realized derivative/status; optional ALIGN extension separate | Local multiplier and response-basis/companion state | Local, complete upstream, chain, companion, full-procedure, whole-chain are separate | Compatible domains; cumulative CAL-grid object missing |
| kernel | Exact RTC-local derivative on aligned grid | Matrix/multiplier operator; not necessarily called kernel | Domain-qualified response operator | Terminology mismatch, not synonym by default |
| response companion | Fixed-state source/perturbation object through realized operation | Companion multiplied by same exact CAL operator/support | Source-domain or CAL-grid companion with exact parent domain | Conditionally equivalent |
| support | Full RTC transitive support and influence | Admitted factor/atmosphere/operator domain restriction | Distinct fit, loading, application, output, coefficient, response supports | Stage-specific; consumer narrowing visible |
| influence | Typed transitive causal closure | No full RTC-specific disposition | Accumulated causes plus named-use predicates | RTC/PTC coherent; CAL gap |
| valid | Multiple independent axes; not finite/eligible | Valid occurrence/operator state | Product/state validity independent of eligibility and availability | Compatible only when axis named |
| eligible | Direct RTC base plus consumer policy | Science-qualification-eligible is a claim layer, not sample eligibility | Use-specific conjunctive support decision | Same word, different scopes; terminology collision avoided by qualification |
| finite | Numerical property only | Numerical property only | Numerical property only | Exactly non-equivalent to valid/available |
| available | Required authority/state exists for named object/claim | Operator/product/uncertainty availability | Independent product/response/covariance/evidence axes | Compatible when object axis named |
| unavailable | Typed absence; never zero/default | No calibrated output where required authority absent | Unknown is not pass; no repair | Exactly compatible |
| invalid | Failed validity predicate | Invalid occurrence/factor/operator state | Distinct from unavailable/rejected | Compatible, axis-specific |
| rejected | Candidate/policy disposition | Unsupported/ambiguous request or association disposition | Estimator/candidate/fallback disposition | Genuinely different object scopes |
| disabled | Stage not active; RTC can terminate validly without PTC | Calibration not an identity fallback; no calibrated number where disabled/unavailable | No PTC product | Compatible no-product semantics by stage |
| not requested | Distinct request state | Distinct requested/effective state | Distinct product role state | Exactly compatible |
| not computed | Evidence/operation absent, not pass | No numerical evaluation | Distinct from unavailable and not produced | Compatible when cause named |
| not produced | Required product absent/failure | No calibrated output | No PTC product | Compatible product-existence axis |
| learned | Evidence/candidates from named learning population | No internal learned estimator in base operation | Fitted candidate/evidence state | Genuinely different ownership |
| resolved | Immutable RTC plan selected from evidence/policy | Observation-resolved factor/operator state | Resolved model/support/application map | Same lifecycle layer, different contents |
| frozen | RTC scientific authority or immutable apply plan | Rationale template only; scientific authority not frozen | Scientific authority or frozen learned subspace | Terminology collision unless object named |
| realized | Exact operation, coefficients, support, exceptions, output | Exact factor/operator/output record | Exact selected model, operation, response, and products | Compatible lifecycle layer |
| conditional covariance | Fixed RTC state and declared components | Same exact CAL multiplier, fixed supplied state | Fixed learned/resolved PTC state | Composable with conditioning stated |
| total uncertainty | Only with complete declared components/cross terms | Withheld unless complete nuisance coverage | Not supplied by default | Exactly compatible withholding rule |
| selection uncertainty | Separate RTC selector term | Nuisance/selection ledger term | Separate learned-selection/full-procedure term | Compatible but not numerically supplied |
| null space | Not an RTC output role | Not a CAL output role | PTC-owned removed/preserved operator state | Genuinely PTC-specific |
| additive reference | RTC affine/replacement/boundary terms, not PTC centering | Optional pre-CAL baseline `b`, order open | Learned `lambda` is subtracted and not restored | Same generic phrase would be a collision; objects differ |

## Representative chain-state trace

The current CAL packet forbids numerical calibrated output under `SCI-CAL-ASM-011`. Cases that begin with “if an admitted CAL product exists” are therefore semantic counterfactuals used to test the handoff contract, not assertions that such a product currently exists.

| Case | Numerical/scientific product | Response-dependent use | Cause and uncertainty | Next action / termination |
|---|---|---|---|---|
| 1. Ordinary admitted RTC sample, complete response | RTC product exists; current CAL numerical product does not under ASM-011. Conditional future CAL/PTC path is defined. | RTC-local use allowed; complete-chain use waits on HCF-004 and CAL availability. | Ordinary validity/support plus typed RTC conditional uncertainty survive. | CAL must resolve OD-001/003/004; current chain terminates at RTC. |
| 2. Exact RTC representative occurrence synthesized by ALIGN | RTC output may numerically exist but is not an independent measurement. | Independent-measurement use forbidden; other uses require explicit policy. | Direct synthesis cause survives; uncertainty remains separately typed. | CAL disposition missing; no strengthening. |
| 3. Exact RTC representative occurrence donor-replaced | RTC output may numerically exist but is not independent. | Same direct exclusion; response must include donor mixing. | Replacement cause and influence survive. | CAL disposition missing; PTC cannot rescue exclusion. |
| 4. Nonrepresentative synthesis/replacement elsewhere in support | RTC output may remain valid/available. | Consumer-specific; not universal rejection. | Full transitive influence survives with uncertainty/status. | CAL must carry cause; PTC applies its named-use policy. |
| 5. Valid RTC values, complete response unavailable | RTC signal product may exist with response unavailable. | Response-dependent use forbidden; product and response axes remain separate. | Response-unavailable cause survives; covariance may still be conditional. | CAL must not strengthen; PTC complete response remains unavailable. |
| 6. CAL factor unavailable or detector association ambiguous | No calibrated CAL number. | Not permitted. | Exact factor/association cause retained; uncertainty unavailable for absent output. | No PTC parent; chain terminates at CAL. |
| 7. Target atmosphere unavailable or outside support | No calibrated CAL number. | Not permitted; no identity atmosphere fallback. | Atmosphere/support cause retained. | No PTC parent; chain terminates at CAL. |
| 8. Calibrated `x` available, systematic uncertainty incomplete | Conditional hypothetical CAL and PTC signal products may exist. | Response use depends independently on response status. | Conditional measurement uncertainty may be available; total/systematic claim unavailable. | PTC preserves nuisance gaps; downstream must not call it total. |
| 9. PTC fit-excluded/application-allowed occurrence | PTC output exists only if exact frozen application inputs are available. | Allowed on declared application support; otherwise unavailable. | Fit-exclusion cause remains; application decision and uncertainty are separate. | Publish exact fit/application status or terminate that occurrence. |
| 10. PTC transformed signal available, complete covariance unavailable | PTC signal product may exist. | Allowed only for uses not requiring missing covariance and subject to response status. | Covariance-unavailable cause survives; no zero or total substitution. | Downstream narrows claim or withholds use. |
| 11. PTC disabled | CAL product may exist; no PTC product. | No PTC response claim. | Disabled state survives; CAL uncertainty unchanged. | Chain terminates at CAL; direct CAL→MAP is outside profile. |
| 12. Companion supplied in source-domain form | Conditional on exact parent/domain, it traverses each admitted RTC, CAL, then PTC operator once. | Complete response claim blocked until HCF-004 closes. | Same masks/support/state and CAL multiplier as signal; uncertainty follows declared companion role. | Apply ordered domains once; never relabel fixed-state as whole-chain. |
| 13. Companion already realized on CAL detector-time grid | PTC companion may be produced if parent and PTC state are available. | Starts with PTC-local operator. | CAL-grid parent identity/support survive. | Do not reapply RTC or CAL response. |
| 14. Optional conditioned-`r` parent absent | Main `x` product unaffected; no `r` diagnostic product. | `r`-dependent diagnostics unavailable only. | `PTC-OWNER-OD-005` cause survives; no fabricated uncertainty. | Continue main chain; terminate optional branch. |
| 15. Conditioned-`r` present but incompatible with CAL grid | Main `x` product unaffected; `r` diagnostic branch unavailable. | No implicit resampling/alignment or reuse of `x` response. | Grid incompatibility, unit/response/support state survive. | Require separate authorized relation or terminate branch. |

## Required contradiction-test dispositions

| Test | Disposition |
|---|---|
| RTC retained sample versus consumer assumption of removal | No removal assumption found; CAL cause disposition remains missing (HCF-006). |
| Direct replacement silently treated as original exposure | RTC/PTC agree on exclusion; CAL does not explicitly govern it (HCF-003/HCF-006). |
| Noncenter influence collapsed to universal rejection | No; PTC explicitly applies named-use policy (HCF-003). |
| CAL peak wording versus PTC response-qualified meaning | Internal CAL conflict found (HCF-005). |
| PTC infers peak/beam/absolute-level fidelity from unit | No; explicitly prohibited. |
| PTC fixed-nominal-beam identity supplied by CAL | No; PTC strengthens CAL's originating response-basis guarantee (HCF-013). |
| Different meanings of valid/eligible/available/support/response | Meanings are compatible only with named axes; CAL cause/response gaps remain. |
| CAL local multiplier assumed to contain whole upstream response | PTC requires a stronger cumulative object; response gap found (HCF-004). |
| Source companion receives RTC/CAL response twice | Profile prevents this by parent-domain rule; current packages support local once-only application. |
| Companion follows different support/operator than signal | CAL/PTC require same realized support/operator; no contradiction found. |
| Calibration or response uncertainty disappears | Types are preserved, but numerical completeness is open (HCF-008). |
| PTC full-procedure called whole-chain | No; PTC separates them. |
| Disabled means identity in one package and no product in another | No contradiction; no-product semantics are coherent (HCF-009). |
| Raw `r` calibrated or conditioned `r` assumed | No; optional producer is explicitly absent (HCF-010). |
| Consumer requires producer-unsupplied field/state | Yes: exact CAL input binding, cumulative response, full RTC parent carriage, numeric atmosphere, and conditioned `r` (HCF-002/004/007/010/011). |
| Producer material state ignored by consumer | Yes: CAL does not fully disposition RTC causes/representative state (HCF-006). |
| Rationale and formal core disagree | Yes: CAL peak versus equivalent wording (HCF-005). |

## Finding counts and audit verdict

- Coherent findings: **2** (`HCF-001`, `HCF-009`), covering **27** coherent invariant rows in the source crosswalk.
- Explicit narrowing findings: **1** (`HCF-003`), covering **3** explicit-narrowing rows.
- Contradictions/internal normative conflicts: **1** (`HCF-005`). No asserted cross-package order contradiction was found.
- Missing handoff guarantees, dispositions, response commitments, or unsupported consumer strengthening: **7 material findings** (`HCF-002`, `HCF-004`, `HCF-006`, `HCF-007`, `HCF-010`, `HCF-011`, `HCF-013`).
- Uncertainty gaps: **1** (`HCF-008`).
- Open owner decisions: **8** in the companion ledger.

**Coherent enough to begin implementation audit?** Not for an end-to-end numerical RTC→CAL→PTC audit. The exact blockers are the CAL input identity/semantics and order, absent numerical atmosphere operator, incomplete cumulative response object and RTC-filter/CAL-atmosphere noncommutation closure, incomplete RTC cause/full-parent carriage through CAL, CAL’s peak/equivalent internal wording conflict, and PTC’s unsupported fixed-nominal-beam strengthening. Package-local audits may proceed only under their separate authority and claim limits.
