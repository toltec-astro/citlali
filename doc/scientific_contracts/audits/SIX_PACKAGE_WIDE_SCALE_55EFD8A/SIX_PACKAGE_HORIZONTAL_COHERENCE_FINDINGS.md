# Six-Package Horizontal Coherence Findings

## Audit identity and disposition

This is an implementation-blind contract audit of SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-PTC, and SCI-VAL at immutable repository commit `55efd8a54464636a24e621f6d1b60486d235b20e`. SCI-MAP is used only as a downstream-consumer reference. The audit makes no implementation-conformity, validation, performance, or production-readiness claim.

The package architecture is mostly acyclic and its principal identity, ordering, cause, and lifecycle distinctions are mutually intelligible. It is not source-closed or complete enough for an authoritative MAP handoff. Two conflicts are inside the frozen SCI-PTC normative core; the current VAL source bindings and usable profile set do not cover the consolidated package state; two required coordinate-boundary bodies are absent; complete response, uncertainty, exposure, coefficient, and MAP-projection prerequisites remain unavailable or owner-open; and CAL, VAL, and MAP are not frozen scientific authorities. These are finite blockers, not permission to infer missing facts.

## F-001 — PTC transformed-signal and centering identities conflict

- **Class:** NORMATIVE INTERNAL CONFLICT
- **Scope:** SCI-PTC frozen v0.1/r0.4; signal, response, and null-space lanes.
- **Producer clause:** `packages/SCI-PTC/v0.1/src/common/equations.tex::eq:ptc-subtraction` defines `Z = Y^CAL - U_hat` with `U_hat = M_hat A_hat^T`.
- **Consumer/competing clause:** `equations.tex::eq:center-scale`; `definitions.tex::SCI-PTC-DEF-003`, `DEF-007`, and `DEF-014`; `requirements.tex::SCI-PTC-REQ-023` and `REQ-083`; `AUTHOR_DRAFT_DECISIONS.md::PTC-AUTH-D027` require nonrestoring centering, so the application is the frozen operation on `Y-lambda`, with `lambda` discarded rather than restored.
- **Explanation:** The two equations are equal only if the fitted component in the first equation is redefined to include the learned additive location, or if another unstated identity places that location in the removed subspace. The normative text instead distinguishes `U_hat`, `lambda`, and the removed subspace. The audit does not choose a repair.
- **Scientific consequence:** The exact transformed value, additive null space, fixed-state derivative, and response companion cannot all be derived from one unambiguous frozen equation set.
- **Affected lane:** main signal; response; uncertainty; provenance.
- **MAP/downstream impact:** A PTC-parented sample and its response/null-space record cannot be certified as one exact logical handoff member while the defining identities disagree.
- **Possible owner dispositions:** reopen r0.4 and explicitly incorporate `lambda` into the subtraction identity; redefine the fitted component and all dependent symbols; or issue a versioned successor that selects another mathematically complete identity and maps the old clauses.
- **Blocker status:** **hard blocker** for the PTC-dependent MAP route and any claim using the exact PTC application/response identity.

## F-002 — PTC named-use eligibility equation omits base-false ineligibility

- **Class:** NORMATIVE INTERNAL CONFLICT
- **Scope:** SCI-PTC frozen v0.1/r0.4; policy/eligibility lane.
- **Producer clause:** `packages/SCI-PTC/v0.1/src/common/equations.tex::eq:cause-support` defines `eligible_U = b_U AND all(p_i,U)` and `ineligible_U = any(NOT p_i,U)`.
- **Consumer/competing clause:** `definitions.tex::SCI-PTC-DEF-013`; `requirements.tex::SCI-PTC-REQ-012--013`; and SCI-VAL `requirements.tex::SCI-VAL-REQ-011--016` require a complete, conjunctive, knowledge-aware disposition in which a decisive false restriction is ineligible and unresolved required permission is decision unavailable.
- **Explanation:** If `b_U` is false while every listed `p_i,U` is true, the displayed equations yield neither eligible nor ineligible. The accompanying prose introduces unknown predicates but does not declare the truth algebra or repair the omitted `NOT b_U` term. A frozen PTC use policy therefore has an uncovered state.
- **Scientific consequence:** Basis-fit, loading-fit, application, output, coefficient/QC, response, empirical, and simulation populations cannot be evaluated from this equation without inventing a rule.
- **Affected lane:** policy/VAL; lifecycle.
- **MAP/downstream impact:** No PTC-derived VAL decision can safely authorize a MAP-facing sample or coefficient using this rule.
- **Possible owner dispositions:** amend the exclusion identity to include the base predicate and bind the exact knowledge algebra; replace the equation with a direct four-state rule; or issue a successor policy source consumed through registered VAL profiles.
- **Blocker status:** **hard blocker** for every use that relies on this PTC-local composite, independent of the separate missing-profile blocker.

## F-003 — VAL source bindings are stale for the consolidated package state

- **Class:** SOURCE-BINDING OR VERSION GAP
- **Scope:** SCI-VAL source-binding structural gate.
- **Producer clause:** frozen package states are ALIGN v0.1/r0.3, AST v0.1/r0.3, RTC v0.1/r0.12, PTC v0.1/r0.4; active CAL is rationale r0.5/ECS r0.4.
- **Consumer clause:** `packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md`; `src/common/requirements.tex::SCI-VAL-REQ-008`, `REQ-010`, `REQ-029`, `REQ-044`, and `REQ-049`.
- **Explanation:** The register binds RTC r0.9 rather than r0.12 and CAL r0.3 rather than the active r0.5/r0.4 pair. It states that ALIGN and AST have no standalone frozen version although both are now frozen v0.1/r0.3. PTC r0.4 matches. MAP r0.3 matches the named revision but MAP scientific authority is not frozen.
- **Scientific consequence:** VAL's own structural rules forbid substituting ambient “current” package meaning. Affected evaluations have unknown applicability and decision unavailable, not eligibility.
- **Affected lane:** policy/VAL; provenance.
- **MAP/downstream impact:** independent-exposure, every PTC use, and MAP admission cannot be replayed against the consolidated sources as presently bound.
- **Possible owner dispositions:** issue exact new binding rows with versions/digests and compatibility statements; explicitly supersede old rows; obtain any required package-owner compatibility approval.
- **Blocker status:** **hard blocker** for every decision importing ALIGN, AST, RTC, CAL, or nonfrozen MAP meaning.

## F-004 — Ordinary-chain PTC and MAP profiles are reserved, not usable

- **Class:** PROFILE REGISTRY GAP
- **Scope:** SCI-VAL Profile Registry and every named-use policy in the PTC-to-MAP route.
- **Producer clause:** `packages/SCI-PTC/v0.1/src/common/requirements.tex::SCI-PTC-REQ-012`, `REQ-015`, `REQ-047`, `REQ-052--068`, and `REQ-089` name distinct uses and keep their actions separate.
- **Consumer clause:** `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md`; `src/common/requirements.tex::SCI-VAL-REQ-009--010`, `REQ-026`, `REQ-044`, and `REQ-046`.
- **Explanation:** Basis-fit admission, loading-fit admission, operator application, output retention, coefficient/QC population, response companion, empirical/simulation population, and `SCI-MAP:map_upstream_admission` are only reserved names. `<PACKAGE>:diagnostic_display` is likewise reserved. A reserved identifier supplies no owner, predicate, domain, missing behavior, compatibility, or exception authority.
- **Scientific consequence:** A requested evaluation cannot produce eligibility for any of these uses. Finiteness, a retained signal, positive coefficient, or another use's decision cannot substitute.
- **Affected lane:** policy/VAL; signal; response; uncertainty.
- **MAP/downstream impact:** no ordinary PTC-dependent MAP sample can receive a usable map-admission decision; PTC fit/application/output/coefficient decisions are also unavailable.
- **Possible owner dispositions:** each actual use owner authors and approves a complete immutable profile; the Registry binds it; VAL evaluates it without acquiring policy ownership.
- **Blocker status:** **hard blocker** for the ordinary MAP route and the named PTC numerical actions.

## F-005 — The sole registered VAL profile is structurally incomplete under the Registry's own rule

- **Class:** PROFILE REGISTRY GAP
- **Scope:** `SCI-VAL:independent_exposure@1`.
- **Producer clause:** `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md::Registry Rule` and `src/common/requirements.tex::SCI-VAL-REQ-030`, `REQ-044`, and `REQ-047` require compatibility, including the aggregate/propagation relation or a declared nonapplicability.
- **Consumer clause:** the canonical row for `SCI-VAL:independent_exposure@1` in `PROFILE_REGISTRY.md`.
- **Explanation:** The row binds the nonexceptionable direct representative-origin restriction but does not state aggregation/propagation compatibility or explicit `not_applicable` for that required field. Its RTC source dependency is also stale under F-003.
- **Scientific consequence:** Registry integrity cannot be demonstrated for the only nominally registered profile; no aggregate may inherit it by name.
- **Affected lane:** policy/VAL; aggregation; provenance.
- **MAP/downstream impact:** independent-exposure decisions cannot be treated as source-current, registry-complete inputs to a MAP policy.
- **Possible owner dispositions:** complete the immutable record with an explicit atomic-only/aggregation disposition and current source bindings, or supersede it with a complete versioned record.
- **Blocker status:** **hard blocker** for claiming the profile is currently usable; it does not erase the underlying ALIGN/RTC facts.

## F-006 — Exact RTC-to-AST sample-grid boundary body is absent

- **Class:** IDENTITY OR GRID GAP
- **Scope:** RTC output `n` to AST RTC-grid coordinate parent.
- **Producer clause:** `packages/SCI-RTC/v0.1/src/common/requirements.tex::SCI-RTC-REQ-028--029`, `REQ-037`, `REQ-041`, and `REQ-114` supply the required facts.
- **Consumer clause:** `packages/SCI-AST/v0.1/src/common/requirements.tex::SCI-AST-REQ-074--079` requires the exact RTC product/plan/grid, representative ALIGN slot, time, phase/delay, segment, support, response, and status.
- **Explanation:** The two cores are conceptually compatible, but no committed artifact named `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md` or an explicitly equivalent digest-bound body exists at the pinned commit. Shape or numerical coordinate equality is expressly insufficient.
- **Scientific consequence:** The boundary cannot be admitted as one exact versioned transfer identity even though its required fields can be enumerated from both cores.
- **Affected lane:** coordinate; response; provenance.
- **MAP/downstream impact:** the logical AST RTC-grid role is under-bound and cannot be frozen as the exact coordinate parent of a MAP handoff.
- **Possible owner dispositions:** approve a boundary artifact that composes, but does not alter, the frozen RTC and AST clauses; bind exact versions, fields, missing states, and compatibility.
- **Blocker status:** **hard source-closure blocker** for exact MAP coordinate handoff; not evidence of a circular numerical architecture.

## F-007 — Detector-geometry and field-rotation boundary body is absent

- **Class:** IDENTITY OR GRID GAP
- **Scope:** external detector geometry/APT realization into SCI-AST.
- **Producer clause:** no admitted boundary body; SCI-BEAM is outside the source packet.
- **Consumer clause:** `packages/SCI-AST/v0.1/src/common/requirements.tex::SCI-AST-REQ-023--034` requires measured geometry authority, exact occurrence association, representation, application counts, pivot/gauge, covariance, and a versioned rotation law.
- **Explanation:** No exact `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md` or explicitly equivalent approved body is committed at the pinned revision. AST correctly refuses to infer geometry from fields, values, design identity, or row order.
- **Scientific consequence:** Geometry-dependent AST coordinate roles remain unavailable unless an external exact artifact satisfying the contract is separately supplied.
- **Affected lane:** coordinate; uncertainty; provenance.
- **MAP/downstream impact:** ordinary MAP coordinates are blocked where their AST parent requires the missing geometry realization.
- **Possible owner dispositions:** name and bind the external geometry/APT authority and exact association artifact; create the approved boundary; retain all unavailable affine/covariance terms.
- **Blocker status:** **role-level hard blocker** for geometry-dependent coordinates, not for unrelated ALIGN signal facts.

## F-008 — RTC reuses `K` for two independent counts

- **Class:** TERMINOLOGY COLLISION
- **Scope:** SCI-RTC frozen v0.1/r0.12 notation and equations.
- **Producer clause:** `packages/SCI-RTC/v0.1/src/common/equations.tex` uses `F_1...F_K` and products through filter stage `K`.
- **Consumer/competing clause:** `src/common/notation.tex` defines `k in {0,...,K}` with `K` the final accepted-plan index; `equations.tex::SCI-RTC-EQ-030` also sets `K = k_(A+1)`.
- **Explanation:** The same bare symbol denotes the final temporal-filter stage and the final accepted refinement plan. Response tensors also use `K` as a kernel family, increasing ambiguity. Context often disambiguates it, but exact cross-clause formal substitution does not.
- **Scientific consequence:** A composed response or replay statement can bind the wrong count without a symbol-renaming map.
- **Affected lane:** response; lifecycle; provenance.
- **MAP/downstream impact:** claim-local ambiguity in the RTC response/plan record; not by itself evidence of numerical nonconformity.
- **Possible owner dispositions:** issue an owner-approved notation correction or successor mapping with distinct filter-stage, accepted-plan, and response symbols.
- **Blocker status:** **claim-local blocker** for formal source closure; the prose architecture remains interpretable.

## F-009 — PTC reuses `C` for causes and estimator candidates

- **Class:** TERMINOLOGY COLLISION
- **Scope:** SCI-PTC frozen v0.1/r0.4 equations.
- **Producer clause:** `packages/SCI-PTC/v0.1/src/common/notation.tex` defines `mathcal C` as the accumulated cause/fact set or graph; `equations.tex::eq:cause-support` unions it.
- **Consumer/competing clause:** `equations.tex` later declares `mathcal C` to be the finite estimator candidate set with order.
- **Explanation:** Cause accumulation and candidate-model selection are different objects with different algebras but share an unqualified symbol in the same normative module.
- **Scientific consequence:** Policy and estimator equations cannot be mechanically composed without an unstated rename.
- **Affected lane:** policy/VAL; signal; lifecycle.
- **MAP/downstream impact:** formal replay ambiguity in PTC selection lineage; subordinate to F-001 and F-002 for route blocking.
- **Possible owner dispositions:** rename one object and publish the exact semantic mapping in a reopened revision or successor.
- **Blocker status:** **claim-local source-closure blocker**.

## F-010 — AST contains unresolved formal-symbol and role-name mismatches

- **Class:** TERMINOLOGY MISMATCH
- **Scope:** SCI-AST frozen v0.1/r0.3 shared notation/equations and approved role maps.
- **Producer clause:** `packages/SCI-AST/v0.1/src/common/notation.tex`, `definitions.tex`, and `equations.tex` use `i` in several `B_i`, `Pi_i`, or `G_pi` contexts and use small-angle `B_ds/f_d` forms.
- **Consumer/competing clause:** `ROLE_FACTORED_PARENTAGE_MAP.md`; `src/common/requirements.tex::SCI-AST-REQ-058`, `REQ-073--083`; canonical geometry uses `g_d/G_gamma`, and atomic-bundle fields include symbols not completely declared in the notation table.
- **Explanation:** The intended roles can be inferred from surrounding prose, but the frozen formal core does not provide one exact symbol-to-role declaration for every occurrence.
- **Scientific consequence:** Parent/type checking of some coordinate, bundle, and delegated projection equations is under-specified.
- **Affected lane:** coordinate; response; provenance.
- **MAP/downstream impact:** prevents a fully mechanical AST-to-MAP handoff proof; F-006, F-007, and F-014 remain stronger numerical-route blockers.
- **Possible owner dispositions:** approve a notation-only correction with parity proof, or issue a successor mapping every old symbol to one canonical role.
- **Blocker status:** **formal-closure blocker**, not a finding that AST reconstructs RTC or MAP.

## F-011 — PTC disabled-route equation is narrower than its disabled-route requirement

- **Class:** DISABLED OR FAILURE CONTRADICTION
- **Scope:** SCI-PTC disabled product semantics.
- **Producer clause:** `packages/SCI-PTC/v0.1/src/common/equations.tex::eq:disabled-route` assigns disabled realization only to the exact set `{Z, Lambda, gamma, MAP_PTC-route}`.
- **Consumer/competing clause:** `requirements.tex::SCI-PTC-REQ-076` says no PTC centering, subtraction, coefficient, transformed product, or PTC-route MAP product is realized; `REQ-088` separates product, response, covariance, and evidence axes.
- **Explanation:** The displayed product set does not explicitly disposition centering state, fitted/model state, or removed-component/subtraction products named by the requirement. It does not affirm the contrary, but the formal implication is incomplete relative to the frozen shall-statement.
- **Scientific consequence:** Disabled replay has no single exhaustive formal product-state mapping.
- **Affected lane:** lifecycle; signal; response.
- **MAP/downstream impact:** a disabled route must remain stopped; no omitted state may be inferred as realized or available.
- **Possible owner dispositions:** expand the exact set and distinguish non-realized learned/model state; or replace the equation with a complete role-indexed product-state map.
- **Blocker status:** **disabled-route source-closure blocker**; the conservative action is no PTC-dependent product.

## F-012 — Complete source-to-map response is not presently closed

- **Class:** RESPONSE GAP
- **Scope:** source/beam through ALIGN, RTC, CAL, PTC, AST geometry, and MAP deposition.
- **Producer clause:** ALIGN `REQ-037`; RTC `REQ-037--041`; CAL `REQ-039--043`; PTC `REQ-061--068` and `REQ-087` each preserve local response or typed unavailability.
- **Consumer clause:** MAP `SCI-MAP-REQ-008`, `REQ-016--017`, `REQ-036`, and `REQ-050`; MAP `OD-003`; AST `REQ-078--083`.
- **Explanation:** The contracts correctly distinguish fixed-state, full-procedure, and complete-chain response, prevent repeated ALIGN/CAL response application, and forbid replacing full temporal support with a nominal coordinate. They also allow typed unavailable links. At the pinned state the external source/beam basis is not admitted here, RTC/ALIGN response may be unavailable by tier, PTC has an internal application conflict, materialized `G_pi` is owner-open, and MAP has no owner disposition for restricted response-unavailable use.
- **Scientific consequence:** A finite signal need not have a complete realized source-to-map response, and the unit label cannot repair that absence.
- **Affected lane:** response; signal; coordinate.
- **MAP/downstream impact:** response-dependent consumers fail closed; literal post-PTC/MAP point-source-peak meaning is not established without the declared realized response or renormalization.
- **Possible owner dispositions:** bind the source/beam basis; resolve PTC; close local response states; decide MAP `OD-003`; and materialize response with the exact same membership, placement, support, and normalization as signal.
- **Blocker status:** **hard blocker** for an exact complete-response MAP handoff; signal-only logical records may remain typed response-unavailable.

## F-013 — The uncertainty chain is typed but numerically incomplete

- **Class:** UNCERTAINTY GAP
- **Scope:** complete conditional and nuisance uncertainty to MAP.
- **Producer clause:** ALIGN `REQ-038--039`; AST `REQ-022` and `REQ-065--072`; RTC `REQ-042--045` and `REQ-135`; CAL `REQ-032--038`; PTC `REQ-057--060`; VAL `REQ-025`.
- **Consumer clause:** MAP `SCI-MAP-REQ-019--024`, `REQ-036`, `REQ-041`, and `REQ-050`; MAP `OD-004`.
- **Explanation:** Every package generally preserves unavailable terms rather than replacing them with zero. However ALIGN timing/model terms can be unavailable; AST pointing/geometry/selection terms and total covariance are conditional; RTC selection/model/donor terms may be absent; CAL explicitly lacks several factor, atmosphere, passband, and cross-array mechanisms; PTC selection and model uncertainty may be absent; and MAP has not chosen the minimum persisted covariance representation.
- **Scientific consequence:** A numerical signal may exist while total uncertainty, significance, full covariance, and several nuisance derivatives remain unavailable.
- **Affected lane:** uncertainty; policy; provenance.
- **MAP/downstream impact:** MAP may preserve a conditional covariance only when supplied; it may not call normalization precision, omit consequential correlations, or claim total significance.
- **Possible owner dispositions:** choose exact representation/persistence boundaries, supply quantified components and correlations, or explicitly restrict consumer claims while preserving typed omissions.
- **Blocker status:** **hard blocker** for total-uncertainty/precision/significance claims; not necessarily for a typed conditional signal bundle.

## F-014 — MAP projection operator `G_pi` has no resolved scientific authority

- **Class:** OWNER DECISION REQUIRED
- **Scope:** downstream-consumer readiness; AST-to-MAP deposition boundary.
- **Producer clause:** AST `SCI-AST-REQ-080--083` supplies continuous coordinates and permits materialization only from a complete MAP-owned request.
- **Consumer clause:** MAP `SCI-MAP-REQ-005`, `REQ-010--011`, and `SCIENTIFIC_OWNER_DECISION_LEDGER.md::SCI-MAP-OD-008`.
- **Explanation:** MAP owns allowed projection classes, normalization, boundary loss, conservation meaning, and the upstream producer of materialized `G_pi`. `OD-008` is open. A continuous coordinate or nominal containing pixel is not an estimator-specific deposition plan.
- **Scientific consequence:** No numerical sample-to-pixel coefficient can be inferred, so there is no contribution set, normalization, support, response, covariance, or final map value.
- **Affected lane:** coordinate; signal; response; uncertainty.
- **MAP/downstream impact:** this is the final direct numerical-route stop even after an eligible handoff exists.
- **Possible owner dispositions:** resolve `OD-008` with exact classes, normalization, boundary/conservation rules, producer, metadata, and evidence obligations; then create a versioned request/materialization relation.
- **Blocker status:** **hard blocker** for every numerical MAP route.

## F-015 — No registered aggregation or observation-coadd admission profile exists

- **Class:** AGGREGATION GAP
- **Scope:** VAL aggregate decisions and MAP observation coaddition.
- **Producer clause:** VAL `SCI-VAL-REQ-030--034`, `REQ-047--048`; MAP `SCI-MAP-REQ-036--041`.
- **Consumer clause:** `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` contains no registered aggregate profile or observation-coadd admission policy.
- **Explanation:** Individual eligibility cannot be averaged into aggregate eligibility. Exact population, atomic source profile, lifecycle, denominator, missing treatment, operator, threshold/polarity, uncertainty, compatibility, and reverse-propagation generation are all policy facts; none is bound for the ordinary coadd route.
- **Scientific consequence:** Individually eligible observations that differ in WCS, shape, response, support policy, units, or parent cannot be coadded by a VAL fraction or implicit compatibility rule.
- **Affected lane:** policy/VAL; lifecycle; MAP aggregation.
- **MAP/downstream impact:** observation-level MAP products may be logically described, but no VAL-owned coadd admission decision is available; MAP's own atomic compatibility checks remain necessary.
- **Possible owner dispositions:** the actual coadd use owner authors a distinct aggregate/coadd profile bound to one exact atomic profile and MAP bundle compatibility proposition.
- **Blocker status:** **hard blocker** for policy-authorized coaddition; not for preserving separate observation bundles.

## F-016 — CAL, VAL, and MAP are not frozen and lack one final text-source binding pattern

- **Class:** SOURCE-BINDING OR VERSION GAP
- **Scope:** source authority and replay.
- **Producer clause:** package READMEs and `doc/scientific_contracts/INDEX.md` state CAL r0.5/r0.4 pending final scientific acceptance, VAL r0.3 manager-reviewed/pending owner review, and MAP r0.3 house rationale frozen but scientific authority not frozen.
- **Consumer clause:** audit source-closure requirement; VAL `SCI-VAL-REQ-029`, `REQ-044`, and `REQ-049`; MAP `SCI-MAP-REQ-001`.
- **Explanation:** CAL, VAL, and MAP do not have the same final canonical text/PDF source manifest discipline as frozen ALIGN/AST/RTC/PTC. Their packet manifests bind author inputs or review outputs rather than one frozen current scientific authority. This does not make their text meaningless, but it caps claims at active/draft status.
- **Scientific consequence:** An authoritative six-package plus MAP handoff cannot be frozen against nonfrozen source authorities.
- **Affected lane:** provenance; all dependent claims.
- **MAP/downstream impact:** a non-authoritative draft handoff can be written; an authoritative frozen handoff cannot.
- **Possible owner dispositions:** complete owner reviews, resolve internal/open items, freeze exact versions, and issue final source manifests binding canonical text and PDFs.
- **Blocker status:** **hard freeze blocker**; independent of numerical-route blockers.

## F-017 — Several required producer facts are owned outside the admitted packet

- **Class:** MISSING PRODUCER GUARANTEE
- **Scope:** Tune/readout, telescope timing/state, pointing selection, geometry/APT, beam/factor, nominal beam, target/source center, frame/EOP/refraction, atmosphere, and source model facts.
- **Producer clause:** no complete external source packet was admitted for these owners; CAL alone content-binds its selected atmosphere operator and passband identities, not observational truth or every nuisance term.
- **Consumer clause:** ALIGN `REQ-001`, `REQ-007--011`, `REQ-016--023`; AST `REQ-013--031`, `REQ-045--046`; CAL `REQ-006--031`, `REQ-040`; PTC `REQ-061`, `REQ-081`; MAP `REQ-005`.
- **Explanation:** The six packages correctly name most required meanings and fail closed when absent, but the audit cannot verify the scientific guarantee of an external package that was not supplied. No consumer may reconstruct these facts from plausible values or shape.
- **Scientific consequence:** Dependent clocks, fields, coordinates, calibration factors, response bases, source masks, or claims remain unavailable at their exact scope.
- **Affected lane:** signal; coordinate; response; uncertainty; provenance.
- **MAP/downstream impact:** one or more MAP-ready members can be blocked even if upstream numerical arrays exist.
- **Possible owner dispositions:** supply exact owner-approved boundary artifacts and source bindings for each required external fact; keep each omission claim-local rather than treating the entire observation as universally invalid.
- **Blocker status:** **conditional hard blocker** for every dependent role; complete inventory is in `EXTERNAL_AUTHORITY_DEPENDENCY_LEDGER.md`.

## F-018 — Disabled PTC does not authorize a direct CAL-to-MAP route

- **Class:** MISSING CONSUMER DISPOSITION
- **Scope:** alternate main-signal routing.
- **Producer clause:** PTC `SCI-PTC-REQ-076--077` disables every PTC-dependent product and explicitly neither prohibits nor authorizes a separately governed direct CAL-to-MAP route.
- **Consumer clause:** MAP `SCI-MAP-REQ-002` names calibrated SCI-CAL ordinary-`xs` input, while MAP's other preconditions require complete eligibility, coefficient, coordinate, response, support, and provenance meanings and do not constitute a CAL-only route profile.
- **Explanation:** The shared phrase “SCI-CAL input” is a quantity boundary, not authority to bypass PTC policy or invent a distinct coefficient/response chain. No route owner, profile, response composition, or disabled-state compatibility artifact is supplied.
- **Scientific consequence:** PTC disabled means no PTC-route MAP product. Whether a different CAL-only estimator exists remains outside this package chain.
- **Affected lane:** signal; policy; response; lifecycle.
- **MAP/downstream impact:** there is no fallback numerical MAP route at the pinned commit.
- **Possible owner dispositions:** separately authorize and contract a CAL-to-MAP route with its own signal parent, policy profile, coefficient, response, uncertainty, and provenance; or explicitly prohibit it for this contract generation.
- **Blocker status:** **route-level hard blocker**; it does not change the semantics of a disabled PTC product.

## F-019 — Exposure has no explicit PTC-to-MAP producer handoff

- **Class:** SUPPORT / CAUSE / INFLUENCE / EXPOSURE GAP
- **Scope:** physical acquired, valid-original, retained, and MAP exposure facts.
- **Producer clause:** ALIGN `SCI-ALIGN-REQ-015`, `REQ-027--030` produces physical and valid-original exposure; RTC preserves support/influence; CAL distinguishes support and weight; PTC preserves causes/support but defines no independent exposure object.
- **Consumer clause:** MAP `SCI-MAP-REQ-007`, `REQ-025`, `REQ-029--030`, and `REQ-036` requires named upstream-eligible and retained exposure under declared projection/accounting.
- **Explanation:** The contracts prohibit converting synthesized support into acquired exposure and prohibit treating weight as exposure. They do not provide one exact occurrence-level rule that carries ALIGN exposure through RTC/CAL/PTC output retention into the MAP bundle and states whether PTC acts as a transparent carrier or a new retained-exposure producer.
- **Scientific consequence:** MAP cannot reconstruct exposure from sample duration, support, hit count, or coefficient.
- **Affected lane:** support/exposure; policy; provenance.
- **MAP/downstream impact:** exposure planes and exposure-based claims are unavailable even if signal contribution later becomes possible.
- **Possible owner dispositions:** name the transparent carrier and exact parent relation; or make a package own the retained-exposure transformation and bind its use-specific policy/profile.
- **Blocker status:** **hard blocker** for MAP exposure members and exposure-dependent policy, but not automatically for a signal-only logical record.

## F-020 — CAL engineering-only/science-qualification facts lack downstream named-use policy

- **Class:** VAL CONGRUENCE GAP
- **Scope:** CAL observation classification into PTC and MAP uses.
- **Producer clause:** CAL `SCI-CAL-REQ-004`, `REQ-027--030`, `REQ-045--049` owns distinct `engineering-only` and `science-qualification-eligible` facts and explicitly does not claim achieved science qualification.
- **Consumer clause:** PTC named-use policies in `SCI-PTC-REQ-012`; VAL `SCI-VAL-REQ-002`, `REQ-009`, `REQ-023`, and `REQ-044`; MAP `SCI-MAP-REQ-004`.
- **Explanation:** No registered PTC or MAP profile states how either CAL-local class acts for basis fit, application, output, coefficient, MAP admission, or coadd admission. VAL cannot turn the producer-local classifier into a universal decision.
- **Scientific consequence:** A supported engineering sample may have a calibrated numerical value but its ordinary scientific uses remain decision unavailable.
- **Affected lane:** policy/VAL; signal.
- **MAP/downstream impact:** no MAP contribution may be inferred from the CAL class alone.
- **Possible owner dispositions:** each named-use owner binds the CAL facts to an exact profile proposition with missing behavior and exceptions.
- **Blocker status:** **use-specific blocker**, subsumed by F-004 for currently reserved profiles.

## F-021 — MAP has no owner disposition for response-unavailable bundle use

- **Class:** MISSING CONSUMER DISPOSITION
- **Scope:** downstream-consumer readiness when signal is available and response is typed unavailable.
- **Producer clause:** upstream packages and MAP `SCI-MAP-REQ-008` permit an explicit typed unavailable response state.
- **Consumer clause:** `packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md::SCI-MAP-OD-003`; MAP `SCI-MAP-PRED-013` says response-dependent consumers reject, but general restricted usability is undecided.
- **Explanation:** The contract intentionally does not decide whether a response-unavailable map bundle is usable by any restricted consumer class. Treating it as universally usable or universally unusable would invent policy.
- **Scientific consequence:** response-independent downstream action has no owner-approved envelope beyond preserving the typed state.
- **Affected lane:** response; policy.
- **MAP/downstream impact:** all response-dependent consumers stop; other use remains unavailable pending `OD-003`.
- **Possible owner dispositions:** identify a restricted consumer list and claims that are response-independent, or require realized response for every scientific map bundle.
- **Blocker status:** **claim/use-level blocker**, additional to F-012.

## F-022 — Frozen ALIGN/AST bytes retain draft-status language

- **Class:** SOURCE-BINDING OR VERSION GAP
- **Scope:** status wording and source identity for ALIGN and AST.
- **Producer clause:** exact r0.3 sources/PDFs contain pre-review draft or pending labels.
- **Consumer/competing clause:** each package `README.md` and `SOURCE_MANIFEST.md` states that Grant Wilson froze the exact unchanged bytes as v0.1/r0.3 on 2026-08-22 and administratively supersedes those embedded labels.
- **Explanation:** This is a status-layer skew, not evidence that the formulas were changed after freeze. A consumer that reads only embedded title/status bytes could nevertheless classify the source incorrectly.
- **Scientific consequence:** source status must be resolved through the explicit freeze navigation/manifest record, never by filename appearance alone.
- **Affected lane:** provenance.
- **MAP/downstream impact:** not a numerical blocker once exact manifest and README authority are carried; it is a source-binding hazard.
- **Possible owner dispositions:** retain as an explicit immutable-byte exception, or issue a separately versioned status-clean successor without pretending the bytes are identical.
- **Blocker status:** **non-numerical provenance warning**; F-003 remains the VAL binding blocker.

## F-023 — No exact MAP-facing analysis coefficient is owner-resolved

- **Class:** OWNER DECISION REQUIRED
- **Scope:** PTC-to-MAP coefficient/weight handoff.
- **Producer clause:** PTC `SCI-PTC-REQ-052--055` distinguishes loadings, diagnostic coefficients, and analysis/gridding coefficients; only the last may face MAP.
- **Consumer clause:** PTC owner ledger `PTC-OD-010` remains open; MAP `SCI-MAP-REQ-006`, `REQ-010--011`, `REQ-013`, and `REQ-020--021` requires a finite positive ordinary coefficient with exact meaning and forbids promotion to precision.
- **Explanation:** No concrete family, index domain, statistic/factors, normalization, lifecycle, support, or permitted MAP use is selected. MAP's `omega_i` cannot be inferred from PTC loadings, CAL `sens`, a variance label, or retained support.
- **Scientific consequence:** Even with signal, coordinate, and eligibility, the estimator contribution coefficient is absent.
- **Affected lane:** signal; support; uncertainty; provenance.
- **MAP/downstream impact:** **hard numerical-route stop** independent of `G_pi`; normalization and contribution membership cannot be formed.
- **Possible owner dispositions:** PTC owner defines one analysis/gridding family and its profile; or another explicitly named upstream owner supplies the coefficient under a compatible route.
- **Blocker status:** **hard blocker** for every numerical MAP contribution.

## F-024 — MAP support-policy numerical domain remains owner-open

- **Class:** OWNER DECISION REQUIRED
- **Scope:** downstream-consumer readiness; dimensionless
  `coverage_cut=c` and support-authorized output membership.
- **Producer clause:** MAP `SCI-MAP-CI-001` resolves only that `c` is
  dimensionless; `SCIENTIFIC_OWNER_DECISION_LEDGER.md::SCI-MAP-OD-007` leaves
  its numerical domain, boundary cases, recommended range, effective-policy
  authority, and failure behavior open.
- **Consumer clause:** MAP `SCI-MAP-REQ-031--032`, `REQ-036`, and
  `REQ-047--048` use the effective value to define support-authorized rows and
  required publication.
- **Explanation:** A profile may bind only an owner-authorized value; it may not
  invent an admissible range or default the value. Negative, zero, nonfinite,
  greater-than-one, and ordinary positive values have no universal disposition
  until the owner resolves OD-007 or an exact approved effective policy admits
  the chosen value.
- **Scientific consequence:** With no authorized effective value, the output
  row set and every dependent signal/response/covariance/exposure/validity state
  are unavailable.
- **Affected lane:** support; policy/VAL; lifecycle/publication.
- **MAP/downstream impact:** **hard numerical-route stop** before support rows or
  live required-product mutation; does not alter upstream package facts.
- **Possible owner dispositions:** approve one domain and boundary table;
  approve distinct versioned profile domains; or retain fail-closed
  unavailability.
- **Blocker status:** **hard blocker** for a route lacking an explicitly
  admitted `coverage_cut`; tracked by XOD-019.

## F-025 — Canonical-grid preparation and reprojection ownership are unresolved

- **Class:** OWNER DECISION REQUIRED
- **Scope:** downstream-consumer readiness; incompatible-grid observation coadd
  and future grid-changing products.
- **Producer clause:** no admitted package is named as an authorized
  crop/pad/reprojection/mosaic producer.
- **Consumer clause:** MAP `SCI-MAP-REQ-037--045` and
  `SCIENTIFIC_OWNER_DECISION_LEDGER.md::SCI-MAP-OD-009`.
- **Explanation:** MAP correctly admits only exact compatible centered-integer
  placement and atomically rejects odd shape differences or different grids.
  OD-009 does not authorize a producer to alter an observation merely to make
  it compatible, and it does not assign future reprojection or mosaic science.
- **Scientific consequence:** No grid-changing operator, response/covariance
  propagation, validity transition, or provenance relation may be inferred.
- **Affected lane:** coordinate; response; uncertainty; aggregation; provenance.
- **MAP/downstream impact:** incompatible-grid coadd and future
  reprojection/mosaic are blocked; this decision alone does not block an
  otherwise compatible single-observation map.
- **Possible owner dispositions:** authorize an exact canonical preparation
  profile and producer; retain strict rejection and create a separate future
  transform package; or leave these products unavailable.
- **Blocker status:** **coadd/grid-role blocker**, tracked by XOD-020; independent
  of F-015's missing aggregate policy.

## Coherent results that survived challenge

These are not repairs and do not cancel any finding:

- The two committed ALIGN-to-AST boundary copies are byte-identical at SHA-256 `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36`.
- The ALIGN–AST–RTC relationship admits an acyclic two-role diamond: AST can construct an ALIGN-grid role before RTC resolution, RTC produces a distinct `n`-grid, and AST constructs a separately parented RTC-grid role afterward. Numerical equality never establishes parent equality.
- ALIGN slot `s`, local storage row `j`, RTC output `n`, detector occurrence `d`, and MAP pixel `p` are contractually distinct.
- RTC sends conditioned raw `x` to CAL; CAL applies `flxscale` and target atmosphere exactly once; PTC then acts in the calibrated unit. No audited clause authorizes a second ALIGN, CAL factor, target-atmosphere, or PTC application.
- Direct representative synthesis/replacement and nonrepresentative transitive influence remain distinct. A cause is not a universal action; PTC is allowed to author stricter named-use policy, but it must be bound through the missing profiles.
- Angular coordinates are not filtered as signal. The full RTC sky-response relation retains the temporal operator and all contributing ALIGN-grid coordinates; a nominal RTC coordinate does not replace that response support.
- Optional conditioned `r` remains same-grid, uncalibrated, non-Stokes, and numerically unable to alter conditioned `x`. Donor-induced conditioned-`r`, response, and dependent-covariance unavailability stays within the optional branch but spans the complete donor causal influence while preserving the grid, raw-`r` parent, causes, and otherwise valid `x`.
- Producer validity, VAL eligibility, consumer numerical contribution, support, and final MAP validity remain separate concepts. Required publication/provenance failure propagates without a false completion claim.

The row-level evidence and statuses for all 26 directive-required interfaces,
three additional closure interfaces, 22 directive-required candidate
invariants, and two additional MAP owner-gate invariants are in
`SIX_PACKAGE_SOURCE_CROSSWALK.csv`.
