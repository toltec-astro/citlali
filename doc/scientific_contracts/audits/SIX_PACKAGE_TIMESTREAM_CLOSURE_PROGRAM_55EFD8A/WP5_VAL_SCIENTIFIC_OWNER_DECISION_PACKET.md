# WP-5 VAL Scientific-Owner Decision Packet

Opened: `2026-08-24`

Status: `WP5-OWNER-D001--D011` approved; remaining registry work awaits
review one at a time

Scope: non-MAP processed-timestream source bindings and VAL profiles. VAL Core
is not reopened. MAP and coadd profiles remain deferred.

## WP5-OWNER-D001 — Current Non-MAP Source Bindings

Question:

> Should VAL's continuing source-binding register replace its stale
> adjacent-package rows with the exact frozen timestream authorities, while
> leaving VAL Core unchanged and MAP explicitly deferred?

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. The continuing register now binds frozen ALIGN r0.3, AST r0.3, RTC r0.12,
   CAL r0.5-r0.4, and PTC r0.5, together with the exact approved Tune/readout
   interface and the approved WP-2--WP-4 manifests.
2. ALIGN/RTC direct representative-origin semantics remain compatible with
   `SCI-VAL:independent_exposure@1`; completion of that profile's registry row
   remains a separate decision.
3. AST coordinate validity remains distinct from signal validity and
   independent exposure.
4. CAL classification remains a producer fact whose consequence is owned by
   each named scientific use.
5. Source binding alone registers no PTC policy.
6. SCI-MAP remains explicitly deferred and unbound; no MAP profile becomes
   evaluable.
7. Historical Core-view source tables remain snapshots. The continuing
   register changes no prior VAL evaluation identity and requires no Core
   rewrite.

## WP5-OWNER-D002 — Complete The Atomic Independent-Exposure Profile

Question:

> Should `SCI-VAL:independent_exposure@1` be explicitly atomic-only, with
> aggregation and reverse propagation marked `not_applicable` under this
> identity, while any future detector, scan, or observation aggregate requires
> a separate complete owner-bound profile and creates a new derived lifecycle
> generation?

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. `SCI-VAL:independent_exposure@1` is complete as an atomic-only profile
   bound to the current source register, frozen ALIGN r0.3, frozen RTC r0.12,
   and the approved WP-2 exposure-lineage boundary.
2. Aggregation and reverse propagation are `not_applicable` under this exact
   identity. No aggregate is registered by this decision.
3. A future aggregate must have its own immutable owner-bound registry record,
   population and support, operator and denominator, missing behavior,
   threshold if any, uncertainty and failure rules, and propagation authority.
4. Any future authorized propagation creates a new derived proposition and
   lifecycle generation. It cannot overwrite the atomic decision or recreate,
   increase, or rewrite SCI-ALIGN physical-acquisition and valid-original
   facts.
5. This completion defines no generic usable exposure, detector fraction,
   retained or projected exposure, coadd quantity, numerical threshold, or
   MAP policy.
6. The existing atomic truth rule, missing/conflict behavior, and absence of
   inferred response or uncertainty roles are unchanged. VAL Core is not
   modified.

## WP5-OWNER-D003 — Compact Common PTC Policy Factoring

Question:

> May the seven distinct PTC named-use profiles reference one immutable common
> restriction fragment, provided the fragment itself grants no permission and
> every use remains a complete separately registered proposition?

Owner response: **approved with binding interpretation**.

Disposition:

1. The common restriction fragment is compact, PTC-owned, immutable, and
   versioned. It contains only restrictions proved to apply without scientific
   variation to all seven named PTC uses.
2. The common fragment is not a VAL profile, is not independently evaluable,
   produces no eligibility result, and grants no permission.
3. Every PTC use `U` remains a complete, separately registered proposition:

   \[
   \mathcal R_U
   =
   \mathcal R_{\rm common}
   \cup
   \Delta\mathcal R_U.
   \]

   Each profile retains its own named action, use-specific restrictions,
   applicability, and unknown or fail-closed behavior.
4. Permission never transfers between named uses:

   \[
   E_U \not\Rightarrow E_V,
   \qquad U\ne V.
   \]

5. A restriction enters `\mathcal R_{\rm common}` only when it is genuinely
   required by all seven uses. Any scientific variation keeps the restriction
   in `\Delta\mathcal R_U`.
6. CAL's `engineering-only` classification remains a preserved producer fact.
   Whether it excludes a PTC use belongs to that named PTC policy; it is not a
   universal CAL or VAL veto on PTC mathematics.
7. Every registered PTC profile references the common fragment by exact
   identity and digest. Changing the fragment creates a new version and cannot
   silently change an existing profile or evaluation identity.
8. This decision creates no runtime common-policy object, inheritance
   mechanism, serialization requirement, sidecar, duplicated provenance
   payload, or separate engineering route. A separate route is required only
   if a later genuinely different scientific policy requires one.
9. D003 approves the factoring rule, not the fragment's contents. The common
   fragment remains unbound until the remaining use-by-use decisions establish
   which restrictions, if any, satisfy the universal-use test. No PTC profile
   becomes usable through D003 alone.

## WP5-OWNER-D004 — Basis-Fit Admission

Question:

> Should `SCI-PTC:basis_fit_admission@1` authorize only whether an exact
> occurrence may influence ordinary-route detector centering and PCA
> basis/subspace estimation, without implying loading-fit, application,
> output-retention, coefficient/QC, response, or empirical/simulation
> permission?

Owner response:

> agreed

Disposition: **approved with the frozen ordinary-centering identity made
explicit**.

Consequences:

1. The profile is supported for the ordinary configured-rank PCA route and is
   atomic at the exact CAL sample-detector occurrence within one immutable PTC
   segment and configured network or array group.
2. Eligibility requires the exact CAL parent, detector/sample identity,
   segment, group, calibrated `x`, basis-fit mask, and every other fact required
   specifically for basis fitting to be resolved and numerically admissible.
   Missing or conflicting required facts yield `decision_unavailable`; a
   decisive false yields `ineligible`; all required restrictions true yields
   `eligible`.
3. A nonfinite or unavailable required calibrated `x`, an occurrence outside
   the exact segment or group, a rejecting basis-fit mask, or a direct
   ALIGN-synthesized or RTC-replaced occurrence is basis-fit ineligible. Facts
   and causes remain preserved.
4. CAL `engineering-only` classification is not by itself a basis-fit
   exclusion. It remains a preserved producer fact and successful basis
   fitting does not create or imply CAL science qualification.
5. Nonrepresentative RTC influence remains visible and is not an automatic
   basis-fit veto unless an exact basis-fit restriction names it as decisive.
6. For the frozen ordinary route, `SCI-PTC-REQ-023`, `SCI-PTC-REQ-093`,
   `SCI-PTC-DEF-007`, and the normative centering equation define
   `\lambda_{g,d}` from exactly the finite basis-fit-admitted occurrences with
   binary centering influence. Therefore:

   \[
   \text{basis-fit ineligible}
   \Longrightarrow
   w^{\rm ctr}_{gtd}=0
   \]

   for this route. A directly synthesized or replaced occurrence teaches
   neither `\lambda_{g,d}` nor the PCA basis.
7. This shared-support identity is PTC mathematics, not a VAL inference. It is
   not a universal rule for every possible centering method. A future robust
   or otherwise different centering family requires its own declared
   population, support, estimator, policy, and version.
8. Eligibility authorizes influence only on ordinary-route `\lambda_{g,d}` and
   basis/subspace estimation. It decides neither whether the fitted operator
   may act on the occurrence nor whether a transformed value may be retained:

   \[
   E_{\rm basis}\not\Rightarrow E_{\rm application},
   \qquad
   \neg E_{\rm basis}\not\Rightarrow\neg E_{\rm application}.
   \]

9. Response and uncertainty states are explicit typed structural facts, but
   numerical response or covariance availability is not a required permission
   for ordinary basis fitting. The profile is atomic-only.
10. Until all seven use decisions establish the contents of the common
    fragment, these restrictions remain in
    `\Delta\mathcal R_{\rm basis}`. D004 alone does not produce a final
    digest-bound usable registry record.

## WP5-OWNER-D005 — Loading-Fit Admission

Question:

> Should `SCI-PTC:loading_fit_admission@1` authorize only whether an exact
> occurrence may influence estimation of a detector's coupling to an already
> fixed basis or template, without requiring its population to equal the
> basis-fit population or importing unrelated metadata as restrictions?

Owner response: **approved with binding interpretation**.

Disposition:

1. The scientific distinction is:

   \[
   \boxed{
   \text{Basis fitting chooses the model;}\quad
   \text{loading fitting estimates a detector's coupling to that fixed model.}
   }
   \]

2. The profile is supported for the ordinary configured-rank PCA route and is
   atomic at the exact CAL sample-detector occurrence within one immutable PTC
   segment and configured network or array group.
3. Eligibility requires only the inputs explicitly required by the frozen PTC
   loading estimator. For the ordinary route these include the exact fixed
   basis/template identity, frozen centering state, detector/group/segment
   identity, loading support and mask, fitting coordinate, declared metric and
   gauge, and the numerical quantities actually used to estimate that
   loading.
4. Generic upstream or downstream metadata, optional facts, provenance that is
   merely reachable, or fields that happen to exist are not loading-estimator
   inputs and cannot become restrictions, decisive exclusions, or missing-fact
   blockers by inference.
5. A required loading-estimator input that is missing, conflicting, nonfinite,
   or unavailable yields `decision_unavailable` or `ineligible` according to
   the complete T/F/U/C rule. A direct ALIGN-synthesized or RTC-replaced
   occurrence is loading-fit ineligible on the calibrated PTC science branch.
   Facts and causes remain preserved.
6. CAL `engineering-only` classification is not by itself a loading-fit
   exclusion. It remains a preserved producer fact and successful loading
   estimation does not create or imply CAL science qualification.
7. Loading fitting uses the already-frozen `\lambda_{g,d}` and fixed model. It
   cannot re-estimate `\lambda_{g,d}`, change the basis, alter basis membership,
   change configured rank, or change the resolved subspace.
8. Basis-fit and loading-fit populations may overlap completely, partially, or
   not for a particular occurrence. No equality and no permission transfer is
   inferred:

   \[
   E_{\rm basis}\not\Rightarrow E_{\rm loading},
   \qquad
   \neg E_{\rm basis}\not\Rightarrow\neg E_{\rm loading}.
   \]

9. Eligibility authorizes influence only on the fitted detector loading. It
   grants no operator-application, output-retention, coefficient/QC, response,
   or empirical/simulation permission.
10. Every fitted loading retains its basis/template identity, gauge, unit,
    configured group, fit support, lifecycle, and numerical use. It is not a
    weight, precision, significance, sensitivity, or independent-noise
    measure.
11. Numerical response or covariance availability is not required for loading
    estimation. The profile is atomic-only.
12. Until all seven use decisions establish the common fragment, these
    restrictions remain in `\Delta\mathcal R_{\rm loading}`. D005 alone does
    not produce a final digest-bound usable registry record.

## WP5-OWNER-D006 — Frozen Operator Application

Question:

> Should `SCI-PTC:operator_application@1` authorize whether the exact resolved
> frozen group-local operator may act at one configured group-time, while
> allowing fit-excluded inputs only when every operator-defined quantity is
> available and forbidding every silent fallback?

Owner response: **approved with an explicit rank-deficiency rule**.

Disposition:

1. The profile is supported for the ordinary configured-rank PCA route. Its
   decision object is the exact coupled group-time application event with its
   target occurrence roles: network-time for network-level PCA and array-time
   for array-level PCA.
2. Application requires only the inputs named by the frozen operator,
   including resolved `\Theta_g`, frozen `\lambda_g`, strictly positive
   configured rank, fitted subspace and required loadings, metric, tolerance,
   generalized inverse, application support and mask, detector bindings,
   finite time-local coefficient-recomputation inputs, and every required
   coordinate transform and boundary state within that same configured group.
3. The mandatory fail-closed numerical boundary is

   \[
   \boxed{
   \operatorname{rank}(N_{g,t}) < k_{{\rm req},g}
   \Longrightarrow
   \text{the frozen group-time operator application is unavailable}.}
   \]

   No implementation may replace that failed application by a lower-rank,
   interpolated, zero-filled, reconstructed, or cross-group alternative while
   labeling it as the same operator.
4. A missing, conflicting, nonfinite, or unavailable required operator input
   yields `decision_unavailable` or `ineligible` under the complete T/F/U/C
   rule and no numerical action is authorized. Failure remains scoped to the
   configured group: one network's failure does not alter another network's
   operator, while array mode has array-group scope.
5. Fit exclusion does not imply application exclusion. An occurrence may be
   acted upon when the exact frozen group-local operator defines the result
   completely:

   \[
   \neg E_{\rm basis}\not\Rightarrow\neg E_{\rm application},
   \qquad
   \neg E_{\rm loading}\not\Rightarrow\neg E_{\rm application}.
   \]

6. Direct synthesized/replaced origin and CAL `engineering-only`
   classification are preserved producer facts but are not universal vetoes
   on performing the frozen mathematics. Application eligibility does not
   relabel either fact, create an independent exposure, or authorize retention
   in an ordinary science product.
7. Eligibility authorizes only data application of the exact frozen operator.
   It grants no output-retention, coefficient/QC, response-companion, or
   empirical/simulation permission. Acting on a response kernel is governed by
   the separate response-companion profile.
8. Numerical complete-chain response or covariance availability is not a
   required permission for data application. Product realization, response,
   covariance, and validation/evidence remain independent typed axes.
9. These restrictions remain in
   `\Delta\mathcal R_{\rm application}` until final common-fragment comparison.
   D006 alone does not produce a final digest-bound usable registry record.

## WP5-OWNER-D007 — Output Retention

Question:

> Should `SCI-PTC:output_retention@1` decide whether an exact transformed
> occurrence is scientifically legitimate as a member of the ordinary PTC
> science-signal support, while remaining independent of both numerical
> calculation and every later scientific-use admission decision?

Owner response: **approved with binding interpretation**.

Disposition:

1. The governing distinctions are:

   \[
   \boxed{
   \text{PTC may calculate more values than it is entitled to call ordinary
   science samples.}}
   \]

   and

   \[
   \boxed{
   \text{PTC retention says the PTC result is scientifically legitimate as a
   signal; it does not say every later scientific use must accept it.}}
   \]

2. The profile is supported for the ordinary configured-rank transformed
   `x` product. It decides membership in ordinary PTC science-signal support;
   it does not prescribe whether or how an implementation serializes a
   calculated value.
3. Retention requires the exact group-time operator application to have been
   realized successfully, the transformed value to be finite, the exact
   output mask to admit the occurrence, and every output-specific required
   fact to be resolved. A failed group-time application yields no retained
   ordinary occurrence for that group-time and no substitute value may be
   relabeled as its result.
4. Application permission does not imply output retention, while retention of
   a transformed numerical value requires realized application:

   \[
   E_{\rm application}\not\Rightarrow E_{\rm output},
   \qquad
   E_{\rm output}\Rightarrow R_{\rm application}.
   \]

5. Direct ALIGN-synthesized or RTC-replaced occurrences are ineligible for the
   ordinary PTC science-signal support under `SCI-PTC-REQ-016`. Their computed
   values may remain reachable as explicitly flagged diagnostic or ineligible
   values, but cannot be represented as retained ordinary science samples.
6. CAL `engineering-only` occurrences may remain in the transformed PTC
   product with the classification preserved. Retention does not create a
   science-quality claim; every later scientific use applies its own exact
   admission profile.
7. Missing complete-chain response or covariance does not erase an otherwise
   valid transformed signal. Product realization, response, covariance, and
   validation/evidence remain independent axes.
8. Retention preserves the admitted CAL unit and calibration convention but
   does not assert preservation of point-source peak, absolute level,
   extended-source response, detector combination, or beam shape.
9. Retention creates no exposure, cannot recreate or increase valid-original
   exposure, and does not imply coefficient/QC membership, response-companion
   admission, MAP contribution, or any other later permission.
10. These restrictions remain in `\Delta\mathcal R_{\rm output}` until final
    common-fragment comparison. D007 alone does not produce a final
    digest-bound usable registry record.

## WP5-OWNER-D008 — Coefficient/QC Population Disposition

Question:

> Should the broad reserved name `SCI-PTC:coefficient_qc_population@1` remain
> unsupported as a generic ordinary-route profile, while PTC may define useful
> informational diagnostics directly and VAL policy is required only when a
> diagnostic receives admission or decision authority?

Owner response: **approved with anti-overengineering clarifications**.

Disposition:

1. The governing rule is:

   \[
   \boxed{
   \text{PTC may calculate whatever well-defined diagnostics are useful;}\quad
   \text{VAL policy is required when one of them is given decision authority.}
   }
   \]

2. No usable generic `SCI-PTC:coefficient_qc_population@1` profile is
   registered for the ordinary route. The underlying quantities do not share
   one universal scientific meaning or population rule.
3. Fitted detector loadings remain governed by loading-fit admission and their
   own declared semantics. Time-local application coefficients remain part of
   exact frozen operator application. Neither role is reclassified as generic
   coefficient/QC population membership.
4. Analysis/gridding coefficients remain MAP-facing and deferred. No PTC
   loading, application coefficient, or diagnostic becomes a MAP coefficient,
   weight, precision, sensitivity, or significance by inference.
5. Purely informational diagnostics require no VAL profile. PTC may calculate
   and record quantities such as maximum mode amplitude, admitted-sample
   fraction, residual RMS, or a convergence statistic when each is
   scientifically well defined.
6. Every diagnostic defines its estimator, population and support,
   normalization where applicable, lifecycle, and intended policy role. Its
   uncertainty or statistical interpretation is specified only when
   scientifically required for the claimed use; no uncertainty estimate is
   manufactured merely to satisfy the Registry.
7. A diagnostic requires a separate complete VAL profile when it gains
   admission, exclusion, thresholding, classification, routing, or other
   decision authority. That profile is family- and use-qualified rather than
   inferred from the broad reserved name.
8. Ordinary-route post-fit diagnostics remain advisory. They cannot modify
   fitted support, grouping, configured rank, `\Theta_g`, operator application,
   output retention, or a previously realized transformed signal.
9. Missing or unavailable optional diagnostics do not invalidate an otherwise
   legitimate transformed signal. No excluded occurrence receives a silent
   numerical-zero coefficient.
10. D008 is an explicit unsupported disposition for the broad generic profile,
    not a prohibition on useful PTC diagnostics. It contributes no restriction
    to the future common fragment and creates no runtime abstraction.

## WP5-OWNER-D009 — Existing Tracked-Kernel Response Admission

Question:

> Should `SCI-PTC:response_companion@1` serve only as VAL's admission profile
> for the tracked-kernel propagation already defined by frozen PTC r0.5, without
> creating any new response object, computation, runtime type, or serialization
> requirement?

Owner response:

> approved

Disposition:

1. `SCI-PTC:response_companion@1` is the VAL admission profile for the existing
   frozen PTC tracked-kernel role defined by `SCI-PTC-DEF-016`,
   `SCI-PTC-REQ-062`, `SCI-PTC-REQ-087`, and `SCI-PTC-REQ-097`. It introduces
   no scientifically distinct response object, estimator, computation,
   payload, sidecar, or serialization rule.
2. The existing operation is

   \[
   \underbrace{K^{\rm in}_g}_{\text{existing tracked kernel}}
   \xrightarrow[\text{existing computation}]{J_{\Theta_g}[Y_g]}
   \underbrace{K^{\rm out}_g}_{\text{existing propagated-kernel result}}.
   \]

   The profile answers only whether the exact existing companion is authorized
   to participate in that role.
3. The companion declares its parent domain. A detector-time companion already
   on the CAL grid enters the PTC-local fixed-state operator directly and does
   not receive `K_{\rm up\to CAL}` a second time. A source-domain companion
   composes the existing admitted upstream chain exactly once.
4. Propagation uses the same frozen group-local state as the data: group,
   coordinates, metric, support, masks, positive rank, subspace, tolerance,
   generalized inverse, time-local full-rank guard, detector classes, and
   boundaries. The companion never enters or alters learning or the science
   result, and frozen `\lambda_g` is not subtracted from the perturbation.
5. A group-time unavailable for data application is unavailable for its
   tracked kernel. No different support, lower rank, interpolation, borrowed
   solve, or alternative operator may be substituted under the same identity.
6. Missing complete upstream response makes the source-domain complete-chain
   claim unavailable but does not erase an otherwise supported CAL-grid local
   PTC response or transformed signal. Unrequested or unavailable response
   remains distinct from product realization.
7. The contract requires response authority and recoverability, not dense
   serialization of every kernel element. An implementation may use any
   representation that exactly recovers the authorized response role.
8. Response uncertainty or statistical interpretation is required only when
   scientifically necessary for the claimed use. Propagation alone establishes
   no PSF/beam recovery, photometric validity, MAP authority, or science
   qualification.
9. Any required upstream pointing identity inherits the approved fail-closed
   observation rule; PTC cannot repair unrecoverable pointing.
10. Full-procedure response remains a separate role and is unavailable on the
    ordinary route. D009 authorizes only existing fixed-state tracked-kernel
    propagation.
11. These restrictions remain in `\Delta\mathcal R_{\rm response}` until final
    common-fragment comparison. D009 alone does not produce a final
    digest-bound usable registry record.

## WP5-OWNER-D010 — No Ensemble-Inference Use

Question:

> Does PTC v0.1 make any scientific inference from an ensemble of alternative
> realizations requiring VAL admission under
> `SCI-PTC:empirical_or_simulation_population@1`?

Owner response:

> approved

Disposition:

1. The governing scientific statement is

   \[
   \boxed{
   \text{PTC v0.1 makes no scientific inference from an ensemble of
   alternative realizations.}
   }
   \]

2. There is therefore no present PTC scientific proposition for VAL to admit
   under `SCI-PTC:empirical_or_simulation_population@1`. The reserved identity
   remains explicitly unsupported and unbound for PTC v0.1; no placeholder
   profile or vacuous eligibility result is created.
3. This disposition does not prohibit PTC-owned descriptive diagnostics of the
   realized data under D008. Such diagnostics acquire a VAL profile only if a
   later decision policy gives them admission or other decision authority.
4. Existing fixed-state tracked-kernel propagation under D009 is response
   propagation, not scientific inference from an ensemble of alternative
   realizations.
5. D010 creates no simulation, surrogate, randomization, resampling, ensemble
   uncertainty, runtime, payload, or serialization requirement.
6. If a future PTC revision makes a scientific inference from empirical,
   simulated, surrogate, or alternative realizations, its owner must define a
   separately named exact use and complete policy before VAL can evaluate it.
7. Because no present proposition exists, D010 contributes no restriction to
   the future common fragment.

## WP5-OWNER-D011 — Minimal Common Semantics Fragment

Question:

> After comparing all seven named PTC-use dispositions, what may be factored
> into their common policy fragment without collapsing the uses or allowing
> unrelated metadata to acquire admission authority?

Owner response:

> approved, with the scientific-relevance rule sharpened so that the mere
> existence, availability, or unknown state of unrelated metadata has no
> admission consequence

Disposition:

1. The common fragment is a compact, PTC-owned, immutable, versioned, and
   digest-bound semantics fragment. It is not a VAL profile, cannot be
   evaluated independently, grants no permission, and produces no eligibility
   result.
2. Every named PTC use remains a distinct scientific proposition. Permission
   for one use never implies permission for another:

   \[
   E_U \not\Rightarrow E_V \qquad (U\ne V).
   \]

3. PTC preserves upstream facts and classifications and does not upgrade them.
   In particular, `engineering-only` remains a preserved producer fact, while
   its consequence for admission belongs to each named use-specific policy.
4. Only facts that the named use explicitly declares scientifically relevant
   to its decision may affect that decision. The mere existence,
   availability, or unknown state of other metadata has no admission
   consequence.
5. The common fragment creates no runtime common-policy object, inheritance
   mechanism, sidecar, payload, serialization requirement, duplicated
   provenance, or separate engineering route.
6. Direct-origin exclusions, fit populations, loading-estimator inputs,
   group/rank guards, output-retention rules, response requirements,
   uncertainty requirements, and missing/conflict behavior are not common
   restrictions. They remain in the complete use-specific propositions where
   scientifically applicable.
7. The five supported profiles—basis fit, loading fit, operator application,
   output retention, and response companion—shall each bind the exact common
   fragment identity and digest while remaining complete propositions with
   their own action, applicability, restrictions, and fail-closed behavior.
8. The generic coefficient/QC and empirical/simulation identities remain
   explicit unsupported dispositions, not fabricated profiles. They need not
   reference the common fragment because there is presently no proposition to
   evaluate.
9. No MAP or coadd profile is introduced. Authoring the exact common artifact
   and five source-current registry records is the next bounded registry step.
