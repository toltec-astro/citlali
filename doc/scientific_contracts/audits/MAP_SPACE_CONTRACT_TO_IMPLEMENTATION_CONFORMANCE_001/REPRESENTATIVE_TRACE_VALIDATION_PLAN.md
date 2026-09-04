# Representative Trace Validation Plan

Status: **planning matrix only; no trace was executed by this study**

The trace IDs and route meanings are preserved exactly from CTI-S007.  Each
row states the minimum future evidence needed after source contradictions and
authority gaps are closed.  Passing a legacy test is not a pass for this
matrix.

<!-- BEGIN-TRACE-PLAN -->
| Trace ID | Original representative route | Current source state | Minimum deterministic fixture | Required positive/negative oracle | Planned gate and retained evidence |
| --- | --- | --- | --- | --- | --- |
| MSP-T001 | MSP-E001 ordinary MAP ingress | `MISSING_AUTHORITY` | One transformed occurrence with exact PTC/AST parent, typed coefficient/QC, finite and unavailable variants | Admit only exact requested/applicable/eligible/realized record; reject missing family, inferred unity, stale generation, nonfinite value, or mismatched same-sample coordinate | Unit predicate plus serialized decision/bundle identity and exact contribution census |
| MSP-T002 | MSP-E005 JINC ingress | `MISSING_AUTHORITY` | One array with signed kernel support and exact PTC/AST occurrence identities | Produce only N,C,Q,m,C²-time; reject missing coefficient choice, foreign coordinate, extra numerical roles, and partial bundle publication | Five-role schema gate, per-pixel arithmetic oracle, absence-state assertions |
| MSP-T003 | MSP-E007 forbidden MAP-to-JINC | `UNAVAILABLE_BY_DESIGN` | Attempt to provide MAP observation/coadd object to JINC API | Compile-time/type rejection or deterministic runtime rejection before mutation; no adapter or alias | Negative unit/API gate with pre/post state digest equality |
| MSP-T004 | MSP-E004 MAP coadd | `CONTRADICTORY` | Two centered integer-aligned observation bundles with deliberately unequal pixel coefficients and overlapping original occurrences | Coadd result invariant to pixel-coefficient inequality under `u_op=1`; exposure uses unique-original union; reject odd shifts/foreign identity atomically | Exact numeric truth test, identity/union ledger, mutation-before-rejection guard |
| MSP-T005 | MSP-E023 direct MAP -> POINT | `MISSING_AUTHORITY` | Known isolated source in one exact MAP observation parent with one eligible and one unavailable array | Preserve MAP unit/response/covariance limitations; per-array failure stays local; no detection/catalog action | POINT compatibility/policy evaluation plus per-array atom serialization |
| MSP-T006 | MSP-E009 + MSP-E025 FIXED -> POINT | `CONTRADICTORY` | Immutable MAP parent, declared full-footprint fixed operator, known source, mixed array outcomes | Exact `J_full L_Theta m`; unavailable outside domain is typed, not zero; POINT retains ancestry and effective-shape meaning | Operator truth test, covariance/response state checks, per-array POINT comparison |
| MSP-T007 | MSP-E012/MSP-E014 + MSP-E026 MATCHED -> POINT | `CONTRADICTORY` | Immutable MAP parent and exact external template for separately selected route A and route C | Fixed-anchor amplitude identity; no method mixing/fallback/detection; POINT amplitude retains matched-parent meaning | Route-specific estimator oracle, template digest binding, negative fallback tests |
| MSP-T008 | MSP-E009 + MSP-E021 FIXED -> NOI | `CONTRADICTORY` | Frozen MAP signal plus deterministic NOI ensemble through one fixed operator | Every ensemble member receives bit-identical operator/support/edge treatment as signal; variance/weight planes are not filtered as signal | Operator-identity digest and member-by-member numeric oracle |
| MSP-T009 | MSP-E012/MSP-E014 + MSP-E022 MATCHED -> NOI | `CONTRADICTORY` | Frozen parent/template/method and deterministic ensemble | Predeclared compatible method only; no outcome-adaptive compatibility; each member follows exact estimator | Method/profile identity gate, complete ensemble census, negative compatibility cases |
| MSP-T010 | MSP-E027/MSP-E028/MSP-E029 POINT named use | `MISSING_AUTHORITY` | One complete, one partial, and one failed per-array measurement atom | Separate existence, completeness, eligibility, realization and owner action; `diagnostic_display_only` is consumer action; unavailable profile never falls back | Registered-profile replay test and immutable SCI-VAL evaluation record |
| MSP-T011 | per-array partial success | `IMPLEMENTED_LEGACY_SEMANTICS` | Three arrays: valid fit, numerical failure, unavailable parent | Valid sibling retained; failed/unavailable arrays carry distinct reasons; no whole-observation success assertion | Per-array lifecycle/unit test and output-schema round trip |
| MSP-T012 | JINC missing response/covariance -> POINT | `MISSING_AUTHORITY` | Valid five-role JINC parent with explicitly unavailable response and covariance | Base fit only if selected role permits; dependent amplitude/uncertainty claims unavailable; never insert zero or independence | Typed-state compatibility tests and negative zero-default assertions |
| MSP-T013 | FIXED no-support rows `NOT_PRODUCED` | `CONTRADICTORY` | Parent with rows outside declared full-footprint/operator support | Emit `not_produced`/unavailable with cause; never a zero-valued produced map or valid POINT input | Full-footprint support truth test and serialized absence-state check |
| MSP-T014 | MSP-E031 NOI -> coefficient forbidden | `CONTRADICTORY` | MAP coefficient plane and NOI products with strong empirical scale | MAP/PTC coefficient digest unchanged; attempted promotion rejected before mutation | Pre/post digest invariant and negative integration test |
| MSP-T015 | MSP-E030 MATCHED -> FRUIT | `NOT_APPLICABLE` | None in this study | Only envelope identity may be checked; no FRUIT execution or implementation conclusion | Future separately authorized attachment review; no present test |
| MSP-T016 | MSP-E008 JINC ordinary coadd forbidden | `UNAVAILABLE_BY_DESIGN` | Two valid five-role JINC observation bundles offered to ordinary coadd | Deterministic rejection before mutation; no MAP coadd identity or output publication | Negative API/integration gate with state-digest equality |
<!-- END-TRACE-PLAN -->

## Evidence levels required for future closure

1. Source-level types and invariants must first remove every applicable
   `CONTRADICTORY` and `MISSING_IMPLEMENTATION` state.
2. The scientific owner must close each applicable `MISSING_AUTHORITY` state
   through immutable package/boundary/profile records.
3. Focused deterministic unit tests must exercise positive, unavailable,
   contradictory, nonfinite, and no-mutation-on-rejection cases.
4. Integration tests must preserve exact parent/profile/product identities in
   outputs.
5. Application and Unity evidence, if later authorized, remain separate gates
   and cannot repair a failed source or unit-level invariant.
