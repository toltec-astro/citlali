# MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001 Product And Boundary Graph

Status: completed under owner disposition `MSP-OD-001`; findings
`MSP-F-001`--`MSP-F-004` remain MAJOR and require shared-source repair.
Stable IDs in this file are reused by the matrix, findings, report, traces,
and verifier.

## Product registry

| Product ID | Authoritative producer | Scientific identity and required companions |
| --- | --- | --- |
| MSP-P001 | SCI-PTC boundary | One transformed occurrence with immutable parent/generation, calibrated signal, a typed but currently unselected MAP/JINC coefficient slot, response/uncertainty state, causes, and policy facts.  PTC is not reopened. |
| MSP-P002 | SCI-AST boundary | Exact same-processed-sample RTC-grid coordinate for MAP signal placement. |
| MSP-P003 | SCI-AST boundary | Exact stable-original ALIGN-grid coordinate for MAP original-footprint exposure; distinct from MSP-P002 and descendant membership. |
| MSP-P004 | SCI-MAP | Atomic ordinary observation bundle: nonpolarimetric total-intensity-equivalent signal in inherited `mJy/beam`, numerator/normalization and exact operator identity, original-footprint exposure roles, response and covariance states, WCS/support/lifecycle/parentage. |
| MSP-P005 | SCI-MAP | Atomic centered-integer equal-observation coadd bundle with dimensionless `u_op=1`, exact admitted members, response/covariance states, unique-original exposure, count, WCS/support/lifecycle/parentage. |
| MSP-P006 | SCI-JINC | Atomic observation/array bundle with exactly five numerical roles: `N`, signed `C`, `Q`, derived `m=N/C` with local support/validity, and coefficient-squared time.  Its compact generative record is information state, not a sixth product. |
| MSP-P007 | SCI-FLT-FIXED | Atomic same-grid deterministic full-footprint transform `y=J_full L_Theta m` of one immutable MSP-P004, MSP-P005, or MSP-P006 parent, retaining exact filter/parent/support/response/covariance/lifecycle identity. |
| MSP-P008 | external template boundary | Immutable matched-template authority with exact units, response, support, coordinates, normalization, and lineage required by the selected route. |
| MSP-P009 | SCI-FLT-MATCHED | Atomic fixed-anchor normalized template-amplitude signal bundle.  It is an estimator, not detection, a posterior/Wiener map, generic convolution, or universal source flux. |
| MSP-P010 | SCI-NOI-GEN | Randomized realization ensemble produced under one exact frozen-map operator and immutable parent facts. |
| MSP-P011 | SCI-NOI-UNC | Conditional detector-sign-randomization marginal second moment in squared signal units; not covariance, precision, a reciprocal weight, or significance. |
| MSP-P012 | SCI-NOI-STD | Dimensionless empirical-scale standardized signal formed from an independent immutable MAP signal and `sqrt(MSP-P011)`; not significance. |
| MSP-P013 | SCI-POINT | Per-array fit result for one known isolated bright source and one exact observation-local eligible parent route. |
| MSP-P014 | SCI-POINT | Per-array source-associated measurement atom: displacement, parent-unit amplitude, effective processed-source shape, formal-uncertainty state, support, method, parent, and diagnostics.  Displacement is not a correction. |
| MSP-P015 | named-use policy owner | Immutable child decision for a named use; `diagnostic_display_only` is a consumer action, not a producer lifecycle state. |
| MSP-P016 | SCI-VAL | Registry/source-bound evaluation record preserving request, applicability, eligibility, realization, action, causes, and immutable reasons; VAL evaluates but does not author policy. |
| MSP-PX01 | excluded external boundary | Deferred terminal FRUIT attachment envelope only.  It is not an audited product or an authorized route realization. |

## Topology

```text
                           MSP-P008 TEMPLATE
                                  |
                               MSP-E014
                                  v
MSP-P001 PTC --MSP-E001--> MSP-P004 MAP observation --MSP-E012--> MSP-P009 MATCHED
      |             ^             |     |        |                     |     |
      |             |             |     |        +--MSP-E023----------+     +--MSP-E022--> NOI
      |          MSP-E002         |     +--MSP-E009--> MSP-P007 FIXED --MSP-E025--> POINT
      |          MSP-P002         +--MSP-E004--> MSP-P005 COADD              |
      |          MSP-E003                     |       |                       |
      |          MSP-P003                     +E010 FIXED                     |
      |                                                                         v
      +--MSP-E005--> MSP-P006 JINC --MSP-E011--> FIXED                     MSP-P013
                 ^        |                 |                                 |
              MSP-E006    +--MSP-E024-------+                              MSP-E027
                           +--MSP-E020--> NOI                                 v
                                                                          MSP-P014
MSP-P001/frozen MAP facts --MSP-E017--> MSP-P010 --MSP-E018--> MSP-P011      |
MSP-P004 + MSP-P011 ----------------------MSP-E019---------> MSP-P012       MSP-E028
                                                                          MSP-P015
                                                                             |
                                                                          MSP-E029
                                                                             v
                                                                          MSP-P016

MSP-P009 --MSP-E030--> MSP-PX01 FRUIT (deferred envelope; excluded)
```

MAP and JINC are sibling mapmaking alternatives.  `MSP-E007` explicitly
forbids MAP-to-JINC serialization; `MSP-E008` forbids applying ordinary MAP
coaddition to JINC.  FLT-FIXED and FLT-MATCHED are distinct transformations;
`MSP-E015` and `MSP-E016` forbid an implicit cascade or identity substitution.

## Route and edge catalog

`CONDITIONAL` means the frozen type-level route exists but all exact route
conditions must be met.  `UNAVAILABLE` means a required present authority or
registered evaluation is absent.  `NOT_AUTHORIZED` and `NOT_APPLICABLE` are
negative traces, not missing implementation features.  `OWNER-RESOLVED`
means the frozen scientific meaning now controls under `MSP-OD-001` while the
conflicting shared-conventions clause remains unrepaired.

<!-- BEGIN-GRAPH-EDGES -->
| Edge ID | Producer -> consumer | Frozen status at audit base | Guarantees that cross | Guarantees that do not cross; gates | Evidence |
| --- | --- | --- | --- | --- | --- |
| MSP-E001 | MSP-P001 -> MSP-P004 | `UNAVAILABLE`; scientific identity `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | occurrence/parent/generation, calibrated nonpolarimetric total-intensity-equivalent signal, unit/nominal-beam lineage, typed coefficient slot, response/uncertainty state, causes | PTC selects no MAP-facing numerical coefficient; MAP may not infer unity, loading, `sens`, scatter, inverse variance, or precision; numerical `coverage_cut` also unresolved | MAP PTC boundary:45-75; MAP requirements:4,8; `MSP-F-001`, `MSP-U-001`, `MSP-OD-001` |
| MSP-E002 | MSP-P002 -> MSP-P004 signal | `CONDITIONAL` | same stable processed sample `n`, exact chain and full frame/WCS/coordinate validity | row/time/shape equality cannot substitute for the exact join; availability follows MSP-E001 | MAP PTC boundary:40-43; MAP requirements:7; `MSP-U-001` |
| MSP-E003 | MSP-P003 -> MSP-P004 exposure | `CONDITIONAL`; scientific meaning `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | one stable original's exact AST ALIGN-grid coordinate, lineage, accounting unit/convention and deduplication identity | never descendant signal membership, processed/filter/interpolation footprint, operator/response support, normalized influence, statistical weight, precision, complete temporal support, or an inferred coordinate | AST-to-MAP boundary:23-68; MAP requirements:9,31-32; `MSP-F-003`, `MSP-OD-001` |
| MSP-E004 | MSP-P004 -> MSP-P005 | `CONDITIONAL`; coefficient meaning `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | exact compatible atomic bundle, centered-integer placement, observation-level dimensionless `u_op=1` per admitted observation row, member response/covariance states, unique-original exposure and count | the observation-level coefficient does not replace or flatten sample-, pixel-, numerator-, denominator-, validity-, or coverage-level information; no empirical/precision weight, JINC coadd generalization, reprojection, interpolation, fractional shift, hidden response subset, zero covariance block, or inferred independence | MAP coadd profiles:14-109; MAP requirements:39-43; `MSP-F-002`, `MSP-U-002`, `MSP-OD-001` |
| MSP-E005 | MSP-P001 -> MSP-P006 | `UNAVAILABLE`; product closure `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` | exact PTC/JINC coefficient declaration slot, signal identity/unit, immutable occurrence/array/generation, typed states and causes | no permitted coefficient family or TolTEC parameter set; no base response/covariance/formal-weight/physical-exposure/standalone-support/diagnostic/generalized-provenance numerical role is conferred or downstream-inferable | PTC-to-JINC boundary:58-92,124-148; JINC requirements:193-231,290-309; `MSP-F-004`, `MSP-U-003`, `MSP-OD-001` |
| MSP-E006 | AST coordinate -> MSP-P006 | `CONDITIONAL` | exact same-processed-sample continuous coordinate, frame/WCS, array and validity state | no time/row fallback; JINC owns rounded-center geometry, phase, square support and edge crop | AST-to-JINC boundary:32-142; `MSP-U-003` |
| MSP-E007 | MSP-P004/MSP-P005 -> MSP-P006 | `NOT_AUTHORIZED` | none | JINC consumes PTC and AST directly; a MAP product is not a parent | JINC definitions:5-10; requirements:17-21; `MSP-U-004` |
| MSP-E008 | MSP-P006 -> MSP-P005 | `NOT_AUTHORIZED` | none | base JINC has no cross-observation combination and cannot inherit ordinary MAP coaddition | JINC requirements:257-260; `MSP-U-004` |
| MSP-E009 | MSP-P004 -> MSP-P007 | `CONDITIONAL`, currently `UNAVAILABLE` | complete immutable MAP observation parent, selected signal, exact same-grid transform, parent response/covariance states | filter cannot learn/select from payload, relabel units/beam, treat unavailable outside rows as zero, or turn marginal variance into independence | FLT-FIXED MAP boundary:21-85; shared core:116-164,362-567; `MSP-U-005` |
| MSP-E010 | MSP-P005 -> MSP-P007 | `CONDITIONAL`, currently `UNAVAILABLE` | complete immutable MAP coadd parent and same guarantees as MSP-E009 | exact coadd parent identity/response/covariance limits remain visible | same as MSP-E009; `MSP-U-005` |
| MSP-E011 | MSP-P006 -> MSP-P007 | `CONDITIONAL`, currently `UNAVAILABLE` | exact atomic five-role JINC bundle; only selected map signal is transformed | no synthesis of JINC response/covariance; no separation or loss of atomic sibling roles | FLT-FIXED JINC boundary:21-67; `MSP-U-005` |
| MSP-E012 | MSP-P004 -> MSP-P009 | `CONDITIONAL`, currently `UNAVAILABLE` | exact MAP observation bundle, one selected A or C method, one template, support/weight/response/covariance states | JINC excluded; no mixing/fallback among methods; no concrete numerical weighting or registered role profile is supplied by freeze | MATCHED MAP boundary:14-42; freeze:27-51; `MSP-U-006` |
| MSP-E013 | MSP-P005 -> MSP-P009 | `CONDITIONAL`, currently `UNAVAILABLE` | exact MAP coadd bundle under the same method-specific obligations | same exclusions as MSP-E012 | same as MSP-E012; `MSP-U-006` |
| MSP-E014 | MSP-P008 -> MSP-P009 | `CONDITIONAL`, currently `UNAVAILABLE` | exact immutable template identity, units, normalization, support, coordinates and response lineage | no hidden template choice, fit, adaptation, or source detection | template boundary:14-32; `MSP-U-006` |
| MSP-E015 | MSP-P007 -> MSP-P009 | `NOT_AUTHORIZED` | none | MATCHED parents are exact MAP observation/coadd bundles, not a FIXED product | MATCHED definitions:37-52; `MSP-U-004` |
| MSP-E016 | MSP-P009 -> MSP-P007 | `NOT_AUTHORIZED` | none | FIXED parents are MAP observation/coadd or JINC observation, not MATCHED | FLT-FIXED shared core:116-164; `MSP-U-004` |
| MSP-E017 | MSP-P001 plus frozen MAP facts -> MSP-P010 | `UNAVAILABLE` | exact fixed-state frozen-MAP contribution and lossless design-balance coefficient lineage | design-balance coefficient is not precision, empirical weight, exposure, validity, or a replacement for PTC `gamma`; r0.5 `@2` profiles unregistered and numerical MAP parent unavailable | NOI PTC boundary:89-116,166-194; `MSP-U-007` |
| MSP-E018 | MSP-P010 -> MSP-P011 | `CONDITIONAL`, currently `UNAVAILABLE` | exact complete ensemble, conditioning law, member identities and squared signal unit | ensemble does not automatically imply variance, covariance, precision, weight or significance | NOI definitions:5-36; requirements:125-163; `MSP-U-007` |
| MSP-E019 | MSP-P004 + MSP-P011 -> MSP-P012 | `UNAVAILABLE` | independent immutable MAP signal plus direct square root of matching marginal second moment; output unit 1 | not significance; cannot mutate MAP/PTC coefficients or validity; no `@1` fallback for changed r0.5 semantics | NOI requirements:165-217; successor table:19-29; `MSP-U-007` |
| MSP-E020 | MSP-P006 -> NOI family | `UNAVAILABLE` | if later bound, complete atomic JINC parent and exact method/array identities | current numerical JINC route unavailable; no invented response/covariance or partial bundle | NOI JINC boundary:33-108; `MSP-U-007` |
| MSP-E021 | MSP-P007 -> NOI family | `CONDITIONAL`, currently `UNAVAILABLE` | every NOI member must receive the identical frozen FIXED operator and parent identity | no filtering a variance/weight plane; no re-resolution or member-specific operator | FIXED-to-NOI boundary:19-84; `MSP-U-005`, `MSP-U-007` |
| MSP-E022 | MSP-P009 -> NOI family | `CONDITIONAL`, currently `UNAVAILABLE` | predeclared compatible transform/method facts only | no outcome-adaptive compatibility, profile fallback, or inferred covariance | MATCHED-to-NOI boundary:10-33; `MSP-U-006`, `MSP-U-007` |
| MSP-E023 | MSP-P004 -> MSP-P013 | `UNAVAILABLE` | distinct eligible POINT parent family; amplitude retains exact MAP unit/calibration/response and widths/angle describe processed shape | boundary mapping is draft/unbound; POINT compatibility method and profile evaluations unavailable | POINT MAP boundary:3-27; POINT assumptions:82-102; `MSP-U-008` |
| MSP-E024 | MSP-P006 -> MSP-P013 | `UNAVAILABLE` | distinct eligible JINC parent family if exactly mapped | no MAP alias; no response/covariance invented; boundary and compatibility method unavailable | POINT JINC boundary:3-28; `MSP-U-008` |
| MSP-E025 | MSP-P007 -> MSP-P013 | `UNAVAILABLE` | distinct eligible FIXED parent with exact MAP/JINC ancestry and filter response state | no universal flux/beam claim; exact upstream and filter response are required where the claim depends on them | POINT FIXED boundary:3-29; `MSP-U-008` |
| MSP-E026 | MSP-P009 -> MSP-P013 | `UNAVAILABLE` | distinct eligible MATCHED amplitude-field parent | no source detection; no extension to a hidden upstream parent; boundary/compatibility unavailable | POINT MATCHED boundary:3-28; `MSP-U-008` |
| MSP-E027 | MSP-P013 -> MSP-P014 | `CONDITIONAL`, no current numerical instance | one per-array fit may yield displacement, parent-unit amplitude, effective shape, uncertainty/support/method/diagnostics with exact parent | a failed array does not erase valid siblings; no whole-observation success; unavailable formal method blocks only dependent roles | POINT definitions:98-120; assumptions:4-102,196-211; `MSP-U-008` |
| MSP-E028 | MSP-P014 -> MSP-P015 | `CONDITIONAL`, currently `UNAVAILABLE` | immutable measurement facts enter a separately owned named-use policy | result existence/completeness is not eligibility; `diagnostic_display_only` is a consumer action | POINT notation:47-82; definitions:138-166; `MSP-U-009` |
| MSP-E029 | MSP-P015 -> MSP-P016 | `UNAVAILABLE` | VAL would preserve four axes, action, causes and exact source/profile identity | all four POINT profiles are draft/unregistered; VAL cannot author or silently substitute policy | POINT policy records; VAL Registry; `MSP-U-009` |
| MSP-E030 | MSP-P009 -> MSP-PX01 | `NOT_APPLICABLE` to this audit | only the exact deferred attachment-envelope identity is recorded | no FRUIT source, algorithm, validation, policy, route realization, worktree or branch is admitted | MATCHED-to-FRUIT envelope:3-31; `MSP-U-010` |
| MSP-E031 | MSP-P011/MSP-P012 -> PTC or MAP coefficients | `NOT_AUTHORIZED` | immutable lineage/correlation information only | NOI assignment, uncertainty, reciprocal, standardized scale or weight never becomes PTC/MAP coefficient, validity, support, exposure or mapmaking authority | NOI requirements:206-217; `MSP-U-004` |
| MSP-E032 | MSP-P013/MSP-P014 -> source detection/catalog | `NOT_AUTHORIZED` | fit and measurement facts for the one known source only | POINT is not detection, search, catalog construction, universal photometry, intrinsic beam inference, or pointing-correction construction | POINT README:22-44; requirements:1-3; `MSP-U-004` |
<!-- END-GRAPH-EDGES -->

## Companion and failure invariants

- A complete MAP consumer boundary is not a bare signal plane.  Response and
  covariance may be honestly limited, partial, symbolic, lineage-resolvable,
  or unavailable where the exact role permits; unavailable is never zero.
- MSP-P006 has exactly five numerical roles.  Response, covariance,
  formal-weight, exposure, standalone support and diagnostics are not base
  JINC products.
- MSP-P007 propagates declared covariance as `A C_parent A^T`; it applies the
  same fixed operator to admitted NOI members and never convolves a weight or
  variance plane as though it were signal.
- MSP-P009 route A has a local-GLS theorem only under its exact local covariance
  assumptions; route C carries no such covariance/optimality claim.
- MSP-P011 is a marginal second moment.  It is not joint covariance, zero
  cross-covariance, independence, a precision product, or a POINT formal-error
  method.
- MAP, JINC, FLT-FIXED and FLT-MATCHED products retain distinct route identity
  through POINT.  Amplitude keeps the exact parent unit/calibration/response;
  width and angle remain effective processed-source-shape quantities.
- Per-array atomicity is preserved at JINC and POINT.  One array's failure does
  not erase a valid sibling and does not establish whole-observation success.

## Owner-disposition overlay and repair boundary

`MSP-OD-001` resolves MSP-E001, MSP-E003, MSP-E004 and MSP-E005 in favor of
the frozen MAP/JINC meanings shown above.  Scientific/package topology is
coherent under that disposition; all independent `MSP-U-*` gates remain in
force.  The four findings remain MAJOR because the contradictory clauses in
`doc/SCIENTIFIC_CONVENTIONS.md` are still present.  Their exact clause-level
repair specification is in `FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md` and is
outside this audit's edit scope.  No additional consequential authority
conflict appeared during completion.
