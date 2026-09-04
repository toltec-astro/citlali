# Route Availability Classification

Status: **32-edge source-level classification; owner review required**

“Available” below means only that a bounded source path exists.  It does not
mean scientifically usable, validated, performant, production-ready, or
authorized.  A primary state from the work-order vocabulary is assigned to
every original edge.

<!-- BEGIN-ROUTE-CLASSIFICATION -->
| Edge ID | Frozen route | Primary state | Source path present? | Blocking reason / direct evidence |
| --- | --- | --- | --- | --- |
| MSP-E001 | MSP-P001 -> MSP-P004 | `MISSING_AUTHORITY` | legacy path only | `naive_mm.h:436-454` consumes `in.weights`, but no owner-selected MAP coefficient family/QC proposition is carried. |
| MSP-E002 | MSP-P002 -> MSP-P004 signal | `IMPLEMENTED_LEGACY_SEMANTICS` | native predecessor route | `timestream_native_science_projection.h:314-326,439-494` checks native identity during construction; its consumer guard at `211-279` compares payload/inventory but not incoming occurrence/application generation or exact target-WCS identity.  `naive_mm.h:376-390` uses the stored pointing; the frozen AST same-sample join is not established.  MSP-E001/MSP-P004 remain additional blockers. |
| MSP-E003 | MSP-P003 -> MSP-P004 exposure | `MISSING_IMPLEMENTATION` | no | `naive_mm.h:400-434,522-530` places exposure at the processed signal coordinate; no original-coordinate/deduplication path was found. |
| MSP-E004 | MSP-P004 -> MSP-P005 | `CONTRADICTORY` | yes, legacy | `observation_coadd_accumulation.h:681-750,829-856` applies per-pixel `omb.weight`, not observation-row `u_op=1`, and sums exposure. |
| MSP-E005 | MSP-P001 -> MSP-P006 | `MISSING_AUTHORITY` | legacy path only | `jinc_mm.h:638-760` consumes the same legacy detector weight; permitted PTC/JINC coefficient family and exact upstream bundle remain unavailable. |
| MSP-E006 | AST coordinate -> MSP-P006 | `IMPLEMENTED_LEGACY_SEMANTICS` | native predecessor route | `jinc_mm.h:370-375,607-617` checks then consumes the native projection.  Construction-local identity checks exist, but `timestream_native_science_projection.h:211-279` does not bind the incoming occurrence/application generation or exact target JINC WCS.  The frozen same-processed-sample AST join is not established; the downstream JINC product remains contradictory. |
| MSP-E007 | MSP-P004/MSP-P005 -> MSP-P006 | `UNAVAILABLE_BY_DESIGN` | no authorized path | JINC dispatch consumes PTC and coordinates directly; the frozen graph forbids MAP serialization into JINC. |
| MSP-E008 | MSP-P006 -> MSP-P005 | `UNAVAILABLE_BY_DESIGN` | no authorized path | Ordinary coadd is not a JINC operation; base JINC has no cross-observation coadd authority. |
| MSP-E009 | MSP-P004 -> MSP-P007 | `CONTRADICTORY` | legacy path | `wiener_filter.h:1054-1104,1314-1449` mutates the parent buffer, derives learned/empirical support, filters weight-like planes, and zeros unavailable output. |
| MSP-E010 | MSP-P005 -> MSP-P007 | `CONTRADICTORY` | legacy path | The same mutable filter accepts coadd buffers without a frozen immutable coadd parent or response/covariance transport. |
| MSP-E011 | MSP-P006 -> MSP-P007 | `CONTRADICTORY` | legacy path | The same filter can act on the shared JINC buffer, whose extra forbidden numerical roles are inseparable from the mutable storage. |
| MSP-E012 | MSP-P004 -> MSP-P009 | `CONTRADICTORY` | legacy path | `wiener_filter.h:1324-1377` implements legacy Wiener filtering, not one frozen MATCHED A/C amplitude route. |
| MSP-E013 | MSP-P005 -> MSP-P009 | `CONTRADICTORY` | legacy path | Coadd input is accepted by the same legacy filter without exact frozen coadd/template/method binding. |
| MSP-E014 | MSP-P008 -> MSP-P009 | `IMPLEMENTED_LEGACY_SEMANTICS` | partial | `post_processing_config.h:19-24,145-160` and `map_filter_config_policy.h:12-51` select/generate templates but do not carry immutable external-template authority. |
| MSP-E015 | MSP-P007 -> MSP-P009 | `UNAVAILABLE_BY_DESIGN` | no authorized path | `MapFilterType` selects one filter mode; the frozen graph forbids implicit FIXED-to-MATCHED cascade or substitution. |
| MSP-E016 | MSP-P009 -> MSP-P007 | `UNAVAILABLE_BY_DESIGN` | no authorized path | The frozen graph forbids MATCHED-to-FIXED substitution; no authorized product adapter was found. |
| MSP-E017 | MSP-P001 + frozen MAP facts -> MSP-P010 | `MISSING_AUTHORITY` | legacy generator only | Sign generation exists, but the PTC design-balance coefficient and frozen-map parent/profile are not selected or registered. |
| MSP-E018 | MSP-P010 -> MSP-P011 | `IMPLEMENTED_LEGACY_SEMANTICS` | yes, legacy | `map.cpp:1382-1397` computes centered `S_R/R`; the ensemble/conditioning identity is predecessor `SCI-NOI-002-v1`, so the frozen route is not available. |
| MSP-E019 | MSP-P004 + MSP-P011 -> MSP-P012 | `IMPLEMENTED_LEGACY_SEMANTICS` | partial | `map.cpp:1495-1525` emits coefficient-standardized and scatter-ratio products, neither bound to the exact frozen parent pair. |
| MSP-E020 | MSP-P006 -> NOI family | `MISSING_AUTHORITY` | legacy path only | Noise storage can accompany JINC, but the numerical JINC route, response/covariance states, and compatible NOI profile are unavailable. |
| MSP-E021 | MSP-P007 -> NOI family | `CONTRADICTORY` | legacy path | `noise_execution_plan.h:53-56` explicitly records that signal and realization edge handling lack strict parity; the frozen route requires the identical fixed operator. |
| MSP-E022 | MSP-P009 -> NOI family | `CONTRADICTORY` | legacy path | The same declared parity gap and absent frozen MATCHED method/profile prevent conformant member transport. |
| MSP-E023 | MSP-P004 -> MSP-P013 | `MISSING_AUTHORITY` | legacy fit only | Raw MAP fits run (`pointing_fit_maps_impl.h:32-68`), but the exact POINT boundary/compatibility method/profile is unbound. |
| MSP-E024 | MSP-P006 -> MSP-P013 | `MISSING_AUTHORITY` | legacy fit only | Shared-buffer fit plumbing cannot substitute for the unavailable JINC-specific boundary and method. |
| MSP-E025 | MSP-P007 -> MSP-P013 | `MISSING_AUTHORITY` | legacy fit only | Filtered observation fits run, but frozen parent ancestry, response state and POINT compatibility policy are absent. |
| MSP-E026 | MSP-P009 -> MSP-P013 | `MISSING_AUTHORITY` | legacy fit only | No exact MATCHED amplitude-field parent or registered POINT compatibility decision is carried into the fit. |
| MSP-E027 | MSP-P013 -> MSP-P014 | `IMPLEMENTED_LEGACY_SEMANTICS` | yes, legacy | `pointing_fit_maps_impl.h:32-99` preserves per-map attempts/validity and does not erase sibling results, but lacks the atomic frozen measurement/lifecycle object. |
| MSP-E028 | MSP-P014 -> MSP-P015 | `MISSING_AUTHORITY` | no | No registered owner-bound POINT named-use policy exists; fit results cannot self-author eligibility. |
| MSP-E029 | MSP-P015 -> MSP-P016 | `MISSING_AUTHORITY` | no | Without an immutable POINT policy/profile, SCI-VAL cannot evaluate this edge; no runtime VAL evaluator was found either. |
| MSP-E030 | MSP-P009 -> MSP-PX01 | `NOT_APPLICABLE` | not inspected | Only the frozen terminal attachment envelope is retained.  FRUIT implementation and active branch are excluded. |
| MSP-E031 | MSP-P011/MSP-P012 -> PTC or MAP coefficients | `CONTRADICTORY` | forbidden flow exists | `noise_weight_policy.h:7-17` and `map.cpp:1465-1493` permit an empirical NOI-derived scale to replace the live MAP `weight` plane. |
| MSP-E032 | MSP-P013/MSP-P014 -> detection/catalog | `UNAVAILABLE_BY_DESIGN` | no direct authorized edge | `source_finding_execution_impl.h:9-66` is a separate detect-then-fit pipeline.  Reuse of the same fitter type is not a POINT-to-catalog authority and must remain separate. |
<!-- END-ROUTE-CLASSIFICATION -->

## Availability summary

| Primary state | Edge count |
| --- | ---: |
| `IMPLEMENTED_CONFORMANT_AT_SOURCE_LEVEL` | 0 |
| `IMPLEMENTED_LEGACY_SEMANTICS` | 6 |
| `DECLARED_NOT_IMPLEMENTED` | 0 |
| `UNAVAILABLE_BY_DESIGN` | 5 |
| `MISSING_AUTHORITY` | 10 |
| `MISSING_IMPLEMENTATION` | 1 |
| `CONTRADICTORY` | 9 |
| `NOT_APPLICABLE` | 1 |
| `INDETERMINATE` | 0 |
| Total | 32 |

MSP-E002 and MSP-E006 retain useful construction-local native behavior under
legacy classifications; the missing consumer identity/WCS binding prevents a
frozen coordinate-boundary conformance claim.  All other edge classifications
are unchanged.  This packet identifies **zero complete conformant end-to-end
map-space routes** at the inspected source tree.
