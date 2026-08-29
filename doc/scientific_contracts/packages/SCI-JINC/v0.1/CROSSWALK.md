| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |
| `SCI-JINC-REQ-001` | Rationale Sec. 2-3; definition "SCI-JINC observation estimator" | Bind a one-observation, per-array JINC route directly to PTC/AST and reject ordinary MAP inheritance. |
| `SCI-JINC-REQ-002` | Rationale Sec. 3; definitions "Exact occurrence" and "Exact AST association" | Require an exact same-processed-sample relation; inferred row/time/order joins do not conform. |
| `SCI-JINC-REQ-003` | Rationale Sec. 3; `SCI-JINC-ASM-002` | Preserve the complete producer-owned transformed-signal identity and meaning without reinterpretation. |
| `SCI-JINC-REQ-004` | Rationale Sec. 3; `SCI-JINC-ASM-003` | Bind the exact PTC registry family, explicit JINC permission, lifecycle, payload, QC, and statistical semantics. |
| `SCI-JINC-REQ-005` | Rationale Sec. 10; definition "Typed numerical unavailability" | Fail closed for absent selection/registration/permission/payload; no unity or alternate-family fallback. |
| `SCI-JINC-REQ-006` | Rationale Sec. 3; canonical notation for `omega_i` | Deposit only a finite strictly positive producer coefficient and preserve other typed dispositions. |
| `SCI-JINC-REQ-007` | Rationale Sec. 4 and 10; `SCI-JINC-ASM-005` | Require one complete authorized array parameter identity, positive pixel size, and integer `subpixel_n>=1`. |
| `SCI-JINC-REQ-008` | Rationale Sec. 2; Eq. `peak-jinc` | Implement the peak-normalized Bessel limit exactly and finitely at zero. |
| `SCI-JINC-REQ-009` | Rationale Sec. 2; Eq. `analytic-kernel` | Preserve the ordered analytic family, peak, zeros, and finite signed lobes. |
| `SCI-JINC-REQ-010` | Rationale Sec. 4; definition "Point phase" | Realize rounded-center, quantized point-phase evaluation; pixel-area integration is nonconformant. |
| `SCI-JINC-REQ-011` | Rationale Sec. 4; `SCI-JINC-ASM-007` | Disclose single-valued discrete choices and show their combined error meets the scientific adequacy scale. |
| `SCI-JINC-REQ-012` | Rationale Sec. 4; definition "Square support" | Populate the full square cache, including corners beyond radial `r_max`, with no circular mask. |
| `SCI-JINC-REQ-013` | Rationale Sec. 4; definition "Rounded-center domain admission" | Reject an outside rounded center before footprint evaluation, including an overlapping-square case. |
| `SCI-JINC-REQ-014` | Rationale Sec. 4; `SCI-JINC-ASM-006` | Crop an in-map square at the finite boundary without wrap, completion, or renormalization. |
| `SCI-JINC-REQ-015` | Rationale Sec. 2 and 4; canonical notation for `kappa_ip` | Preserve zero/negative analytic meanings and fail the affected bundle on non-finite kernel evaluation. |
| `SCI-JINC-REQ-016` | Rationale Sec. 5; definition "Sample-pixel membership" | Use one exact membership and coefficient realization for every coupled accumulator term. |
| `SCI-JINC-REQ-017` | Rationale Sec. 5; Eqs. `numerator`-`quadratic` | Maintain distinct signed `N`, signed `C`, and quadratic `Q` accumulators. |
| `SCI-JINC-REQ-018` | Rationale Sec. 5; Eqs. `map-estimate` and `operator` | Normalize with `N/C`; a `Q` denominator or another local model does not conform. |
| `SCI-JINC-REQ-019` | Rationale Sec. 5 and 7; definition "Local formal support" | Apply every finite, positivity, nonzero, and numerical-conditioning gate before publishing a local map value. |
| `SCI-JINC-REQ-020` | Rationale Sec. 5; Eq. `normalization` | Accept finite negative normalization and reject exact cancellation without a substitute value. |
| `SCI-JINC-REQ-021` | Rationale Sec. 5; Eqs. `absolute-sum` and `cancellation-ratio` | Retain dimensionless `rho`; support near cancellation only with an adequate numerical-error demonstration. |
| `SCI-JINC-REQ-022` | Rationale Sec. 5; `SCI-JINC-ASM-007` | Do not promote a universal cutoff, floor, summation algorithm, fixed order, or bitwise result to scientific policy. |
| `SCI-JINC-REQ-023` | Rationale Sec. 4-5; `SCI-JINC-ASM-007` | Budget function, phase/cache, and accumulation errors together against the approximately `10^-3` scale. |
| `SCI-JINC-REQ-024` | Rationale Sec. 5; notation table; `SCI-JINC-PRED-004`-`005` | Preserve both coefficient unit classes and required common-scale/unit-rescaling invariances. |
| `SCI-JINC-REQ-025` | Rationale Sec. 7; Eqs. `conditional-variance`-`general-covariance` | Permit formal statistical labels only when the exact family and covariance assumptions establish them. |
| `SCI-JINC-REQ-026` | Rationale Sec. 6; Eq. `kappa-time` | Accumulate seconds from `kappa^2/f_s`; do not square `omega` or the complete signed weight. |
| `SCI-JINC-REQ-027` | Rationale Sec. 6; definition "Complete observation bundle" | Label coefficient-squared time only as method-specific accounting, never exposure, precision, or validity. |
| `SCI-JINC-REQ-028` | Rationale Sec. 6; fixed-bundle definition | Publish exactly the five required roles and no inferred companion role. |
| `SCI-JINC-REQ-029` | Rationale Sec. 6; definition "Whole-bundle failure" | Suppress a bundle on required whole-role failure while retaining pixel-local invalidity as ordinary map content. |
| `SCI-JINC-REQ-030` | Rationale Sec. 6; `SCI-JINC-ASM-008` | Produce zero through three bundles, at most one per requested/admitted stable array, with no placeholders. |
| `SCI-JINC-REQ-031` | Rationale Sec. 6; definition "Scientific destination identity" | Bind each bundle independently and prohibit cross-array or cross-destination merging. |
| `SCI-JINC-REQ-032` | Rationale Sec. 6; definition "Scientific destination identity" | Resolve unique destination ownership before mutation and fail ambiguity without a partial winner. |
| `SCI-JINC-REQ-033` | Rationale Sec. 6; `SCI-JINC-ASM-008` | Allow same-observation chunking only under one complete scientific identity; implementation partitions are not products. |
| `SCI-JINC-REQ-034` | Rationale Sec. 6; `SCI-JINC-ASM-008` | Reject cross-observation combination until a separate complete-bundle coadd boundary exists. |
| `SCI-JINC-REQ-035` | Rationale Sec. 3 and 7; definition "JINC sample admission" | Keep producer availability, profile admission, geometry, support, conditioning, and bundle validity as separate gates. |
| `SCI-JINC-REQ-036` | Rationale Sec. 3 and 7; sample-pixel membership definition | Apply only named restrictions, preserve causes, and do not invent causes for zero, negative, outside-support, or outside-center states. |
| `SCI-JINC-REQ-037` | Rationale Sec. 3; `SCI-JINC-ASM-002` | Enforce the producer-transformer-consumer ownership split without reconstructing upstream or downstream science. |
| `SCI-JINC-REQ-038` | Rationale Sec. 7; conditional-math subsection | Preserve conditional response/covariance reasoning but publish no base response, variance, formal-weight, covariance, or significance role. |
| `SCI-JINC-REQ-039` | Rationale Sec. 6 and 10; Eq. `kappa-time` | Publish no physical-exposure role and never reinterpret coefficient-squared time as exposure. |
| `SCI-JINC-REQ-040` | Rationale Sec. 6 and 10; missing-facts table | Do not add standalone availability/support, optional-role, diagnostic, detailed-cause, or generalized provenance machinery. |
| `SCI-JINC-REQ-041` | Rationale Sec. 10; `SCI-JINC-ASM-005` | Keep every TolTEC numerical value unavailable, including inherited 45 m, shape, and mode-dependent `r_max` values. |
| `SCI-JINC-REQ-042` | Rationale Sec. 7 and 9; claim-layers and evidence-layers tables; predictions appendix | Assess every stable prediction under a disclosed policy and report each evidentiary claim layer separately. |
