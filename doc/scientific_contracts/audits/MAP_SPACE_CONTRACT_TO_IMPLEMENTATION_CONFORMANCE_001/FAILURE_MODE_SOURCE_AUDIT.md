# Failure-Mode Source Audit

Status: **source-level hazards at exact base; no fixes authorized**

This audit searches for failure modes that could make a legacy path appear
scientifically usable even when required authority or products are absent.
Severity is about scientific misrepresentation risk, not measured operational
frequency.

| Finding | Severity | State | Affected graph | Direct source evidence | Failure mode and required invariant |
| --- | --- | --- | --- | --- | --- |
| CTI-FM-001 | CRITICAL | `CONTRADICTORY` | MSP-P004, MSP-E003 | `include/citlali/core/mapmaking/naive_mm.h:400-434,522-530` | Eligible and retained exposure use the processed signal coordinate and processed membership.  A stable original may be duplicated, moved, or omitted.  Exposure must instead use the distinct original-coordinate occurrence ledger and unique-original accounting. |
| CTI-FM-002 | CRITICAL | `CONTRADICTORY` | MSP-P005, MSP-E004 | `include/citlali/core/pipeline/observation_coadd_accumulation.h:681-750,829-856` | Coadd multiplies by each pixel's live coefficient and sums exposure.  This silently substitutes a weighted mean for the frozen equal-observation `u_op=1` aggregate and lacks original-occurrence union. |
| CTI-FM-003 | CRITICAL | `CONTRADICTORY` | MSP-P006 | `include/citlali/core/mapmaking/jinc_contract.h:28-47,275-304,331-369`; `src/citlali/core/mapmaking/map.cpp:68-82` | JINC is labeled as a MAP contract and requires extra formal-weight, support, response, exposure/coverage, denominator-absolute-sum and contributor-count roles.  The five-role atomic bundle is not preserved. |
| CTI-FM-004 | CRITICAL | `CONTRADICTORY` | MSP-P006 | `src/citlali/core/mapmaking/map.cpp:176-220,224-267` | Unsupported, unresolved, nonfinite, or missing-companion JINC pixels are converted to numerical zeros in signal, coefficient, exposure, kernel and realizations.  Unavailable is not zero and missing roles must not be fabricated. |
| CTI-FM-005 | CRITICAL | `CONTRADICTORY` | MSP-E031 | `include/citlali/core/pipeline/noise_weight_policy.h:7-17`; `src/citlali/core/mapmaking/map.cpp:1465-1493` | NOI-derived empirical scaling may replace the live MAP normalization coefficient.  Frozen NOI products never become PTC/MAP coefficients, validity, support, exposure, or mapmaking authority. |
| CTI-FM-006 | MAJOR | `MISSING_AUTHORITY` | MSP-E001, MSP-E005, MSP-E017 | `include/citlali/core/timestream/ptc/ptcproc.h:2358-2411,2947-2958`; `include/citlali/core/mapmaking/naive_mm.h:436-454` | A generic `weights` field is populated from several legacy formulas and consumed by MAP/JINC.  No exact owner-selected coefficient family, named-use profile, QC proposition, or typed unavailability reaches the consumer.  Numeric positivity is not authority. |
| CTI-FM-007 | MAJOR | `CONTRADICTORY` | MSP-P007, MSP-E009--MSP-E011 | `include/citlali/core/mapmaking/wiener_filter.h:1062-1104,1314-1449` | The fixed-transform analogue learns/derives support from live weight/coverage, modifies the shared parent in place, propagates a reciprocal weight as variance, and writes zero outside support.  The frozen full-footprint operator requires immutable parentage and typed availability/covariance transport. |
| CTI-FM-008 | MAJOR | `CONTRADICTORY` | MSP-P009, MSP-E012--MSP-E014 | `include/citlali/core/config/post_processing_config.h:13-24,145-160`; `include/citlali/core/mapmaking/wiener_filter.h:1054-1057,1324-1377` | One legacy mode switch and template family are treated as the filtering identity.  Neither one selected frozen MATCHED A/C method nor immutable template/anchor/normalization/response lineage is represented.  A Wiener name cannot stand in for the frozen amplitude estimator. |
| CTI-FM-009 | MAJOR | `CONTRADICTORY` | MSP-E021, MSP-E022 | `include/citlali/core/pipeline/noise_execution_plan.h:53-56` | Source itself declares strict signal/realization filter-edge parity not established.  Frozen filtered NOI requires every member to receive the identical frozen operator; zero-centered realization handling cannot be assumed equivalent to signal-background affine handling. |
| CTI-FM-010 | MAJOR | `MISSING_AUTHORITY` | MSP-E023--MSP-E029 | `include/citlali/core/engine/detail/pointing_fit_maps_impl.h:32-99`; `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md:280-302` | Fit success and stored errors exist, but exact parent-route compatibility, formal-error method, named use, owner action and VAL profile are absent.  A valid numerical fit cannot silently become eligible measurement or consumer policy. |
| CTI-FM-011 | MAJOR | `IMPLEMENTED_LEGACY_SEMANTICS` | MSP-E027 | `include/citlali/core/engine/detail/pointing_fit_maps_impl.h:32-94`; `include/citlali/core/pipeline/pointing_execution_plan.h:160-188` | Per-map attempts and valid siblings are retained, which is compatible with per-array atomicity, but the output has only counts/rows rather than typed per-array failure causes and immutable parent/lifecycle.  Whole-observation success must not be inferred from any valid row. |
| CTI-FM-012 | MAJOR | `UNAVAILABLE_BY_DESIGN` | MSP-E007, MSP-E008 | `include/citlali/core/pipeline/observation_coadd_accumulation.h:761-875`; `include/citlali/core/mapmaking/jinc_mm.h:370-384` | Ordinary MAP coadd and JINC are implemented through different entry points.  No adapter may serialize MAP into JINC or ordinary-coadd JINC; future refactoring must preserve this negative boundary. |
| CTI-FM-013 | MAJOR | `UNAVAILABLE_BY_DESIGN` | MSP-E015, MSP-E016 | `include/citlali/core/config/post_processing_config.h:13-17,145-160` | A single filter-type choice currently prevents an explicit cascade, but no frozen product type enforces the prohibition.  Future code must reject implicit FIXED/MATCHED substitution even when arrays and units appear compatible. |
| CTI-FM-014 | MAJOR | `UNAVAILABLE_BY_DESIGN` | MSP-E032 | `include/citlali/core/engine/detail/source_finding_execution_impl.h:9-66`; `include/citlali/core/engine/detail/pointing_fit_maps_impl.h:9-68` | Detection/catalog fitting and known-source pointing fitting reuse the Gaussian fitter family.  They are separate call paths, but type/name reuse creates a future conflation hazard.  POINT results must never authorize detection, catalog construction, universal photometry, or correction. |
| CTI-FM-015 | MAJOR | `MISSING_IMPLEMENTATION` | MSP-P016 | bounded search of CTI-S018--CTI-S038 | No runtime SCI-VAL object preserves request, applicability, eligibility, realization, action, causes, exact profile/source versions, and immutable reason.  Boolean flags and fit-valid counters are not a four-axis evaluation. |
| CTI-FM-016 | MODERATE | `IMPLEMENTED_LEGACY_SEMANTICS` | MSP-P011, MSP-P012 | `src/citlali/core/mapmaking/map.cpp:1378-1397,1495-1525` | Similar numerical moments/ratios exist under predecessor identities.  Without exact ensemble/parent/profile binding, consumers can confuse marginal second moment, standardized signal, dynamic range, fit S/N, precision, and significance. |

## Bounded negative-search result

Within CTI-S018--CTI-S038, no source object was found for a stable-original
MAP exposure coordinate/deduplication ledger, the frozen MAP r0.7.1 bundle,
the exact five-role JINC bundle, either frozen filter product, an immutable
matched-template authority, a registered POINT named-use decision, or a
runtime SCI-VAL four-axis evaluation.  This is bounded absence evidence, not a
claim about uninspected branches, worktrees, external systems, or future code.

## Safety conclusion

The exact source cannot safely advertise frozen map-space conformance.  The
highest-risk issue is not that products are wholly absent; it is that legacy
products with plausible names and numerical fields can be mistaken for the
frozen identities while applying prohibited defaults or transformations.
