# SCI-RTC v0.1/r0.3 scientific-owner decision ledger

Status: implementation-blind author register; every entry below is unresolved
unless superseded by later explicit scientific-owner approval.

This ledger records choices that the approved author packet deliberately does
not answer. It does not infer values from software, configuration, prior use,
or generic practice. The `Unavailable while open` column is normative package
consequence: an open choice blocks only the named operation or claim, not
unrelated raw RTC products.

## State vocabulary

- `OPEN`: v0.1 needs explicit owner authority before the dependent operation
  or claim can be applied.
- `CONDITIONAL`: the default path is already defined; this decision is needed
  only if the optional alternative is requested.
- `DEFERRED`: explicitly outside v0.1 and unavailable in this revision.

## Unresolved v0.1 choices

| ID | State | Exact scientific-owner decision required | Unavailable while open | Contract dependencies |
| --- | --- | --- | --- | --- |
| `SCI-RTC-OWNER-001` | OPEN | Name every v0.1 product role, if any, that is authorized to import CAL and produce `mJy/beam`, including target reference plane/domain and required CAL availability. Raw Beammap is already fixed as `Delta f/f`. | Any calibrated RTC product whose role is not explicitly authorized; raw Beammap remains available subject to its other inputs. | REQ-002--005, REQ-048 |
| `SCI-RTC-OWNER-002` | OPEN | Select the admitted fixed-mode integer factor set, exact realizable prefilter family, and role/downstream compatibility policy. Phase remains zero and point selection remains fixed. | Fixed downsampling beyond `M=1` for a role lacking this selected policy. | REQ-028--031 |
| `SCI-RTC-OWNER-003` | OPEN | Select the despike detector definition: statistic/estimator, support, threshold convention and value(s), boundary treatment, source/flag use, non-finite handling, version, and deterministic precision/tie semantics. | Despike target selection; pass-through RTC without despiking may remain available if explicitly selected. | REQ-012, REQ-017, REQ-026 |
| `SCI-RTC-OWNER-004` | OPEN | Select donor eligibility, detector topology, score, deterministic tie rule, same-time versus neighboring-time support, number of donors, weights/normalization, and reuse limits. | Donor replacement; spike detection may still produce a cause under an explicitly selected no-replacement failure policy. | REQ-017--018, REQ-042 |
| `SCI-RTC-OWNER-005` | OPEN | Select the no-donor/invalid-transfer fallback and whether any separately authorized donor-to-target transfer besides compatible `flxscale` is admitted, with direction, domain, support, uncertainty, validity, and provenance. | Replacement whenever the approved `flxscale_q/flxscale_d` route is unavailable; no zero, stale donor, unchanged spike, or legacy responsivity fallback is implied. | REQ-014--018, REQ-049 |
| `SCI-RTC-OWNER-006` | OPEN | For every coordinate-dependent role, select source catalog identity, frame/topology, geometry/radius or shape, inclusive/exclusive boundary, temporal dilation, detector binding, and interaction with despike/filter policy. | The affected source mask or coordinate-dependent response operation. Invalid coordinates still fail closed. | REQ-009, REQ-024--025 |
| `SCI-RTC-OWNER-007` | OPEN | Select the FIR/IIR/notch stage roster and chronological order, exact coefficient or design authority, coefficient convention, normalization, precision, sampling rate, direction, and stability acceptance. | Any filter stage lacking the selected policy; an explicitly selected identity/no-filter chain may remain available. | REQ-012, REQ-021--023, REQ-037--040 |
| `SCI-RTC-OWNER-008` | OPEN | Select filter initial/final state, reset/carry boundaries, context interval, boundary extension, output guard/edge rule, short/empty-scan disposition, and any finite IIR tail criterion. | Ordinary-valid output on incomplete context and every stateful stage lacking a complete state policy. | REQ-021, REQ-023, REQ-027, REQ-041 |
| `SCI-RTC-OWNER-009` | OPEN | Select non-finite rejection, unavailable-footprint, authorized prior replacement, reset/recovery, and failure behavior separately for samples, factors, coordinates, coefficients, and state. | Recovery from a required non-finite input; silent zero/coercion remains forbidden. | REQ-026--027, REQ-049 |
| `SCI-RTC-OWNER-010` | OPEN | Select the exact flag/cause aggregation and compact transitive-influence representation over full support, including required cause precedence only where RTC itself needs it. | Any aggregated flag product beyond the unambiguous retained causes; scientific ineligibility from synthesis/replacement influence remains fixed. | REQ-019--020, REQ-029, REQ-046--047 |
| `SCI-RTC-OWNER-011` | OPEN | Select learned-mode candidate integer factors and the admitted realizable filter family for each candidate, including coefficient precision and coefficient-production authority. | Learned resolution and learned apply. | REQ-032--035 |
| `SCI-RTC-OWNER-012` | OPEN | Define the smallest admitted beam used for learned safety: beam quantity/width convention, role and artifact identity, array/detector aggregation, uncertainty treatment, validity, and missing-beam policy. | Learned resolution and any learned safe-factor claim. | REQ-033--034 |
| `SCI-RTC-OWNER-013` | OPEN | Define maximum valid in-scan speed: coordinate/frame, scan set, support, derivative/estimator, treatment of turnarounds and gaps, precision, validity threshold, uncertainty, and insufficient-support policy. Percentiles remain diagnostic only. | Learned resolution and any learned safe-factor claim. | REQ-033--034 |
| `SCI-RTC-OWNER-014` | OPEN | Select the astronomical beam-times-realized-filter transfer metric, frequency/scan-crossing domain, tolerance value, uncertainty margin, and pass/fail boundary. | Learned astronomical-transfer admission. | REQ-033, EQ-021 |
| `SCI-RTC-OWNER-015` | OPEN | Select the phase-zero alias metric, input spectral envelope/band, every folded-image evaluation rule, attenuation bound, uncertainty margin, and pass/fail boundary. | Learned alias admission and any learned alias-safe claim. | REQ-030, REQ-033, EQ-014, EQ-021 |
| `SCI-RTC-OWNER-016` | OPEN | Select the sampling-sufficiency criterion for the smallest-beam/maximum-speed crossing, including the sample-location or samples-per-feature metric, numeric bound, and equality convention. | Learned sampling admission. | REQ-033, EQ-021 |
| `SCI-RTC-OWNER-017` | OPEN | Select filter-realizability limits: allowed length/order, coefficient quantization/precision, stability and normalization tolerances, computational constraints that have scientific effect, and failure boundary. | Learned filter-realizability admission. | REQ-021--023, REQ-033, EQ-021 |
| `SCI-RTC-OWNER-018` | OPEN | Name the required downstream consumers and select their cadence, grid, response, support, and serialization compatibility predicates for a common observation plan. | Learned downstream-compatibility admission. | REQ-033--035, REQ-052--053, EQ-021 |
| `SCI-RTC-OWNER-019` | OPEN | Select learned-request behavior when no candidate passes or required learning input is unavailable: fail the request or resolve native cadence under an explicitly named fallback, with distinct state and diagnostics. | Learned resolution and apply; no silent native fallback is authorized. | REQ-032, REQ-035 |
| `SCI-RTC-OWNER-020` | OPEN | Define the admitted observation/scan set and completeness rule used to learn the one common plan, including invalid scans, missing arrays, partial support, and when the plan is frozen. | Learned resolution and claim that the plan covers the observation. | REQ-032--036 |
| `SCI-RTC-OWNER-021` | OPEN | Select the required diagnostic roster by product role and classify each diagnostic as inert, advisory, or a declared selected-policy input, with estimator, unit, support, validity, and availability. | A role's required diagnostic completion and any policy that consumes an unclassified diagnostic. | REQ-048--051 |
| `SCI-RTC-OWNER-022` | OPEN | For each requested statistical product, select the admitted input mean/covariance or nuisance model, modeled correlations, supported subset, selection treatment, consumer exclusions, and whether only conditional weight is requested. | Covariance, total-uncertainty, significance, or weight claims requiring the missing model; conditioned TOD may remain available. | REQ-042--045 |
| `SCI-RTC-OWNER-023` | OPEN | Select the exact required-output member roster and atomic completion policy for each product role, including which response, uncertainty, provenance, and diagnostics are required versus optional. | A complete role-specific bundle claim when the roster is not otherwise fixed by the core minimum. | REQ-048--051 |
| `SCI-RTC-OWNER-024` | CONDITIONAL | If an alternative to calibration-before-replacement is requested, approve the complete affine equivalence evidence and its validity domain; otherwise retain the fixed default order. | Only the proposed alternative order; the default calibration-before-replacement path is unaffected. | REQ-013, EQ-004 |

## Deferred successor choices

| ID | State | Deferred scope | V0.1 consequence |
| --- | --- | --- | --- |
| `SCI-RTC-OWNER-025` | DEFERRED | Noise-aware learned optimization. | Unavailable; noise may be diagnostic but cannot select the v0.1 plan. |
| `SCI-RTC-OWNER-026` | DEFERRED | Per-array or per-scan learned factors/filters and heterogeneous downstream time grids. | Unavailable; one common immutable observation plan is required. |
| `SCI-RTC-OWNER-027` | DEFERRED | Continuous or within-apply adaptation. | Unavailable; apply consumes one frozen resolved plan. |
| `SCI-RTC-OWNER-028` | DEFERRED | Enabled polarimetry or measured R-channel conditioning. | Unavailable; v0.1 is Stokes-I primary `xs`. |

## Additional r0.2 learn--resolve--apply choices

| ID | State | Exact scientific-owner decision required | Unavailable while open | Contract dependencies |
| --- | --- | --- | --- | --- |
| `SCI-RTC-OWNER-029` | OPEN | For every operation that may learn a threshold, mask, donor grouping, line, filter, or sampling parameter, name the learning population, exclusions, completeness rule, freeze point, quality/uncertainty estimator, and same-observation data-dependence treatment. | The affected learned operation and any claim that its plan is resolved. | DEF-021--024, REQ-032, REQ-055--057 |
| `SCI-RTC-OWNER-030` | OPEN | Select each admitted notch purpose and applicability plus center/width/depth/Q or pole/order constraints, false-detection control, uncertainty, line-drift tolerance, phase/direction, overlap rule, state, guard, and astronomical-transfer tolerance. | The affected fixed or learned notch and any preserved-response claim. | REQ-021--023, REQ-061, REQ-069 |
| `SCI-RTC-OWNER-031` | OPEN | Name which v0.1 roles, if any, admit low-pass, high-pass, or band-pass operations and select their pass/transition/stop, ripple, attenuation, phase, retained-mode, source-class, and edge requirements. | Any broad-band filter branch lacking a role-specific design specification. | REQ-059--062, REQ-069 |
| `SCI-RTC-OWNER-032` | OPEN | Select the scan-speed/direction, projected-beam, source-extent/class, prior-response, and cadence domain plus exact metric, tolerance, uncertainty margin, and equality convention for astronomical filter admission. | Any filter-preservation or science-band claim lacking this domain. | EQ-023, REQ-060, REQ-066, REQ-069 |
| `SCI-RTC-OWNER-033` | OPEN | Select FIR family, order/tap design objective, pass/stop metrics, phase and delay limits, alias bound, maximum support/guard, coefficient precision, realizability limit, and deterministic tie rule among passing designs. | Selection of FIR taps/order and any claim that the selected length is scientifically justified. | EQ-025--026, REQ-063, REQ-069 |
| `SCI-RTC-OWNER-034` | OPEN | By product role, select zero-phase, shifted-time, coordinate-corrected, response-aware, or bounded-negligible temporal registration, including centroid/PSF metric, tolerance, scan domain, and AST handoff. | Temporal-centering and scan-direction coordinate/centroid claims for a delayed response. | EQ-026--027, REQ-040, REQ-067 |
| `SCI-RTC-OWNER-035` | OPEN | Define complete RTC-plan compatibility for transferring Beammap calibration to a target and select the required response calculation, renormalization, or observational evidence when plans differ. | Cross-plan point-source, PSF, or photometric calibration transfer beyond the exact matched-plan domain. | REQ-005, REQ-013, REQ-068 |
| `SCI-RTC-OWNER-036` | OPEN | Select preregistered observational-performance and science-impact study domains, metrics, tolerances, uncertainty treatment, and acceptance authority for notch, broad-band/FIR, donor, and decimation design sweeps. | Observational-performance and science-impact qualification; algebraic contract review remains separately assessable. | REQ-054, REQ-069--070, PRED-030--038 |

## Additional r0.3 bounded-refinement choices

| ID | State | Exact scientific-owner decision required | Unavailable while open | Contract dependencies |
| --- | --- | --- | --- | --- |
| `SCI-RTC-OWNER-037` | OPEN | Select the maximum number of refinement cycles and the equality convention at the bound. | Iterative notch-plan refinement. | DEF-027/029, REQ-071/080 |
| `SCI-RTC-OWNER-038` | OPEN | Select the number of consecutive complete-plan comparisons required for stability and whether an unchanged rejected cycle counts. | Plan-stability or convergence disposition. | REQ-079--080 |
| `SCI-RTC-OWNER-039` | OPEN | Select candidate-line significance, uncertainty treatment, multiple-candidate/false-line control, and response-floor exclusion. | Successor-line admission. | REQ-077--078 |
| `SCI-RTC-OWNER-040` | OPEN | Select the line-identity and matching rule across cycles, including split, merge, interval, detector, and frequency-convention cases. | Complete plan-stability comparison. | REQ-079 |
| `SCI-RTC-OWNER-041` | OPEN | Select the minimum interference-metric improvement and equality convention required to continue or admit a successor. | Successor admission and improvement-based stopping. | REQ-078/080 |
| `SCI-RTC-OWNER-042` | OPEN | Select the maximum number of admitted notches, including how overlapping or detector/interval-specific notches count. | Refinement beyond the admitted notch-count budget. | REQ-071/078 |
| `SCI-RTC-OWNER-043` | OPEN | Select maximum cumulative removed or strongly attenuated bandwidth, metric, frequency convention, and overlap accounting. | Successor admission beyond the bandwidth budget. | REQ-078 |
| `SCI-RTC-OWNER-044` | OPEN | Select the cumulative astronomical-transfer budget over the admitted scan/beam/source domain, including point-source peak, extended response, and uncertainty margin. | Successor admission and preserved-transfer claim. | REQ-076/078 |
| `SCI-RTC-OWNER-045` | OPEN | Select cumulative phase, group-delay, centroid, PSF, and coordinate-registration budgets by product role. | Successor admission and temporal/coordinate validity. | REQ-076/078 |
| `SCI-RTC-OWNER-046` | OPEN | Select cumulative ringing and transient metrics, support, tolerances, and equality convention. | Successor admission and ordinary-valid interior after transients. | REQ-076/078 |
| `SCI-RTC-OWNER-047` | OPEN | Select cumulative edge/guard-loss metric, usable-support floor, interval aggregation, and failure boundary. | Successor admission and usable-support claim. | REQ-076/078 |
| `SCI-RTC-OWNER-048` | OPEN | Select when held-out or otherwise independent scans are required and what same-data assessment may support. | Independent confirmation claim; same-data consistency remains explicitly data-dependent. | REQ-076--077 |
| `SCI-RTC-OWNER-049` | OPEN | Select original-input replay as the universal default or name bounded roles that authorize sequential cascade, with parent and equivalence requirements. | Any non-replay final product or cascade claim. | EQ-029, REQ-074--075 |
| `SCI-RTC-OWNER-050` | OPEN | Select fallback after a rejected successor: retain the last accepted plan by default or fail/cancel under named causes. | Final-plan selection for a rejected successor cause not covered by policy. | DEF-029, REQ-080 |

## Exact unavailable-state rule

An unresolved entry must be referenced by ID in the observation-resolved
state when it blocks an operation. It must not be replaced by a numerical
sentinel, a software default, a value recovered from prior products, or a
claim that the unavailable operation was disabled after being requested.
Resolution requires explicit scientific-owner approval and a successor ledger
revision; this author draft does not provide that approval.
