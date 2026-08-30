# WP-7 RTC Scan/Array Planning Authority Crosswalk

Date: 2026-08-29

Controlling decision:
[WP-7 RTC Scan/Array Planning Scientific-Owner Authority](WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)

Controlling numerical policy:
[`wp7-rtc-scan-array-numerical-policy-v1`](WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md),
approved by the scientific owner on 2026-08-30

Source authority is the frozen SCI-RTC v0.1/r0.12 package bound by
`validation/wp7_timestream_successor_authority.json`, plus the accepted
network-timing correction. Frozen text remains historical and is not silently
edited. This crosswalk states only the bounded successor dispositions.

## Supersession map

| Frozen entry or topic | Bounded successor disposition |
| --- | --- |
| OWNER-002 fixed factor/filter mode | The accepted `M=1` identity mode remains. A single fixed `M=2` witness independent of scan and array is not the next scientific plan. Nonidentity factor/filter selection is deterministic per scan and array under the approved numerical policy. |
| OWNER-011 candidate factors and filter families | Resolved to every integer `M` in `[1, 256]` and the one approved centered symmetric Type-I Kaiser-windowed-sinc FIR family with the exact construction and smallest-odd-tap tie rule. |
| OWNER-012 beam aggregation/model | Resolved to the approved 272/214/150 GHz array frequencies, exact 50.0 m clear circular aperture, normalized unobscured Airy intensity profile, and coefficient `1.028993969962188`. |
| OWNER-013 scan speed statistic | Resolved to scalar valid realized on-sky speed and the actual maximum over admitted science occurrences. The exact admission boundary is `v >= 1 arcsec/s`; no percentile is permitted. |
| OWNER-014 passband criterion | Resolved to full Airy temporal support at `v_plan`, independent `1e-3` peak/flux/shape/FWHM/centroid/calibration limits, `1e-4` passband ripple, centered zero phase, and `1e-12` DC-gain error. |
| OWNER-015 alias criterion | Resolved to a worst-retained-frequency total folded-alias power-gain bound of `1e-6`, with the exact image-sum norm and per-image design allocation in the numerical policy. |
| OWNER-016 beam sampling | Resolved to at least four output samples per approved Airy intensity FWHM. |
| OWNER-017 automatic realization | Resolved to the approved Kaiser family, smallest certified odd tap count, five-second maximum half-support, and declared binary64 construction and ordered accumulation. Prototype tap estimates are not coefficient authority. |
| OWNER-018 output cadence | Superseded where it required one ordinary observation-wide cadence. Arrays may differ by scan; every output remains network-timed. A common analysis cadence requires a named synchronous consumer under ADR 0015. |
| OWNER-019 fallback/failure | If no `M > 1` passes, use factor `M=1` without sampling change only when the input cadence still meets the science-band and four-samples-per-FWHM requirements. Otherwise publish no admitted ordinary astronomical product with `input_cadence_inadequate_for_science_band`. This does not alter the separate accepted identity conformance route. If a scan has no admitted run at or above `1 arcsec/s`, it has no admitted ordinary astronomical timestream product. No science-band relaxation is permitted. |
| OWNER-020 plan stability | The plan is immutable per scan, array, and exact cadence before Apply. It is invariant to detector values and engineering chunks; an observation-common plan is not required. |
| OWNER-026 deferred per-array/per-scan factors and heterogeneous grids | Explicitly reopened and superseded. Per-scan/per-array filters and factors are the ordinary model. Their products retain independent network axes; heterogeneous cadences do not imply a common grid. |
| OWNER-029 planning population | Planning consumes only admitted AST-valid science-scan trajectory occurrences and immutable array/cadence/policy facts. Detector timestream values are excluded. |
| OWNER-031--034 response, beam, and passband policy | Resolved by `wp7-rtc-scan-array-numerical-policy-v1`, including full ideal-aperture temporal support, exact product/response bounds, and the 5% velocity plus 100 ppm cadence margins. |
| OWNER-036 observational qualification | Unchanged and open. Structural approval does not establish observational performance or production readiness. |
| OWNER-052 low-speed and paired validity consequences | `v < 1 arcsec/s` is a pair-wide astronomical occurrence exclusion with typed cause `below_minimum_science_scan_speed`. It does not erase or merge member-local producer validity and causes. Distinct invalid AST/telemetry facts retain distinct causes. |

## Preserved authority

- Network-specific timing and explicit common-analysis-grid placement remain
  controlled by the 2026-08-29 timing correction and ADR 0015.
- Paired `x/r` identity, independent member validity and local causes,
  conservative pair-wide action, complete support, immutable lifecycle
  binding, chunk invariance, and compact realization remain authoritative.
- `M=1` exact identity behavior and its accepted implementation/evidence are
  unchanged.
- Unlisted SCI-RTC requirements, equations, predictions, and owner-ledger
  entries retain their prior state.

## Implementation obligations under the closed numerical policy

| Owner consequence | Implementation obligation |
| --- | --- |
| Slow motion is inadmissible | Preserve the raw producer facts, add the exact typed RTC exclusion, split admitted runs, and prohibit filter influence across the boundary. |
| AST owns trajectory validity | Consume bounded immutable AST facts; invalid velocity spikes fail or remain excluded before the maximum is formed. |
| Beam authority is fixed | Bind a versioned immutable array-model artifact; never substitute fitted or observation-local beams. |
| Planning is deterministic | Resolve the complete scan/array plan before Apply and make it independent of chunking and detector values. |
| Largest conforming factor wins | Evaluate the complete approved factor set with exact passband, phase, alias, sampling, support, and paired-operator tests; fall back only to `M=1`. |
| Arrays may differ | Bind output occurrence/time/support separately to each source network; do not introduce common-grid dependencies. |
| Boundary support is scientific | Exclude any output lacking complete approved support and retain the exact transitive support relation and cause. |

## Still unavailable

Numerical authority is closed, but no nonidentity implementation is authorized
by this crosswalk alone. Conforming AST science-scan membership, scalar
velocity, derivative validity and cause, telemetry-defect disposition, and
actual-maximum authority must be present before the first real plan can be
formed. Exact coefficient construction/certification, implementation,
representative-data evidence, and fresh exact-SHA review also remain pending.
The feasibility sweep's tap counts and nominal observation-152390 header speed
are not promoted.
