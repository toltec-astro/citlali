# RTC learned-sampling Stage A successor metrics

Date: 2026-08-10

Status: bounded completion candidate; deterministic local gates pass;
independent re-audit has not been launched

## Authority and scope

This successor is implemented on
`codex/repair-rtc-learned-sampling-stage-a-successor` from exact application
base `6cbe119a59f8915c5aecf5eaf333425dd592993d`. Its frozen repair authority is
handoff `SCI-RTC-001-LEARNED-SAMPLING-STAGE-A-REPAIR-READY-006` at
coordination commit `3132d5d8c001ef32f185d4ece2038aa6d7ce1b5c`, plus the
2026-08-10 owner clarification recorded for the beam-transfer and coherent
tone conventions.

The product remains observe-only. It enumerates, characterizes, and reports
every admitted factor without ranking, recommending, selecting, or applying
one. RTC/PTC/mapmaking inputs, flags, samples, timestamps, grids, weights,
maps, and all science products other than the formative `rtcdiag` diagnostic
are unchanged.

## Source motion and cadence authority

Source-telescope motion is captured from authoritative raw telescope rows
before detector-grid interpolation. A typed observation-scoped carrier in
`TimestreamAlignmentState` owns only interval support needed by `rtcdiag`; it
is replaced/reset for every observation and is never read by RTC, PTC, or
mapmaking.

The non-HWPR statistic is the empirical `v95` of valid eligible tangent-plane
source-telescope speed magnitudes. `p99`, `p99.5`, and the raw maximum are
diagnostics. Intervals above `3600 arcsec/s`, gaps above `0.1 s`, non-finite or
non-positive intervals, and tangent-plane pointing steps above `0.01 rad` are
rejected. These guards are versioned as
`native-source-row-gap-jump-and-one-row-scan-boundary-v1`. Source columns must
have equal lengths; unequal lengths fail closed without truncation. Valid
intervals at or above `1.0 arcsec/s` are eligible; exactly `1.0 arcsec/s` is
included. If none remains, dependent metrics use
`unavailable_low_velocity`. Counts, durations, exclusions, coverage, and the
pre-interpolation coordinate identity are persisted. A compact interval
ledger records start/stop boundary identity, duration, speed, validity,
eligibility, and stable cause-specific reason; it does not embed raw telescope
samples.

Scan support is assigned by native source-row timestamps, not interval
midpoints. For each scan, the first source row at or after the scan start and
the last row at or before the scan stop are found, exactly one native row is
excluded at each end, and only intervals whose two endpoint rows lie inside
the remaining inclusive row range contribute. Partial overlaps, boundary
exclusions, and their durations are persisted separately. Consecutive scans
apply this rule independently.

Requested, effective, and realized cadence and filter states are distinct.
Actual `telescope.fsmp` is the realized native cadence; the raw plan owns the
requested/effective cadence and filter configuration. The writer records the
consistency result. Exact realized FIR coefficients come from the RTC filter
object, or the explicit identity vector `[1]` when filtering is disabled.

Requested, effective, ignored, and realized hardware-presence HWPR facts are
also separate. Physical HWPR data do not imply scientific enablement. An
effectively enabled HWPR observation retains only the factor-1 reference row;
dependent astronomical metrics are unavailable with
`unavailable_hwpr_sampling_contract`.

## Beam and transfer identity

The per-array beam-size authority is the fixed diffraction-derived Airy
**intensity FWHM** `1.028 lambda / 50 m`: `4.66`, `5.94`, and `8.48 arcsec`
for `a1100`, `a1400`, and `a2000`. These values only set the scale of the
owner-selected circular-Gaussian temporal intensity model. The transfer shape
is not an Airy-profile transform and no APT or measured-beam fallback exists.
Unknown arrays are unavailable.

For FWHM `theta` and eligible motion statistic `v95`, the persisted model is

```text
sigma_t = theta / (2 sqrt(2 ln(2)) v95)
B(f) = exp(-2 pi^2 sigma_t^2 f^2)
B(0) = 1; phase(B) = 0; beam power = |B|^2.
```

For factor `M`, the phase-zero time-domain unit-complex-tone response folds
exactly `M` unique periodic images over an explicit half-open interval. Every
input tone has unit amplitude; there is no artificial `1/M` tone factor. Each
image uses `B(f)` and the exact centered complex response of the realized RTC
FIR. Amplitude, phase, power, and complex distortion are relative to the
corresponding unaliased baseband. Factor 1 alias is exactly zero and its
stopband status is `not_applicable_no_decimation`.

The counterfactual is always `(M, phase=0, H_RTC_realized)`. Stage A neither
synthesizes a candidate FIR nor infers candidate count from a configured
filter edge. For each admitted scan-array pair:

```text
N_beam(M) = theta_FWHM fs / (M v95)
Mmax = floor(theta_FWHM fs / v95)
candidates = {1, ..., max(1, Mmax)}.
```

The coefficient vector, SHA-256 digest, requested/effective/realized filter
parameters, factor/phase binding, and identity-vector convention are
persisted. The digest is SHA-256 over an unsigned 64-bit little-endian
coefficient count followed by every realized IEEE-754 binary64 coefficient in
little-endian realized order. Its convention is
`sha256-u64le-count-then-ieee754-binary64le-realized-order-v1`.

## Bounded characterization and applicability

Alias, relative-response, distortion, and stopband maxima use the
deterministic `uniform-partition-global-lipschitz-enclosure-v1` method with 256
partitions. The sampled maximum is a lower bound; an analytical derivative
bound times the half-cell radius produces a conservative upper enclosure.
The alias/relative-response domain is half-open
`[-fs/(2M), fs/(2M))`, implemented with `nextafter` at the upper endpoint; the
FIR stopband is closed `[fs/(2M), fs/2]`. The file records method, domain,
partitions, analytic Lipschitz identity and per-metric value, evaluation
counts, coefficient identity, lower/upper bounds, error enclosure, and
independent per-metric status/reason. A zero enclosure is
`numerical_converged`; a finite nonzero rigorous enclosure is
`numerical_bounded_not_converged`, not an exact, worst-case, or
tolerance-converged claim; evaluation failure is `numerical_failed`.

Resource preflight limits the implementation to 8192 factors per scan-array,
8,000,000 total candidate rows, 50,000,000 estimated complex evaluations, and
536,870,912 estimated `rtcdiag` bytes, using checked arithmetic. Every
scan-array retains its full derived `Mmax` and its own range status. A pair
above the 8192 limit is isolated while unaffected pairs remain evaluable when
the rectangular product fits. If a global row, evaluation, or byte guard
prevents the table, explicit product/range unavailability is emitted with no
candidate dimension or factor-1 pseudo-table; the range is never silently
truncated.

Finite-scan applicability uses the exact realized FIR tap count and centered
left/right context, factor, phase zero, outer and science-scan boundaries,
eligible assigned-grid support, and internal gaps. It reports candidate
outputs, fully supported outputs, boundary/gap/other incomplete counts,
longest run, duration, and fraction. `N_full == 0` is the sole hard Stage A
boundary and yields `candidate_unusable_no_complete_context`. One full output
means mathematical evaluability only. The separately reported actually
applied RTC operator uses `scan_unusable_for_applied_rtc_operator` with
`no_complete_context` at zero support; Stage A does not enforce that result.

## `rtcdiag` v2 semantic breaks

`RTC_DIAG_SCHEMA_VERSION` is now `rtcdiag-v2`, and algorithm identity is
`rtc-learned-sampling-stage-a-v2`. The old maximum-speed/APT-projected-beam,
filter-edge candidate enumeration, attenuation/broadening, incoherent
alias-power, transition-margin, and software-delay Stage A fields are removed
without compatibility aliases. They are replaced by:

- raw pre-interpolation motion support and `v95`/`p99`/`p99.5`/maximum;
- fixed per-array FWHM plus explicit Gaussian-model and FWHM-authority IDs;
- exact factor range, phase-zero unit-tone convention, FIR vector/digest, and
  requested/effective/realized cadence, filter, and HWPR facts;
- candidate, plan-transfer, applied-operator, alias, and stopband
  status/reason fields using the persisted stable numeric vocabulary;
- bounded coherent amplitude/phase/power/distortion, alias, and FIR stopband
  fields with numerical evidence; and
- exact complete-context counts and fractions.

Metric values are non-authoritative unless their own domain-specific status
is available or converged. The frozen vocabulary preserves prerequisite
available/unavailable, candidate-not-evaluated prerequisite,
candidate-range/table availability, plan-transfer available/unavailable,
applied-operator not-applicable, numerical
converged/bounded-not-converged/failed, and exact
`not_applicable_no_decimation` distinctions. The product notice explicitly
says no factor was ranked, recommended, selected, or applied.

## Completion-candidate evidence and exclusions

The dedicated translation unit and production CLI compile, and all 26 focused
tests pass. They cover unequal-column failure,
negative azimuth wrap, exact native-row boundary/partial/consecutive-scan
support, gap and low-velocity identity, empirical percentiles, fixed beam
authority, Gaussian normalization, coherent odd/even folding without `1/M`,
factor-1 identity, factor-range independence from filter edges, analytic,
narrow-extremum, long-FIR, broad-valid, singular and per-metric numerical
fixtures, the independent byte-level coefficient digest, checked resource
overflow/row/evaluation/byte/isolation/no-truncation behavior, exact complete
context, requested/effective/realized HWPR and cadence matrices, file-presence
negative control, observation reset and A/B state sentinels, atomic
create/write/sync/rename cleanup, and production helper writer/reopen joins.

The local build cache was configured with the installed Homebrew OpenMP header
and library, without a source-tree or product change. The full CTest inventory
passes all 649 enabled tests; the pre-existing
`MapFitterLifecycle.ExactProductSequence` is the sole disabled test. The
baseline-tool suite passes 172/172 tests. The complete config preflight passes,
including 127 Python tests, four mode kits, eight compact-compatibility cases,
100% compact-surface coverage, all authority/boundary audits, and the unchanged
45-record raw-execution census at digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`.
The 60-record validation ledger and validation-profile registry validate; the
Phase 5 readiness report executes successfully and truthfully remains
`preparing`/not promotion-ready because its documented external promotion
evidence is still absent.

Gate-driven corrections remained inside the approved paths: generic
pre-interpolation source columns now normalize cardinalities to `size_t` before
comparison; the unchanged product-registry identity remains v2 while the
`rtcdiag` product alone advances to v2; and requested/effective/realized filter
facts plus exact realized FIR coefficients are captured once by the rtcdiag
output owner into a typed adapter before numerical consumption. No local
science reduction, Unity work, push, merge, re-audit, or downstream launch
occurred.

The checked-in `candidate_metrics.csv` is a compact deterministic
output-format illustration. It is not observation evidence, a numerical gate,
or a factor recommendation.
