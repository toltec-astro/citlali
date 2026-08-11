# RTC learned-sampling Stage A successor metrics

Date: 2026-08-11

Status: successor repair 2 locally validated completion candidate; independent
re-audit has not been launched

## Authority and scope

This narrow SRA-001 through SRA-009 repair is implemented on
`codex/repair-rtc-learned-sampling-stage-a-successor-2` from exact application
base `66c96757164af2c83ee1449d00fea30d131a7e3f`. Its frozen successor repair
authority is coordination commit
`3fe0aa30eaa0d8848dbb39eb720457326c0b43ba`; all unaffected predecessor
scientific authorities and positive controls remain binding.

The product remains observe-only. It enumerates, characterizes, and reports
every admitted factor without ranking, recommending, selecting, or applying
one. RTC/PTC/mapmaking inputs, flags, samples, timestamps, grids, weights,
maps, and all science products other than the formative `rtcdiag` diagnostic
are unchanged.

## Source motion and cadence authority

Source-telescope motion is captured from authoritative raw telescope rows
before detector-grid interpolation. The reduction-observation input owner
resets the typed `TimestreamAlignmentState` carrier before loading an
observation, captures it immediately after that observation's telescope rows
are loaded, and binds observation index, obsnum, and telescope path. The
`rtcdiag` writer consumes the carrier only when all three identities still
match. It is never read by RTC, PTC, or mapmaking.

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
Realized native cadence is measured from exact `TelTime` differences; missing,
non-finite, non-positive, or irregular grids fail dependent metrics with a
cause-specific reason. Valid Stage A calculations use realized cadence even
when requested-to-effective or effective-to-realized consistency is
`mismatch`; both consistency results remain separately persisted. Exact
realized FIR coefficients come from the RTC filter object, or the explicit
identity vector `[1]` when filtering is disabled.

Citlali remains total-intensity only. Supported diagnostics use an explicit
`total_intensity` analysis mode and do not consult `calib.run_hwpr`, serialized
legacy state, or HWPR-file presence. The existing capability gate rejects an
enabled polarimetry request before scientific execution. If the diagnostic API
is explicitly given `hwpr_dependent`, it reports `unsupported_hwpr` and emits
no candidate dimension or candidate rows; no HWPR lifecycle or transfer model
is invented.

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
8,000,000 total candidate rows, 500,000,000 actual work units, and 536,870,912
estimated `rtcdiag` bytes, using checked add/multiply throughout. With
`Q = 257`, factor `M`, realized FIR tap count `L`, production-valid detector
count `D`, and native sample count `N`, the charged work is

```text
numerical(M=1) = L
numerical(M>=2) = L [ M(6Q+1) + 2Q+4 ]
context(M) = D [ floor((N-1)/M)+1 ] L
actual = sum over every admitted scan-array-factor of numerical + context.
```

Serialized storage charges every rectangular candidate cell at exactly 40
int32 plus 43 binary64 fields (`504 bytes`) and adds every realized FIR vector
at `8L` bytes. Every scan-array retains its full derived `Mmax` and its own
range status. A pair above the 8192 limit is isolated while unaffected pairs
remain evaluable when the rectangular product fits. Overflow, global row,
actual-work, or storage rejection occurs before partial evaluation and emits
explicit unavailability with no candidate dimension or factor-1 pseudo-table;
the range is never silently truncated.

Finite-scan applicability uses the exact realized FIR tap count and centered
left/right context, factor, phase zero, boundaries, motion eligibility, exact
per-detector validity, finite values, science flags, and the separately
captured realized filter-guard mask. Input construction assigns each
detector-time cell one category. Deterministic precedence is motion gap,
low-velocity, invalid/over-limit, per-detector invalid, realized filter guard,
residual/pre-guard science flag, non-finite input, then fully supported. The
science category is explicitly `flag && !guard`, so a guard cannot be swallowed
by the flag it produced. Candidate detector-output category counts are mutually
exclusive, sum exactly to the detector-output-cell total, and require
`unclassified == 0`.

Temporal `N_full` counts an output when at least one production-valid detector
has complete context; detector-output counts retain exact per-detector
validity. `N_full == 0` remains the sole hard Stage A boundary and yields
`candidate_unusable_no_complete_context`. One full output is mathematical
evaluability only, with no adequacy threshold. The separately reported actually
applied RTC operator retains its prior status semantics; Stage A does not
enforce the diagnostic result.

## `rtcdiag` v3 successor contract and publication

`RTC_DIAG_SCHEMA_VERSION` is `rtcdiag-v3`, algorithm identity is
`rtc-learned-sampling-stage-a-v3`, embedded generic product declaration is
`sci-rtc-001-stage-a-successor-products-v1`, point validation-profile ID is
`sci-rtc-001-stage-a-successor-v1`, and contract epoch is
`sci-rtc-001-stage-a-successor-2026-08-11`. The profile remains `preparing`.
The OOF, Beammap, and science profiles use deterministic mode suffixes. Four
thin `sci-rtc-001-stage-a-successor-{mode}-products-v1` wrappers inherit the
matching established mode contract and override only its native RTC entry:
`rtc-diagnostics` for point/OOF, `source-crossing-rtc-diagnostics` for
Beammap, and `observation-rtc-diagnostics` for science. Every override changes
only `check_id` to `rtc_diagnostics_stage_a_successor_v3`; file scope, pattern,
classification, cardinality, and every non-rtcdiag entry remain unchanged.

The old maximum-speed/APT-projected-beam,
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
- exact detector-output complete-context counts, mutually exclusive exclusion
  categories, and temporal counts/fractions.

Candidate availability is an executable conditional contract. Available (`1`)
requires the candidate dimension, declared count equality, and all 81
candidate arrays with that exact trailing dimension. Unavailable (`0`) forbids
the dimension and every candidate array. Integer scalar conditions are
normalized before comparison. Both malformed available and residual-row
unavailable products are adversarial validator fixtures. The contract also
requires the exact 40-lowercase-hex `RTC_SAMPLING_CITLALI_COMMIT`, rejects a
dirty production build identity, and embeds the canonical raw-input-manifest
bytes, SHA-256, and reference into the self-contained product.

The output owner reserves a unique adjacent
`.citlali-stage.<pid>.<sequence>` file with `O_EXCL` mode `0600`. All setup,
scan rows, candidate tables, identity, and manifest provenance are written to
that staging path; RTC diagnostic append refuses any published path. The
finalizer reopens and validates the complete artifact, syncs, closes, verifies
the regular file, and performs one same-directory atomic rename without
pre-deleting the prior final. Create, write, scan append, provenance,
validation, sync, close, and rename failures propagate, clean only the
recognized task staging artifact, advertise no partial generation, and
preserve any prior good final. Nothing appends after publication.

Metric values are non-authoritative unless their own domain-specific status
is available or converged. The frozen vocabulary preserves prerequisite
available/unavailable, candidate-not-evaluated prerequisite,
candidate-range/table availability, plan-transfer available/unavailable,
applied-operator not-applicable, numerical
converged/bounded-not-converged/failed, and exact
`not_applicable_no_decimation` distinctions. The product notice explicitly
says no factor was ranked, recommended, selected, or applied.

## Completion-candidate evidence and exclusions

Focused evidence covers independent source-motion interval lookup,
same-observation identity reset/match, explicit total-intensity/HWPR diagnostic
routing, realized-cadence measurement, checked actual-work and storage
formulas, detector-level context precedence and exact sums, full source
identity, unique adjacent staging, prior-good preservation, publication
ordering, and executable candidate-table cardinality. The contract fixture
derives its malformed cases independently of the production writer.

The production CLI and every registered test target compile. The local build
uses an unmodified Ceres 2.1 source snapshot, matching Citlali/kidscpp's legacy
parameterization API, and the installed macOS 13.3 SDK to avoid newer-library
format-overload drift; these are build inputs only and make no source or
product change. All 653 enabled CTests pass; the pre-existing
`MapFitterLifecycle.ExactProductSequence` remains the sole disabled test. The
baseline-tool suite passes 177/177, including 38/38 focused product-contract
and validation-profile tests. The complete config preflight passes its 127
Python tests, four mode kits, eight compact-compatibility cases, 100% compact
surface coverage, all authority/boundary audits, and the unchanged 45-record
raw-execution census at digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`.
The 60-record validation ledger, three-change/five-integration-commit intended
science ledger, four-mode preparing profile registry, and session-exit audit
(`library_exits=0`, `cli_exits=0`, `growth=0`) also pass. No skipped test is
counted as successful evidence.

No local science reduction, Unity work, push, merge, re-audit, downstream
launch, or production authorization occurred.

The checked-in `candidate_metrics.csv` is a compact deterministic
output-format illustration. It is not observation evidence, a numerical gate,
or a factor recommendation.
