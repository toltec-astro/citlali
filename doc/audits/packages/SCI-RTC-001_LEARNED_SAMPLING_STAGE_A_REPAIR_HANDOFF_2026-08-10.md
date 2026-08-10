# SCI-RTC-001 learned-sampling Stage A repair handoff

Date: 2026-08-10

Handoff ID: `SCI-RTC-001-LEARNED-SAMPLING-STAGE-A-REPAIR-READY-006`

Status: owner decisions consolidated; documentation-only repair handoff
prepared; repair not authorized or launched

## Exact authority and proposed repair identity

- Coordination drafting base: `codex/coordinate-rtc-ptc-queue` at
  `c078179df5916c54b6ab0ee3789fcde925b43d87` (parent
  `6136f65ca81956cec5ceb9866303580409182257`, tree
  `5ed3283ec85fb8a4b5c8c712bc2f925fb651b9c2`).
- Returned Stage A application candidate:
  `6cbe119a59f8915c5aecf5eaf333425dd592993d` (parent
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
  `153a327445f1eac03db7220478bc4cd44cd93ac2`).
- Candidate patch SHA-256:
  `f640bc8627317f46be76c85873fabf76858ebe7d999f0665e833c5e751b64e04`.
- Independent return-for-repair audit:
  `5ecd3c4b57cdd24c9fc263a054106516b2781d71` (parent exact candidate
  `6cbe119a59f8915c5aecf5eaf333425dd592993d`, tree
  `5ec42039e0e93f824671be8c0808d0fcf5d797ad`).
- Audit report object:
  `5ecd3c4b57cdd24c9fc263a054106516b2781d71:handoff/RTC_LEARNED_SAMPLING_STAGE_A_REAUDIT_2026-08-10.md`,
  SHA-256
  `a6af5d6a14850283cdda420b3d99831b33378dda0c67244f589517ff1caa53c3`.
- Audit evidence object:
  `5ecd3c4b57cdd24c9fc263a054106516b2781d71:handoff/RTC_LEARNED_SAMPLING_STAGE_A_REAUDIT_EVIDENCE_2026-08-10.md`,
  SHA-256
  `0bb678240d78c26315a213eccd54e4812bf2d07f985a5ffb3fb237ed7170a6bf`.
- Audit findings object:
  `5ecd3c4b57cdd24c9fc263a054106516b2781d71:handoff/RTC_LEARNED_SAMPLING_STAGE_A_REAUDIT_FINDINGS_2026-08-10.csv`,
  SHA-256
  `d71e70fbc4a1b7f60b001789c888229bfc8ff24dc5651e1add57f1e1a0dab2fb`.
- Prior owner-approved design object:
  `cbb676d84bc58da4239a906a420a04a326968309` (parent coordination commit
  `c078179df5916c54b6ab0ee3789fcde925b43d87`, tree
  `168f7f28656ecc31e507f603b99160bc1e1a5a59`).
- Owner-approved `RTC-A-008` resolution source supplied on 2026-08-10:
  `/Users/gwilson/.codex/attachments/74b0b74f-8a66-425a-b720-b19bf3fee35a/pasted-text.txt`,
  SHA-256
  `3d46bc8771e1f28fa658db82de980a26433755a9852e9b3696b4cc5739b0f8cf`.
- Documentation-only successor finding ledger:
  `doc/audits/proposals/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_REPAIR_FINDING_LEDGER_2026-08-10.yaml`.

The exact proposed repair base is the returned application candidate
`6cbe119a59f8915c5aecf5eaf333425dd592993d`. The proposed successor branch is
`codex/repair-rtc-learned-sampling-stage-a-successor`, with a fresh worktree at
`/private/tmp/citlali-repair-rtc-learned-sampling-stage-a-successor`.
Selecting these proposed identities does not accept, integrate, merge, or
launch the candidate or repair.

## Supersession boundary

The owner decisions in this handoff replace the conflicting Stage A portions
of the design at `cbb676d8...`, including the prior APT/elliptical-beam
authority, maximum-speed primary statistic, configured-filter-edge candidate
range, and incoherent alias quantity. The prior design and audit remain
immutable historical evidence. This handoff does not revise the mature RTC
operator contract, authorize Stage B, or establish a Stage A numerical
precision gate, factor ranking, candidate selection, or application.

The separate production low-velocity mask task is excluded. The Stage A
low-velocity threshold below controls diagnostic motion eligibility only. It
does not change flags, validity delivered to RTC/PTC/mapmaking, detector
samples, map inputs, weights, maps, or production behavior.

## Approved Stage A scientific and product contract

### 1. Beam authority

Use a fixed per-array diffraction-limited Airy **intensity** FWHM

```text
theta_FWHM = 1.028 lambda / 50 m
```

with nominal array wavelengths and these authoritative angular values:

| Array | `theta_FWHM` |
| --- | ---: |
| `a1100` | `4.66 arcsec` |
| `a1400` | `5.94 arcsec` |
| `a2000` | `8.48 arcsec` |

There is no APT or measured-beam fallback. An unknown array makes the beam and
dependent metrics unavailable. Per-detector full Gaussian covariance is future
`SCI-BEAM-001` work and is not part of Stage A.

### 2. Motion authority and source support

Requested, effective, and realized HWPR state are distinct and must be
recorded. Physical HWPR input or file presence alone does not make an
observation HWPR-enabled. When HWPR is absent or explicitly ignored by the
effective observation plan, apply the approved non-HWPR contract below.

For those non-HWPR or effectively HWPR-ignored observations, the provisional
primary motion statistic is `v95`, the empirical 95th percentile of valid,
eligible source-telescope speed magnitudes from authoritative source rows
before detector-grid interpolation. It must not be called an upper bound.
Report `p99`, `p99.5`, and the raw maximum as diagnostics.

Preserve source-row support, gaps, boundary identity, valid and rejected
counts, and coverage. Speeds greater than `3600 arcsec/s` are invalid.
Approximately `500 arcsec/s` is contextual normal-operation information, not a
validity bound.

Only source-telescope intervals with `v >= 1.0 arcsec/s` are eligible for
`v95` and the astronomical sampling metrics; exactly `1.0 arcsec/s` remains
eligible. Persist the threshold, HWPR state, excluded count, excluded duration,
excluded fraction, and remaining support. If no interval remains, return the
stable reason `unavailable_low_velocity` deterministically.

When HWPR is scientifically enabled, telescope `v95` and the diffraction beam
alone do not define the astronomical passband. RTC precedes mapmaking, and a
polarized signal can occupy the `4*f_HWPR` carrier plus scan-induced sidebands.
For an effectively HWPR-enabled observation, all Stage A dependent
astronomical sampling and transfer metrics are fail-closed unavailable with
stable reason `unavailable_hwpr_sampling_contract`.

Do not apply the non-HWPR `1 arcsec/s` eligibility rule or the telescope-only
candidate range to enabled HWPR. Retain the factor-1 explicit identity-plan row
as provenance and reference, with every HWPR-dependent metric unavailable. It
is not a sampling recommendation.

Future HWPR enablement requires separately approved authority for rotation
frequency and phase support, validity and gap handling, the `4*f_HWPR`
convention, scan-sideband extent, the combined beam/FIR/modulation/decimation
transfer, and an enabled-polarimetry validation dataset and product gate. This
handoff infers or authorizes no HWPR correction, threshold, candidate
selection, or production behavior. Enabled polarimetry/HWPR remains
unavailable pending that scientific contract and reference gate.

### 3. Astronomical alias operator

Use coherent complex phase-zero astronomical folding of the admitted beam
response multiplied by the observation's exact effective/realized RTC filter
response under the enumerated `M` and phase counterfactual, through the
phase-zero decimator. Report amplitude, phase, power response, and distortion
relative to unaliased baseband.

Enumerate exactly `M` unique periodic images using one explicit half-open
convention. Factor `M=1` has exactly zero alias. Do not publish an incoherent
PSD interpretation unless a separate PSD model and authority are approved.

### 4. RTC-A-007 numerical error control and scientific-selection boundary

Separate filter characterization from scientific acceptability. Fixed-grid
results are not `exact` or `worst case`. Use analytical results where
available and deterministic bounded or adaptive calculations otherwise.

Report conservative numerical bounds for aliasing and stopband extrema using
the exact realized FIR coefficients. A broad but valid enclosure is acceptable
and must be reported honestly as bounded evidence. Persist enough method and
provenance to reproduce the result:

- method and version;
- exact coefficient identity;
- the domain and partition, or adaptive state required by the method contract;
- resource and evaluation information;
- bounds and error enclosure;
- convergence, status, and stable reason; and
- relevant numerical settings.

Per-metric validity is independent. Failure of one numerical metric does not
invalidate unrelated candidate diagnostics. Factor 1 has exactly zero alias
contribution and no decimation stopband; its stopband metric status is
`not_applicable_no_decimation`.

Stage A stopband rejection and astronomical alias transfer are diagnostic
characterizations only. No absolute stopband-rejection threshold ranks,
recommends, accepts, or rejects a factor.

Any later candidate selection must evaluate noise-relative alias variance with
an empirical or otherwise approved representative detector-noise PSD, not a
white-noise assumption or attenuation alone:

```text
P_alias = sum_alias_bands integral S_x(f) |H_M(f)|^2 df
epsilon_alias = P_alias / P_noise_retained
```

The provisional future design target is `epsilon_alias <= approximately 0.01`:
order 1% or less additional retained-band noise variance, about 20 dB below
retained noise in power, and approximately 0.5% RMS increase. This is a future
science-driven candidate-selection target only. It is not a Stage A gate,
rank, acceptance criterion, or numerical-precision requirement.

Stage A must provide trustworthy bounded transfer and alias metrics from which
that later noise-relative calculation can be made. Stage A does not need or
authorize an arbitrary precision threshold.

### 5. Exact candidate enumeration and existing-filter counterfactual

Never infer candidate count from a configured filter edge. Use exact
effective/realized observation cadence `fs`; preserve the requested cadence
and its consistency result. `488 Hz` is approximate hardware context or a
maximum, never a substitute for the actual cadence.

For each admitted array and factor `M`, calculate

```text
N_beam(M) = theta_FWHM * fs / (M * v95)
Mmax = floor(theta_FWHM * fs / v95)
candidates = {1, ..., max(1, Mmax)}
```

Factor 1 is always present. If native `N_beam < 1`, report it truthfully. One
sample per FWHM is only a deliberately conservative diagnostic endpoint; it is
not acceptance, recommendation, authorization, or selection.

Evaluate every candidate without ranking. Stage A does not synthesize a new FIR
for each hypothetical `M`. It characterizes the existing-filter
counterfactual

```text
(M, H_RTC^realized)
```

in which each physically enumerated `M` and its phase convention are bound to
the observation's exact effective/realized RTC filter. When filtering is
disabled, use the explicit identity coefficient vector `[1]`; never consume
dormant filter settings.

For every `M`, persist:

- the coefficient vector and digest;
- configured filter parameters;
- requested, effective, and realized filter state;
- the exact `M`/phase/filter binding; and
- relevant filter and observation identity/provenance.

Calculate the approved astronomical transfer, bounded aliasing, stopband
characterization, and other Stage A metrics for that exact combination. This
includes obviously poor combinations, such as an output Nyquist lying inside
the realized FIR passband. Do not rank, recommend, accept, or reject a factor.

Poor performance means only that `M` performs poorly with the FIR actually
realized for this observation. It does not establish that `M` itself is
scientifically unsuitable or estimate the best achievable performance with an
appropriately designed anti-alias filter. This distinction must be explicit in
the `rtcdiag` Stage A product contract and provenance.

Later learned-plan work, not Stage A, may jointly select `M` and a
factor-specific FIR using cadence, scan motion, diffraction beam,
astronomical-transfer requirements, an approved representative detector-noise
PSD, cost, and `epsilon_alias = P_alias / P_noise_retained` with the provisional
future target `<= approximately 0.01`. Stage A adds neither user-supplied FIR
grids nor any implicit FIR-design rule.

Persist `fs`, `v95`, `theta_FWHM`, `Mmax`, the eligibility threshold, status,
and reasons. Missing or invalid cadence, motion, array, or beam yields a
deterministic unavailable state.

A technical resource guard may fail explicitly but must not truncate or
redefine the scientific range. Preserve the derived `Mmax` and report
`candidate_range_resource_limit`. Under the hardware and eligibility context,
the finite extreme is roughly 4,000 candidates for the largest FWHM,
`fs` approximately 488 Hz, and `v95 >= 1 arcsec/s`.

### 6. RTC-A-008 complete-context applicability

Stage A has no arbitrary positive numerical minimum for meaningful
complete-context support. Duration, fraction, beam-crossing, or
downstream-weight thresholds require later observational evidence or a named
downstream scientific requirement. Zero fully supported candidate outputs is
the sole hard Stage A candidate-applicability boundary. Stronger minima are
deferred.

Compute complete-context support from the exact realized operator and output
grid, not from total scan duration. The calculation must use the actual
realized FIR coefficients and length, left and right context, factor and
decimation phase, scan boundaries, valid eligible support, internal gaps,
guards, and candidate output grid.

For each hypothetical candidate:

- zero fully supported candidate outputs yields the stable status
  `candidate_unusable_no_complete_context`, with meaning explicitly bound to
  zero complete-context support;
- the plan-level coefficient/decimator response may still be reported, but
  never as an observation-effective response; and
- one or more fully supported outputs establishes mathematical evaluability
  only. It does not establish adequacy, recommendation, authorization, or
  selection.

Avoid an ambiguous `insufficient` label for this boundary unless the status
explicitly states that the count is zero.

The applied RTC operator is a separate status. If an applied operator has zero
fully supported outputs, report `scan_unusable_for_applied_rtc_operator` with
reason `no_complete_context`. Production RTC must ultimately fail closed and
must not feed such a scan to mapmaking as valid processed RTC data. Stage A
only reports this state: it changes no flags, suppresses no products, changes
no execution, and changes no RTC/PTC/mapmaking input. Enforcement belongs to a
separate RTC sample-validity/production task and is not launched here. One or
more full outputs does not create or imply a stronger production threshold.

Persist these diagnostics per scan and candidate:

- factor and explicit identity/FIR state;
- tap count and left/right context;
- total eligible input support;
- total candidate-output count;
- fully supported output count `N_full`;
- summed complete-context duration;
- `f_full = N_full / N_candidate_outputs`, with the denominator explicit and
  a typed non-value when that denominator is zero;
- incomplete boundary, internal-gap, and other counts and fractions;
- longest contiguous full-output run;
- plan-transfer status;
- observation-applicability status; and
- a stable cause-specific reason.

Beam-crossing count may be diagnostic only. It creates no map-weight or
applicability dependency. The explicit factor-1 identity-filter plan requires
no FIR context. Independent cadence, motion, and identity failures retain
their existing distinct reason codes.

The interior transfer

```text
B(f) H_FIR(f) D(f)
```

is the exact complete-context interior plan transfer. It is never the transfer
of the whole finite scan. Do not average incomplete or unchanged boundary
behavior into it, call it observation-effective when no full output exists, or
imply that one full output is adequate.

### 7. Availability and metric validity

Represent separately:

1. scan-array prerequisite status;
2. candidate status; and
3. validity for each metric.

Ignore metric values unless their validity is true. Infinity is legitimate
only for a metric whose contract explicitly permits it. Reason codes must be
stable and cause-specific. The implementation checkpoint must freeze the exact
reason-code vocabulary before editing; it must include at least
`unavailable_low_velocity`, `candidate_range_resource_limit`,
`candidate_unusable_no_complete_context`,
`scan_unusable_for_applied_rtc_operator`, `no_complete_context`, and
`unavailable_hwpr_sampling_contract`; the factor-1 stopband metric must use
`not_applicable_no_decimation`. The vocabulary must not collapse missing
cadence, motion, array, beam, FIR, support, numerical, HWPR-contract, or
resource failures into one status.

### 8. Diagnostic-product evolution

`rtcdiag` and `ptcdiag` are formative products. For this bounded Stage A
repair, correctness takes precedence over backward compatibility. Existing
`rtcdiag` fields may be corrected or renamed without compatibility aliases,
but the repair must:

- bump the `rtcdiag` schema version;
- document every semantic break; and
- update `validation/product_contracts.json` and applicable executable product
  tests.

Stage A authorizes no unrelated `ptcdiag` change. The truthful scope statement
is that non-`rtcdiag` science products and product cadence are unchanged;
`rtcdiag` schema and diagnostic numerics intentionally evolve.

### 9. Compact exact-rerun provenance

Persist enough to state setup, method, and exact rerun:

- observation, scan, and array identity;
- exact Citlali commit;
- schema and algorithm version;
- a reference to the canonical raw-input manifest;
- exact effective/realized cadence, requested cadence, and consistency result;
- source support and coverage;
- `v95` definition, thresholds, requested/effective/realized HWPR state,
  counts, and exclusions;
- diffraction authority and constants;
- factor/phase and the observation's exact effective/realized RTC coefficient
  vector and digest, configured parameters, requested/effective/realized filter
  state, and counterfactual binding;
- numerical method/version, exact coefficient identity, domain/partition or
  contracted adaptive state, resource/evaluation information, bounds/error
  enclosure, convergence/status/reason, and relevant numerical settings;
- relevant resolved configuration; and
- prerequisite, candidate, and per-metric status/reasons.

Do not embed raw telescope samples, APT beam contents, exhaustive runtime
configuration, intermediate-state serialization, or self-contained archival
replay.

## Mandatory pre-edit checkpoint

A separately launched repair must begin from the exact proposed base in a
fresh clean worktree and return before editing with:

- exact HEAD, parent, tree, branch, worktree, and clean state;
- exact proposed changed paths from the allowlist below;
- finding/decision-to-path and finding/decision-to-test traceability;
- the exact pre-interpolation source-row authority and compact carrier used to
  provide motion support without changing map inputs or flags;
- exact effective/realized cadence owner and requested-cadence consistency
  check;
- exact owners and propagation paths for requested, effective, and realized
  HWPR state, proving that physical input/file presence does not imply enabled
  status;
- exact existing-filter counterfactual binding for every enumerated factor,
  including phase, coefficient vector/digest, configured parameters,
  requested/effective/realized filter state, and identity `[1]` when filtering
  is disabled;
- exact complete-context counting convention derived from the realized FIR,
  actual phase/output grid, boundaries, gaps, guards, and valid eligible
  support, with no positive applicability threshold beyond `N_full == 0`;
- exact `rtcdiag` schema version and complete stable reason-code vocabulary;
- analytical or deterministic bounded/adaptive numerical method and
  conservative error-control plan for every retained metric, including exact
  coefficient identity, reproducibility state, independent metric validity,
  and factor-1 `not_applicable_no_decimation` semantics;
- resource preflight proving that the full derived range is evaluated or that
  `candidate_range_resource_limit` is returned without truncation;
- proof that no recommendation, selection, factor/filter application, Stage B,
  production low-velocity mask, PTC/VAL/MAP/BEAM change, or science-output
  consumer is proposed; and
- confirmation that Unity, local science reductions, external contact,
  delegation, merge, push, repair launch, and re-audit launch remain
  prohibited.

Silence prohibits a capability. A needed new state owner, broad framework,
public `Engine` field, HWPR behavior beyond the approved fail-closed contract,
new FIR synthesis or FIR-grid input, numerical acceptance or
candidate-selection policy beyond the approved characterization boundary, or
path outside the allowlist is a stop for
coordinator/owner review.

## Initial path allowlist

Candidate Stage A implementation and diagnostic wiring:

- `include/citlali/core/pipeline/rtc_learned_sampling_metrics.h`
- `include/citlali/core/pipeline/rtcdiag_scan_summary.h`
- `include/citlali/core/pipeline/rtcdiag_netcdf.h`
- `include/citlali/core/engine/detail/rtcdiag_output_impl.h`

Existing source-row and cadence boundaries, only for a compact read-only
diagnostic authority before detector-grid interpolation:

- `include/citlali/core/pipeline/initial_observation_setup.h`
- `include/citlali/core/pipeline/timestream_alignment_helpers.h`

Focused tests and executable product contract:

- `tests/test_rtc_learned_sampling_metrics.cpp`
- `tests/CMakeLists.txt`
- `validation/product_contracts.json`

Candidate documentation and non-authoritative format example:

- `doc/RTC_LEARNED_SAMPLING_STAGE_A_METRICS_2026-08-09.md`
- `doc/REFACTOR_STATUS.md`
- `validation/rtc_learned_sampling_stage_a_example_2026-08-09/candidate_metrics.csv`

No other application, configuration, test, validation, product, or
documentation path is authorized by this handoff. A necessary compact carrier
that cannot be implemented within an existing allowlisted owner requires a
stop and exact path amendment before creation.

## Finding-to-repair traceability

| Audit findings | Required bounded repair | Required focused evidence |
| --- | --- | --- |
| `RTC-A-001`, `RTC-A-002` | Replace APT/ellipse aggregation with exact fixed Airy intensity FWHM table; unknown array unavailable; no measured/APT fallback. | Exact array constants/formula, unknown array, Beammap placeholder non-use, no APT fallback, units/identity/status fixtures. |
| `RTC-A-003`, `RTC-A-004` | Record distinct requested/effective/realized HWPR state. For absent or effectively ignored HWPR, compute motion from valid eligible pre-interpolation source rows, reject invalid bounds, apply non-HWPR `v>=1.0` eligibility and `>3600` invalid rule, and persist support/coverage/counts. For effectively enabled HWPR, fail dependent metrics closed as `unavailable_hwpr_sampling_contract`; retain factor-1 identity provenance only. | Raw-gap versus interpolated-gap, exact threshold equality, below-threshold exclusion, all-low-velocity unavailable, over-limit, invalid bounds, partial support, absent/ignored/enabled HWPR state matrix, file-presence negative control, enabled-HWPR unavailable metrics, factor-1 reference/non-recommendation, and no non-HWPR threshold/candidate reuse. |
| `RTC-A-005`, `RTC-A-006` | Implement coherent complex phase-zero beam-times-realized-FIR folding with exactly `M` half-open images and zero factor-1 alias. | Independent analytic complex fixtures for factor 1, odd/even factors, DC, output/native Nyquist, amplitude, phase, power, and distortion; negative control against incoherent power sum. |
| `RTC-A-007` | Separate diagnostic characterization from scientific acceptability. Remove fixed-grid exact/worst claims; use analytical or deterministic bounded/adaptive methods on exact realized coefficients; publish conservative enclosures and reproducibility state with independent per-metric validity. Factor 1 has zero alias and stopband status `not_applicable_no_decimation`. No absolute rejection threshold or future `epsilon_alias` target ranks/selects a Stage A factor. | Narrow-extremum and long-filter adversaries; analytic-limit controls; certified enclosure/convergence; deliberately broad valid bounds; nonconvergence/resource status; independent per-metric failure; coefficient-identity and method-state writer/reopen; factor-1 exact-zero alias and not-applicable stopband; negative controls against fixed-grid exact/worst language, white-noise selection, absolute attenuation gating, or candidate ranking. |
| `RTC-A-008` | Bind each factor and phase to the observation's exact effective/realized RTC filter as the existing-filter counterfactual; never synthesize an `M`-specific FIR. Compute complete-context applicability on the actual phase/output grid. Zero full outputs is the sole hard Stage A candidate boundary; separate plan transfer, hypothetical candidate applicability, and applied-operator scan status. Report the complete diagnostic set without a positive minimum-support or map-weight rule. | Counterfactual-to-realized-filter identity joins; coefficient vector/digest and requested/effective/realized state; disabled-filter `[1]`; deliberately poor filter/`M` combinations; no new FIR grid or synthesis; `N=0,1,L-1,L,L+1`; exact first full output at actual phase; internal gaps/invalid samples; multiple short regions; boundary impulses; hypothetical zero-support rejection without condemning the scan; applied zero-support fail-closed status; plan-transfer versus observation-applicability separation; counts derived from actual RTC conventions. |
| `RTC-A-009` | Enumerate exact beam/cadence/motion range independent of filter edge; evaluate all candidates or fail without truncation and preserve `Mmax`. | Formula/range properties, factor-1 floor, native undersampling truth, approximate 4,000 extreme, filter-edge independence, resource rejection, and no ranking/selection. |
| `RTC-A-010` | Separate prerequisite, candidate, and per-metric validity with stable reasons; values ignored unless valid. | Every missing/invalid prerequisite, partial metric failure, legitimate infinity, invalid infinity, empty support, resource guard, and status/value-consistency fixture. |
| `RTC-A-011` | Persist the compact exact-rerun provenance above and no prohibited payload. | Writer/reopen joins, requested/effective cadence mismatch, manifest reference, method/error identity, coefficient order/digest, repeatability, and prohibited-payload absence. |
| `RTC-A-012` | Bump `rtcdiag` schema; rename/correct fields without aliases; document semantic break; update executable contract; leave `ptcdiag` alone. | Schema-version, old-name absence where renamed, units/dimensions/status, product-contract gate, explicit semantic-break record, and unchanged `ptcdiag` blob/path inventory. |
| `RTC-A-013`, `RTC-A-014` | Replace circular/self-composed tests with independent adversarial production-path evidence; prove atomic required output, repeat/reset, and non-interference. | Engine/setup or exact-base A/B samples/flags/time/PTC/map invariance; writer create/write/sync/rename failure; no partial complete artifact; repeat/sequential observation; bounded setup/cardinality; full deterministic repository gates. |

## Required validation and handback

After the pre-edit checkpoint and one first-viable-artifact checkpoint, the
repair may run only focused deterministic local implementation tests and the
repository-required deterministic gates:

1. independent focused fixtures for every traceability row above;
2. explicit RTC-A-008 fixtures for `N=0,1,L-1,L,L+1`, the exact first fully
   supported output at the actual phase, internal gaps and invalid samples,
   multiple short regions, boundary impulses, the identity filter,
   hypothetical-candidate failure without scan condemnation, applied
   zero-support fail-closed reporting, and plan-transfer versus
   observation-applicability separation, with expected counts derived from
   actual RTC conventions;
3. explicit RTC-A-007 analytic and bounded/adaptive numerical fixtures,
   including narrow extrema, long filters, conservative enclosure,
   convergence/nonconvergence, deliberately broad valid bounds, exact
   coefficient and method-state reproduction, independent per-metric failure,
   factor-1 exact-zero alias and `not_applicable_no_decimation`, and negative
   controls against fixed-grid exact/worst claims or candidate selection;
4. production `rtcdiag` writer/reopen, schema, status, provenance, and injected
   failure fixtures;
5. exact-base A/B non-interference for samples, flags, timestamps, assigned
   grid, RTC/PTC/map inputs, weights, maps, non-`rtcdiag` product contents, and
   product cadence;
6. sequential/OpenMP and repeated/sequential-observation determinism;
7. full CTest with disabled/skipped tests reported;
8. `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`;
9. applicable baseline, validation-ledger, authority, and product-contract
   gates; and
10. `git diff --check`, exact changed-path inventory, artifact digests, and
   clean state.

Return one exact repair commit/parent/tree, changed paths, all test commands
and results, candidate-range/cardinality and resource results, provenance and
schema identities, finding traceability, every remaining unavailable state,
and confirmation that no science samples, flags, RTC/PTC/map inputs, maps,
weights, non-`rtcdiag` products, or cadence changed. Stop for coordinator review
before any re-audit, integration, merge, push, production change, Stage B, or
downstream action.

## Explicit exclusions and non-authorization

- no production low-velocity mask task, flag mutation, or map-input mutation;
- no arbitrary positive complete-context duration, fraction, beam-crossing,
  downstream-weight, or other minimum-support threshold;
- no Stage A enforcement of the applied-operator production fail-closed rule,
  product suppression, execution change, or mapmaking-input change;
- no HWPR correction, threshold, enabled-HWPR candidate range or selection,
  rotation/phase convention, production behavior, or polarimetry enablement;
- no APT/measured-beam fallback or per-detector Gaussian covariance;
- no incoherent PSD claim without a separately approved PSD model;
- no absolute stopband-rejection threshold, white-noise assumption, attenuation-only
  acceptability claim, or use of the provisional future `epsilon_alias` target
  as a Stage A gate, rank, precision requirement, acceptance, or rejection;
- no `M`-specific FIR synthesis, user-supplied FIR grid, implicit FIR-design
  rule, or inference of best-achievable performance for any factor;
- no tolerance, rank, recommendation, selection, resolved/applied plan, factor,
  FIR, cadence, or RTC execution change;
- no unrelated `ptcdiag`, RTC/PTC/VAL/MAP/BEAM, science-product, or production
  change;
- no physical event/timing/astrometric correction or source-mask task;
- no raw-sample embedding, exhaustive configuration/state serialization, or
  archival replay expansion;
- no local science reduction, Unity request/access, external contact,
  delegation, costly campaign, merge, or push; and
- no repair, validation campaign, re-audit, Stage B, or downstream launch.

Only a separate owner instruction may launch a repair against the exact
proposed base, branch, and a subsequently frozen/verified version of this
handoff.
