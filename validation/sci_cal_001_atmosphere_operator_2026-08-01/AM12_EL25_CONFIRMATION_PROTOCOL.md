# SCI-CAL-001 AM 12.2 EL25 numerical-confirmation protocol

## Immutable preregistration status

This standalone protocol is frozen on 2026-08-02 before any confirmation AM
process is launched, before a confirmation cache exists, and before any new
confirmation result is inspected. It authorizes one bounded numerical study
only. It does not adopt an atmosphere operator, authorize an operational
domain or production calibration, modify Citlali or TolTECA, resolve the open
SCI-ALIGN-001 interface dependency, replace the frozen v2 evidence, authorize
CAL repair, or launch a re-audit.

The machine preregistration is
`am12_el25_confirmation_preregistration.json`, SHA-256
`66c9583d67c3696ac03d1edbd6eade95884dbdc77dd93ef890226594f210da70`.
The decision schema is `am12_el25_confirmation_result.schema.json`, SHA-256
`a28e738970b2a462fd1fb68c78aad552e32cbd396f8f60956f8615e4be2a3965`.
The frozen evidence driver is `run_am12_el25_confirmation_study.py`, SHA-256
`bcc4bc9f59574424e1daab652ab0316f8a694998155d9c3daa246e1e6260fb22`. The Git commit containing this
protocol, manifest, schema, and driver is the preregistration commit; its
identity cannot be self-recorded and must be bound by the execution context
before AM execution.

## Authority and predecessor identities

The work begins from clean evidence head
`f4014d3669b94b1eceb8158da7993737efc908f2`, parent
`742a3c263faf68c2de7d5b8db0d3423127f60480`. The exact application/repair base
is `9aae0e669384c5c0c0dda93debc194d6b8dac787`.

Authority is read from coordination commit object
`8fc9263a2f502656b51d32cb60655481f83509f1`, whose parent is immutable decision
commit `f513f410b88d147be6bd016d4c79ac1d3a5b2a8e`. The binding commit changes only
the audit-ledger binding; unrelated coordination working-tree state is
excluded. The controlling decision and passband records are:

- `SCI-CAL-001_ATMOSPHERE_CONFIRMATION_DECISION_2026-08-02.md`, SHA-256
  `24a2e12d37689999e1070bec4329aeead47c8e24082b20f183bf0ad8e29494b0`;
- `SCI-CAL-001_PASSBAND_AUTHORITY_001.json`, SHA-256
  `2756908181cc466550399ec0a869e6671de7912bd3a935f9aeebf63e3e826617`;
  and
- `SCI-CAL-001-XAUD-001.yaml`, SHA-256
  `2248422a507455e972c70c221c214b40fec68566011d27a9d8827952e43087d5`.

The complete f401 package and its canonical native, P1, and v2 caches passed
cache-only replay before registration. Its `SHA256SUMS` SHA-256 is
`bafd34e4a3d5bffb95b3af1fdbcfb7c993146248b2bccd1d0333bae91fd3caad`.
The frozen v2 manifest, decision, candidate nodes, truth rows, and runner have
SHA-256 values `c9f6aea80851fb7726b8845d4697af1cb270cb7ff7ce51d3d5fc63828f793b3a`,
`976c6c6a269a1b5dabde2b5eba89cb6176b02b837ea2b7b0e26a64307fe9ee59`,
`8005c8ae1d4ab1c8de39f06a632d76d3e8f248939dc63c616dd176bcbd2f6fe2`,
`ad74d19ef0bc915255b9cc7a507e8977f96435fb37ce0d0bd7cb385991c1802c`,
and `ace8e08a037535260b6b1d889f83dbf722ffc932e05bc1f7f83f0565ef0ff47c`.
None may be rewritten or relabeled.

## Confirmatory question and fixed candidate roles

The primary question is whether the previously nominated simple candidate

```text
fixed_djf25_v1 + am12_piecewise_linear_los_tau_eval_v0
```

meets every structural, provenance, coverage, warning, support, and maximum
one-percent numerical representation-fidelity gate on wholly new direct-AM
truth over the confirmation support. This pair became the conditional
simplicity candidate in the f401 owner brief before the present results and is
therefore the sole primary confirmatory candidate.

The other three frozen pairs are secondary descriptive candidates:

```text
fixed_djf25_v1       + am12_pchip_los_tau_eval_v0
conditioned_djf_v1   + am12_piecewise_linear_los_tau_eval_v0
conditioned_djf_v1   + am12_pchip_los_tau_eval_v0
```

All four are evaluated without tuning. A secondary result cannot rescue a
primary failure, and no candidate, interpolation rule, gate, or tie-break may
change after registration.

## Closed support and sample identity

The confirmation-only closed support is:

```text
0 <= zenith tau225 <= 0.158313198574890929
25 deg <= aligned sample elevation <= 80 deg
q95 excluded
```

The physical correction uses the full modified-secant airmass of each eligible
sample and top-of-atmosphere pivot `X_ref=0`. There is no clamping,
nearest-model fallback, extrapolation, or use of zenith opacity as if it were
line-of-sight opacity. Candidate truth and representation are compared in
line-of-sight band optical depth, `lambda = -log(T_eff)`, at the complete
sample elevation.

The evidence wrapper must fail closed unless it receives all of: a nonempty
sample identity, the aligned elevation, a nonempty timing-gap or interpolation
origin, finite positive duration, an explicit original-versus-synthesized
eligibility state, and eligibility exactly true. Missing, invalid, or
ineligible state fails. This is an abstract confirmation probe; it does not
claim that the current ALIGN interface supplies the contract.

The accepted eligibility-state literals are exactly `original_eligible` and
`synthesized_eligible`. The aligned-elevation value in the state must exactly
equal the elevation passed to the operator. Valid original and synthesized
states are tested at EL25, EL50, and EL80. Reject each required field when
removed individually; empty sample identity, timing/interpolation origin, or
eligibility string; any unknown eligibility string; `eligible` other than the
Boolean value true; an aligned-elevation mismatch; duration zero, negative, or
non-finite; each coordinate immediately outside a closed endpoint; and every
non-finite opacity or elevation.

Endpoint probes must accept exact tau values 0 and q75 and elevations 25 and
80 with eligible state. They must reject the binary64 neighbor below or above
each opacity/elevation endpoint, NaN/non-finite inputs, incomplete operator
support, unknown candidate/passband/spectral identities, and every absent or
ineligible ALIGN field.

## Independent confirmation tuples

No v2 truth row is reused. The opacity rule is fixed without reference to any
candidate error: take exact affine trisections `1/3` and `2/3` of each adjacent
q0/q25, q25/q50, and q50/q75 zenith-opacity interval using 80-digit Decimal
arithmetic. The six requested coordinates, seven-significant-digit AM T225
literals, and achieved coordinates are:

```text
q0:  tau225=0                     T225(EL80)=1.0
q25: tau225=0.0504874104674104401 T225(EL80)=0.9500275
q50: tau225=0.0883393725904400573 T225(EL80)=0.9142065
q75: tau225=0.158313198574890929  T225(EL80)=0.8515054
```

| Coordinate | Requested tau225 | T225 literal at EL80 | Achieved tau225 |
| --- | ---: | ---: | ---: |
| q0-q25 1/3 | `0.0168291368224701467` | `9.830571e-01` | `0.016829094692792866028095833034107594099807326682113141316107807600180529588238238` |
| q0-q25 2/3 | `0.0336582736449402934` | `9.664012e-01` | `0.033658252426569561178189965239103731896041242555314415055126326210836780629567994` |
| q25-q50 1/3 | `121161083856167/1920000000000000` | `9.379339e-01` | `0.063104700540512686321847773467542976759139168650091580323635320748709109259984728` |
| q25-q50 2/3 | `2271661556482905547/30000000000000000000` | `9.259942e-01` | `0.075722042030700389119626121070791857500491361410683665359233247051421228619914645` |
| q50-q75 1/3 | `0.1116639812519236812` | `8.928092e-01` | `0.11166401570834966115343660054904395552668301345765783648306027667115637367003112` |
| q50-q75 2/3 | `0.1349885899134073051` | `8.719128e-01` | `0.13498855802183467889100714475992808246559076086155667163732523489052384928879709` |

Use Decimal precision 80 with `ROUND_HALF_EVEN`. For displayed transmission
`T` and half-step `h=5e-8`, freeze

```text
negative bound = log((T+h)/T) / X80
positive bound = log(T/(T-h)) / X80
pass iff -negative_bound <= achieved_tau-requested_tau <= positive_bound
```

The exact `(negative, positive)` bounds in coordinate order are:

```text
q0-q25 1/3:  (5.0090908858107033827835573239810291304422737160115895888810676192355105584876663e-8,
               5.0090911405818155183991594225401065268392246982618032136751583496413970730661938e-8)
q0-q25 2/3:  (5.0954224370615562895356932124441836111550577060191613807299022672429358629448971e-8,
               5.0954227006902939647334992799666188814863255310041522931301354365233511256478793e-8)
q25-q50 1/3: (5.2500739698361731252061096500674759683253594175273224851347734490661077834711246e-8,
               5.2500742497105923019404829462579831704186556246697124931152045806992313252374157e-8)
q25-q50 2/3: (5.3177680293511263491630173321576765658340019355600575557783659169232269994913374e-8,
               5.3177683164894352565127278860744803456181814351145769986909971798262308943268477e-8)
q50-q75 1/3: (5.5154251851158612156615494015961717851096582949772038896213002566463504312754885e-8,
               5.5154254939962658730097503035489235220981487451408403603038791204659130864364942e-8)
q50-q75 2/3: (5.6476087332111139087678129120029262288238843956424739130911842042689435459684850e-8,
               5.6476090570742854978706910123960762606245350419194784125029785366346481208877829e-8)
```

Recomputed stored Decimal values must agree within `5e-78`; the scientific
gate uses the recomputed asymmetric bounds, not the stored maximum shortcut.

For q25-q50 the rationals above are authoritative; their repeating decimal
expansions are recorded in the machine preregistration. The driver must
recompute every analytic transmission, displayed literal, achieved coordinate,
residual, and asymmetric half-display-step bound before cache creation. A
residual outside the registered bound stops before AM execution.

Both coordinates in each interval use every frozen predecessor profile role:

- q0-q25: `LMT_DJF_5`, `LMT_DJF_25`;
- q25-q50: `LMT_DJF_25`, `LMT_DJF_50`; and
- q50-q75: `LMT_DJF_50`, `LMT_DJF_75`, `LMT_annual_25`, `LMT_MAM_25`.

This gives 16 independently solved scale cases. Every case is run at every
integer elevation from 25 through 80 inclusive. That elevation rule follows
the declared closed support and does not target a measured error feature. It
gives 56 elevations, 896 full AM grids, 10,752 direct band/alpha truth values,
and 43,008 four-candidate comparison rows.

Before creating the cache, the driver must construct the 336 unique
`(requested_tau225, elevation)` keys and the corresponding 336 achieved-
coordinate keys, then anti-join both sets against candidate fit coordinates
(q anchors at even EL20--80) and every inspected v2 discovery coordinate
(arithmetic midpoints at odd EL21--79). The comparison uses the exact
registered Decimal coordinate and integer elevation; both anti-joins must have
zero overlap. Profile remains an additional retained identity. Any overlap,
missing predecessor artifact, or digest mismatch stops before AM.

## AM inputs, scale solver, and raw evidence

Use the copied AM 12.2 source/input family only. The source payload aggregate
SHA-256 is
`0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8`.
The locally built Mach-O AM 12.2 executable SHA-256 is
`78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`.
The six permitted AMC file digests are recorded in the machine
preregistration. Only `Nscale troposphere h2o` through AMC argv `%9` varies.

Reuse the canonical 48-bisection plateau solver from
`probe_am12_h2o_scale_hypotheses.py`, SHA-256
`caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`,
with at most 64 bracket expansions. Each case must match its exact parsed T225
literal at EL80 and preserve the scale as round-trip Decimal and binary64 hex,
inside/outside plateau bounds, complete scale trace, and trace SHA-256.

Full spectra use exactly 0--500 GHz inclusive at 10 MHz, hence 50,001 rows.
Every run preserves raw combined output and a canonical sidecar containing the
request, argv, working directory role, AMC and executable digests, locale,
OMP setting, cache shard, execution context, return status, diagnostic counts,
raw and normalized digests, numeric digest, row count, and AM version.

Execution requires a fresh nonoverlapping cache and an exclusive whole-cache
lock. Cache-only regeneration and check modes use a shared lock and cannot
launch AM. The run inventory must bind every raw output, sidecar, and scale
trace used. `failed_attempts` must be empty for a valid confirmation. A
cache-only replay must leave the evidence-file aggregate digest unchanged.
The registered cache basename is
`sci_cal_001_am12_el25_confirmation_v1_20260802_root`; execute with eight jobs,
eight deterministic cache shards, and one OMP thread per AM process. The path
must be absent before `--run-confirmation` creates it.

## Warning admission

Status 0 is admissible only with no warning header and no error record. Status
1 is admissible only for a complete 50,001-row full spectrum with exactly the
known unresolved-narrow-line warning structure, one corresponding canonical
summary whose unresolved count is 86, 87, or 88, zero unknown warning classes,
zero error records, and no cache-mutation diagnostic. It is labeled
warning-bearing numerical evidence, never clean success. A status-1 scale
search or any other incomplete-grid status-1 result fails because it cannot
satisfy the 50,001-row requirement. Every other nonzero status fails closed.

The only permitted four-line summary grammar is exactly:

```text
! Warning: Encountered in-band lines narrower than the frequency
!          grid spacing.  The output configuration data includes
!          the unresolved line count after each column definition
!          for which this occurred.  Count: (86|87|88)
```

The frozen imported parser and this protocol admit literal LF line endings
only for that summary; CRLF is not canonical confirmation evidence. At least
one exact per-column warning record must accompany a status-1 summary.

The only other permitted warning header matches exactly
`! Warning: Column included [0-9]+ unresolved lines.` There must be exactly one
summary, and every line beginning `! Warning:` must be either that summary
header or a permitted per-column record. Every line beginning `! Error:`, any
cache-mutation warning, and every other warning header invalidate the evidence.
The two known mutation diagnostics are exactly
`! Warning: Unable to rename file in insert_as_mru().` and
`! Warning: Unable to rename file in promote_to_mru().`; each is counted
separately as cache mutation and also remains disallowed by the warning-header
allowlist.
The decision records the 86/87/88 histogram, wrong-row status-1 count,
scale-search status-1 count, unknown-warning count, error count, and
cache-mutation-warning count.

The driver must apply this policy to newly executed and cache-loaded results;
the sidecar cannot override parsed raw diagnostics.

## Exact passband and spectral convention

The sole confirmation passband identity is
`toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`
at TolTECA commit `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`.
The set hash is recomputed in lexical relative-path order from path bytes, NUL,
the raw 32-byte member digest, and NUL. Its 1,297,803 bytes comprise:

| Member | SHA-256 |
| --- | --- |
| `index.yaml` | `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5` |
| `data/a1100_passband.ecsv` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `data/a1400_passband.ecsv` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `data/a2000_passband.ecsv` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

Use the exact ECSV `f` nodes in GHz and `throughput` values as supplied; do not
peak- or area-normalize the curves. Use top-of-atmosphere
`S_nu proportional to nu^alpha` for `alpha={-1,0,2,4}`. Linearly interpolate
AM transmission only between adjacent 10-MHz nodes, with no spectral
extrapolation. On the exact ECSV nodes use composite-trapezoid quadrature and

```text
T_eff = integral R_nu S_nu T_nu dnu / integral R_nu S_nu dnu
lambda = -log(T_eff)
C = exp(lambda) = 1/T_eff
```

Both integrals must be finite, the denominator strictly positive, and
`0<T_eff<=1`. This energy-weighted convention is limited to this confirmation.
No FTS challenger participates after the owner's exact ECSV choice.

## Candidate representation and structural gates

Candidate bytes come from `am12_successor_operator_nodes.csv`, SHA-256
`8005c8ae1d4ab1c8de39f06a632d76d3e8f248939dc63c616dd176bcbd2f6fe2`.
The only selected node identities are `tolteca_v1_a1100`,
`tolteca_v1_a1400`, and `tolteca_v1_a2000`, mapped to their same-named arrays;
every FTS or otherwise nonselected node identity is rejected.
The two lanes and two interpolation rules remain exactly those defined by the
frozen v2 driver. At each nonzero opacity anchor, elevation is the frozen PCHIP
through even EL20--80 node values, evaluated only on confirmation EL25--80.
The clear-to-q25 segment remains exactly linear in line-of-sight optical depth.
Above q25 the registered operator is either piecewise linear or PCHIP in
opacity. No extrapolation or coefficient refit is permitted.

Internal candidate-node integrity is checked directly at every frozen source
node, including EL20, EL22, and EL24, without routing those internal checks
through the public confirmation wrapper. The public wrapper independently
rejects every elevation below EL25. Thus source-node identity does not broaden
the declared public support.

For each of the 48 candidate/array/alpha identities, evaluate dense structural
support. The opacity grid is the sorted unique binary64 union of
`numpy.linspace(0,q75,1001)`, all four q anchors, and all six achieved
confirmation coordinates. The elevation grid is the sorted unique binary64
union of `numpy.linspace(25,80,551)`, the exact endpoints, and every frozen
even candidate-node elevation inside the closed support. Required properties
are:

- finite values, nonnegative line-of-sight optical depth, positive
  transmission and correction;
- opacity monotonicity and elevation monotonicity within `1e-12` numerical
  tolerance;
- exact anchor residual at most `1e-12` and exact q0-q25 linear-identity
  residual at most `1e-12`;
- relative correction continuity across internal opacity knots at most
  `1e-10`; and
- the fail-closed support and ALIGN probes above.

At q25 and q50, continuity is evaluated at every structural elevation using
the two pairs `(nextafter(knot,-inf), nextafter(knot,+inf))` and
`(knot-1e-12*(q75-q0), knot+1e-12*(q75-q0))`. The reported residual is the
maximum absolute relative correction difference across both pairs.

Every structural property is a separate gate; numerical fidelity cannot hide
a structural failure.

## Representation metric, threshold, and success rule

For each new direct truth grid, array, alpha, and candidate, compute without
rounding:

```text
signed fractional correction error = C_operator / C_direct_AM - 1
absolute fractional correction error = abs(C_operator / C_direct_AM - 1)
```

The provisional fidelity threshold is inclusive `<=0.01`. It measures only
the numerical representation of the declared AM calculation. It is not a
one-percent physical photometry claim and does not replace the later 5--10%
absolute-flux or approximately 5% repeatability gates.

The 48-row decisive metric table has one row for every candidate, array, and
alpha, summarized across all 896 registered truth grids. It records count,
signed minimum/maximum and bias, RMS, p95 absolute, median absolute, maximum
absolute error, and a deterministic worst location. The location tie-break is
lexical on `(coordinate_id, truth_profile, elevation_deg, array, alpha)` among
exactly equal unrounded binary64 absolute errors. The complete 43,008-row table
retains all inputs, truth, prediction, signed error, and raw/sidecar digests.
Signed bias is `numpy.mean(error)`; RMS is
`sqrt(numpy.mean(error**2))`; p95 absolute error is
`numpy.quantile(abs(error),0.95,method="linear")`; and median absolute error is
`numpy.median(abs(error))`, including the arithmetic mean of the two central
sorted values for even `n`. All gates use unrounded binary64 values; serialized
binary64 values use Python format `.17e`, meaning 17 digits after the decimal
point and 18 significant digits.

The primary verdict is `primary_confirmation_gate_pass` if and only if:

1. every authority, executable, AMC, passband, predecessor, execution-context,
   cache, sidecar, and warning binding passes;
2. the tuple anti-join is exactly zero;
3. all 16 profile/coordinate scale cases are conditioned within their frozen
   bounds;
4. all primary structural checks pass;
5. the primary maximum error is at most 0.01 for every array/alpha row and
   across their union; and
6. coverage is exactly 16 scale cases, 896 full grids, and 43,008 comparison
   rows with no missing, unexpected, or duplicate key.

The verdict is `confirmation_invalid` if any of G0, G1, G2, G4, or G6 is
false, or execution cannot produce complete registered evidence. It is
`primary_confirmation_gate_fail` only when those five validity gates are true
and primary G3 and/or G5 is false. It passes only when all G0--G6 are true.
For valid pass/fail evidence, software status is `pass_clean` when every run is
status 0 and `pass_warning_bearing_evidence` when at least one exactly admitted
status-1 run exists. Invalid evidence has software and numerical status
`invalid`; numerical status is otherwise `primary_pass` or `primary_fail` in
lockstep with the verdict. Secondary candidates receive the same metrics and
structural checks, but cannot change the primary verdict. Results cannot tune
candidates or gates.

## Output schema, deterministic artifacts, and replay

Canonical JSON uses sorted keys, compact separators, UTF-8, and one trailing
LF. CSV columns and rows are fixed by the driver, ordered lexically by their
declared identity key, use Unix LF, and serialize binary64 metrics with 17
digits after the decimal point (`.17e`, 18 significant digits). The driver
digest above is the authoritative
low-level column/order/schema implementation where this protocol names an
algorithm rather than enumerating code statements; changing it creates a new
study version. Generated outputs are exactly:

The exact CSV field orders and row sort keys are:

```text
scales fields:
coordinate_id,interval,fraction_numerator,fraction_denominator,truth_profile,
requested_tau225,achieved_tau225,coordinate_residual,
analytic_transmission_decimal,target_transmission_literal,
achieved_transmission_el80,h2o_scale_decimal,h2o_scale_hex,
negative_lower_tau_half_step,positive_upper_tau_half_step,
plateau_lower_outside_scale,plateau_lower_inside_scale,
plateau_upper_inside_scale,plateau_upper_outside_scale,
trace_path_relative_to_cache,trace_sha256
sort: coordinate_id,truth_profile

run-inventory fields:
run_class,coordinate_id,truth_profile,cache_id,stage,scale_decimal,elevation_deg,
zenith_angle_deg,frequency_min_centi_ghz,frequency_max_centi_ghz,argv_json,
working_directory_role,profile_sha256,am_executable_sha256,omp_threads,
locale_json,execution_host_json,execution_context_sha256,am_cache_shard_index,
am_cache_shard_count,raw_path_relative_to_cache,raw_sha256,
sidecar_path_relative_to_cache,sidecar_sha256,return_code,am_version_identity,
numeric_row_count,unresolved_line_warning_count,
unresolved_column_warning_line_count,unresolved_summary_warning_line_count,
other_warning_line_count,error_line_count,scale_trace_path_relative_to_cache,
scale_trace_sha256,scale_trace_evaluation_index,scale_trace_role
sort: run_class,coordinate_id,truth_profile,elevation_deg,cache_id

comparison-row fields:
coordinate_id,interval,fraction_numerator,fraction_denominator,truth_profile,
requested_tau225,achieved_tau225,coordinate_residual,h2o_scale_decimal,
h2o_scale_hex,target_transmission_literal,achieved_transmission_el80,
elevation_deg,airmass,candidate_id,candidate_role,lane,operator,passband_set_id,
passband_id,array,alpha,direct_effective_transmission,
direct_line_of_sight_optical_depth,direct_extinction_correction,
operator_line_of_sight_optical_depth,operator_extinction_correction,
signed_fractional_correction_error,absolute_fractional_correction_error,
raw_sha256,sidecar_sha256
sort: candidate_id,array,alpha,coordinate_id,truth_profile,elevation_deg

metric fields:
candidate_id,candidate_role,lane,operator,passband_id,array,alpha,n,
signed_min_fractional_correction_error,signed_max_fractional_correction_error,
signed_bias_fractional_correction_error,rms_fractional_correction_error,
p95_absolute_fractional_correction_error,
median_absolute_fractional_correction_error,
max_absolute_fractional_correction_error,gate_threshold,gate_pass,
worst_coordinate_id,worst_truth_profile,worst_requested_tau225,
worst_achieved_tau225,worst_elevation_deg,
worst_signed_fractional_correction_error
sort: candidate_id,array,alpha

physical-metric fields:
candidate_id,candidate_role,lane,operator,passband_id,array,alpha,
all_evaluated_quantities_finite,minimum_line_of_sight_optical_depth,
minimum_lambda_tau225,minimum_lambda_elevation_deg,
maximum_effective_transmission,minimum_extinction_correction,
minimum_tau_direction_delta,maximum_elevation_direction_delta,
tau_wrong_way_step_count,elevation_wrong_way_step_count,
maximum_tau_wrong_way_fractional_correction_excursion,
maximum_elevation_wrong_way_fractional_correction_excursion,
maximum_internal_anchor_absolute_residual,maximum_low_segment_absolute_residual,
maximum_relative_correction_continuity_residual,positivity_pass,domain_pass,
tau_monotonicity_pass,elevation_monotonicity_pass,continuity_pass,
fail_closed_pass,internal_anchor_pass,exact_low_segment_pass,
physical_contract_pass
sort: candidate_id,array,alpha
```

The scale, comparison, metric, and physical tables contain exactly 16, 43,008,
48, and 48 data rows. Run-inventory cardinality equals the complete 16 scale
traces plus exactly 896 full-grid runs and is checked by anti-join rather than
preregistering the solver's data-dependent number of bracket evaluations.

1. `am12_el25_confirmation_execution_context.json`;
2. `am12_el25_confirmation_scales.csv` (16 rows);
3. `am12_el25_confirmation_run_inventory.csv` (all scale and full-grid runs);
4. `am12_el25_confirmation_rows.csv` (43,008 rows);
5. `am12_el25_confirmation_metrics.csv` (48 decisive rows);
6. `am12_el25_confirmation_physical_metrics.csv` (48 rows);
7. `am12_el25_confirmation_coverage.json`;
8. `am12_el25_confirmation_decision.json`, validated against the frozen
   result schema;
9. `AM12_EL25_CONFIRMATION_REPORT.md`; and
10. `am12_el25_confirmation_manifest.json`, binding the preceding nine
    artifacts, the external raw-evidence inventory, this protocol, the
    preregistration commit, and every upstream identity.

`SHA256SUMS` must bind every regular file directly in this package directory
except `SHA256SUMS` itself, exactly matching the existing package convention.
A cache-only replay must regenerate all ten artifacts byte-for-byte without
invoking AM, and a separate check mode must compare them byte-for-byte without
writing. The complete package verifier must then pass in proportion to the
f401 replay.

The report and machine decision must keep software correctness, numerical
representation fidelity, and observational performance as separate statuses.
Observational performance is always
`not_evaluated_required_before_production` in this study.

## Stop boundary

After the result and verification commits, return the preregistration commit,
result commit, artifact digests, exact primary maximum and location, gate
verdict, and any blocker to the coordinator. Do not adopt an operator,
authorize the confirmation support operationally, modify application code,
contact Unity, begin repair, launch re-audit, or rewrite predecessor evidence.
