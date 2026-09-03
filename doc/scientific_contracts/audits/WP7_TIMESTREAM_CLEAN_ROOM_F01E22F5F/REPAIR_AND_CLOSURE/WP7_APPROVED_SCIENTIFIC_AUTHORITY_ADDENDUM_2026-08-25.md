# WP-7 Approved Scientific-Authority Addendum

Status: **approved scientific authority; sanitized readability view; no
implementation-conformity or finding-closure claim**

Approval date: `2026-08-25`

Scientific owner: Grant Wilson

Governing disposition:
[`WP7_SCIENTIFIC_OWNER_DISPOSITION_2026-08-25.md`](WP7_SCIENTIFIC_OWNER_DISPOSITION_2026-08-25.md)

This addendum reproduces only the approved scientific content and precedence
needed by a fresh clean-room auditor. The governing disposition and repair
authority manifest bind its exact bytes. It does not expose prior findings or
require any audit conclusion.

## 1. Native paired-readout interface precedence

The exact interface bytes at SHA-256
`f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`
are approved together with their README, source manifest, owner-decision
record, and scientific-owner approval record.

The approval record and source manifest promote those exact retained interface
bytes and supersede only the interface's embedded pre-promotion status and
candidate-only authority wording. They do not change any scientific interface
semantic, transform, convention, runtime payload, or implementation claim.

## 2. CAL numerical authority

The following existing numerical objects are admitted at their exact frozen
identities. They shall not be regenerated from prose or replaced by a similarly
named object.

| Object | Required SHA-256 |
| --- | --- |
| Atmosphere machine contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Atmosphere node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| TolTECA-v1 passband set | `5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433` |

The passband-set identity is the approved canonical member-name/digest
aggregation over these four members in lexical member order:

| Member | SHA-256 |
| --- | --- |
| `index.yaml` | `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5` |
| `data/a1100_passband.ecsv` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `data/a1400_passband.ecsv` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `data/a2000_passband.ecsv` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

### 2.1 Detector-time WVR interpolation

Method identifier: `cal_wvr_tau225_linear_detector_time_v1`.

1. The input is the sequence of producer-valid LMT WVR `tau225` records in the
   current observation's `tel*.nc` stream, using each record's source time.
   The detector-reference sample time is mapped to that time basis by the
   approved SCI-ALIGN mapping. Records from another observation are never
   admitted.
2. At an exact matching source time, return the source value exactly. Multiple
   byte-identical records at one time collapse to one record; conflicting
   duplicate values make opacity unavailable at that time and for any bracket
   that would use it.
3. Otherwise, let `(t0, tau0)` and `(t1, tau1)` be the consecutive valid source
   records that bracket `t`, with `t0 < t < t1`. Evaluate
   `w = (t - t0) / (t1 - t0)` and
   `tau225(t) = tau0 + w * (tau1 - tau0)`.
4. Evaluate in IEEE-754 binary64, round-to-nearest ties-to-even, in the written
   subtraction, division, subtraction, multiplication, addition order. Each
   operation rounds separately; contraction to a fused multiply-add is not
   permitted.
5. The WVR source's declared validity governs whether a bracket and its gap
   are admissible. This rule adds no arbitrary elapsed-time threshold.
6. Do not extrapolate, clamp, hold an endpoint, inherit a prior-observation
   value, or interpolate through an invalid or conflicting record.
7. Record the two source-record identities, source times and values, mapped
   detector time, interpolation weight, result, source-validity interval, and
   method identifier. An exact-match result records its single source record
   and equality disposition.

### 2.2 Unavailable telescope opacity

Method identifier: `cal_wvr_tau225_unavailable_v1`.

1. A detector sample has no admissible telescope opacity when it has neither a
   producer-valid exact-time WVR record nor a valid same-observation bracket
   under `cal_wvr_tau225_linear_detector_time_v1`.
2. Absent records, an unbracketed time, a bracket outside source-declared
   validity, a conflicting duplicate, or an unavailable time mapping yield
   `outside_supported_calibration` for the affected sample. Negative or
   non-finite opacity yields `invalid_atmosphere`.
3. The affected sample is excluded from calibrated-signal support and no CAL
   multiplier or calibrated value is emitted for it. Its upstream value may
   remain available only under its upstream identity and validity; it must not
   be passed onward or relabeled as an ordinary calibrated sample.
4. Do not substitute numeric zero, a unity correction, an observation mean or
   median, nearest-neighbor or hold-last/hold-next state, a scalar-header
   fallback, an AM climatology or profile, a configured default, or opacity
   from another observation.
5. Publish a machine-distinguishable cause from `wvr_tau225_absent`,
   `wvr_tau225_unbracketed`, `wvr_tau225_gap_outside_source_validity`,
   `wvr_tau225_conflicting_duplicate`, `wvr_tau225_negative`,
   `wvr_tau225_nonfinite`, or `wvr_time_mapping_unavailable`, together with the
   affected sample support and available source lineage.
6. Unsupported samples do not invalidate independently supported samples in
   the same observation. If no samples remain supported, CAL publishes the
   truthful no-calibrated-output state rather than an ordinary calibrated
   product. Observation-wide opacity classification cannot restore numerical
   support.

## 3. Observation-wide WVR opacity quality

Method identifier: `cal_wvr_observation_quality_mean_peak_v1`.

### 3.1 Population and observation window

1. The classified window is the closed interval from the first to the last
   detector-reference sample time belonging to the current observation,
   before CAL validity masking. A missing or non-finite endpoint, or
   `t_end <= t_start`, yields `opacity_quality_unavailable`.
2. Map that interval to the WVR source-time basis with the approved SCI-ALIGN
   authority and construct `cal_wvr_tau225_linear_detector_time_v1` from only
   the current observation's `tel*.nc` records. The complete classified
   interval must be covered. An empty source, missing bracket, disallowed gap,
   conflicting duplicate, or unavailable time mapping yields
   `opacity_quality_unavailable`; negative or non-finite required opacity
   yields `invalid_opacity_input`.
3. The ordered classifier breakpoints are the mapped observation endpoints
   plus every admitted WVR source time strictly inside the interval. Evaluate
   the approved interpolant at both endpoints. Do not resample, smooth,
   cadence-weight, fill a gap, or count detector samples as independent WVR
   evidence.

### 3.2 Summary and excursions

4. Compute the duration-weighted mean of the continuous piecewise-linear
   opacity over the complete interval. For chronological breakpoints
   `(t_i, tau_i)`, its area is the composite trapezoid
   `A = sum_i (t_(i+1)-t_i) * (tau_i+tau_(i+1)) / 2`, and
   `tau_mean = A / (t_end-t_start)`. Also record the minimum and maximum; the
   extrema occur at the breakpoints.
5. An excursion is one connected component of the interval on which
   `tau225(t) > 0.15`. Resolve threshold crossings analytically on the same
   linear segments. For a strict interior crossing of threshold `q`, evaluate
   `u = (q - tau_i) / (tau_(i+1) - tau_i)` followed by
   `t_cross = t_i + u * (t_(i+1) - t_i)`; an endpoint exactly equal to `q` is
   the crossing. Record every component's start, end, duration, and peak, plus
   total excursion duration, longest duration, duration fraction, count, and
   integrated excess `integral max(tau225(t)-0.15, 0) dt`. Partition at every
   threshold crossing and compute the excess with the same chronological
   trapezoid rule applied to the nonnegative endpoint excesses.
6. There is no additional duration, count, cadence, or fraction cutoff in v1.
   Here `momentary` has the exact combined meaning that the time-weighted mean
   remains at or below `0.15` and no instantaneous peak exceeds `0.175`.
   Excursion persistence therefore affects the class through its contribution
   to the time-weighted mean without introducing another threshold.

### 3.3 Class mapping and boundary behavior

7. Assign exactly one class in this precedence order:

   - `invalid_opacity_input` if a required opacity state is negative or
     non-finite;
   - `opacity_quality_unavailable` if the complete window or required source
     coverage cannot be resolved under items 1--3;
   - `outside_supported_opacity` if complete valid coverage exists but
     `tau_max > 0.25`;
   - `science_qualification_eligible` if `tau_mean <= 0.15` and
     `tau_max <= 0.175`; or
   - `engineering_only` for every other completely covered, finite,
     nonnegative observation with `tau_max <= 0.25`.

   Equality is inclusive at `0.15`, `0.175`, and `0.25`. A value immediately
   above a boundary takes the next less-favorable class. These are operational
   quality classes, not achieved atmosphere-fidelity,
   observational-performance, `science-qualified`, or `calibrated-science`
   claims.

### 3.4 Determinism and output record

8. Parse source values, mapped times, and the decimal threshold strings
   `"0.15"`, `"0.175"`, and `"0.25"` into IEEE-754 binary64 with correct
   round-to-nearest, ties-to-even conversion. Evaluate interpolation,
   crossings, segment areas, chronological accumulation, duration, division,
   and comparisons in binary64 round-to-nearest ties-to-even with the written
   operation order and a rounding step after every elementary operation.
   Fused contraction and reassociation are prohibited. Exact binary64 equality
   receives the inclusive disposition above. A non-increasing breakpoint,
   non-finite intermediate, or summary outside the finite input range yields
   `opacity_quality_unavailable` with cause `classifier_numeric_failure`.
9. Publish the classifier and interpolation identifiers; observation and
   source identities; mapped interval; ordered input-record identities and
   values; coverage and validity disposition; breakpoint count; minimum,
   maximum, mean, duration, and trapezoid area; the complete excursion
   inventory and aggregate statistics from item 5; threshold constants;
   precision rule; final class; and machine-distinguishable causes.

Sample-level numerical atmosphere support remains independent. An
observation-wide class neither fills an unsupported sample nor authorizes
numerical extrapolation. Conversely, an unavailable, invalid, or
outside-supported observation class does not erase independently supported
sample-level CAL results; their validity and limitations remain explicit.

## 4. RTC logical-stream terminal completion

The terminal endpoint is completion of the consumer-neutral logical RTC output
stream over the declared observation or processing domain, together with
finalization of the RTC facts that genuinely have observation-level scope.

The logical stream is the ordered sequence of conditioned sample outputs plus
the RTC-owned facts needed to interpret them. Its elements may be produced and
consumed incrementally. They need not all coexist in memory, on disk, or in
one file, table, archive, or observation-sized object. “Complete” describes
scientific-content and lifecycle completion, not physical materialization or
serialization. The terms “atomic bundle,” “publish,” and “export” mean
successful availability and completion of that logical content and its
required facts; they do not prescribe a storage form.

Normal operation may partition the stream into chunks, scans, or other
engineering units. An arbitrary chunk boundary is not scientific support. If
an RTC operator is defined over a scan, segment, or other declared domain, its
chunked execution must preserve the same domain-level scientific result as
non-chunked execution, subject only to the operator's declared numerical
tolerance. RTC may retain the state, overlap, guards, learned parameters, or
deferred decisions needed across implementation chunks. A scan boundary has
scientific force only where the governing operator contract assigns it.

The boundary distinguishes:

1. downstream-visible incremental conditioned outputs and their required
   interpretive facts;
2. temporary RTC state used to preserve domain-level semantics across chunks,
   which is not thereby a persistent product; and
3. optional materialization requested explicitly for validation, audit,
   diagnostics, user output, or a separately approved downstream contract.

Persist only facts required by the RTC contract. Do not infer mandatory
per-chunk sidecars, per-sample provenance records, or repeated observation
history. Observation-level facts are finalized once at the terminal boundary
unless a governing requirement explicitly assigns a different cadence.

RTC owns the scientific identity, ordering, support, validity, response, and
lineage facts of its logical outputs. Each downstream consumer owns its own
admission and acceptance requirements. The PTC-disabled terminal route
therefore completes successfully without CAL, PTC, or MAP, claims no
external-consumer acceptance, and requires no unnamed consumer.

If an external handoff is placed inside a governed route later, its separately
approved consumer contract must name the consumer and define the admitted
logical-stream subset, cadence or grid, coordinate and support semantics,
response, uncertainty, lineage, any serialization, failure behavior, and
acceptance criteria.

