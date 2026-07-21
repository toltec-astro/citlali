# NGC4449 Full-Reduction Citlali Investigation

Date: 2026-07-21

This note records the first full NGC4449 reduction investigation from
`~/work_toltec/local_data/2025-C1-COM-01/NGC4449`. It separates observed facts,
inferences, implemented candidate corrections, and validation still required.
It is not an accepted science checkpoint.

## Executive Findings

The run is scientifically promising, but it exposed four Citlali contract
failures:

1. Fruit-loop feedback was enabled for ten iterations with every selection
   gate disabled. Iterations 2--9 reproduced the maps to roundoff while later
   iterations consumed about 6.4 hours.
2. Products named `sig2noise` were emitted without empirical noise products.
   They are signal multiplied by the square root of formal mapmaker weight, not
   calibrated statistical-significance maps.
3. A 200,000-row diagnostic cap also truncated the operational learning state.
   The cap filled in iteration 0, making learning depend on observation and
   iteration order and leaving observation 152433 with no learned sample masks.
4. Expected policy outcomes were logged once per detector or optional field,
   producing thousands of warnings that obscured run-level QA.

The candidate correction rejects statically empty fruit-loop requests,
reserves S/N product names for empirically calibrated estimators, separates
effective learned state from bounded diagnostic history, and aggregates the
dominant warning classes. A full Unity science run remains required because
the uncapped effective learning state intentionally changes flags in this
workload.

## Fruit-Loop No-Op

The merged request had fruit loops enabled with `max_iters: 10`, but
`sig2noise_limit`, every `array_flux_limit`, `peak_fraction_limit`, and
`local_snr_floor` were all zero. Noise-product generation was also disabled.
There was therefore no active pixel-selection mechanism. The maps from
iterations 2--9 differ only at approximately `2e-16` relative scale, and the
runtime convergence flag was never set. Iterations 3--9 alone consumed about
6 hours 23 minutes of the 8 hour 55 minute run.

Candidate contract:

- enabled feedback requires at least two iterations;
- at least one flux, empirical-S/N, peak-fraction, or local-S/N gate must be
  active;
- a nonzero S/N gate requires enabled empirical noise products and a positive
  realization count;
- missing empirical RMS for an active S/N gate is fatal rather than silently
  disabling that gate;
- the main noise-variance FITS HDU carries `MEDRMS`, so feedback does not
  require writing every noise realization; and
- actual detector-sample feedback applications are counted, logged, and a
  source-model iteration that applies zero samples fails before another
  iteration can start.

This prevents the exact NGC4449 failure before reduction work starts. It does
not yet implement a scientific convergence statistic for nonzero feedback;
that remains explicit debt.

## What The Reported S/N Maps Were

With empirical noise products disabled, the writer used the fallback

`formal_standardized_signal = signal * sqrt(formal_mapmaker_weight)`.

Formal mapmaker weight describes the mapmaking variance model. It does not
measure the realized residual distribution or correlated structure after TOD
cleaning and spatial filtering. Consequently the NGC4449 raw standardized-map
widths were approximately 2.23, 2.08, and 1.69, while the filtered widths were
approximately 19.30, 20.03, and 7.99. Those widths are direct evidence that the
planes cannot be interpreted as unit-normal statistical significance.

The current empirical pixel estimator is

`pixel_snr = signal * sqrt(empirical_weight)`,

where jackknife realizations calibrate empirical weight. The filtered
point-source estimator is

`point_source_snr = point_source_flux / point_source_uncertainty`.

The fruit-loop configuration field `sig2noise_limit` has a related but
distinct scalar estimator: map signal divided by the per-array median RMS of
the jackknife realizations. It is an empirical selection statistic, not the
per-pixel S/N FITS plane.

Candidate contract:

- only empirical estimators are written under `sig2noise*` names;
- when empirical products are absent, the fallback is written as
  `formal_standardized_signal_*` with estimator type
  `formal_weight_standardized` and a description stating that it is not a
  statistical-significance map;
- the immutable phase-4 v1 contracts remain historical, and v2 check entries
  describe the truthful successor schema.

## Learning State: Data Model And Growth

### Fundamental unit

A sample-mask event is not a single sample and not a persistent detector
classification. It is a detector-time interval associated with:

- observation and zero-based scan;
- stable detector UID plus array/network diagnostics;
- production stage and reason;
- zero-based raw or PTC sample bounds;
- iteration and optional score/source-protection metadata; and
- the application stage (`pre_rtc` or processed-stage).

When applied, an interval expands into detector-sample flag proposals. A
detector-penalty event is a scan-local detector or network action with factor,
score, producer, and reason. Busy-network summaries and application summaries
are diagnostics, not learned masks.

### Correct interpretation of the NGC4449 counts

The final CSV retained 200,000 sample-mask events and reported 503,612 dropped
events. Iteration 0 generated 351,806 events: the first 200,000 were retained
and 151,806 were dropped. Iteration 1 generated another 351,806 events after
the diagnostic/operational vector was already full, so all were dropped. All
retained sample-mask rows consequently say iteration 0.

The retained rows were distributed as follows:

| Observation | Retained sample-mask events |
| --- | ---: |
| 152390 | 23,854 |
| 152392 | 22,106 |
| 152419 | 57,225 |
| 152431 | 96,815 |
| 152433 | 0 |

RTC produced 142,771 retained rows and PTC produced 57,229. This is direct
order dependence caused by the shared cap, not evidence that 152433 contained
no pathology.

The previously quoted “883 masks applied” is not a mask count. It is the number
of sample-mask application-summary calls in a later iteration. Across those
calls, all 200,000 retained records matched and proposed 994,316 unique
detector-sample flags: 933,122 new and 61,194 already flagged. Similarly, 837
detector-exclusion application calls considered 4,296 candidate records across
stages and proposed 214,025,661 sample flags, most already flagged. The learned
state is therefore much larger than a few thousand actions.

### Unique information in the retained portion

All 200,000 full diagnostic rows are distinct. Ignoring explanatory reason
while retaining operational identity leaves 197,990 unique event keys. Online
union by observation, scan, application stage, and detector UID reduces the
retained sample masks to 178,449 disjoint intervals. The summed intervals cover
1,032,673 detector-samples before union and 994,316 after union.

Thus only about 10.8% of retained interval rows and 3.7% of their sample span
are redundant. The append-only representation did repeat whole learning
iterations, but the within-iteration state contains substantial real interval
complexity; replacing 200,000 with a larger arbitrary cap would not solve the
architecture. The dropped 503,612 events were not serialized, so their unique
interval content cannot be measured exactly from this run.

Detector penalties show clearer duplication: 2,486 diagnostic events reduce
to 1,243 effective records, corresponding to repeated learning iterations.

### Candidate architecture

Operational learned state and diagnostic event history now have distinct
owners:

- effective sample masks are an online interval union keyed by observation,
  scan, application stage, and detector UID;
- effective detector penalties are reduced to one record per scientific action
  identity;
- the operational collections are not subject to the diagnostic cap;
- `max_records_per_type` limits only the optional CSV event history;
- disabling diagnostic output does not disable collection or application of
  operational learning; and
- application reads only the effective state.

This removes cap-induced observation ordering and prevents a repeated learning
iteration from growing operational state when it carries no new intervals.
Memory still scales with the number of disjoint learned intervals, which is the
actual model complexity. The next full run must record effective interval
counts and peak memory so that a natural resource bound can be designed from
evidence rather than another event-count ceiling.

## Warning Stream

The final iteration contained 9,671 warnings. Of these, 9,448 were per-detector
messages saying a local-residual despike proposal exceeded its safety cap; 140
were missing optional telescope/header fields; 55 reported `0/N` samples not
aligned; and 20 concerned noise/fruit-loop behavior.

Candidate logging policy:

- do not warn for zero unaligned samples;
- keep individual absent optional telescope fields at debug level and emit one
  per-file aggregate warning;
- keep individual rejected despike proposals at debug level and emit one
  scan-level informational guard summary; and
- fail active fruit-loop S/N requests when empirical RMS is unavailable rather
  than warning and changing the requested algorithm.

Shape mismatches, invalid numerical state, required missing input, and I/O
failure remain warnings or fatal errors according to their existing contracts.

## Housekeeping Correlation Sidecar

The follow-up review found no array-temperature channel with a statistically
credible run-wide association after correcting the broad channel/derivative
screen. Temperature level also covaries strongly with observation order, so a
simple level correlation is confounded by elapsed time. A few of the most
severe network pathologies did coincide with point changes in PT2, mixing-
chamber, or focal-plane thermometry, but the approximately 60-second HK cadence
is too coarse to establish onset or causality.

The candidate therefore adds diagnostic evidence, not an automatic flagging
rule. For each deduplicated `busy_network_pathology` action it writes
`learning_housekeeping_iter_N.csv` rows for seven focal-plane/array
thermometers and six dilution-fridge channels. Each row identifies the
observation, zero-based scan, network, array, pathology score, PTC chunk
midpoint Unix time, HK file, physical channel, kelvin unit, and explicit match
status. Successful matches publish the nearest sample, signed time offset,
absolute sample age, previous/next values, first differences, and local
three-point excursion.

An explicit `toltec_hk` input is preferred when present; otherwise the writer
looks for exactly one `toltec_hk_*_<obsnum>_*.nc` beside the TolTEC detector
files. Missing, ambiguous, malformed, unavailable-sentinel, and out-of-range
cases are represented in the sidecar rather than amplified into warning spam.
The sidecar is header-only when no qualifying pathology occurs. Its output I/O
is required when learning diagnostics are enabled, but HK availability never
changes the learned state or the science flags.

## Validation And Remaining Work

Local candidate evidence:

- `citlali_cli` builds;
- all 480 CTests pass, including focused learning, fruit-loop activation,
  realized-feedback, and map-semantics tests;
- the full config preflight passes 116 tests and all required audits; and
- all 106 baseline-tool tests, including product-contract,
  validation-profile, and science-change-ledger checks, pass;
- Unity science validation is required before acceptance.

The Unity successor run should verify:

1. the NGC4449 no-op config is rejected before execution;
2. a valid active-gate configuration produces a changing feedback model;
3. observations later in input order contribute effective masks;
4. repeated learning iterations do not duplicate effective intervals or
   penalties;
5. diagnostic-cap overflow does not change the operational state;
6. applied-new-flag safety caps remain respected;
7. no S/N-named HDU exists without empirical calibration;
8. empirical products retain expected S/N semantics and FITS metadata;
9. warning counts are small enough for QA and every remaining warning is
   actionable; and
10. peak RSS and learning-state counts are recorded.

Slurm/OpenMP allocation matching is intentionally deferred to the separate
runtime-resource work requested by the project owner.
