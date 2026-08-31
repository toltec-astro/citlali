# WP-7 RTC D2 PSD and Line Evidence Tooling

Date: 2026-08-31

Status: **bounded D2 measurement tooling implemented and locally exercised;
conforming Beammap, Science, and OOF measurements remain pending; no PSD
envelope, downsampling factor, filter, or production route is selected**

## Scope

This increment implements the first measurement layer for D2 in the accepted
[filter/downsampling certification plan](../doc/WP7_RTC_FILTER_DOWNSAMPLING_CERTIFICATION_TEST_PLAN_2026-08-30.md).
It does not change RTC numerics or the terminal route. It supplies:

- one network-scoped PSD/line evidence builder;
- one corpus aggregator for the required Beammap, Science, and OOF route
  families;
- one explicitly non-conforming legacy-TOD adapter for discovery and sizing;
- focused fail-closed synthetic tests; and
- an exact distinction between pre-decimation line evidence and later
  post-cleaning diagnostics.

## Exact execution-order finding

The current legacy RTC call path in
[`rtcproc.h`](../include/citlali/core/timestream/rtc/rtcproc.h) has a real
pre-decimation line path:

1. fixed and learned RTC line-audit notches are considered/applied when the
   pre-filter line audit is enabled;
2. configured TOD filtering and notch sections run;
3. filter-edge support is handled;
4. optional post-filter detector notches run; and
5. only then does the legacy downsampler remove samples.

The model-protected PTC line audit in
[`ptc_line_audit_impl.h`](../include/citlali/core/engine/detail/ptc_line_audit_impl.h)
runs later at the PTC boundary. It is useful established line-strategy
evidence, but it cannot by itself protect a line before RTC decimation. The D2
artifact therefore permits a foldable line to clear the ordering gate only
when an explicit interval declares both:

- `effective_before_decimation: true`; and
- a nonempty `operator_evidence_id` binding the claim to a realized operator
  test or trace.

Post-cleaning detections are labeled diagnostic and never produce a factor
line gate.

## Network-scoped input contract

[`rtc_filter_psd_line_evidence.py`](../tools/wp7/rtc_filter_psd_line_evidence.py)
accepts a small JSON manifest whose arrays are separate `.npy` files. This is
an offline evidence interface, not a persistent RTC TOD schema. One manifest
describes exactly one observation/network/stage and declares:

- case, route, observation, scan, network, array, and signal units;
- either `network_native` timing with a native stage or
  `legacy_rectangular_discovery` timing with a legacy stage;
- the exact cadence-domain identity, nominal interval, and allowed measured
  deviation;
- compact native occurrence, time, physical-run, and detector axes;
- one contiguous sample-by-detector signal plane;
- original validity and an explicit source-exclusion mask;
- an explicit established-line-strategy mask and any realized
  pre-decimation operator evidence; and
- hashes of the manifest and every referenced array.

The tool rejects duplicate or decreasing occurrence identities, nonfinite or
decreasing times, reused noncontiguous run identities, samples outside the
declared cadence domain, malformed axes, missing masks, an unrecognized line
strategy, and insufficient contiguous support. Invalid sentinels are inserted
between physical runs before calling the established masked Welch estimator,
so no PSD window crosses a gap or run boundary. The requested frequency grid
is fixed; independent windows may be pooled from multiple physical runs, but
the total minimum is enforced after that pooling.

The artifact records the exact estimator, Hann window, per-window median
detrend, one-sided convention, equivalent-noise bandwidth, PSD units, source
and line-mask policies, detector/window counts, cadence summary, line
inventory, and all factor `M=1..256` fold mappings for a true native prefilter
stream. It stores unmasked detector PSDs plus a separate broadband-frequency
eligibility mask; lines are not silently absorbed into or erased from the
broadband evidence.

## Aggregation sensitivity without selection

[`rtc_filter_psd_line_corpus.py`](../tools/wp7/rtc_filter_psd_line_corpus.py)
groups only native post-cleaning residual artifacts with identical array,
cadence-domain, and units. It writes all five required aggregation
alternatives—median, 90th, 95th, 99th percentile, and maximum—plus contributing
detector counts and per-case integrated broadband summaries. It also reports
maximum-to-95th-percentile sensitivity and the worst available prefilter line
gate for every factor.

The corpus result remains explicitly unselected. It reports missing route
families, missing raw/residual stages, pending line masks, and legacy discovery
inputs instead of converting them into a pass. Legacy artifacts never enter a
native aggregation group.

## Legacy discovery adapter and real-file smoke test

[`export_legacy_tod_psd_line_discovery.py`](../tools/wp7/export_legacy_tod_psd_line_discovery.py)
exists only to exercise existing local products while the conforming native
producer is built. It hard-codes `legacy_rectangular_discovery`, uses legacy
stage names, records the source file hash and embedded filter/downsampling/line
configuration, and supplies no source or line-mask authority.

The adapter and analyzer were exercised on the local pointing-152391 PTC TOD,
network 7. The compact result truthfully reported:

- disposition `discovery_only_non_native_timing`;
- array `a1400`, 12 published physical scans, 3,468 selected rows;
- observed rectangular-container rate about `61.03558 Hz` even though the
  source header retains `SAMPRATE=122.0703125 Hz`;
- embedded `CONFIG.TODFILTERED=1`, `CONFIG.DOWNSAMPLED=1`, and line audit
  disabled;
- 279 of 420 detector rows with enough fixed-grid contiguous support; and
- four detector-local line clusters, labeled
  `diagnostic_only_postcleaning_stream` with no pre-decimation factor gate.

Those facts demonstrate why an existing rectangular TOD cannot be relabeled
as the required native-rate prefilter or cleaned-residual evidence. They are a
tool/data-shape smoke test only and are not committed as D2 science evidence.

## Focused validation

The four focused modules currently provide 26 deterministic cases covering:

- native versus legacy timing disposition;
- independent network time origins;
- exact run-boundary separation;
- declared cadence-domain enforcement;
- fixed-grid support rejection;
- source and established line-mask requirements;
- established detector line discovery;
- protected and unprotected foldable lines;
- post-cleaning ordering non-authority;
- source-order guards for prefilter-before-downsample and later PTC audits;
- byte-stable artifacts;
- corpus route completeness, duplicate identity, legacy exclusion, and
  aggregation alternatives; and
- an in-process synthetic legacy NetCDF adapter exercise.

Repository verification also passes:

- the local `citlali_cli` build;
- all 894 runnable CTests, with the one established disabled test unchanged;
- all 207 baseline-tool tests; and
- the complete required config preflight: 129 unit tests, all four mode kits,
  8/8 compact-compatibility cases, complete surface coverage, and every
  authority/boundary audit.

## Remaining D2 work

D2 is not closed. The next implementation task is a bounded producer/observer
that exposes native-rate prefilter values and native-rate post-cleaning
residuals on each network's accepted D1 occurrence/time/run axes without
creating a new persistent TOD product. It must also supply the approved
route-specific source mask and realized pre-decimation line-operator evidence.

After that producer exists, run Beammap `148670`, Science `152390` and
`152392`, and OOF `152385`--`152387`, build the native corpus, and return any
material aggregation choice or unprotected foldable line to the owner. Do not
begin FIR design or select a factor while either issue remains unresolved.
