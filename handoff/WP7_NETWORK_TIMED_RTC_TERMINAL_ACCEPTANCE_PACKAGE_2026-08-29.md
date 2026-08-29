# WP-7 Network-Timed RTC Terminal Publication Acceptance Package

## Disposition

The bounded network-timed RTC terminal-publication implementation is complete
at exact source revision
`cddfea28f89d3ca51ba52930a82f51c270905874`. Its local repository gates and
representative observation 152390 execution pass. This package is ready for
the required fresh independent exact-SHA read-only review.

The first review of implementation `ff00ff4a6` and its v7 record returned
HOLD because finalization admitted a replay from a different context with the
same scalar identity and cardinalities, and because the validator did not pin
the exact package SHA, executable bytes, dataset slice, and representative
cardinalities. The repaired finalization retains the exact immutable context
handle and run identity. The v8 validator hard-pins the complete observation
152390 witness and requires explicit expected source and executable hashes;
it can also hash the executable file directly. The v7 record is therefore
historical and superseded.

This is not complete-successor qualification, production activation, or
authorization for a nonidentity RTC method. The provisional
integration-center native timing assignment remains calibration-pending.

The retained
[v8 observation 152390 record](WP7_NETWORK_TIMED_IDENTITY_RTC_ACCEPTANCE_152390_V8_2026-08-29.json)
has SHA-256
`c54004c9ec64a9a4a83d0e0c2abd5889efc8d3c827db505b8b3c7bd64556d3d4`
and passes
[`verify_identity_rtc_acceptance.py`](../tools/wp7/verify_identity_rtc_acceptance.py).
The executable has SHA-256
`82585320cff3c12df04a2b61b377d55f81932f9cca3defbde77fcd346a4bba13`.

## Bounded claim

For the exact `M=1`, no-despike, no-level-shift, no-filter witness:

- ordinary RTC consumes one immutable application context over native
  per-network paired `x/r` occurrences;
- one exact-cover engineering schedule enters context admission, Learn,
  Resolve, Apply, finalization, and publication exactly once;
- the terminal product preserves the native network occurrence, reconstructed
  time, primitive interval, detector identity, `x/r` values, support,
  independent member validity and causes, and conservative pair-wide
  disposition;
- a compact finalization binds completion to the exact admitted context
  handle, run identity, and logical occurrence and detector-occurrence
  cardinalities;
- a compact realization records what the immutable plan realized;
- one complete product is atomically published to an inspectable, no-replace,
  in-memory slot; and
- RTC owns no numerical plane and publishes no persistent RTC TOD schema.

Ordinary RTC does not request, construct, call, or depend on a cross-network
common analysis grid. No CAL, VAL, PTC, MAP, production-route activation, or
nonidentity numerical method is included.

## Representative evidence

The real-data run uses the approved observation 152390 APT bundle, all 11
TolTEC networks, each bound raw file and Tune report, the exact
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1` artifact, and the
owner-assigned provisional native-occurrence support artifact. The slice is
native rows 20000 through 22047 on each network.

The run reports:

- 11 networks, 5,518 detectors, 22,528 native occurrences, and 11,300,864
  native detector occurrences;
- all 22,528 native occurrences retained on their originating network axes;
- 22,601,728 bitwise `x/r` comparisons at paired ingress and another
  22,601,728 through the RTC terminal product;
- 11,300,864 identity, pair-disposition, and causal-evidence comparisons;
- 22,528 native-time, interval-support, and representative-native
  correspondence comparisons;
- one two-partition execution compared with the single-partition execution
  across all 11,300,864 detector occurrences and the exact realized operator;
- 227,328 compact evidence events occupying 3,637,248 bytes, zero dynamic plan
  bytes, and zero RTC-owned numeric bytes; and
- zero scientific, value, identity, validity/cause, support, timing,
  correspondence, operator, or chunk-partition mismatches.

The primary paired-ingress-through-terminal-publication execution measured
0.363 seconds wall and 0.226 seconds CPU. Process-lifetime peak RSS was
607,567,872 bytes; that is a harness/process measurement, not route-local
allocation.

The record explicitly states:

- `exact_context_handle_binding_verified = true`;
- `rtc_timing_scope = network-specific`;
- `common_analysis_grid_requested = false`;
- `persistent_rtc_tod_published = false`;
- interval identity
  `wp7-provisional-integration-center-152390-v1:network-native-intervals`;
- producer-defined `kidscpp-xs` and `kidscpp-rs` coordinate meanings; and
- validity domain `delivered-finite-payload-and-producer-validity`.

The interval assignment is an owner-authorized engineering assignment for
this gate, not a claim that producer timestamp semantics have been calibrated.
Its disposition remains
`replace_with_calibrated_producer_relation_when_available`.

## Local gates

The exact implementation revision passed:

- 139 focused WP-7 and SCI-ALIGN CTests;
- all 870 runnable repository CTests, with the one established disabled test
  unchanged;
- all 207 baseline-tool unit tests;
- all 22 v8 acceptance-validator unit tests;
- all 129 required config unit tests and the complete config preflight audits;
- public-header isolation compilation; and
- the exact clean-source and dependency-state acceptance build gate.

The exact acceptance build verified:

- Citlali source revision and executable revision equal the full candidate
  SHA;
- no tracked, staged, untracked, or ignored source-tree content contaminated
  the executable;
- design commit `46824f7de` and ALIGN repair `d55deefb3` are ancestors;
- Kidscpp and Tula match the approved base revisions plus checked-in local
  build patches; and
- the runner executable hash matches the hash written into the record.

Validate the retained record with:

```sh
$HOME/tolteca/bin/python -B tools/wp7/verify_identity_rtc_acceptance.py \
  handoff/WP7_NETWORK_TIMED_IDENTITY_RTC_ACCEPTANCE_152390_V8_2026-08-29.json \
  --expected-source-revision cddfea28f89d3ca51ba52930a82f51c270905874 \
  --expected-executable-sha256 82585320cff3c12df04a2b61b377d55f81932f9cca3defbde77fcd346a4bba13 \
  --executable build/bin/citlali_wp7_identity_rtc_acceptance
```

The command rejects a substituted source revision or record hash even when the
record remains internally self-consistent, and rejects an executable whose
bytes do not match the exact package expectation.

## Independent review boundary

The fresh reviewer should evaluate exact implementation revision
`cddfea28f89d3ca51ba52930a82f51c270905874` together with the v8 record and
this package. The review should determine whether:

1. terminal completion is truthfully context-bound and fail-closed;
2. publication is atomic, no-replace, in-memory, and inspectable;
3. finalization and realization remain compact rather than duplicating the
   product, support, or provenance history;
4. network-specific occurrences and times remain authoritative and no
   common-analysis relation enters ordinary RTC;
5. the representative evidence validates the complete bounded claim without
   circular expectations; and
6. failure behavior, exact source/dependency binding, memory claims, and
   unchanged out-of-scope behavior are adequately covered.

Acceptance and readiness remain pending until that exact-SHA independent
review returns its disposition.
