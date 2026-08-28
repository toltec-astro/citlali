# WP-7 Identity RTC Representative-Data Acceptance Package

## Status

The bounded implementation is locally constructed and synthetic gates pass.
Representative real paired-data acceptance is **pending owner execution**. This
package does not claim implementation conformity, observational performance,
science qualification, production readiness, or successor activation.

The implementation branch must retain both accepted inputs as exact ancestors:

- WP-7 design commit `46824f7de`;
- ALIGN strict-half repair `d55deefb3`.

The three review boundaries are the base merge, paired native ingress/product
semantics, identity RTC learn-consider-apply, and RTC-only in-memory route and
publication. The paired product retains each network's native occurrence axis.
Only the separately owned ALIGN relation supplies RTC's common-slot admission.
No AST, CAL, VAL, PTC, or MAP operation belongs to this acceptance run.

## Local exact-revision gates

Run from a clean checkout of the exact candidate revision:

```sh
git merge-base --is-ancestor 46824f7de HEAD
git merge-base --is-ancestor d55deefb3 HEAD
cmake --build build --target citlali_cli -j 8
cmake --build build --target citlali_wp7_timestream_test citlali_sci_align_test -j 8
ctest --test-dir build --output-on-failure -R '^citlali::(wp7|sci_align)::'
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
build/bin/citlali --version
git rev-parse HEAD
```

The executable revision must match exact `HEAD`. Skipped required data, an
unexpected error-level record, a partial product, or a synthetic-only run does
not satisfy the representative-data gate.

## Owner-run invocation

Use a representative real observation whose Tune/readout producer can supply
the exact approved
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1` binding (artifact SHA-256
`f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`).
Observation 152390 is a useful workload candidate because its existing
operational evidence covers all TolTEC networks, but the owner must confirm
that the run supplies the approved paired mapping and primitive occurrence
facts; finite arrays or matching shapes are not a substitute.

At the application seam, construct one `PairedReadout` from each atomic KIDs
solver `x/r` result, preserving the producer mapping handle, detector binding,
native timing, primitive occurrence intervals, independent member validity,
and support. Then invoke:

```cpp
citlali::pipeline::RtcOnlyProductSlot publication;
const auto outcome = citlali::pipeline::run_identity_rtc_only(
    {{run_id}, paired_readout, align_plan, first_slot, past_last_slot},
    publication);
```

Inspect `publication.snapshot()` in memory. Do not serialize a new RTC TOD
schema for this gate. Record wall time, CPU time, and peak RSS around the whole
paired-ingress-through-publication route, not only the identity view creation.

Run at least two engineering chunk partitions over the same scientific
occurrences. Product, buffer, plan-instance, and transaction identities may
differ. The exact scientific occurrence identities, `x/r` values, primitive
support, local causes, pair decisions, selected times, and representative
native correspondence must not.

## Required comparisons

The acceptance record must demonstrate:

- all required networks and detector identities came from the admitted
  participant inventory;
- both `x` and `r` were compared bitwise with their native mapped parent at
  every mapped detector occurrence;
- occurrence identity, primitive support, member-local causes, pair-wide
  decisions, selected time, and representative native occurrence were checked
  at every aligned detector occurrence;
- direct `r` evidence can make `x` ineligible, direct `x` evidence can make `r`
  ineligible, and the member-local cause remains on its true coordinate;
- the identity operator has factor one, diagonal coefficients one, and cross
  coefficients zero;
- `RtcTimestream` owns no duplicate numerical plane;
- RTC-only completion is published exactly once and failure publishes no false
  completion;
- no AST interpolation and no CAL, VAL, PTC, or MAP operation was invoked; and
- no unexpected error- or critical-level record occurred.

## Evidence record

Write one JSON record with schema
`citlali-wp7-identity-rtc-acceptance-v1` and the fields required by
[`tools/wp7/verify_identity_rtc_acceptance.py`](../tools/wp7/verify_identity_rtc_acceptance.py).
The validator intentionally requires real paired data, an owner run, exact
ancestry, full-cell comparisons, at least two chunk partitions, positive timing
and RSS measurements, and zero scientific mismatches or out-of-scope calls.

Validate it with:

```sh
$HOME/tolteca/bin/python tools/wp7/verify_identity_rtc_acceptance.py acceptance.json
```

A passing record is representative execution evidence for this bounded
identity route. Fresh independent read-only conformance review remains a
separate gate on the completed vertical increment and must assess the exact
implementation revision plus this evidence. Legacy activation and retirement
remain separate owner decisions.
