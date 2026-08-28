# WP-7 Identity RTC Representative-Data Acceptance Package

## Status

The bounded implementation is locally constructed, synthetic gates pass, and
the owner-directed representative real paired-data gate **passes** at exact
source revision `4d0ec46ee19267b351b8a3ca015964e0400cdfd4`. The retained
[observation 152390 evidence record](WP7_IDENTITY_RTC_ACCEPTANCE_152390_2026-08-28.json)
passes the repository validator and has SHA-256
`10ef712c4f2a4ad79d227a1cf376cc0ff4f51090baea03455a681ec7befe9a26`.
This package establishes representative
execution evidence only; it does not claim independent implementation
conformity, science qualification, production readiness, or successor
activation.

The implementation branch must retain both accepted inputs as exact ancestors:

- WP-7 design commit `46824f7de`;
- ALIGN strict-half repair `d55deefb3`.

The four review boundaries are the base merge, paired native ingress/product
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
cmake --build build --target citlali_wp7_identity_rtc_acceptance -j 8
ctest --test-dir build --output-on-failure -R '^citlali::(wp7|sci_align)::'
$HOME/tolteca/bin/python tools/wp7/test_verify_identity_rtc_acceptance.py
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
build/bin/citlali --version
git rev-parse HEAD
```

The executable revision must match exact `HEAD`. Skipped required data, an
unexpected error-level record, a partial product, or a synthetic-only run does
not satisfy the representative-data gate.

## Owner-directed local invocation

Use a representative real observation whose Tune/readout producer can supply
the exact approved
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1` binding (artifact SHA-256
`f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`).
The opt-in `citlali_wp7_identity_rtc_acceptance` target performs this binding
for observation 152390. It verifies the canonical APT bundle and exact raw-file
byte counts and SHA-256 digests; discovers and hashes each Tune fit report from
the raw KIDs metadata; verifies the Tune observation, network, and acquisition
metadata; binds both the Tune and science-readout accumulation lengths into a
temporary metadata-normalized view so Kidscpp does not consult its legacy
default; reconstructs native timing from raw packet facts; runs the configured
sequential `gainlintrend` KIDs transform; atomically moves each solver's paired
`x/r` result into `PairedReadout`; and runs the identity RTC route as one full
partition and two chunks. The normalized Tune view preserves the original
numeric rows and is removed after execution; the mapping identity retains the
original Tune hash, both accumulation lengths, the adapter revision, and the
Kidscpp revision. The local Kidscpp and Tula checkouts intentionally carry the
repository's tracked macOS build overlays, so the runner also verifies both
patch artifacts and records their SHA-256 values, the full dependency base
revisions, and the acceptance executable's own SHA-256. The runner rejects any
source other than observation 152390 and requires the approved
producer-interface artifact at its recorded SHA-256.

Build the target from a clean exact candidate revision, then invoke it with the
local observation 152390 paths:

```sh
candidate_revision=$(git rev-parse HEAD)
build/bin/citlali_wp7_identity_rtc_acceptance \
  --data-dir /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --apt-manifest /Users/gwilson/work_toltec/local_data/2026-refactor/projects/SCI_ALIGN_STAGE7_NGC4449_152390/apts/v2/apt_152390_matched.apt-v2/manifest.ecsv \
  --config /Users/gwilson/work_toltec/local_data/2026-refactor/projects/SCI_ALIGN_STAGE7_NGC4449_152390/toltec_umass_edu/NGC4449/reduced/redu04/citlali_merged_config.yaml \
  --producer-interface-artifact /Users/gwilson/Documents/Codex/2026-08-26/wp7-1-clean-room-audit/work/packet/WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D/sources/doc/scientific_contracts/producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md \
  --kidscpp-build-patch patches/local/kidscpp-local-build.patch \
  --tula-build-patch patches/local/tula-local-build.patch \
  --output acceptance.json \
  --source-revision "$candidate_revision" \
  --owner-run --design-is-ancestor --align-repair-is-ancestor
```

The default representative slice is native rows 20000 through 22047. The
`--first-native-row` and `--native-row-count` options exist for bounded
diagnostic runs; a smaller smoke run is not acceptance evidence.

At the application seam, the runner constructs one `PairedReadout` from each
atomic KIDs solver `x/r` result, preserving the producer mapping handle,
detector binding, native timing, primitive occurrence intervals, independent
member validity, and support. It then invokes:

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

The focused suite plus acceptance record must demonstrate:

- all required networks and detector identities came from the admitted
  participant inventory;
- both `x` and `r` were compared bitwise with their native mapped parent at
  every mapped detector occurrence;
- occurrence identity, primitive support, member-local causes, pair-wide
  decisions and causal evidence, selected time, and representative native
  occurrence were checked at every aligned detector occurrence;
- direct `r` evidence can make `x` ineligible, direct `x` evidence can make `r`
  ineligible, and the member-local cause remains on its true coordinate;
- the identity operator has factor one, diagonal coefficients one, and cross
  coefficients zero;
- `RtcTimestream` owns no duplicate numerical plane;
- RTC-only completion is published exactly once and failure publishes no false
  completion;
- no AST interpolation and no CAL, VAL, PTC, or MAP operation was invoked; and
- no unexpected error- or critical-level record occurred.

The representative 152390 slice contains 227,106 ineligible pairs, all caused
by validity evidence present on both `x` and `r`; it does not happen to contain
a one-sided invalid pair. The focused identity-RTC tests therefore remain the
evidence for the required asymmetric `r`-to-`x` and `x`-to-`r` conservative
consequences and retained member-local causes. The real-data record verifies
the same pair-wide resolution and cause carriage exhaustively for the evidence
origins actually present in this slice.

## Recorded execution

The retained exact-revision run covers 11 networks, 5,518 detectors, 2,048
native rows, 11,300,864 native detector occurrences, and 11,289,828 aligned
detector occurrences. It compared 22,579,656 paired values and performed
11,289,828 comparisons each for identity, support, pair decision, and causal
evidence. All scientific, chunk-partition, selected-time, native-
correspondence, and out-of-scope call mismatch counts are zero.

Paired ingress reports 226,608,108 logical owned bytes, including the two
180,813,824-byte numerical planes. RTC owns zero numerical bytes. Measured
paired-ingress-through-publication time is 4.497 seconds wall and 4.411 seconds
CPU, with peak RSS 629,358,592 bytes. These are one representative local run,
not a general performance qualification.

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
