# WP-7 Identity RTC Representative-Data Acceptance Package

## Status

The bounded implementation and repository-local gates are available. The
project owner has assigned reconstructed native event time provisionally to
the integration center, with primitive duration given by raw
`AccumLen / FpgaFreq`. This is an engineering convention for the first route,
not a claim that the producer's true timestamp semantics are known.
Calibration remains explicitly pending.

The representative real paired-data gate **passes** under that assignment at
exact source revision `434919a84406ef84afaaebc1169cf0430accf3f3`.
The retained
[v4 observation 152390 record](WP7_IDENTITY_RTC_ACCEPTANCE_152390_V4_2026-08-29.json)
passes the repository validator and has SHA-256
`0c2f96047419af086656e46bd3b2922474e83f02ae1c450c466cc7b5a6839ff4`.
The v3 record remains a historical native-axis diagnostic whose gate did not
prove exact source/dependency state or independent ingress expectations. The
earlier v1 and v2 records also remain historical diagnostics and projected the
identity RTC product through ALIGN's common grid.

The exact provisional policy is the
[support assignment](WP7_NATIVE_OCCURRENCE_SUPPORT_ASSIGNMENT_2026-08-28.yaml).
The v4 evidence record must preserve its exact SHA-256, identity, provisional
status, integration-center role, and
calibration disposition. This package can therefore produce representative
execution evidence while the timing calibration is investigated; it does not
turn the assignment into producer authority and still will not by itself
claim independent implementation conformity, science qualification,
production readiness, or successor activation.

The implementation branch must retain both accepted inputs as exact ancestors:

- WP-7 design commit `46824f7de`;
- ALIGN strict-half repair `d55deefb3`.

The four review boundaries are the base merge, paired native ingress/product
semantics, identity RTC learn-consider-apply, and RTC-only in-memory route and
publication. The paired product and RTC terminal product retain each network's
native occurrence axis. This route does not consume, invoke, or publish an
ALIGN common-slot association. No AST, CAL, VAL, PTC, or MAP operation belongs
to this acceptance run.

## Local exact-revision gates

Run from a clean checkout of the exact candidate revision:

```sh
git merge-base --is-ancestor 46824f7de HEAD
git merge-base --is-ancestor d55deefb3 HEAD
cmake --build build --target citlali_cli -j 8
cmake --build build --target citlali_wp7_timestream_test citlali_sci_align_test -j 8
cmake --build build --target citlali_wp7_identity_rtc_acceptance -j 8
ctest --test-dir build --output-on-failure -R '^citlali::(wp7|sci_align)::'
ctest --test-dir build --output-on-failure
$HOME/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'
$HOME/tolteca/bin/python -m unittest tools.wp7.test_verify_identity_rtc_acceptance
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
build/bin/citlali --version
git rev-parse HEAD
```

The acceptance target refuses a dirty Citlali source tree, a dependency tree
other than the exact approved base plus checked-in patch, or an executable
revision different from exact `HEAD`. Skipped required data, an
unexpected error-level record, a partial product, or a synthetic-only run does
not satisfy the representative-data gate.

## Owner-directed local invocation

Use a representative real observation whose Tune/readout producer can supply
the exact approved
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1` binding (artifact SHA-256
`f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`).
It must also supply the checked-in
`citlali-native-occurrence-support-assignment-v1` artifact scoped to observation
152390. That artifact assigns `integration_center`, binds primitive duration to
`Header.Toltec.AccumLen / Header.Toltec.FpgaFreq`, and states that calibration
is pending and may replace or correct the relation.
The opt-in `citlali_wp7_identity_rtc_acceptance` target performs this binding
for observation 152390. It verifies the canonical APT bundle and exact raw-file
byte counts and SHA-256 digests; discovers and hashes each Tune fit report from
the raw KIDs metadata; verifies the Tune observation, network, and acquisition
metadata; binds both the Tune and science-readout accumulation lengths into a
temporary metadata-normalized view so Kidscpp does not consult its legacy
default; reconstructs native timing from raw packet facts; runs the configured
sequential `gainlintrend` KIDs transform; atomically moves each solver's paired
`x/r` result into `PairedReadout`; and runs the identity RTC route directly on
each network's native occurrence support as one full partition and two chunks.
It does not construct or pass an ALIGN common-slot plan. The normalized Tune
view preserves the original
numeric rows and is removed after execution; the mapping identity retains the
original Tune hash, both accumulation lengths, the adapter revision, and the
Kidscpp revision. The local Kidscpp and Tula checkouts intentionally carry the
repository's tracked macOS build overlays, so the runner also verifies both
patch artifacts, proves the resulting Git tree identities, and records their
SHA-256 values, the full dependency base revisions, the exact clean Citlali
revision, and the acceptance executable's own SHA-256. The runner rejects any
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
  --occurrence-support-assignment handoff/WP7_NATIVE_OCCURRENCE_SUPPORT_ASSIGNMENT_2026-08-28.yaml \
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
const auto native_spans =
    citlali::pipeline::full_native_occurrence_spans(*paired_readout);
const auto outcome = citlali::pipeline::run_identity_rtc_only(
    {{run_id}, paired_readout, native_spans},
    publication);
```

Inspect `publication.snapshot()` in memory. Do not serialize a new RTC TOD
schema for this gate. Record wall and CPU time for the primary
paired-ingress-through-publication route. Record process-lifetime peak RSS
truthfully as a harness/process measurement, not route-local allocation.

Run at least two engineering chunk partitions over the same scientific
occurrences. Product, buffer, plan-instance, and transaction identities may
differ. The exact scientific occurrence identities, `x/r` values, primitive
support, local causes, pair decisions, native reconstructed times, and
representative native correspondence must not.

## Required comparisons

The focused suite plus acceptance record must demonstrate:

- all required networks and detector identities came from APT and the admitted
  participant inventory, with an independent detector-axis comparison;
- every native occurrence interval was bound to the explicit provisional
  event-time assignment and raw duration relation;
- both `x` and `r` were compared bitwise at paired ingress and again through
  the RTC product at every native detector occurrence;
- admitted member state was independently recomputed from Tune validity and
  solver finiteness before RTC; occurrence identity, primitive support,
  member-local causes, pair-wide
  decisions and causal evidence were checked at every native detector
  occurrence, while native time, support, and representative correspondence
  were checked at every native occurrence;
- direct `r` evidence can make `x` ineligible, direct `x` evidence can make `r`
  ineligible, and the member-local cause remains on its true coordinate;
- the identity operator has factor one, diagonal coefficients one, and cross
  coefficients zero;
- `RtcTimestream` owns no duplicate numerical plane;
- RTC-only completion is published exactly once and failure publishes no false
  completion;
- the route entered native admission, Learn, Consider, Apply, and publication
  exactly once, while the route implementation headers and source guards do not
  name or invoke common-grid, AST, CAL, VAL, PTC, or MAP entry points; and
- no unexpected error- or critical-level record occurred.

The representative 152390 slice contains 227,328 ineligible pairs, all caused
by validity evidence present on both `x` and `r`; it does not happen to contain
a one-sided invalid pair. The focused identity-RTC tests therefore remain the
evidence for the required asymmetric `r`-to-`x` and `x`-to-`r` conservative
consequences and retained member-local causes. The real-data record verifies
the same pair-wide resolution and cause carriage exhaustively for the evidence
origins actually present in this slice.

## Recorded native-axis provisional-assignment execution

The passing v4 run covers 11 networks, 5,518 detectors, 2,048 rows per
network, 22,528 native occurrences, and 11,300,864 native detector
occurrences. Every native occurrence is retained and matches the provisional
support assignment. The runner compared 22,601,728 values at paired ingress,
all 5,518 admitted detector bindings, all 22,601,728 admitted member states,
and another 22,601,728 values through the RTC product. It checked identity,
decision, local causes, and causal evidence at all 11,300,864 native detector
occurrences, and checked native time, support, and representative native
correspondence at all 22,528 native occurrences. All scientific,
chunk-partition, timing, correspondence, and assigned-support mismatch counts
are zero.

Paired ingress reports 226,608,108 logical owned bytes. Compact RTC evidence
owns 3,637,248 bytes for 227,328 events. The identity plan owns no dynamic
bytes and RTC owns zero numerical bytes. Measured paired-ingress-through-
publication time is 0.318 seconds wall and 0.222 seconds CPU. Process-lifetime
peak RSS is 610,385,920 bytes. These measurements characterize this
representative local harness run rather than a general performance
qualification or route-local allocation claim.

The superseded v2 run selected 11,289,828 detector occurrences through an
ALIGN common grid and therefore omitted 11,036 native detector occurrences
from the RTC product. Its otherwise useful zero-mismatch measurements remain
historical diagnostics, not acceptance evidence for the corrected native-axis
claim.

## Prior v1 diagnostic execution (superseded as acceptance)

The retained v1 run covers 11 networks, 5,518 detectors, 2,048
native rows, 11,300,864 native detector occurrences, and 11,289,828 aligned
detector occurrences. It compared 22,579,656 paired values and performed
11,289,828 comparisons each for identity, support, pair decision, and causal
evidence. All scientific, chunk-partition, selected-time, native-
correspondence, and out-of-scope call mismatch counts are zero. It did not bind
primitive occurrence support to an explicit declared assignment or record the
pending calibration disposition, so these otherwise useful results do not
close the revised gate.

Paired ingress reports 226,608,108 logical owned bytes, including the two
180,813,824-byte numerical planes. RTC owns zero numerical bytes. Measured
paired-ingress-through-publication time is 4.497 seconds wall and 4.411 seconds
CPU, with peak RSS 629,358,592 bytes. These are one representative local run,
not a general performance qualification.

## Evidence record

Write one JSON record with schema
`citlali-wp7-identity-rtc-acceptance-v4` and the fields required by
[`tools/wp7/verify_identity_rtc_acceptance.py`](../tools/wp7/verify_identity_rtc_acceptance.py).
The validator intentionally requires real paired data, an owner run, exact
ancestry, exact clean source and dependency-tree bindings, the one approved
provisional support assignment and its SHA-256, an explicit calibration-pending
disposition, independent ingress identity/member-state checks, complete
assigned-support binding, full-cell native-axis comparisons, at least two chunk
partitions, positive timing and process-RSS measurements, exact allowed-stage
entry counts, zero plan and RTC numeric allocation, and zero scientific
mismatches.

Validate it with:

```sh
$HOME/tolteca/bin/python tools/wp7/verify_identity_rtc_acceptance.py acceptance.json
```

A passing record is representative execution evidence for this bounded
identity route. Fresh independent read-only re-review remains a separate gate
on the completed vertical increment and must assess exact implementation
revision `434919a84406ef84afaaebc1169cf0430accf3f3` plus the v4 record.
Legacy activation and retirement remain separate owner decisions.
