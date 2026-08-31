# WP7-REPLAY-001: Canonical D2 Evidence-Tooling Replay

Date: 2026-08-31

Status: **owner-accepted and canonically integrated at exact
`f8ba732bc4072e918c2521a013305be354ed7b53`; no application route or scientific
factor/filter choice activated**

## Owner acceptance and integration

On 2026-08-31 the project owner reviewed this unit's scope and reported
validation, accepted `WP7-REPLAY-001` at exact commit `f8ba732bc...`, and
authorized its integration into canonical application mainline. The owner
explicitly excluded the preserved producer/prefilter/residual prototype,
filter design, downsampling implementation, and every additional scientific
or architectural choice. Local `codex/refactor-mainline` was fast-forwarded to
the accepted commit. No remote push occurred.

## Authority and identities

- Work order: `WP7-REPLAY-001`
- Canonical base: `f6c9033f80810da255a9bfa987e0fba8a082b785`
- Replay branch locator: `codex/wp7-g4-replay-001`
- Exact divergent source commit:
  `49fe73e757daa1885cd23127e8441cba47e648d2`
- Exact source parent: `28cbd5daec513588f9553f7e0b34a817c31b1517`
- Scientific/engineering authority: ADR 0020 and
  `doc/WP7_RTC_FILTER_DOWNSAMPLING_CERTIFICATION_TEST_PLAN_2026-08-30.md`
- Governing program: `doc/WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`

This unit replays offline D2 PSD/line measurement and aggregation tooling. It
does not replay a filter bank, select a factor, design a filter, change an RTC
route, publish a new product, or claim application conformance.

## Exact source-path disposition

Source commit `49fe73e757...` changes exactly ten paths. All ten are accounted
for here; the commit was not cherry-picked wholesale.

| Source path | Source action | Canonical disposition |
| --- | --- | --- |
| `doc/REFACTOR_STATUS.md` | Modified | Divergent status prose was not applied directly. Its D2 tooling result and remaining producer boundary were reconciled into the current G4 status after the owner-approved governance closeout. |
| `doc/WP7_RTC_FILTER_DOWNSAMPLING_CERTIFICATION_TEST_PLAN_2026-08-30.md` | Modified | The bounded item-2 progress note was replayed with links to both the source handoff and this canonical replay record. |
| `handoff/WP7_RTC_D2_PSD_LINE_EVIDENCE_TOOLING_2026-08-31.md` | Added | Imported from exact source blob `8bcbb23061e6ecb9db3c75ddb2e6d7fe4e31f60b`, then labeled as source-lane evidence and linked to this canonical record. |
| `tools/wp7/export_legacy_tod_psd_line_discovery.py` | Added | Exact source blob `9a07e16549b8b0f2b5ff825263ae6126bc88bb79`; unchanged. |
| `tools/wp7/rtc_filter_psd_line_corpus.py` | Added | Exact source blob `ba1a90b886cdea86a12cbbf130bd1d3dbaf3d9a8`; unchanged. |
| `tools/wp7/rtc_filter_psd_line_evidence.py` | Added | Exact source blob `5520eebfc531cc5f6bae4b058831d5a1ef58293c`; unchanged. |
| `tools/wp7/test_export_legacy_tod_psd_line_discovery.py` | Added | Exact source blob `fc959232e1edf768ecb35dafb7670ad03e05442a`; unchanged. |
| `tools/wp7/test_rtc_filter_line_order_trace.py` | Added | Exact source blob `97af34e1b29b587696a6ca516d69481dccd6baac`; unchanged. |
| `tools/wp7/test_rtc_filter_psd_line_corpus.py` | Added | Exact source blob `279fceb694981e44fb7ea51b8cc33b5bbfb9f78d`; unchanged. |
| `tools/wp7/test_rtc_filter_psd_line_evidence.py` | Added | Exact source blob `2fe8caa8c99ea26414da72ec7a16f9dd82284192`; unchanged. |

Canonical-only reconciliation also updates the integration ledger, successor
program status, machine-readable successor authority, and this handoff. Those
paths are G4 control records, not silently attributed to the divergent commit.

## Preserved out-of-scope work

The following uncommitted divergent paths remain unchanged in
`/private/tmp/citlali-wp7-network-timed-rtc-repair` and are not source identity
for this unit:

- `tests/CMakeLists.txt`
- `include/citlali/core/pipeline/rtc_filter_d2_measurement.h`
- `tests/rtc_filter_d2_measurement_header.cpp`
- `tests/test_rtc_filter_d2_measurement.cpp`

They form a possible network-native prefilter/residual producer prototype, but
they require a separate exact identity, review, work order, and canonical
replay decision. Nothing in `WP7-REPLAY-001` approves their design.

## Fresh canonical validation

The replay candidate passes:

- Python syntax compilation for all seven imported tool/test modules;
- 26 focused deterministic D2 tests;
- 130 configuration unit tests;
- all four TolTECA mode kits;
- 8/8 compact compatibility cases;
- 100% compact surface coverage with zero gaps;
- all 207 baseline-tool tests;
- validation ledger: 60 records valid;
- science-change ledger: 3 changes and 5 integration commits valid;
- the local `citlali_cli` build and Git-version identity gate; and
- 832/832 runnable fallback-build CTests, with the one established disabled
  test unchanged.

The first baseline-suite invocation in the fresh replay worktree ran all 207
tests but reported 11 failures because that worktree had no local
`build/bin/citlali`; every failure stopped at the same missing-executable
precondition. The canonical parent executable was then built at exact
`f6c9033f8`, temporarily bound into the replay worktree, and the complete
207-test suite passed. The temporary binding was not added to Git.

The fallback executable reports Citlali revision `f6c9033f8` and Kidscpp
`04088da-dirty`. These build and CTest results are supplemental local
regression evidence, not a reproduction of the accepted Spack-backed V2
campaign. No affected-mode reduction is required for this unit because it
changes no application route, configuration, numerical operator, or product.

## Stop boundary

This candidate stops before the producer prototype and before all filter
selection/design work. D2 remains open until a separately reviewed
network-native in-memory producer supplies native-rate prefilter and
post-cleaning residual planes, route-specific source masks, and realized
pre-decimation line-operator evidence for the required Beammap, Science, and
OOF corpus.

This unit is closed. Do not begin the producer or a filter-design unit merely
because these offline evidence tools pass their local gates. A next bounded
G4 increment requires a separate owner-reviewed proposal.
