# TIMESTREAM-SUCCESSOR-RTC-PERFORMANCE-001 — 2026-09-18

Owner direction: repair performance RTC-wide and inspect storage/lifetimes,
using bound arrays and shared axes without changing scientific processing.
The [work order](../doc/TIMESTREAM_SUCCESSOR_RTC_PERFORMANCE_001_2026-09-18.md)
records effective governance, exact live base and the single owned branch.

Implementation `f34f9d9d896132b4ba6bb9d30380b9d34e94d867`, tree
`424a9c4235da4116167aba23d10bc5706801d9a1`, is a direct child of canonical
`fe448860741c543b731faaa76c0eea5194e7599f`. Branch:
`codex/timestream-successor-rtc-performance-001`; worktree:
`/private/tmp/citlali-timestream-successor-cal-001`. Earlier branches, failed
evidence, accepted implementation and the frozen old-route baseline remain.

The ordinary 152390/0/2 network12 Science/Lissajous NGC4449 invocation completes
RTC → CAL → PTC → VAL. Same input, 414 requested identities, 363 with output,
51 fully unavailable detectors, 27,157,906 retained PTC samples and 124
converged rank-10 fits. All 5,968 binary products / 3,862,256,724 bytes match
the baseline bit-for-bit. All 12 existing YAML documents preserve scientific
contents. Timings, executable identity and load entropy are explicitly
normalized; both changed CAL-receipt hash references are independently checked
against their actual files. The initial strict comparator failure is preserved,
not overwritten or treated as a scientific failure.

| Measured interval | Baseline seconds | Candidate seconds |
|---|---:|---:|
| Entire invocation | 631.43 | 311.27 |
| Frozen RTC plan construction | 47.90 | 9.79 |
| RTC numerical Apply | 26.41 | 27.00 |
| Grid preparation/inspection | 36.30 | 0.87 |
| Conditioned relearning, matched outcomes and exports | 165.57 | 67.92 |
| CAL Learn + Consider + Apply/VAL, before publication | 57.56 | 1.33 |
| PTC including publication | 31.56 | 19.94 |

Candidate CAL publication adds 9.49 seconds. The earlier accepted run's
55.43-second CAL subtotal and 618.69-second total remain valid; this table
uses a new matched baseline rerun. One local A/B observation, including short
baseline stack captures, is not a repeated benchmark. Ingress remains about
128 seconds, including the existing 11-network/6.757-GB timing-input recovery.
Its internal hashing/read/reconstruction costs are not individually attributed.

The repair indexes sparse event support, validates array identities at product
boundaries, uses direct indices and compact state inside them, reuses the exact
prepared grid, shares timing/telescope calculations and array/schedule CAL
atmosphere evaluations, reuses only spectra with identical full centering
support, and streams large diagnostic sequences. Original replay remains
immutable; exact plane digests replace redundant verification copies. Full
occurrence/support access remains available. No generic cache/framework or
new scientific authority is introduced.

Comparable one-second RSS peaks are 13.841 → 7.288 GB; candidate internal
high-water is 7.311 GB. Native storage is 62,736,318 detector-samples: original
x/r 1.004 GB, original states 0.251 GB, conditioned+filtered x/r 2.008 GB,
plan/result masks 0.314 GB. One shared native axis is 4.85 MB; the shared
output-time axis is 0.61 MB and all grid descriptors together are 19.9 KB.
AST output pointing/elevation is 0.503 GB; CAL entries and output cells are
0.753 and 0.502 GB. There is no persistent full identity or filter-dependency
record per output sample.

After CAL Apply, the measured logical arrays/evidence total 5.862 GB, live
allocator storage is 6.470 GB and RSS is 6.700 GB. The 0.609-GB allocator
remainder includes capacity and other metadata, explicitly not fully assigned.
PTC group arrays add 0.532 GB while CAL parents remain alive. After releasing
the RTC arm, live allocator storage drops to 1.788 GB; RSS retains freed pages.
Known eliminated payloads are 1.004 GB of original copies and approximately
0.502 GB from the old AST optional record layout. Streamed YAML removes large
window node graphs, but no invented exact split of baseline RSS is claimed.
The durable report lists element sizes, buffer counts, lifetimes and limitations.

Gates: 43 focused checks; all 1,261 runnable CTests; configuration preflight
(130 tests and audits); 207 baseline Python, 86 successor Python and four
source-graph tests PASS. Existing disabled MapFitterLifecycle test unchanged.
Independent fresh-context exact-SHA review reproduced all 43 focused tests and
passed scientific/behavioral, architecture/ownership and repository/evidence
axes with recorded limitations. No unexpected error-level output in the run.

Actual environment is Release AppleClang 21.0.0.21000334, C++20/Homebrew;
preexisting `kids 04088da-dirty` banner retained. No Unity access or new Unity
qualification. Input SHA256:
`bbb85e2b943c0f6d21ebebf990aef1286906fab8f6971c3026020603e9dcd8e3`.
Executable SHA256:
`33a095962efc7f52914d74209cf7459731fb89a7383e625d8d2edc15b3d992e9`.

Durable evidence:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-rtc-performance-f34f9d9d8-20260918`.
Working origin: `/private/tmp/citlali-rtc-performance-001-20260918`.
`RESULT.md`, `MEASUREMENTS.json`, `FINAL_COMPARISON.json`,
`INDEPENDENT_REVIEW.md`, both complete run products, executable copies and
the comparison scripts preserve the result. `performance.jsonl` in the
candidate output holds per-stage RSS, allocator live bytes/block counts and
logical payload counters. Unknown allocation multiplicities are explicit.

Canonical promotion and owner push remain pending; verify live canonical
before promotion rather than assuming the base has not moved. Any executable
reconciliation needs its own exact tests/review. The implementation default
continues through PTC/VAL with unchanged filters, masks, timing, validity,
donors, source protection, method/rank and qualification limits. No MAP,
FRUIT, detector rescue or new scientific qualification. The next pipeline
owner remains typed MAP consumption under its own contract; remaining
ingress/spectral performance work would be separately measured and bounded.
