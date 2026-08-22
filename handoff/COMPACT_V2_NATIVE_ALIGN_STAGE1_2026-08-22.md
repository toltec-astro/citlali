# Compact-v2 Native ALIGN Stage 1

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `da9e1deac139f4904d059822e8518259838e45c0`
- Implementation tree: `41897db2ef8468d408153c7da2110812b880c36e`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`

## Result

Stage 1 is implemented and locally validated. A matched compact-v2 bundle can
now publish an immutable verified bundle-to-detector-column relation in the
same transaction as the presentation-ranked numeric `Calib::apt` compatibility
view. Rejected raw bytes, bundle tamper, relation drift, or `Calib::setup`
failure leave the prior APT, typed relation, metadata, groupings, and derived
calibration state unchanged.

The relation retains exact bundle and relation-component identities,
observation and parent identities, raw source bindings, distinct source,
application, and presentation ranks, output and target row keys, raw
network/channel, disposition, selected seed, and the baseline-governed signed
`flag`. Only verified unmatched or ambiguous rows retain that flag as absent.
`kids_flag`, sample flags, and `flag2` cannot substitute for it.

Science and Pointing loads retain the typed relation. Detector/automatic
Beammap continues to build its APT from raw input and never calls this consumer
admission. The existing non-detector Beammap calibration-table lane continues
to load its numeric APT view but explicitly discards the typed consumer
relation. A baseline bundle is rejected by the consumer admission API.

No RTC, PTC, pointing, naive, JINC, product, or native-time consumer is
activated by this commit.

## Focused rejection coverage

The Stage 1 matrix covers presentation permutation; exact int64 identity above
the legacy double-safe range in the typed relation; complete detector and raw
channel coverage; matched, unmatched, and ambiguous states; target/output and
baseline-seed joins; component and row identities; and atomic rejection of
component/raw tamper, stale scope, duplicates, omissions, wrong network or
channel, foreign target or seed rows, unauthorized nulls, incomplete raw
coverage, governed-flag drift, and flag substitutes. The new public header is
compiled without the test precompiled header.

## Validation

The fresh local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 1 relation executable | 4/4 passed |
| Complete SCI-ALIGN executable | 15/15 passed |
| Compact-v2, Calib guardian, and pipeline-routing selection | 66/66 passed |
| Complete CTest | 736/736 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| CLI build and version boundary | passed |
| Diff hygiene | `git diff --check` passed |

The first `check` invocation reached the established
`citlali_safety_test_NOT_BUILT` placeholder because that executable is not a
dependency of the `check` target. Building the separately declared safety
target and rerunning the complete discovered suite produced the recorded
736/736 runnable pass.

## Stop boundary

Stage 1 stops here as required by the accepted plan. Stage 2 native alignment
and pointing carriers have not begun. There is no Unity gate for Stage 1 and
no production or validation-profile activation claim.
