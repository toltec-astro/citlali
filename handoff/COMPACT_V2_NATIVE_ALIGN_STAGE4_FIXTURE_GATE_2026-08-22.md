# Compact-v2 Native ALIGN Stage 4 Fixture Gate

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Fixture commit: `6d65b151eb836e2bbd5f5f1d3bf381427800528a`
- Fixture tree: `406d1784d64d6fb41e567aab083798324f871634`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`
- Fixture identity: `urn:citlali:sci-align:native-gap:v1`
- Fixture SHA-256: `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701`

## Result

The accepted plan's final prerequisite before Stage 4 is satisfied. A small,
owner-reproducible native-gap fixture is frozen at
`tests/fixtures/sci_align/native_gap_v1.yaml`. A reusable test-only loader
checks the raw-file SHA before parsing, preserves exact signed and unsigned
integer identities, validates the complete source/channel inventory, and
materializes the fixture through the current Stage 2 native alignment
contracts. Any unreviewed byte change fails before semantic use.

The fixture reconstructs the relevant bounded evidence from historical commit
`fd3627fc7` without replaying its superseded production bridge. Its current
authority is the accepted compact-v2 plan and the exact fixture commit above.

## Frozen topology and Stage 4 oracle

The fixture contains networks 0 and 7 with two raw channels each and an
interleaved four-column compact-v2 presentation. Output UIDs include zero,
`9007199254740993`, and signed-int64 maximum. Five common relational slots run
at a realized cadence of 0.01 seconds.

Network 0 delivers all five rows with counters 100 through 104. Network 7
delivers four rows: counters 700, 701, 703, and 704. Its missing counter 702
produces no row, value, flag entry, or native identity. Common slot 2 is
explicitly absent for network 7; delivered native row 702 maps to common slot
3. The network-7 packet runs are exactly `[700,702)` and `[702,704)`, separated
by the `701 -> 703` counter discontinuity.

The complete two-network cohort intervals are exactly `[0,2)` and `[3,5)`.
For factor-2 RTC dispatch, the fixture pins one output support per network in
each interval:

| Segment | Network | Common slots | Selected anchor row | ORed original flags by raw channel |
| --- | ---: | --- | ---: | --- |
| 0 | 0 | `[0,2)` | 100 | `[5, 8]` |
| 0 | 7 | `[0,2)` | 700 | `[10, 48]` |
| 1 | 0 | `[3,5)` | 103 | `[96, 384]` |
| 1 | 7 | `[3,5)` | 702 | `[2560, 5120]` |

This is an input/support oracle only. It does not approve an RTC
implementation or numerical result in advance.

## Owner reproduction

From the repository root:

```bash
shasum -a 256 tests/fixtures/sci_align/native_gap_v1.yaml
cmake --build build --target citlali_sci_align_test -j 8
ctest --test-dir build -R sci_align_native_gap_fixture --output-on-failure
```

The digest must equal the value recorded above and the four focused fixture
tests must pass. The full SCI-ALIGN suite remains available with
`ctest --test-dir build -R sci_align --output-on-failure`.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Native-gap fixture cases | 4/4 passed |
| Complete SCI-ALIGN executable | 32/32 passed |
| Complete CTest | 753/753 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| CLI build and exact version boundary | `v4.0.0-3670-g6d65b151e`; passed |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 754 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
753 runnable tests passed.

## Stop boundary

The fixture commit and this record contain no RTC processor call, PTC/PCA
adapter, mapmaking path, product publication, runtime activation, or numerical
kernel change. Stage 4 has not begun. The fixture prerequisite is now closed,
so Stage 4 may begin as a separate implementation commit under the accepted
contiguous-run dispatch and stop contracts. No Unity run is required for this
local fixture gate.
