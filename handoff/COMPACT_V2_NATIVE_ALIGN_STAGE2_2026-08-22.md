# Compact-v2 Native ALIGN Stage 2

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `838f50249ac07bd90308f90f49397d3a38c4cd4a`
- Implementation tree: `f38ef2fe1e1cacc676a250a21474dd1877208114`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`

## Result

Stage 2 is implemented and locally validated. Immutable alignment carriers now
retain every delivered per-network timestamp, its exact signed packet counter,
explicit counter discontinuities, and scan-bounded contiguous runs. Exact
counter continuity means signed `after == before + 1`; repeats, decreases,
jumps, and maximum-to-minimum transitions split runs without an inferred
rollover or synthesized sample.

Each network also retains the relational association between delivered native
rows and the existing common-time compatibility slots. Gap association uses
the realized cadence, one `std::round` candidate with half-way cases away from
zero, an inclusive `abs(delta) <= dt/2` edge, injective row/slot ownership, and
exact parity with the established legacy presence mask. A subcadence packet
drop therefore remains absent and cannot be reconstructed as detector data.

The measured telescope trajectory and the existing observation pointing-
offset model are evaluated separately at each network's exact native detector
times. Pointing admission requires exact network, row-interval, timestamp, and
alignment-handle identity. The observation owner validates the complete
alignment/pointing candidate before one immutable-pointer swap; a rejected
candidate leaves the prior accepted pair pointer-identical.

The only edit to an established helper is replacing `fmt/core.h` with
`fmt/format.h`, which makes its existing `fmt::format` use self-declaring under
the current fmt dependency. It was exposed by the new public-header isolation
translation units and does not alter runtime behavior.

No existing common-time compatibility product or numerical route is changed
or activated by this commit.

## Focused rejection and identity coverage

The Stage 2 matrix covers subcadence drops without synthesis; repeat,
decrease, jump, maximum-counter transition, and scan boundaries; prohibition
of cross-run association; exact half-away rounding and inclusive half-cadence
tolerance; legacy presence parity; association collision; network-input
permutation; exact identical-native-time equality; native-time telescope and
pointing-offset evaluation; and atomic rejection of duplicate, absent,
nonfinite, fractional or out-of-range counter, stale-handle, and foreign-scope
candidates. The public alignment and observation-carrier headers compile
without the test precompiled header.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 2 native-carrier cases | 7/7 passed |
| Complete SCI-ALIGN executable | 22/22 passed |
| Complete CTest | 743/743 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| CLI build and version boundary | passed |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 744 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
743 runnable tests passed.

## Stop boundary

Stage 2 stops here as required by the accepted plan. It does not carry detector
values, gather or scatter mutable data, alter RTC/PTC, activate naive or JINC
mapmaking, publish products, bind the Stage 7 native-ready consumer relation,
or change runtime routing. Stage 3 has not begun. There is no Unity gate or
production-profile activation claim for this carrier-only stage.
