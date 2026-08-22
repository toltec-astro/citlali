# Compact-v2 Native ALIGN Stage 4 RTC Run Dispatch

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `23a5cabe9fa6ec6579c91ec7c7a344339d06c993`
- Implementation tree: `bd2c74f4dde97e3eb9f99b845d2f4741ee458d76`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`
- Fixture prerequisite commit: `6d65b151eb836e2bbd5f5f1d3bf381427800528a`
- Fixture SHA-256: `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701`

## Result

Stage 4 is implemented and locally validated. The new
`timestream_rtc_run_adapter.h` consumes the immutable Stage 3 measured scan,
derives maximal complete temporal segments, partitions each segment into
packet-contiguous network runs, and presents each network's measured detector
columns to an established run-local RTC numerical body. The adapter invokes
the existing `timestream::Downsampler` independently for every run, so stride
anchors reset at each discontinuity.

The adapter preserves the distinction between temporal presence and sample
validity. A delivered row with nonzero original flag bits remains present and
participates in support; the flag does not erase the measured sample. A
nonfinite measured value is rejected before any numerical-body invocation.
The numerical body receives owned run-local matrices through a const input
view and must preserve shape, finite values, and every original flag bit. It
may add flag bits but may not remove any delivered bit.

Every downsampled row records:

- the temporal segment and common-slot interval;
- the exact `NativeContiguousRun` and detector-column partition;
- ordered common slots and exact native row/time identities in its support;
- the selected run-local anchor; and
- the exact bitwise OR of actual input flags over that support.

All candidate runs and inputs are prepared and validated before the first
numerical-body call. Cross-run-window requests fail closed. The Stage 3 scan,
ledger, and operation sequence remain unchanged by dispatch or rejection.

## Focused contract coverage

The five new cases prove:

1. exact factor-2 output for the frozen native-gap oracle, including run-local
   anchors, ordered support, interleaved detector partitions, ORed original
   flags, and no gap bridging;
2. bitwise legacy rectangular-stride equivalence for a complete identical-time
   fixture, including signed zero, exact flags, and final short support;
3. exact repeated equality at OpenMP thread counts 1, 2, 4, and 8 while a
   numerical body changes values and adds a flag bit;
4. rejection of invalid factors, cross-run-window requests, and nonfinite
   measured values before the numerical body and without ledger mutation; and
5. rejection of numerical-body shape drift, removed flag bits, or nonfinite
   output.

The public adapter header also compiles in isolation.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 4 focused cases | 5/5 passed separately at OpenMP thread counts 1, 2, 4, and 8 |
| Complete SCI-ALIGN executable | 37/37 passed |
| Public-header isolation | passed |
| Complete CTest | 758/758 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| Frozen fixture identity | SHA-256 unchanged at `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701` |
| CLI build and exact implementation boundary | `v4.0.0-3672-g23a5cabe9`; binary SHA-256 `88bc483ca7fe9a3ee8a26be73e1505cf6504e3be8391a737be9f0358d412c89a` |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 759 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
758 runnable tests passed.

## Stop boundary

Stage 4 stops at the accepted boundary. The adapter does not call
`RTCProc::run`, alter an established RTC numerical kernel, gather or scatter
PTC/PCA cohorts, invoke naive or JINC mapmaking, publish products, add public
`Engine` state, or activate a runtime route. RTC product writing remains
disabled. Consequently, the native-required processing mode still cannot
enter production RTC through this commit.

No Unity run is required for this bounded local stage. Stage 5 may now begin
as a separate commit: PTC/PCA cohort gather and transactional scatter, stopping
before mapmaking. The owner-run Unity campaign remains a prerequisite for
accepting Stage 7, not Stage 4 or Stage 5.
