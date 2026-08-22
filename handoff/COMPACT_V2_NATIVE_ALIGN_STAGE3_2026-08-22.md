# Compact-v2 Native ALIGN Stage 3

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `6008ec6330e7058c7c87f3a6a7e568165763f35b`
- Implementation tree: `de5edb31e645bf662e4dfe82daf890f4ba38863f`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`

## Result

Stage 3 is implemented and locally validated. Each admitted raw KIDs network
input carries exact source UID, network, interface, native-row origin, measured
value-matrix ownership, and original native flag-bit ownership. Admission
requires exact observation scope, relation/alignment handles, source inventory,
network, interface, channel cardinality, native row interval, and complete raw
channel coverage. Every raw channel joins exactly one compact-v2 detector
column; no presentation order, contiguous-network assumption, or floating
identity conversion participates in that join.

The resulting immutable measured scan mapping retains exact raw
source/channel, detector column, output UID, disposition, and baseline `flag`
identity. Its relational cell view distinguishes mapped-valid,
mapped-invalid, and absent cells for complete and partial cohorts. Nonfinite
values and samples carrying nonzero original flags remain present but invalid;
they are not reclassified as absent, filled, or synthesized.

The existing input matrices remain the sole measured-value owners. The network
inputs and mapping retain shared ownership handles and detector/sample lookup
metadata, but do not allocate another O(rows x detectors) value matrix. The
scan/chunk ledger stores only fresh per-sample revisions and reads values and
original flags through the admitted owners.

A scan transaction builds and validates the complete immutable mapping, fresh
ledger, and operation sequence before one live-owner swap. The sequence begins
at zero and is monotonic within that transaction. Invalid candidates and an
attempt to replace an active transaction fail before lifecycle mutation.
Commit, rollback, and boundary destruction remove the mapping, ledger, and
sequence so no mutable revision or operation state crosses a scan/chunk or
observation boundary.

No existing compatibility product, numerical kernel, or runtime route is
changed or activated by this commit.

## Focused rejection and identity coverage

The six Stage 3 cases cover exact source/channel, detector-column, UID,
disposition, baseline-flag, value, and original-flag joins; complete and
partial cohorts; mapped-valid, mapped-invalid, and absent cells; noncontiguous
network membership; interleaved presentation columns; network-input
permutation; exact zero, greater-than-2^53, and maximum-int64 output UIDs;
retained input-owner pointer identity without a second value copy; fresh
ledger revisions and operation sequence; commit, rollback, and retry reset;
and atomic rejection of null owners, shape mismatch, interface mismatch,
channel-count mismatch, invalid slot windows, cross-network sample keys,
inactive candidates, and active-transaction replacement.

The public measured-scan header compiles in its own translation unit without
the test precompiled header.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 3 measured-ingress cases | 6/6 passed |
| Complete SCI-ALIGN executable | 28/28 passed |
| Complete CTest | 749/749 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| CLI build and exact version boundary | `v4.0.0-3668-g6008ec633`; passed |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 750 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
749 runnable tests passed.

## Stop boundary

Stage 3 stops here as required by the accepted plan. It does not dispatch RTC,
gather or scatter PTC/PCA working groups, invoke naive or JINC mapmaking,
publish products, bind the Stage 7 native-ready consumer relation, or activate
runtime routing. The native-required processing mode remains unable to enter
RTC. Stage 4 has not begun.

No Unity run is required for this admission-only stage. Before Stage 4 begins,
the project owner must freeze one small owner-reproducible native-gap fixture
locally, as required by the accepted plan.
