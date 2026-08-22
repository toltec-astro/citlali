# Compact-v2 Native ALIGN Stage 5 PTC/PCA Cohorts

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `35a61eaaf91722ac7167bfb90d1029f09b4d1df2`
- Implementation tree: `11b7c62726f3678fb54ef5f0513935dc0d0e0383`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`
- Stage 4 prerequisite commit: `23a5cabe9fa6ec6579c91ec7c7a344339d06c993`
- Fixture SHA-256: `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701`

## Result

Stage 5 is implemented and locally validated. The new
`timestream_ptc_cohort_adapter.h` consumes the immutable Stage 3 measured scan
and a completed Stage 4 RTC dispatch. Before issuing a PTC operation, it
reconstructs the expected run inventory from the admitted scan and requires
exact equality of segment identity, run boundaries, raw input values and flag
bits, detector partitions, ordered common-slot and native-row support,
run-local anchors, and internal consistency of the recorded actual ORed flag
support. Missing, repeated, reused, or foreign RTC support fails before a
grouping or numerical body.

Every PTC matrix is confined to one contiguous Stage 4 segment. The frozen gap
therefore yields independent `{0,1}` and `{3,4}` cohort matrices; no PCA group
can span the absent slot or either packet-run boundary.

Detector membership comes only from the verified typed relation:

- `all`, `nw`, `array`, and per-detector ordinary groups preserve exact
  presentation-ranked columns, including interleaved/noncontiguous members;
- enabled `corr_nw` invokes its established grouping body separately for each
  network and segment, requires injective subgroups of at least two detectors,
  and retains every ungrouped detector as an explicit pass-through group; and
- group identities without retained exact typed membership fail closed rather
  than consulting a legacy matrix or inferring membership.

Delivered RTC flags, operation-local exclusion bits, and typed APT flags form
the actual detector-cell exclusion state. Excluded cells receive only a
checked finite private placeholder. The shared established PCA compatibility
classifier rejects incompatible optional modes before grouping or cleaning.
PCA-invalid and pass-through cells preserve the exact finite RTC value that
entered PTC; only PCA-valid cells receive cleaner replacements.

The scan/chunk-owned measured ledger now supports an issued operation followed
by one transactional scatter. The complete destination set, identity,
timestamp, expected revision, action, and finite projected value are validated
before sparse values or dense revisions are swapped. Rejected nonfinite,
stale, duplicate, or foreign batches leave the ledger unchanged and can be
corrected and retried with the same issued operation. A successful transaction
advances every affected anchor exactly once and records the committed monotonic
operation. All operation and revision state remains bounded by the existing
scan/chunk transaction.

## Focused contract coverage

The six Stage 5 cases prove:

1. exact interleaved `nw` and `array` membership, two independent gap-bounded
   segments, private placeholder invariance, and no revision of the orphaned
   network-0 sample at the missing common slot;
2. noncontiguous `corr_nw` grouping, one cleaner call only for the selected
   subgroup, and exact preservation of non-identity RTC pass-through values;
3. rejection of incomplete RTC inventory, incompatible optional modes,
   second-pass/windowed requests, and unsupported group identities before any
   grouping body or operation issue;
4. atomic nonfinite, already-committed/stale, and duplicate-destination
   rejection with successful corrected retry;
5. bitwise equality with the existing rectangular ordinary-PCA path when all
   networks have identical times; and
6. exact repeated output at OpenMP thread counts 1, 2, 4, and 8.

The public adapter header also compiles in isolation. Existing Stage 1 relation
and Stage 3 measured-binding tests now pin the retained typed array identity.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 5 focused cases | 6/6 passed separately at OpenMP thread counts 1, 2, 4, and 8 |
| Complete SCI-ALIGN executable | 43/43 passed |
| Public-header isolation | passed |
| Complete CTest | 764/764 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| Session-exit audit | 719 dependencies; zero library/CLI exits; zero growth |
| Frozen fixture identity | SHA-256 unchanged at `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701` |
| CLI build and exact implementation boundary | `v4.0.0-3674-g35a61eaaf`; binary SHA-256 `e9ff67aa3a30a6da293054f4d38a70e34e671680c703fc51e07917a9f957ac02` |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 765 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
764 runnable tests passed.

## Stop boundary

Stage 5 stops at the accepted boundary. The adapter does not alter an
established RTC or PTC numerical algorithm, enter naive or JINC mapmaking,
publish products, add public `Engine` state, or activate a runtime route. RTC
and PTC product writing remain disabled pending Stage 7 provenance.

No Unity run is required for this bounded local stage. Stage 6 may now begin as
a separate commit: route exact measured detector cells with exact native
pointing into the existing naive and JINC population paths, stopping before
output-lineage claims or ordinary mode activation. The owner-run Unity campaign
remains a prerequisite for accepting Stage 7, not Stage 5.
