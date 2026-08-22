# Compact-v2 Native ALIGN Plan Acceptance — 2026-08-22

## Verdict

Independent Stage-0 re-review accepts exact plan commit
`a3f2bf465a26048b24017ebd50876c4a2684b1b8`, tree
`3ef26b7f05413dd3a48139fb0be3fd0586a59a2b`. No blocking findings remain.

The reviewer made no file, ref, commit, or tracked working-tree change. This
record is a documentation child of the accepted plan and does not alter its
reviewed content or identity.

## Six required answers

1. **Compact-v2 identities are sufficient.** Bundle, relation, output-row,
   target-row, source, network/channel, disposition, and baseline-governed
   `flag` facts are sufficient and correctly scoped.
2. **The mode matrix is correct.** Science and Pointing are candidate
   consumers; both existing Beammap calibration lanes remain without
   matched-consumer lineage; OOF and other modes fail closed.
3. **Common-slot and run-boundary rules are correct.** Native timestamps retain
   authority; exact signed-counter continuity, absence, injectivity, `dt / 2`
   association, and legacy presence-mask parity are frozen.
4. **Lifecycle owners are explicit.** Immutable relation/alignment/pointing
   state is observation-owned; measured mapping, ledger, revision, and
   operation state is scan/chunk-owned; publication state remains with the
   existing output owner.
5. **Numerical and product claims are bounded.** Mature numerical bodies and
   accumulation order remain frozen; identical-time equivalence, thread-count
   gates, staged product disablement, and owner-run Unity evidence remain
   required.
6. **Exact verdict: `accept`** for
   `a3f2bf465a26048b24017ebd50876c4a2684b1b8`.

## Nonblocking implementation notes

- Stage 1 should add the non-detector Beammap no-retained-native-consumer-
  relation test immediately, rather than relying only on Stage 7 routing.
- Stage 2 should pin `std::round` half-way behavior and inclusive
  `abs(delta) <= dt / 2` edge semantics.

These notes refine the already accepted tests; they do not change the plan's
authority, stage order, or claim boundary.

## Authorization boundary

This acceptance opens Stage 1 only: the immutable verified compact-v2
bundle-to-detector-column relation and atomic `Calib::get_apt` publication,
without runtime native-consumer activation. It does not accept later stages,
authorize a push, or change production status.
