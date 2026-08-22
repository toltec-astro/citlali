# Compact-v2 Native ALIGN Plan Independent Review — 2026-08-22

## Disposition

An independent delegated reviewer inspected exact plan commit
`82b086856f891873167760534b64a0811840f3cb` without editing files, commits,
refs, or working-tree state. Its Stage-0 verdict is **`revise`**.
Implementation remains blocked until an independent review accepts an exact
revised plan commit.

## Six required answers

1. **Compact-v2 identity sufficiency — yes, with a flag clarification.**
   `VerifiedBundle`, bundle `ComponentIdentity`, `AptRow`, `RelationRecord`,
   `RelationTable`, target `ScopedRowReference`, and verified source records
   contain the facts needed for the detector-column adapter. The plan needed
   to name baseline-derived `flag` and its authorized typed-missing state.
2. **Mode matrix — revise.** Science and Pointing activation and OOF/other
   fail-closed routing were correct. Beammap's matched-consumer prohibition was
   correct, but the plan overgeneralized every Beammap as raw-only instead of
   retaining the existing non-detector calibration-table lane.
3. **Common-slot and run boundaries — revise.** Common-slot nonauthority,
   immutable timestamps, absence, and no-row-reuse rules were correct. Exact
   packet-counter continuity and boundary cases were not frozen.
4. **Transaction and lifecycle owners — revise.** Atomic behavior was defined,
   but the mutable ledger, operation sequence, and reset boundaries were not
   assigned to one scan/chunk owner.
5. **Numerical and product bounds — yes.** The plan preserves mature numerical
   bodies, requires identical-time equivalence, prevents premature product
   claims, and routes final acceptance through owner-run Unity evidence.
6. **Exact verdict — `revise`** for
   `82b086856f891873167760534b64a0811840f3cb`.

## Blocking findings and resolution route

1. Distinguish detector/automatic Beammap raw production from the unchanged
   existing non-detector calibration-table lane. Neither may activate
   matched-v2 native-consumer lineage.
2. Define the consumer-selection field as baseline-governed exact-int64
   `flag`, with typed missing only where the verified unmatched/ambiguous
   compact-v2 rule authorizes it. Never substitute `kids_flag`, sample flags,
   or Beammap `flag2`.
3. Define contiguous packet counters as an exactly representable signed
   `before + 1 == after` transition only. Repeats, decreases, jumps, rollover,
   and scan boundaries close support. Test every case.
4. Assign verified relation/alignment/pointing handles to the observation;
   measured mapping, ledger, and operation sequence to one scan/chunk; and
   atomic product/index publication to the existing output owner. Reset or
   destroy each state at its named boundary.

The reviewer also recommended freezing the established `dt / 2` gap-slot
tolerance, single rounded candidate, injectivity, and legacy presence-mask
parity. The revised plan adopts this recommendation.

## Claim boundary

This record is review evidence only. It does not accept the revised plan,
activate a consumer, admit application code, authorize a push, or change any
APT, ALIGN, Beammap, JINC, or production status.
