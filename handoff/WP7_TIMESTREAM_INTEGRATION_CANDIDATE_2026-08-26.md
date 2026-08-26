# WP-7 Timestream Integration Candidate - 2026-08-26

## Outcome

The local candidate branch `codex/wp7-timestream-integration-candidate`
descends directly from exact Unity-tested application commit
`3ebc2a67fc32bad69759ff45638484efabf91773`. It contains two coherent
code/tool commits before this evidence update:

1. `aa85a2287` — bounded inactive-source cleanup;
2. `ff7899668` — focused reduction-auditor evidence repairs.

No Unity access, push, ref deletion, tag creation, untracked-file deletion,
build-adaptation merge, or numerical algorithm change occurred in preparing
this candidate.

## Mainline Fast-Forward

At the exact tested application SHA, remote
`origin/codex/refactor-mainline` was
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`. It is an ancestor of
`3ebc2a67f`; `git rev-list --left-right --count
origin/codex/refactor-mainline...3ebc2a67f` returned `0 71`. At code/tool
candidate `ff7899668`, the relation was `0 73`. Subsequent evidence-only
descendants add only candidate-side history, so the owner can use an ordinary
guarded fast-forward after fetching and recomputing the relation.

## Exact Unity Stage 7 Evidence

The completed owner-run Stage 7 NGC4449 152390 result identifies:

- application commit `3ebc2a67fc32bad69759ff45638484efabf91773`;
- executable version `v4.0.0-3712-g3ebc2a67f`;
- four completed fruit-loop iterations, 124 scans per iteration;
- zero warning, error, or critical records in the downloaded final log;
- 124 started and 124 completed scan lines in final `redu04`;
- raw v3 completion with 5,518 detectors reconciled as 4,512 APT-eligible plus
  1,006 APT-excluded, `detector_sample_expansion: false`, and 124 bounded scan
  summaries;
- canonical publication `status: validated_complete` and
  `bounded_provenance_validated: true`; and
- a recursively complete artifact tree.

The downloaded final root index is
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/SCI_ALIGN_STAGE7_NGC4449_152390/toltec_umass_edu/NGC4449/reduced/redu04/index.yaml`.
Its SHA-256 is
`e72c7c8f326d514ec0025691ca8b31206c40d0f08b7b50729fc093d7a5273040`.
Independent recursive verification checked exact directory membership, all six
indexes, 33 immutable files totaling 134,493,770 bytes, and the one declared
operational-mutable log; there were no missing, extra, size-mismatched, or
digest-mismatched artifacts.

The final 124-by-5,518 `rms`, `stddev`, `median`, `flagged_frac`, and `weights`
matrices contain 684,232 finite cells apiece. Nonzero counts are 660,300 for
RMS, standard deviation, and median, 326,995 for flagged fraction, and 484,755
for weights. All flagged fractions lie in [0,1], and weights are nonnegative.

The configured native busy-row rule applied 36 factor-zero suppressions across
22 scans and seven networks. The owner explicitly judged the restored
suppression not overly aggressive. This is the bounded disposition of the
correctness repair at `3ebc2a67f`; it is not a general accepted successor
baseline claim.

## Reduction Auditor Repairs

The downloaded result confirmed both reported false negatives:

- PyYAML parses unquoted ISO `build_timestamp` as `datetime`, which the raw v3
  validator previously rejected despite complete revision/version identity.
- TolProj's current project path is outside the historical validation-root
  regex even though the config unambiguously records `reduction_type: science`.

Commit `ff7899668` admits a nonempty string or PyYAML `datetime` only for
`build_timestamp`; revision and version remain required nonempty strings, and
unrelated timestamp types remain invalid. It also derives mode from exactly one
recognized config reduction type (`pointing -> point`, `oof`, `beammap`, or
`science`) and combines that with legacy path evidence. It never derives mode
from the requested expectation, and the regression proves a mismatched request
still fails.

Focused regressions pass 2/2; the complete audit module passes 92/92. The real
`redu04` audit exits zero with recognized mode `science`, valid raw v3
provenance, all required science sidecars valid, and no path-label claim.

## Bounded Source Cleanup

Before deletion, the active graph comprised eight library and three CLI
translation units. None of the 13 census targets was compiled or included.
Recursive include analysis reached 733 repository headers, including all
corresponding engine, Wiener, and utility headers. The Wiener source was
exactly a one-include placeholder. Commit `aa85a2287` removed:

- legacy mains `main_old.cpp`, `mpi_main.cpp`, `kids_main.cpp`, and
  `lali_main.cpp`;
- empty/commented engine sources `todproc.cpp`, `kidsproc.cpp`, `engine.cpp`,
  `lali.cpp`, `pointing.cpp`, and `beammap.cpp`;
- the unbuilt `wiener_filter.cpp` placeholder;
- empty `utils.cpp` and `dummy.cpp`;
- seven stale commented CMake source entries; and
- four now-obsolete legacy-main config-audit exceptions.

No header or mature RTC/PTC/JINC/Wiener algorithm changed. Post-cleanup static
reachability remains 11 active translation units and 733 repository headers,
with no deleted or other non-target `.cpp` reachable.

## Science-Ledger Disposition

`validation/accepted_runs.json` remains byte-for-byte unchanged at SHA-256
`4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb`.
`validation/intended_science_changes.json` remains byte-for-byte unchanged at
SHA-256
`b6635d5f0e0282f7db716c89dd5cfbae1c4700040b85d410ba724f2a334cd450`.

The current science-change schema admits only accepted entries and requires at
least one referenced `accepted_run`. The Stage 7 campaign has complete bounded
execution/provenance evidence and the owner's narrow busy-row disposition, but
it lacks a same-input accepted successor comparison and no explicit owner
decision promotes it to the accepted-run ledger. Adding either record would
invent scientific acceptance, so both ledgers are intentionally preserved.

## Repository Preservation

The durable [ref/worktree inventory](REPOSITORY_REF_WORKTREE_INVENTORY_2026-08-26.md)
classifies all ref families, 21 worktrees, eight release tags, three stashes,
470 untracked user files, 41 thin transfer bundles, and unreachable/reflog-only
commits. It identifies only 19 contained local branches as mechanically
retirable after integration. All divergent branches, all remote topic refs,
the tested-SHA repair ref, forensic pointers, Codex-managed refs, stashes,
untracked artifacts, and active worktrees remain retained.

The inventory also records an unresolved authority-name discrepancy: governing
documents name `codex/conan2-adaptation`, while the active build worktree is
`codex/build-adaptation` at `d9843e85`, five commits ahead of its remote. Both
must remain until the owner reconciles the name and path. This build lane is
not part of the present candidate.

## Validation And Handoff

The definitive clean-build and gate results must be run at the final evidence
commit and reported with the task handoff so the embedded executable revision
matches exact candidate HEAD. One established disabled CTest,
`MapFitterLifecycle.DISABLED_ExactProductSequence`, is reported separately and
is not treated as a runnable pass or unexpected skip.

Phase 5 readiness remains `preparing`. Its 2026-08-26 reevaluation updates the
stale build-review blocker to the adopted Conan 2 Adapt disposition while
retaining the real successor build/dependency gates. The same-SHA four-mode
matrix and accepted-successor-baseline blockers also remain. No validation
profile, fixture gate, accepted run, or promotion condition was weakened.

A short Unity point smoke at final candidate HEAD is recommended before local
branch retirement. It validates the cluster build/reachability path after
source deletion and the final embedded revision. Repeating the long Stage 7
science campaign is not required to validate these inactive-source, auditor,
and documentation changes; its scientific evidence remains bound to exact
`3ebc2a67f`.

The inventory provides the safe user sequence: first preserve unreachable tips
and untracked artifacts in verified archives; create forensic tags for exact
`3ebc2a67f` and final candidate HEAD; fetch and re-prove the fast-forward;
dry-run and perform the owner-controlled mainline/tag push; run the short Unity
point smoke; then delete only the guarded 19 local ancestor branches. No remote
topic-ref deletion is recommended in this cycle.
