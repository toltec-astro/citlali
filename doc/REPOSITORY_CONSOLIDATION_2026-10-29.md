# Citlali working-folder consolidation — 2026-10-29

> Historical preparation record. Grant expanded this to grouped, backed-up
> 2026-refactor consolidation on October 30, excluding 4.x/5.x work.
> See the [completed operation record](REPOSITORY_CONSOLIDATION_2026-10-30.md).

Status: preservation completed; first one-folder removal batch proposed, **not executed**.
Owner: Grant Wilson. This is a bounded cleanup work order, not a governance amendment.

## Plain-language purpose

Reduce 20 working folders to a core of five, with a separate owner approval for
small batches of physical folder removal. A working folder is one usable copy
of files. A branch is a saved line of work in Git; removing a working folder
need not remove that branch or its history. All 104 local branches, three
stashes (parked unsaved changes), and existing recovery evidence remain protected.
Historical work will have an indexed recovery location instead of requiring
15 extra folders to remain open-ended development locations.

The earlier 33-registration prune is complete. It removed stale bookkeeping for
folders already absent; it did not remove any of these 20 folders. This new
work order prepares physical-folder consolidation and has a separate approval gate.

## Preflight and scope

- Read `AGENTS.md`, `CURRENT_STATE.md`, `REFACTOR_STATUS.md`,
  `governance/ENGINEERING_GOVERNANCE.md`,
  `governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`, and
  `governance/REVIEW_AND_CONFORMANCE.md`. Existing canonical effectiveness is
  recorded in `INTEGRATION_LEDGER.md`; this cleanup changes none of those rules.
- Risk: Tier 2 evidence cleanup. Owned branch/worktree:
  `codex/citlali-engineering-20261029` at `/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor`.
  Clean starting source: `3579f1055872aab546179c6d28556bb2ece1986b`.
  Canonical: `e32c4584dec07617d6caa857d39bf13047b6c4b0`.
- Scope: inventory; independent history and unsaved-file backups; restoration
  tests; dependency and chat-ownership checks; staged proposals and documentation.
- Expected source changes: this record, `CURRENT_STATE.md`, and `REFACTOR_STATUS.md`.
  No application implementation slot is opened by this evidence-preparation work.
- Excluded: scientific policy/algorithm changes, merges, canonical integration,
  pushes, Unity access, reductions, branch/stash deletion, garbage collection,
  reflog expiration, and modification of frozen runtimes or products.
- User authority: safe preparation is authorized now. Each deletion batch needs
  its own explicit approval, exactly as Grant requested. No approval is inferred
  from this plan or from a folder being clean or old.
- Gates: full independent backup restore and object integrity, exact refs and
  stash-object identities, dirty-file byte checks, candidate source restore,
  ignored-payload and runtime-consumer audit, independent exact-commit review,
  then fresh pre/post checks for each approved batch.
- Stop if candidate identity, dirt, ignored files, locks, ownership, process use,
  dependency assumptions, backup hashes, or protected evidence differ. Reassess
  the exact batch; do not broaden deletion or use force to overcome a stop.

## The five-folder destination

| Retain | Purpose |
| --- | --- |
| `/Users/gwilson/GitHub/citlali` | Main repository; owns shared Git history and preserves legacy baseline. |
| `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor` | Active pointing and scientific assessment; live work is protected. |
| `/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor` | Engineering and consolidation; reuse this existing folder. |
| `/Users/gwilson/.codex/worktrees/884f/citlali-refactor` | Pinned scientific-contract chat; retain its detached source identity. |
| `/Users/gwilson/GitHub/citlali-refactor` | Pinned/recent science chats and current builds’ shared dependency sources. |

The older shared science folder is a real dependency: engineering, pointing and
pilot builds each refer to 18 source trees under its `build/_deps`. Keep it.
PTC remains a temporary sixth hold until its unsaved report, ignored binaries,
and saved runtime/source bindings have a reviewed restoration path. The five
folders are a destination, not a claim that the other 15 are ready to delete.

## Preservation completed

Evidence directory (outside the working folders being consolidated):

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-consolidation-20261029`

Its `RECOVERY.md` is the human entry point. `history-manifest.json` records all
371 original refs, all 104 branch tips, three stash entries, the detached
checkout, and the two recovery-only commits from the prior metadata cleanup.
`repository-history.bundle` is self-contained, with no prerequisite repository.
It was restored into an independent bare Git repository without shared-object
links, all saved refs compared exactly, and full strict object integrity passed.

| Artifact | SHA-256 | What it preserves |
| --- | --- | --- |
| `repository-history.bundle` | `7c91740468b2d73fb1ebb49b6d5a427aeaaa95d374e252cd5c97f4cebcd6be2b` | All captured named history, all three stash objects/parents, detached HEAD, and prior two recovery commits; 1,519,389,242 bytes |
| `existing-worktree-recovery.bundle` | `6c9f1d3e15bc501f1f3f1c4f55dc885cd3656f797be200ca8e1406ed3c754b1b` | Nine additional commits retained only by three older folders’ Git recovery records; 4,243,513 bytes; restore main bundle first |
| `batch-01-paired-guard-git-admin.tar.gz` | `30057d507e80f8aabc356fec4a7ae2d506b1f1d30ff3808c31f15de78c4865fd` | Exact first candidate’s Git administration, index and local recovery records |

The nine extra commits belong to collaborator-pilot (one), fruit-adaptive (five),
and PTC (three). The supplemental bundle was fetched into the independent
restore, every supplemental ref compared, and full strict integrity passed again.
The earlier 41 original recovery bundles and two metadata-prune preservation
artifacts stay in place. All 41 original hashes and their 31 distinct prerequisite
histories were checked; the full backup contains their required history.

Separate archives preserve the three older folders’ unsaved files, with
individual source/archive/restored byte checks and index plus binary patches:

| Folder | Preserved entries | Archive |
| --- | --- | --- |
| PTC rank-response | 1 modified report | `ptc-rank-response-unsaved.tar.gz` |
| Citlali performance | 1 untracked command note | `citlali-perf-unsaved.tar.gz` |
| Older shared science checkout | 465 modified/untracked paths | `citlali-refactor-older-unsaved.tar.gz` |

These 467 entries include the original recovery bundles in the older science
checkout. Their manifests contain all paths and digests. Ignored build/runtime
payloads are deliberately a separate gate; these archives do not claim to cover
them. Eleven historical build caches refer to a dependency tree that is now
absent, so their existing binaries must be saved, not assumed reproducible.
Backups are local, not an off-machine disaster-recovery copy.

The pointing chat stayed active. The inventory observed
`9bcbfcf539355a36faf18c5114492fd56c600fb0` with new local edits; the later history snapshot at
`2026-10-29T22:45:38.692333+00:00` includes committed pointing source
`a8b720e17ef5b72afc2824d55570ca0ec1e159b7`. These are separate observations. No moving pointing
files were archived as a supposedly stable snapshot, and subsequent pointing
commits/edits are outside that backup cutoff. Its live folder, trials and
products remain untouched. There is no claim of an atomic whole-machine snapshot.

## First approval batch: one source-only folder

- Folder: `/Users/gwilson/.codex/worktrees/basis-paired-guard-20261016/citlali-refactor`.
- Branch kept: `codex/basis-paired-guard-20261016`.
- Exact HEAD: `386b6af4fd23fadc306865e2838845463f59f36a`.
- Purpose: an older paired-worker supervision experiment. Its committed stop
  record withheld operational readiness after a November RTC failure; no
  external paired launcher was issued by that work.
- Unique work: three commits, all retained in the branch and verified backup.
  The work is preserved as historical source/evidence, not declared scientifically
  accepted, integrated, or obsolete.
- Local files: clean; zero untracked or ignored paths; no symlinks. About
  1.71 GiB (1.84 GB) reported folder allocation; actual freed disk space can vary.
- Recovery proof: independent checkout of this exact source from the restored
  backup, with 29,324 tracked paths, matching index entries and clean status.
- Dependency scan: no inbound reference found in the bounded scan of 108 current
  POINT metadata files and 145 historical campaign setup/launch/state files.
  This is not an exhaustive host-wide absence claim.
- Ownership: its Git administration names the chat “OOF Optimizing”
  (`01a0df4e-e19f-7ea3-bfb3-31453711cb9d`). Direct read-only inspection found
  that chat not loaded and its current folder is the retained older science
  checkout. The current task only has the engineering worktree attached. This
  does not prove absence of every historical app attachment; no other chat was
  messaged, archived, or changed.
- Active use: `lsof` reported no open-file/cwd consumer at the recorded inspection;
  repeat this and the app ownership check immediately before removal.

**Proposed action after explicit approval only:** ordinary, non-force
`git worktree remove` for this exact folder. That removes its checked-out files
and its per-worktree Git registration; it keeps the branch, shared history,
stashes and backups. Use the app’s archive operation instead only if a matching
managed attachment is established and its scope is proven equivalent. Do not
unlink unrelated chats or improvise a force-removal fallback.

Expected result: 20 existing working folders become 19. All five core folders,
PTC hold, branch tips, stash entries, original recovery artifacts, shared
reflogs and frozen runtime identities remain protected. Before/after checks
must allow separately recorded changes by the active pointing chat and this
engineering lane; unrelated differences are a stop, not evidence of preservation.

Recovery under the unchanged branch tip is:

```sh
git -C /Users/gwilson/GitHub/citlali worktree add /Users/gwilson/.codex/worktrees/basis-paired-guard-20261016/citlali-refactor codex/basis-paired-guard-20261016
```

First verify that branch still equals the approved full HEAD and the destination
is absent. If it moved, restore the exact approved commit detached at a fresh
path; never reset a moved branch. `RECOVERY.md` covers independent backup recovery.
This is a recovery instruction, not a command executed during preparation.

## Remaining small batches: preparation queue

This grouping makes all 15 eventual candidates visible. Only batch 1 is proposed
for approval now. Later batches need ignored-payload archives, byte-verified
restoration, exact source/path bindings, current ownership/use checks, independent
review, and a fresh small approval. Reorder or split a group if dependencies
require it; no fixed deadline overrides preservation.

| Batch | Folder labels | State | Folder count if eventually approved and completed |
| --- | --- | --- | --- |
| 1 | `basis-paired-guard-20261016` | Prepared for exact owner approval | 19 |
| 2 | `basis-fit-storage-20261014`, `basis-signed-20261014`, `basis-sph-rtc-padding-20261015` | Preparation only; no deletion approval requested | 16 |
| 3 | `basis-rtc-padding-20261014`, `basis-sph-hero-20261016`, `basis-sph-support-20261020` | Preparation only; no deletion approval requested | 13 |
| 4 | `november-a1100-20261023`, `a1100-hero-20261025`, `a1100-exclude-20261026` | Preparation only; no deletion approval requested | 10 |
| 5 | `fruit-adaptive-cleaning-20261007`, `fruit-artifact-replay-20261010` | Preparation only; no deletion approval requested | 8 |
| 6 | `collaborator-pilot-20261027`, `citlali-perf` | Preparation only; no deletion approval requested | 6 |
| 7 | `ptc-rank-response` | Preparation only; no deletion approval requested | 5 |

For batch 2, saved launchers import fit-storage and signed source, require exact
Git identities, and bind signed/padding executables. Archive their ignored
payloads (about 9.65 MB, 612.69 MB and 450.71 MB respectively) and explicitly
preserve how to restore the original absolute paths before proposing deletion.
Closed historical runs do not make saved replay scripts independent of these
folders. PTC has additional binary families and 25 external metadata references;
its hold is last. Other later groups still need equally specific consumer checks.

No branch deletion is included. After folder consolidation, a separate owner
review can decide which saved lines should be marked historical or retired.
The 104 branch names should not be presented as 104 active projects.

## Complete starting folder inventory

Exact HEADs below are the earlier inventory snapshot, not a claim that active
pointing or engineering work stopped. Full statuses, ignored roots, retaining
refs, unique commits and recovery object origins are in the evidence JSON.

| Folder | Inventory HEAD | Planned batch or keep |
| --- | --- | --- |
| `/Users/gwilson/GitHub/citlali` | `ffc6b9070f4744f9778f3db71cdc468846d1da89` | keep |
| `/Users/gwilson/.codex/worktrees/884f/citlali-refactor` | `ae953ed4d87d1f693d2bbf42aebbc25ef730c771` | keep |
| `/Users/gwilson/.codex/worktrees/a1100-exclude-20261026/citlali-refactor` | `13a715a8847c97b3516826de308939b3c146c690` | 4 |
| `/Users/gwilson/.codex/worktrees/a1100-hero-20261025/citlali-refactor` | `58a9805dfa8a7305c2b762853f81e2b99405f77c` | 4 |
| `/Users/gwilson/.codex/worktrees/basis-fit-storage-20261014/citlali-refactor` | `ccbd57eec9158f6f3774eb096ccfe6741308868e` | 2 |
| `/Users/gwilson/.codex/worktrees/basis-paired-guard-20261016/citlali-refactor` | `386b6af4fd23fadc306865e2838845463f59f36a` | 1 |
| `/Users/gwilson/.codex/worktrees/basis-rtc-padding-20261014/citlali-refactor` | `15f3e957b29086e06d7ed9e4e228dbf5808a778c` | 3 |
| `/Users/gwilson/.codex/worktrees/basis-signed-20261014/citlali-refactor` | `660c509f25b0539167a079564ec91ed17d4f124d` | 2 |
| `/Users/gwilson/.codex/worktrees/basis-sph-hero-20261016/citlali-refactor` | `de971ef721ca783649dc336e9e2695a75250310c` | 3 |
| `/Users/gwilson/.codex/worktrees/basis-sph-rtc-padding-20261015/citlali-refactor` | `986de4af8e9534b6c7d4db2457bf7fb320ceedca` | 2 |
| `/Users/gwilson/.codex/worktrees/basis-sph-support-20261020/citlali-refactor` | `c5551bcabc5e1bfae323cdd57a844a5591347e38` | 3 |
| `/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor` | `3579f1055872aab546179c6d28556bb2ece1986b` | keep |
| `/Users/gwilson/.codex/worktrees/collaborator-pilot-20261027/citlali-refactor` | `16d1b1aaeaed3526cb6383672d47f1fc770e78c4` | 6 |
| `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor` | `9bcbfcf539355a36faf18c5114492fd56c600fb0` | keep |
| `/Users/gwilson/.codex/worktrees/fruit-adaptive-cleaning-20261007/citlali-refactor` | `3a74a5c535c80cf666a3ef652497b04a10fc422d` | 5 |
| `/Users/gwilson/.codex/worktrees/fruit-artifact-replay-20261010/citlali-refactor` | `c26f7d924dcfd5939c800c0200a2b7252441d34d` | 5 |
| `/Users/gwilson/.codex/worktrees/november-a1100-20261023/citlali-refactor` | `7121fdb63b88712aade9ec8739a64aea14fc2b4f` | 4 |
| `/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor` | `db04757dd00933e26d3a6c679d175122f1db2cc6` | 7 |
| `/Users/gwilson/GitHub/citlali-perf` | `d01ffa5981551345eeb1c765c24125200b896847` | 6 |
| `/Users/gwilson/GitHub/citlali-refactor` | `dd4a3599279a0d09eeccaf3c63a4ac2782198d57` | keep |

## Conformance and completion boundary

Scientific/behavioral conformance: unchanged; no algorithm, configuration,
observation or scientific acceptance changed. Architecture/ownership: unchanged;
real build, runtime and chat ownership dependencies are documented and held.
Repository/evidence hygiene: preservation and independent restoration completed;
physical consolidation remains pending exact approval. Native/application tests
and Spack/Unity qualification are not triggered by documentation and backup work.
No reduction or runtime replay was executed. Independent review and exact final
commit binding are stored with the evidence; approval and execution results must
be appended when a batch actually occurs.
