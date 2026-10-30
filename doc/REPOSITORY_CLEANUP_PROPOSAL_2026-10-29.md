# Repository cleanup proposal — 2026-10-29

Status: **completed under exact owner approval; all protected state verified intact**.

## Approved cleanup execution — 2026-10-29

Grant explicitly approved: "prune only the 33 missing worktree registrations in
the proposal, then verify that all protected branches, stashes, existing
worktrees, and recovery artifacts remain intact."

The exact proposal was executed at source
`941ccf476da011fa72043488fa566f7aafe45785`. Immediately before execution,
all 33 paths were still absent, branch/HEAD identities and all 233 administrative
files matched the approved manifest, and Git's complete dry run matched the
33 administrative IDs. The immutable administrative archive and reflog-commit
bundle retained their recorded hashes. All 414 administrative commit IDs and
10,611 index objects were rechecked; the only two commits outside named-ref
reachability were covered by the verified recovery bundle. All original bundle
prerequisites remained reachable.

`git worktree prune --verbose --expire now` removed exactly those registrations,
including their archived per-worktree logs. Registrations decreased from 53 to
20. A fresh dry run is empty. The immediate before/after comparison proved:

- All 371 refs, including 104 local branch tips, and all three stashes unchanged.
- All 20 existing worktrees retained, with identical HEADs, status, indexes,
  dirty-file contents and checkout/ignored-root identities. The three original
  dirty worktrees and detached checkout remained protected.
- All shared reflog bytes, 41 original recovery bundles, both preservation
  artifacts and the checked frozen runtime files unchanged.

Independent preflight and post-operation verification passed without findings,
including 467 dirty entries, all 33 combined recovery prerequisite identities,
and the continued presence of both preserved reflog-only commit objects.
No branch/tag deletion, existing-worktree removal, shared reflog expiration,
garbage collection, remote action, trial modification or reduction occurred.

The approval covers this completed batch only. The original proposal and
before-operation inventory remain historical evidence; they are not permission
for future cleanup. Execution records and both independent snapshots are retained
under this task's artifact directory in `cleanup-execution-20261029/`:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-engineering-20261029`

`execution-result.json` binds the exact command, owner approval, proposal digest
and Git output; `before.json`, `after.json`, `verification.json` and the two
`citlali-independent-cleanup-*.json` files bind preservation checks. The original
backup retention/prerequisite obligations continue after pruning.

Scientific/behavioral conformance: preserved; no application change or
observational gate triggered. Architecture/ownership: unchanged. Repository and
evidence hygiene: pass, independently verified. This follow-up changes only the
cleanup/status documentation; its commit advances the owned engineering branch
after the unchanged-ref comparison, retaining the reviewed source as ancestry.

## Original approved proposal

The proposal below is retained as the exact pre-operation record. Its prerequisites were rechecked before the approved execution; it grants no authority for another cleanup batch.

The shared repository is `/Users/gwilson/GitHub/citlali/.git`. The reviewed candidate is `77063575c4fac2bb207ca462435009358be396f6`; canonical `codex/refactor-mainline` remains `e32c4584dec07617d6caa857d39bf13047b6c4b0`. The candidate is 262 commits ahead with zero canonical-only commits. Ancestry does not establish scientific acceptance. Engineering continues separately on `codex/citlali-engineering-20261029`.

## Protected material

The initial refresh recorded 104 local branches, 121 cached remote refs, 22 tags, 18 archive refs, 104 Codex refs, 53 worktree registrations, 20 existing paths, and 33 missing registrations. Ref and dirty counts are snapshot values; see the final machine-readable inventory for later engineering/doc commits. No remote query or fetch was performed.

| Protected item | Disposition |
| --- | --- |
| Three original dirty worktrees | Preserve contents and ownership; no cleanup approval. `ptc-rank-response/citlali-refactor`: one modified report; `GitHub/citlali-perf`: one untracked command file; `GitHub/citlali-refactor`: modified `AGENTS.md` and 464 fully enumerated untracked files. |
| Detached checkout `/Users/gwilson/.codex/worktrees/884f/citlali-refactor` | Retain exact HEAD `ae953ed4d87d1f693d2bbf42aebbc25ef730c771`; disposition unresolved. |
| Three stashes | Preserve unchanged: `75cc620256a460d6002ad367cecf17f7df1dbd1b`, `1c98f93376776b4dd51494f230c559cf1e031a5b`, `b27271fc76ffea063341465a5a62112167f50e01`. |
| 41 thin recovery bundles, 19,014,724 bytes | Retain originals and 31 exact prerequisite commits. All prerequisites were reachable from named refs at inventory time; original bundles were not fully restore-tested. |
| Uniquely retaining branches | Preserve all; initial refresh found 15 such branches after the new engineering branch mechanically made the candidate tip nonexclusive. The original 16-branch preservation set remains protected; exclusivity changes are not disposal authority. |
| Ignored builds, dependency environments and frozen runtimes | Preserve in place. Inventory lists ignored roots, not full backups. |
| Pointing candidate, trial inputs/configs/products, reference reductions | Preserve; scientific product review remains owned by the original thread. |

The previous `fruit-pipeline-freeze/citlali-refactor` worktree registration disappeared outside this audit; a new Codex snapshot ref retained its HEAD `5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5`. This audit did not archive or remove it and does not attribute the actor. The new engineering worktree accounts for the replacement registration.

## Preservation required before the proposed batch

| Artifact | Exact identity |
| --- | --- |
| Missing-registration administrative archive: `missing-worktree-admin-records.tar.gz` | 6,047,496 bytes; SHA-256 `97199075cbc204cbbe7667611c549d0b0849b4c9528fbb736dab71478ef5d2e7`. Includes exact administrative files, indexes and per-worktree logs. Every archived file and unchanged source hash was verified. |
| Reflog-only commit preservation: `missing-worktree-reflog-commits.bundle` | 4,414 bytes; SHA-256 `bb0afa55e3bcd4a7fc940547d8aaedbaefa6b019b439094a92f1106ae68dc921`. Thin bundle; prerequisite retention is mandatory. |

The administrative audit checked 414 commit IDs. Two exist outside named-ref reachability and are included in the object bundle:

- `b79f3451116abc3978daba94c96387e444b2486b`, from the `citlali-timestream-successor-cal-001` HEAD log.
- `b5a05c424c12ace0be3d317900a6ef819360c1b1`, from the `citlali-timestream-successor-rtc-event-background-001` HEAD log.

Their required prerequisite commits are `98380090e770f960b0decb07ebe3cabc77993132` and `8ef77477823a5393f9cc7860d8f963509aee0fc8`. The preservation manifest enumerates named refs retaining each prerequisite; retain at least one such named ref for each exact commit or a separately verified complete backup. This thin bundle is not self-contained.

Restoration passed in a separate bare repository without alternates, seeded with 48,219 prerequisite-history objects. Both preservation tips were absent before import. Bundle verification, exact commit bytes and recursive-tree comparison, and full strict `git fsck --no-reflogs` passed. No shared refs were written. All 10,611 unique objects referenced by the 33 administrative indexes were reachable from named refs; no index-only object was found.

Both preservation artifacts and manifests have been copied with digest verification to the retained task artifact directory:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-engineering-20261029`

The temporary preparation directory is not the retained copy. Keep this artifact directory until a separately verified owner-controlled archive replaces it; verify these digests again before any approved pruning. This is local recovery preservation, not an off-machine disaster-recovery claim. Full backup/index/object details are in `cleanup-batch.json`, `repository-inventory.json`, and `admin-object-preservation.json`.

## Exact proposed metadata batch

The following 33 missing paths match the prior review. Every branch remains protected. The administrative ID is the directory under `/Users/gwilson/GitHub/citlali/.git/worktrees/`. No existing checkout is included.

| Administrative ID | Missing checkout path | Retained branch ref | Exact HEAD |
| --- | --- | --- | --- |
| `citlali-cti-od-002-coefficient-authority-2026-09-05` | `/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05` | `refs/heads/codex/cti-od-002-coefficient-authority-2026-09-05` | `867879b992598484e1bc5b481e5e14ff91c04dab` |
| `citlali-integrate-map-profile-admission-2026-09-08` | `/private/tmp/citlali-integrate-map-profile-admission-2026-09-08` | `refs/heads/codex/integrate-map-profile-admission-2026-09-08` | `3efd079bb5c28f28f6f2d49296a29e966bde050b` |
| `citlali-integrate-map-space-conformance-001-2026-09-05` | `/private/tmp/citlali-integrate-map-space-conformance-001-2026-09-05` | `refs/heads/codex/integrate-map-space-conformance-001-2026-09-05` | `00b974c9039d4c3025dcce18f26bca69d36af9c3` |
| `citlali-integrate-map-v02-r04-2026-09-08` | `/private/tmp/citlali-integrate-map-v02-r04-2026-09-08` | `refs/heads/codex/integrate-map-v02-r04-2026-09-08` | `b675bb64a7054f7b24403c79898965e8765cfd02` |
| `citlali-integrate-timestream-rtc-assessment-001` | `/private/tmp/citlali-integrate-timestream-rtc-assessment-001` | `refs/heads/codex/integrate-timestream-rtc-assessment-001` | `0f52e1a421416eadaf7d8d99d64f537132395c0c` |
| `citlali-map-contract-reconciliation-2026-09-07` | `/private/tmp/citlali-map-contract-reconciliation-2026-09-07` | `refs/heads/codex/map-contract-reconciliation-2026-09-07` | `b972d2850c2521662af59bba31d71a3ee525033c` |
| `citlali-map-successor-first-forward-001` | `/private/tmp/citlali-map-successor-first-forward-001` | `refs/heads/codex/map-successor-first-forward-001` | `ac1bf4eecc9973515195151bf865e25893e2e5d5` |
| `citlali-noi-coefficient-nonmutation-work-order-2026-09-06` | `/private/tmp/citlali-noi-coefficient-nonmutation-work-order-2026-09-06` | `refs/heads/codex/noi-coefficient-nonmutation-work-order-2026-09-06` | `34a378e7298b2ee69cc2a81a56f8bebc2f835657` |
| `citlali-point-additive-20260923` | `/private/tmp/citlali-point-additive-20260923` | `refs/heads/codex/map-successor-first-forward-001` | `ac1bf4eecc9973515195151bf865e25893e2e5d5` |
| `citlali-refactor-mainline-after-map-space-repair-2026-09-04` | `/private/tmp/citlali-refactor-mainline-after-map-space-repair-2026-09-04` | `refs/heads/codex/refactor-mainline` | `e32c4584dec07617d6caa857d39bf13047b6c4b0` |
| `citlali-rtc-disturbance-burden-census-001` | `/private/tmp/citlali-rtc-disturbance-burden-census-001` | `refs/heads/codex/rtc-disturbance-burden-census-001` | `e7ccd3da4bf2ac3412508dd07118f136272655c5` |
| `citlali-rtc-real-data-audit-001` | `/private/tmp/citlali-rtc-real-data-audit-001` | `refs/heads/codex/rtc-real-data-audit-001` | `4a7e7775d3d8350519d75b05a8b8dc0bdf017641` |
| `citlali-sci-fruit-method-scope-20260908` | `/private/tmp/citlali-sci-fruit-method-scope-20260908` | `refs/heads/codex/sci-fruit-historical-control-method-scope-r0.1` | `5fc4ce839a5deb0d44096f17d0b7c2dbddcaedc2` |
| `citlali-spack-release-contract-reconciliation` | `/private/tmp/citlali-spack-release-contract-reconciliation` | `refs/heads/codex/spack-release-contract-reconciliation` | `a586fd2cb11d427053205109bf5d7a9a617a5a58` |
| `citlali-timestream-successor-cal-001` | `/private/tmp/citlali-timestream-successor-cal-001` | `refs/heads/codex/timestream-successor-multinetwork-001` | `686c280037762450b1870308c920e9191216dc16` |
| `citlali-timestream-successor-d2-canonical-admission` | `/private/tmp/citlali-timestream-successor-d2-canonical-admission` | `refs/heads/codex/timestream-successor-d2-canonical-admission` | `218988dc21cfc279cdfe7a4318b574f070cd45f9` |
| `citlali-timestream-successor-d2-canonical-admission-002` | `/private/tmp/citlali-timestream-successor-d2-canonical-admission-002` | `refs/heads/codex/timestream-successor-d2-canonical-admission-002` | `5244e04638db5aa92180feab52acd782229cfa32` |
| `citlali-timestream-successor-development-default-001` | `/private/tmp/citlali-timestream-successor-development-default-001` | `refs/heads/codex/timestream-successor-development-default-001` | `e633dd3e529897defa8d38d6f189f27289401492` |
| `citlali-timestream-successor-identity-route` | `/private/tmp/citlali-timestream-successor-identity-route` | `refs/heads/codex/timestream-successor-identity-route` | `655160ba59a8abcddb5984d3241857c4ae969589` |
| `citlali-timestream-successor-ptc-spod-001` | `/private/tmp/citlali-timestream-successor-ptc-spod-001` | `refs/heads/codex/timestream-successor-ptc-spod-001` | `4bf470389fe7958f27bcda7a300baf3f31959b86` |
| `citlali-timestream-successor-rtc-donor-fill-001` | `/private/tmp/citlali-timestream-successor-rtc-donor-fill-001` | `refs/heads/codex/timestream-successor-rtc-donor-fill-001` | `5de31ec52cb3b079b22e47fee95b0725662c69a8` |
| `citlali-timestream-successor-rtc-event-background-001` | `/private/tmp/citlali-timestream-successor-rtc-event-background-001` | `refs/heads/codex/timestream-successor-rtc-event-background-001` | `8fff882dd0e8891512b01e6065cdb442981b1e5d` |
| `citlali-timestream-successor-rtc-jump-apply-001` | `/private/tmp/citlali-timestream-successor-rtc-jump-apply-001` | `refs/heads/codex/timestream-successor-rtc-jump-apply-001` | `4d609306c868338e2f5c6b7920b81a4d39ebf967` |
| `citlali-timestream-successor-rtc-line-power-001` | `/private/tmp/citlali-timestream-successor-rtc-line-power-001` | `refs/heads/codex/timestream-successor-rtc-line-power-001` | `4c066f1e771b904ae78ed030c232f055dbc8dd16` |
| `citlali-timestream-successor-rtc-line-transfer-001` | `/private/tmp/citlali-timestream-successor-rtc-line-transfer-001` | `refs/heads/codex/timestream-successor-rtc-line-transfer-001` | `0bcb132df284257418c6d5196b50c549b9775673` |
| `citlali-timestream-successor-rtc-native-spectral-learn-001` | `/private/tmp/citlali-timestream-successor-rtc-native-spectral-learn-001` | `refs/heads/codex/timestream-successor-rtc-native-spectral-learn-001` | `77cc29503ae54484cb6f128744f7d32aef0dd8f2` |
| `citlali-timestream-successor-rtc-notch-recovery-001` | `/private/tmp/citlali-timestream-successor-rtc-notch-recovery-001` | `refs/heads/codex/timestream-successor-rtc-notch-recovery-001` | `af716d166447624c7efbc0575c83b9272e7b472f` |
| `citlali-timestream-successor-rtc-spike-learn-001` | `/private/tmp/citlali-timestream-successor-rtc-spike-learn-001` | `refs/heads/codex/timestream-successor-rtc-spike-learn-001` | `2d41532ce3493ef329e35f438fe433a69c7c93a2` |
| `citlali-timestream-successor-rtc-transient-apply-001` | `/private/tmp/citlali-timestream-successor-rtc-transient-apply-001` | `refs/heads/codex/timestream-successor-rtc-transient-apply-001` | `d1e9bab76dadded53e8dd107947d508142e2b8ed` |
| `citlali-timestream-successor-unity-gcc13-abi-conformance` | `/private/tmp/citlali-timestream-successor-unity-gcc13-abi-conformance` | `refs/heads/codex/timestream-successor-unity-gcc13-abi-conformance` | `2f59ef8b9906b2912c4980fbcf9c7ab3a28c310a` |
| `citlali-timestream-successor-val-d2-prerequisite-001` | `/private/tmp/citlali-timestream-successor-val-d2-prerequisite-001` | `refs/heads/codex/timestream-successor-val-d2-prerequisite-001` | `080df2a1431487e8cabc255beb9ddc59c0721b59` |
| `citlali-timestream-successor-val-native-admission-001` | `/private/tmp/citlali-timestream-successor-val-native-admission-001` | `refs/heads/codex/timestream-successor-val-native-admission-001` | `8eccf8990027b8f65febb5f20d9a415829152322` |
| `citlali-timestream-successor-val-native-target-001` | `/private/tmp/citlali-timestream-successor-val-native-target-001` | `refs/heads/codex/timestream-successor-val-native-target-001` | `37e6fb6ce50acba644f04b6b6b546762bf07f384` |

## Approval boundary and execution conditions

Grant must approve exactly this batch. Approval would cover removal of these archived administrative registrations, including their per-worktree logs, after preservation and a fresh unchanged-state check. It would not cover branch/tag/ref deletion, existing worktree removal, stash changes, recovery-bundle deletion, shared reflog expiration, garbage collection, history rewriting, remote changes, or trial/evidence removal.

Immediately before execution:

1. Verify durable archive and object-bundle digests and retained prerequisite identities.
2. Recheck every absent path with `lexists`, every branch/HEAD identity, and every archived administrative-file hash.
3. Repeat HEAD/ORIG_HEAD/log/index reachability inspection; stop for any missing object or newly unpreserved commit/index object.
4. Compare the complete dry-run set with these exact 33 administrative IDs; stop on any addition, omission or changed identity.

The read-only dry run is:

```bash
git -C /Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor worktree prune --dry-run --verbose --expire now
```

Only after exact approval and all rechecks, the corresponding operation would use `worktree prune --verbose --expire now` on that same repository. It removes bookkeeping, not the retained branches or bulk tracked evidence. No pruning operation had been executed when this proposal was prepared. The later approved execution is recorded above.

## Branch disposition and future storage

The refreshed inventory assigns every branch a provisional category: active integration/engineering, awaiting integration disposition, retained evidence, archive candidate, or retirement candidate. Historical ownership, clean status, containment and cached remote presence do not authorize deletion. Owner disposition is still required for every closure or retirement.

Keep compact summaries, scientific decisions, exact source/build identities, input-association manifests, comparison conclusions, and artifact locations/SHA-256 values in Git. Place future bulk FITS, repetitive plots and large diagnostics/logs in retrievable project-controlled storage with access/retention and restore procedures. Preserve existing comparisons and history. Approximately 1.70 GB of validation data remains reachable in candidate history; deleting branch names will not remove it. No LFS migration, history rewrite or garbage collection is proposed.
