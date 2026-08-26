# Repository Ref And Worktree Inventory - 2026-08-26

## Scope And Reference Point

This is a read-only evidence-preservation census taken at code/tool candidate
`ff7899668c1ddd1fb75988c5ef2eb62ceb337430`. Documentation-only descendants do
not change the ancestry classes below, but all commands must recompute their
guards against the eventual final candidate before changing a ref.

Exact Unity-tested application identity is
`3ebc2a67fc32bad69759ff45638484efabf91773`. Remote application mainline was
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`; its merge base with the candidate
was exactly that same commit, and
`origin/codex/refactor-mainline...ff7899668` was `0 73`. An ordinary
fast-forward is therefore possible.

No branch, tag, worktree, stash, bundle, untracked file, remote ref, or
unreachable commit was changed or removed during this census.

## Ref Census

| Ref family | Count | Relation to `ff7899668` | Disposition |
| --- | ---: | --- | --- |
| Local branches | 125 | 1 equal, 26 ancestors, 98 divergent, 0 descendants | Retire only the guarded 19-branch set below after integration; retain every divergent branch |
| Remote-tracking refs | 108 | 32 ancestors, 76 divergent, 0 equal or descendants; includes symbolic `origin/HEAD` | No remote deletion in this cycle |
| Historical release tags | 8 | Four ancestors and four divergent through older release history | Retain all |
| `refs/codex/snapshots` | 67 | 8 ancestors, 59 divergent | Codex-managed; do not remove manually |
| `refs/codex/turn-diffs` | 14 | One capture and 13 non-commit checkpoints | Codex-managed; do not remove manually |
| Stashes | 3 | User work outside ordinary branch reachability | Retain all |

Sixty-five snapshot tips are duplicated by another ref, one is contained by
another ref, and one is snapshot-only:
`refs/codex/snapshots/7b0194474e74fa2450c74c7e31c555e96ac50765`
at `2aa5c12ba20f5ca043609b9b75713cf014394065`. This is another reason not to
prune Codex-managed refs mechanically.

Local and remote branch-name topology is:

- 77 names with identical local and remote tips;
- 40 local-only names;
- 22 remote-only names; and
- eight shared names with different tips.

| Shared branch name | Local | Remote |
| --- | --- | --- |
| `codex/build-adaptation` | `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046` | `4f9c7e55ab41c2d7ca88ffee86bc9e60e8c5727b` |
| `codex/coherent-iq-sidecar-validation` | `9e739db6d17410497aceaf33a4fd22e9d62ff793` | `762e6c1f131ba53392646424681f60dafe66bf29` |
| `codex/fruit-loop-calibration-reference` | `b02fef613cc7e632828ec762fced6a428906c502` | `f70701ad488444f3e2528c6bbe3e798863c9e301` |
| `codex/integrate-sci-noi-002` | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4` |
| `codex/refactor-mainline` | `9aae0e669384c5c0c0dda93debc194d6b8dac787` | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` |
| `codex/scientific-audit-framework` | `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` | `192e0d9b5e3be4eb20522d3319cae346168c4bce` |
| `codex/structural-refactor` | `17148719663dd9f6aca1234f78ac4877f8535507` | `c495c3a4840af50072ed6ebdcd297045a5739fe1` |
| `v4.x` | `bbf14b57f22e9a1e30f2c156f66b199d64d52f95` | `61dfdc0492ef56bc88e572e295369f5b63f7d91d` |

## Integrated Or Superseded Local Branches

The following 19 local branches are ancestors of the candidate, have no
attached worktree, and lose no commit reachability if deleted after application
mainline advances to the final candidate. They are the only unambiguous local
retirement set in this cycle:

```text
codex/converge-apt-align-jinc
codex/fruitloop-feedback-estimator
codex/integrate-sci-map-001
codex/integrate-sci-map-002
codex/integrate-sci-noi-002
codex/repair-apt-prod-001-canonical-baseline-v1
codex/repair-apt-prod-002-observation-contract
codex/repair-apt-prod-003-compact-v2
codex/repair-apt-v2-typed-null-policy
codex/repair-build-git-provenance-refresh
codex/repair-native-explicit-mjd-pointing-support
codex/repair-sci-align-native-cohorts
codex/repair-sci-map-001
codex/repair-sci-map-002
codex/repair-sci-map-002-successor
codex/repair-sci-map-002-successor-2
codex/repair-sci-map-002-successor-3
codex/repair-sci-noi-002
codex/restore-legacy-apt-admission
```

Retain these other ancestor refs:

- `codex/repair-native-consumer-extinction` is the ordinary local and remote
  ref pinning exact Unity-tested `3ebc2a67f`; no tag currently pins it.
- `codex/refactor-mainline` is the canonical authority and must be advanced,
  not deleted.
- `codex/wp7-timestream-integration-candidate` remains the task branch through
  integration.
- `codex/fruit-loop-calibration-reference` and `codex/structural-refactor` are
  governing historical forensic pointers.
- `v4.x`, all eight release tags, and
  `origin/codex/jinc-redu04-validated` remain historical/validated-run
  authorities.

Two more contained branches have attached clean worktrees and must not be
deleted until those tasks are deliberately closed:

| Branch | Worktree |
| --- | --- |
| `codex/issue-observation-apt-v2` | `/private/tmp/citlali-converge-apt-align-jinc` |
| `codex/integrate-apt-v2-sci-align-native` | `/Users/gwilson/.codex/worktrees/73d1/citlali-refactor` |

## Divergent Retained Lanes

All 98 divergent local branches retain unique history and are excluded from
mechanical deletion. Important authority families are:

- Build adaptation: local `codex/build-adaptation` at
  `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046` in
  `/Users/gwilson/GitHub/citlali-refactor-build`. It is 87 candidate-only / 27
  build-only commits relative to `ff7899668` and five commits ahead of
  `origin/codex/build-adaptation`.
- Build-lane naming discrepancy: governing documents name
  `codex/conan2-adaptation`, whose remote ref is stale at `9aae0e669`; the
  active worktree uses `codex/build-adaptation`. Retain both until the owner
  reconciles the authority name and path.
- Scientific-contract authority: `codex/scientific-contract-library` at
  `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`,
  `codex/scientific-audit-framework`, `codex/science-doc-framework`, both
  Stage-A scientific-contract branches, and every audit, re-audit,
  coordination, registration, and freeze branch.
- SCI-ALIGN evidence: `codex/sci-align-001-lissajous-timestream-fit`,
  `codex/sci-align-001-split-direction-beammap-validation`,
  `codex/sci-align-001-3c273-corpus-tooling`, and related native-consumer,
  diagnostic, and transfer histories.
- CAL/RTC/PTC/MAP audit and repair branches, the performance lane, `gw_dev*`,
  and legacy development/release branches also retain unique divergent
  history.

No divergent branch is a retirement target in this cycle.

## Historical Release Tags

| Tag | Type | Target object/commit | Relation |
| --- | --- | --- | --- |
| `v1.0.0` | annotated | `be41c0d482559f13c6f913626f1b5531c6d3c8d2` | divergent historical release |
| `v1.1.0` | annotated | `147d6701d960c1e0bd0b873647b88c524ea9d16b` | divergent historical release |
| `v1.2.0` | lightweight | `f02372678bb695d5232d24b8a50c6c24da41e8c0` | divergent historical release |
| `v1.2.1` | annotated | `2bc3d42de1bcc5b1a1d8fb326fd9e51c7c784c2a` | divergent historical release |
| `v2.0.0` | lightweight | `2d43c50735c6f362a48484a9c809e25b7085fb8e` | ancestor |
| `v3.0.0` | lightweight | `a6cba03a75d73fd549665ac34f06a323d509ec6a` | ancestor |
| `v3.1.0` | lightweight | `ea49d46393dfcc44d1ec13055026dbe2b2fa8dbf` | ancestor |
| `v4.0.0` | lightweight | `a398581f48200dcd0cf41e1e09d33b5b7922a06f` | ancestor |

Retain all eight. No tag pins `3ebc2a67f` or the integration candidate.

## Worktrees

| Path | Branch/head | Relation | Tracked | Untracked |
| --- | --- | --- | ---: | ---: |
| `/Users/gwilson/GitHub/citlali` | `gw_dev` at `ffc6b907` | divergent | 0 | 0 |
| `/private/tmp/citlali-contracts-consolidated` | `scientific-contract-library` at `20ba6ae5` | divergent | 0 | 0 |
| `/private/tmp/citlali-converge-apt-align-jinc` | `issue-observation-apt-v2` at `cd76cf7d` | ancestor | 0 | 0 |
| `/private/tmp/citlali-legacy-apt-dispatch` | `repair-native-consumer-extinction` at `3ebc2a67` | ancestor | 0 | 2 |
| `/Users/gwilson/.codex/worktrees/0391/citlali-refactor` | detached `46ad2388` | ancestor | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/73d1/citlali-refactor` | `integrate-apt-v2-sci-align-native` at `a71fce41` | ancestor | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/88ed/citlali-refactor` | detached `9c1107b1` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/be67/citlali-refactor` | detached `bbf14b57` | ancestor | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/c9be/citlali-refactor` | detached `96b3c66d` | divergent | 0 | 3 |
| `/Users/gwilson/.codex/worktrees/cf7d/citlali-refactor` | `repair-sci-align-native-cohort-consumers` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/d2d6/citlali-refactor` | `repair-sci-map-002-jinc-ownership-invariant` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/d47f/citlali-refactor` | `repair-apt-stage-d-kmp-overlay` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/dc3e/citlali-refactor` | `implement-sci-map-003-slice-001` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/e41a/citlali-refactor` | `repair-cal-apt-repair-001` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/f3c8/citlali-refactor` | detached `bbf14b57` | ancestor | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/f803/citlali-refactor` | detached `65d1abb8` | divergent | 0 | 0 |
| `/Users/gwilson/.codex/worktrees/fddf/citlali-refactor` | WP-7 integration candidate | current | task edits | 0 |
| `/Users/gwilson/GitHub/citlali-perf` | `perf-map-accumulation-noise-lifecycle` | divergent | 0 | 1 |
| `/Users/gwilson/GitHub/citlali-refactor` | `sci-align-001-lissajous-timestream-fit` | divergent | 0 | 464 |
| `/Users/gwilson/GitHub/citlali-refactor-build` | `build-adaptation` | divergent | 0 | 0 |
| `/Users/gwilson/GitHub/citlali-refactor-sci-align-split-direction` | `sci-align-001-split-direction-beammap-validation` | divergent | 0 | 0 |

The clean detached ancestor worktrees `0391`, `be67`, and `f3c8` contain no
unique commit or untracked file, but they are Codex-managed. Close/archive
their tasks in the app before any manual Git worktree removal.

## Stashes And Untracked User Evidence

Retain all stashes:

```text
stash@{0} 75cc620256a460d6002ad367cecf17f7df1dbd1b  perf naive_mm work
stash@{1} 1c98f93376776b4dd51494f230c559cf1e031a5b  gw_dev RTC work
stash@{2} b27271fc76ffea063341465a5a62112167f50e01  gw_dev JINC/pointing work
```

There are 470 untracked user-owned files in four worktrees:

- two Eigen compatibility configs in
  `/private/tmp/citlali-legacy-apt-dispatch`;
- three WP-7 independent-audit reports in worktree `c9be`;
- `UNITY_PERF_COMPILE_COMMANDS.txt` in the performance worktree; and
- 464 files in `/Users/gwilson/GitHub/citlali-refactor`: 41 Git bundles, two
  WP-7 comparison reports, 213 files under `output/`, and 208 under `tmp/`.

The five WP-7 reports have SHA-256 values:

```text
c20d59d4e2785eec54aa8b8987d3ad09a83d924bcd66023c3de38613120960d5  WP7_INDEPENDENT_REPORT_SHA256SUMS.txt
e12c0a424d8914a9505261de610d4218984eb5b2576cde53e16a3005f493732e  WP7_INDEPENDENT_SCENARIO_SUITE.md
13c903d596782f8ed52d293085819769305dff3e301478f7658d1d07e31e5549  WP7_INDEPENDENT_SCIENTIFIC_CONTRACT_AUDIT.md
446792d75d67ce25254af9832436c8f64bd8dcd1bc49f9676a2b1e8aba9e5396  WP7_TWO_AUDIT_COMPARISON_REPORT.md
c424c881e39df9736419c623bb5dbbd56c305b203eeb181edec8ac92250b18d4  WP7_TWO_AUDIT_FINDING_CROSSWALK.csv
```

All 41 bundles verify in the present repository. They total 19,014,724 bytes,
advertise 38 unique tips, and are thin. Thirty-seven require a prerequisite
outside the application candidate. They are transfer evidence, not standalone
archives. Do not delete them or their divergent prerequisite refs until a
self-contained archive has been verified.

## Unreachable And Reflog-Only Commits

`git fsck --unreachable` finds four commits unreachable even with reflogs:

```text
158d2f80002d0fcb364d59f1a1638c7c4f7469d3  Add source-aware pointing strategy
9579b5b10d6a3106713dd7c997dfd41a7e78a69f  further components
d593dcb318d887df0ba40ad555159fd4d2c238f4  update submodules
eb91fb3370f436b435d87cf523d7ff08c2b9df2b  Initial commit
```

`git fsck --no-reflogs --unreachable` finds 21 commits covered by these 17
maximal tips:

```text
027c5caca90ec636a900b45affa0768139ab4d30
0bde0e385a4fa345cbde82bd8bfee06924087639
158d2f80002d0fcb364d59f1a1638c7c4f7469d3
1c98f93376776b4dd51494f230c559cf1e031a5b
3ac2dcdcc4b5452c6b5944286c51019cfe36427c
5e85cc530c95f52f5d3eb24a8802673e27758b2a
5eb8d859615e4682e537f645ffc62ce20a7836b8
736a8ab3bfe37e9f150ff361afd4b9f15518619b
8918189e413b874cd81b9c4ee66cbd590b043959
9579b5b10d6a3106713dd7c997dfd41a7e78a69f
a91f5492654b0ad8fe8e6b542349da16d1ab337a
b27271fc76ffea063341465a5a62112167f50e01
cd83546280e2aa99ad1f84d1241c8613de8c6209
d593dcb318d887df0ba40ad555159fd4d2c238f4
d5bbedf79eb5c51d81613374768916d070a0b025
d94845cdce4ef7f508b389d340faa414e571b1b5
fb1e605df23e719840c7def6b2af897f314826fb
```

These include old stashes, superseded build-adaptation commits, native ALIGN
commits, and audit evidence. Do not run `git gc`, `git prune`, or reflog
expiration before making them reachable and archiving them.

## Evidence-Preserving User-Run Sequence

These commands are deliberately local until the explicit push step.

First make the 17 maximal reflog/unreachable tips reachable and create a full
Git-ref archive:

```zsh
archive_oids=(
  027c5caca90ec636a900b45affa0768139ab4d30
  0bde0e385a4fa345cbde82bd8bfee06924087639
  158d2f80002d0fcb364d59f1a1638c7c4f7469d3
  1c98f93376776b4dd51494f230c559cf1e031a5b
  3ac2dcdcc4b5452c6b5944286c51019cfe36427c
  5e85cc530c95f52f5d3eb24a8802673e27758b2a
  5eb8d859615e4682e537f645ffc62ce20a7836b8
  736a8ab3bfe37e9f150ff361afd4b9f15518619b
  8918189e413b874cd81b9c4ee66cbd590b043959
  9579b5b10d6a3106713dd7c997dfd41a7e78a69f
  a91f5492654b0ad8fe8e6b542349da16d1ab337a
  b27271fc76ffea063341465a5a62112167f50e01
  cd83546280e2aa99ad1f84d1241c8613de8c6209
  d593dcb318d887df0ba40ad555159fd4d2c238f4
  d5bbedf79eb5c51d81613374768916d070a0b025
  d94845cdce4ef7f508b389d340faa414e571b1b5
  fb1e605df23e719840c7def6b2af897f314826fb
)
for oid in $archive_oids; do
  git update-ref "refs/archive/wp7-pre-retirement-20260826/$oid" "$oid"
done

mkdir -p /Users/gwilson/work_toltec/local_data/2026-refactor/archives
git bundle create \
  /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-pre-wp7-refs-20260826.bundle \
  --all
git bundle verify \
  /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-pre-wp7-refs-20260826.bundle
git fsck --no-reflogs --unreachable --no-progress 2>&1 |
  sed -n '/unreachable commit/p'
```

The last command should print no unreachable commits. Archive the four
untracked sets separately because a Git bundle never contains them:

```zsh
git -C /private/tmp/citlali-legacy-apt-dispatch \
  ls-files --others --exclude-standard -z |
  tar -C /private/tmp/citlali-legacy-apt-dispatch --null -T - -czf \
    /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-legacy-apt-untracked-20260826.tar.gz

git -C /Users/gwilson/.codex/worktrees/c9be/citlali-refactor \
  ls-files --others --exclude-standard -z |
  tar -C /Users/gwilson/.codex/worktrees/c9be/citlali-refactor --null -T - -czf \
    /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-wp7-audit-untracked-20260826.tar.gz

git -C /Users/gwilson/GitHub/citlali-perf \
  ls-files --others --exclude-standard -z |
  tar -C /Users/gwilson/GitHub/citlali-perf --null -T - -czf \
    /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-perf-untracked-20260826.tar.gz

git -C /Users/gwilson/GitHub/citlali-refactor \
  ls-files --others --exclude-standard -z |
  tar -C /Users/gwilson/GitHub/citlali-refactor --null -T - -czf \
    /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-sci-align-untracked-20260826.tar.gz

shasum -a 256 \
  /Users/gwilson/work_toltec/local_data/2026-refactor/archives/citlali-*-20260826.*
```

Create explicit forensic tags before retiring the branch that presently pins
the Unity-tested SHA:

```zsh
candidate_sha=$(git rev-parse \
  refs/heads/codex/wp7-timestream-integration-candidate)

git tag -a stage7-ngc4449-152390-3ebc2a67f \
  3ebc2a67fc32bad69759ff45638484efabf91773 \
  -m 'Unity Stage 7 NGC4449 152390 validated application identity'
git tag -a wp7-timestream-integration-20260826 \
  "$candidate_sha" \
  -m 'WP-7 timestream audit integration candidate'
```

Fetch and prove the fast-forward again immediately before the owner-controlled
push:

```zsh
git fetch origin \
  refs/heads/codex/refactor-mainline:refs/remotes/origin/codex/refactor-mainline

git merge-base --is-ancestor \
  refs/remotes/origin/codex/refactor-mainline \
  "$candidate_sha"

git rev-list --left-right --count \
  "refs/remotes/origin/codex/refactor-mainline...$candidate_sha"

git push --dry-run origin \
  "$candidate_sha:refs/heads/codex/refactor-mainline"
git push origin \
  refs/tags/stage7-ngc4449-152390-3ebc2a67f \
  refs/tags/wp7-timestream-integration-20260826 \
  "$candidate_sha:refs/heads/codex/refactor-mainline"

git branch -f codex/refactor-mainline "$candidate_sha"
git branch --set-upstream-to=origin/codex/refactor-mainline \
  codex/refactor-mainline
```

The count must have zero on the remote-only side. After the mainline push and
recommended final-SHA Unity point smoke, recheck and retire only the 19
unambiguous local branches:

```zsh
retire_branches=(
  codex/converge-apt-align-jinc
  codex/fruitloop-feedback-estimator
  codex/integrate-sci-map-001
  codex/integrate-sci-map-002
  codex/integrate-sci-noi-002
  codex/repair-apt-prod-001-canonical-baseline-v1
  codex/repair-apt-prod-002-observation-contract
  codex/repair-apt-prod-003-compact-v2
  codex/repair-apt-v2-typed-null-policy
  codex/repair-build-git-provenance-refresh
  codex/repair-native-explicit-mjd-pointing-support
  codex/repair-sci-align-native-cohorts
  codex/repair-sci-map-001
  codex/repair-sci-map-002
  codex/repair-sci-map-002-successor
  codex/repair-sci-map-002-successor-2
  codex/repair-sci-map-002-successor-3
  codex/repair-sci-noi-002
  codex/restore-legacy-apt-admission
)

all_merged=true
for branch_name in $retire_branches; do
  if ! git merge-base --is-ancestor \
    "refs/heads/$branch_name" \
    refs/heads/codex/refactor-mainline; then
    print -u2 "not merged: $branch_name"
    all_merged=false
  fi
done

if [[ "$all_merged" == true ]]; then
  git branch -d $retire_branches
fi
```

Do not delete remote topic refs in this cycle. The forensic-pointer policy,
thin-bundle prerequisites, divergent evidence lanes, and unresolved build-lane
name require a separate owner disposition. Do not run Git garbage collection,
pruning, reflog expiration, stash deletion, or worktree removal as a substitute
for that disposition.
