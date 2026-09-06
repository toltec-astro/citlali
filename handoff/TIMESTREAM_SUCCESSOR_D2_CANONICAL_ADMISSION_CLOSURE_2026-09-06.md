# Timestream Successor D2 Canonical Admission Closure

Status: owner-accepted D2 candidate admitted to local canonical;
documentation closeout prepared for independent exact-SHA review;
GitHub push remains owner-controlled

Work order: `TIMESTREAM-SUCCESSOR-D2-CANONICAL-ADMISSION-002`, bounded
admission/closure continuation. Owner: Citlali project owner. Risk tier: 2.

## Owner decision and executed admission

On 2026-09-06 the project owner replied **“I approve, carry on”** to the
explicit request to approve canonical admission of
`aeea0eef04ec70d8142c8a20fd7b09dfb64725ed`. That exact subject had passed
fresh independent review with recorded limitations and no findings. This
records the owner decision required by effective engineering governance;
it does not infer acceptance from passing tests alone.

Read-only preflight reverified clean local and live GitHub canonical at
`00b974c9039d4c3025dcce18f26bca69d36af9c3` and the original D2 feature at
`54c3254aaf8a6639d3fe1e56e23709e29cd34b3b`. The approved candidate was
unchanged and clean; its preservation proof and all 24 task-local evidence
checksums passed. Local `codex/refactor-mainline` was then fast-forwarded
literally from `00b974c9...` to `aeea0eef...` in
`/private/tmp/citlali-refactor-mainline-after-map-space-repair-2026-09-04`.
The resulting canonical tree was clean and exactly matched the accepted tree.
No merge commit was synthesized by the fast-forward and no GitHub push ran.

| Role | Exact identity |
| --- | --- |
| Owner-accepted and locally admitted candidate | `aeea0eef04ec70d8142c8a20fd7b09dfb64725ed` |
| Accepted candidate tree | `751f274f57b13883eadb02ab546105601e1f66dc` |
| Ordered first parent / prior live canonical | `00b974c9039d4c3025dcce18f26bca69d36af9c3` |
| Ordered second parent / Unity-tested repair | `7d57a5acf893ae0f34c3639499484b3f5976768a` |
| Repair tree | `c3c545c826c8e8c948c1d53fbf8301b52d68febe` |
| Parents' common ancestor | `ae953ed4d87d1f693d2bbf42aebbc25ef730c771` |
| Preserved first admission candidate | `218988dc21cfc279cdfe7a4318b574f070cd45f9` |
| Original accepted/pushed D2 closure | `54c3254aaf8a6639d3fe1e56e23709e29cd34b3b` |
| Original D2 implementation | `bb060947175523d6fc6a777ae4ad4606693e9e5f` |
| Literal implementation base | `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` |

## Authority and bounded closeout

Repository `AGENTS.md`, the toltec-context skill and authority routing,
current status/ledger, and effective `ENGINEERING_GOVERNANCE.md`,
`TIMESTREAM_SUCCESSOR_GOVERNANCE.md` and `REVIEW_AND_CONFORMANCE.md` were
read for this continuation. Normative bytes remain those accepted through
`06a3ade51c1b3f38887295433d913811bf25cd14` and made effective through
`77507836325eff9f469062d5884481ea37599594`. Their digests remain unchanged.
Scientific authority remains the WP-7.1 contract source `170ecea9...`, closure
`20ba6ae5...`, unchanged machine-readable authority binding and ADRs 0017--0023
as fully identified in the accepted admission handoff.

The owner-approved admission includes the required durable record of the
decision. This closeout uses the existing
`codex/timestream-successor-d2-canonical-admission-002` branch/worktree at
`/private/tmp/citlali-timestream-successor-d2-canonical-admission-002`.
It appends one documentation commit with the accepted candidate as sole
parent; no new branch or implementation slot is opened. Its four paths are
this handoff, the preserved exact candidate review, the status document and
integration ledger. All earlier status bodies and four D2 handoffs remain
unchanged. The original implementation, closure, prior candidate and repair
refs remain preserved. Advancing this owned admission branch for its records
does not rewrite or replace its accepted ancestor.

The reassessment is **continue within approved admission scope**. There is
no new source, test, CMake, scientific contract, interface, lifecycle, numerical
operator or ownership change. All 1600 tracked inputs outside `doc/` and
`handoff/` remain identical to the Unity-tested repair. All other accepted
documentation, including the MAP study and its owner dispositions, is
preserved. D2 retains mechanical residual storage; VAL retains semantic
sample/coordinate validation. The finite-frequency repair, valid ordering,
193 malformed-input cases and residual nonfinite `x/r` storage are unchanged.

## Accepted review and gate evidence

The complete fresh-context review of `aeea0eef...` is preserved byte-for-byte
in [the exact-SHA review](TIMESTREAM_SUCCESSOR_D2_CANONICAL_ADMISSION_002_EXACT_SHA_REVIEW_2026-09-05.md),
SHA-256 `56909f91662b02dc09254465e393d6e4ac5756675cd62291f1cf1a225d905a4c`.
Its scientific/behavioral and repository/evidence dispositions are PASS WITH
RECORDED LIMITATIONS; architecture/ownership is PASS. There are no blocker,
major or minor findings. Review-time statements about pending admission and
manifest finalization remain historical; this later owner record governs
admission, and the complete 24-file task-local manifest has since verified.

Unity GCC13/Spack job **64018093 PASS** remains bound to exact repair
`7d57a5acf...`, tree `c3c545c8...`: 892/892 runnable CTests, 893 registered
with the established `MapFitterLifecycle.ExactProductSequence` disabled,
8/8 focused D2 tests, matching clean CLI/source binding, clean pinned
first-party revisions, all 21 archive files checksum-verified by the owner,
empty Slurm stderr and Slurm `COMPLETED 0:0`. The run took 9m20s on six CPUs
with 64G requested; peak batch RSS was 22958868K. Full CTest took 34.62s;
focused D2 took 0.07s. The compiled environment identifies GNU 13.3.0,
C++23, Release, OpenMP variants and Spack DAG
`ryu6zhjvp5sh7fsbwrzyykbkgkglzkkx`.

The owner-held archive is:
`/work/toltec/citlali_dev/citlali_acceptance_evidence/TIMESTREAM-SUCCESSOR-D2-ADMISSION-REPAIR-001/7d57a5acf893ae0f34c3639499484b3f5976768a/unity-gcc13-sbatch-64018093`.

- Reviewed runner SHA-256: `0b65dff43288c6490a945251c14089409f27d8c4d8baf8341c501f4ee37478d0`.
- Owner-reported final-status SHA-256: `c73eaf529296dfecdfea0e5013753b415d3453a78334a99596f939e915f72139`.
- Owner-reported archive-manifest SHA-256: `d7bcf27bc7134dbac254be707c958d9d17634950e7c6bf50076f702d39b12b8b`.

Failed job 64008057 remains retained evidence of an external CLI-path error
before tests. The independently reviewed v2 runner corrected that locator
and added an executable guard without altering source or gate thresholds.
Original job 63987273 retains its original implementation subject. Detailed
receipts, dependency revisions and failure records remain recorded or linked
by the accepted
[admission handoff](TIMESTREAM_SUCCESSOR_D2_CANONICAL_ADMISSION_002_2026-09-05.md).

Local source gates remain supplemental at `7d57a5acf...`: builds and 892
runnable CTests, 8 focused D2 tests, 207 baseline-tool tests, 130 config tests,
all kits/audits and both ledgers passed. AppleClang/Homebrew and dirty cached
Kids limitations remain explicit. The exact candidate review independently
verified complete input equivalence, ancestry, governance and artifact
digests, document links, runner syntax/hash and receipt binding. Scientific
layout passed; the unchanged MAP acceptance verifier passed at its original
clean exact subject `00b974c9...`, with no verifier relaxation.
That MAP success remains the pre-admission checkpoint: after the authorized
D2 fast-forward, a repeat at the preserved MAP worktree correctly stops on
its canonical-ref guard. Its snapshot-specific gate is not applicable to the
later canonical state. The accepted packet bytes and original result are
preserved; this closure uses complete-tree preservation rather than changing
the verifier or substituting a false canonical ref.

## Closure verification and publication boundary

This documentation closeout requires an exact four-path/tree preservation
check, retained review digest, complete input equivalence, preserved status
histories, links, whitespace, clean committed state and fresh independent
read-only review of its own full SHA. Its containing SHA/tree, final review
and actual subsequent canonical fast-forward are recorded externally after
commit, avoiding a circular containing-SHA claim. Only after those checks
pass does the approved closeout fast-forward to canonical. GitHub ref state
must be checked again before providing the owner's non-force push command.
Any unexpected ref movement or content difference requires reassessment.

No further Unity build or scientific reduction is triggered: this closeout
changes only records. Evidence reuse proves unchanged tracked inputs; it does
not relabel Unity execution as the later admission/closure SHA or assert
identical generated version strings or executable bytes. Runtime deployment
labels remain unmanaged/unavailable. Full remote logs and checksum results
are owner-supplied evidence, without direct Unity access by writer/reviewer.
Managed deployment, installed-artifact acceptance, representative science,
ordinary route activation and production readiness are not established.

No filtering, factor selection, downsampling, substantive RTC/PTC/AST,
common-grid, MAP, CAL, production, cleanup, legacy retirement or subsequent
Timestream Successor work follows. The admission/repair implementation work
is complete; no next unit is started. All GitHub pushes remain owner-run.

Closeout evidence:
`/private/tmp/citlali-d2-canonical-admission-closure-2026-09-06`.
Accepted candidate evidence:
`/private/tmp/citlali-d2-canonical-admission-002-preflight-2026-09-05`.
