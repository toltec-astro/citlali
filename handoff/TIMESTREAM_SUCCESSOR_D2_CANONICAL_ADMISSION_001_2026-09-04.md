# Timestream Successor D2 Canonical Admission 001

Status: owner-authorized integration candidate prepared with passing local
gates; pending independent exact-SHA review and owner disposition; canonical
movement, push, activation, production, and later implementation remain held

Work order: `TIMESTREAM-SUCCESSOR-D2-CANONICAL-ADMISSION-001`

Owner: Citlali project owner

## Authorization and authority

On 2026-09-04 the owner approved the bounded admission proposal after a
read-only preflight. Approval covers one isolated integration candidate,
preservation of the accepted D2 history and content, three admission-record
paths, local gates, and independent fresh-context exact-SHA review. It does
not authorize moving canonical, pushing, activation, production, cleanup, or
another implementation increment.

Risk tier: Tier 2, canonical integration. This operation occupies neither an
application spine slot nor a scientific-module probe slot.

Applicable governance read:

- `AGENTS.md` at the verified canonical base;
- `doc/governance/ENGINEERING_GOVERNANCE.md`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`; and
- `doc/governance/REVIEW_AND_CONFORMANCE.md`.

The three normative documents retain their accepted SHA-256 identities
`70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76`,
`29fae6f789bb6133c1f5bcdaf0f15437f2eb8c4110f338d3a8de9a4d98ba88dc`, and
`691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1`,
respectively. They are effective through accepted governance commit
`06a3ade51c1b3f38887295433d913811bf25cd14` and canonical effectiveness record
`77507836325eff9f469062d5884481ea37599594`, not through their historical
candidate banners.

Current sequencing is governed by canonical `doc/REFACTOR_STATUS.md` and
`doc/INTEGRATION_LEDGER.md`, including the 2026-09-04 map-space closure.
`doc/APPLICATION_BASELINES.md`, `doc/WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`, and
`validation/wp7_timestream_successor_authority.json` retain their routing and
historical roles. Their older checkpoint identities do not replace the live
canonical base below.

Scientific authority remains the WP-7.1 Timestream Contract Baseline, source
`170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure
`20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, its canonical authority router,
ADRs 0017--0023, and the bounded VAL-ownership dispositions preserved in the
accepted D2 handoff. Architecture and ownership remain governed by
`doc/ARCHITECTURE.md`, `doc/SCIENTIFIC_CONVENTIONS.md`, and the effective
governance. No scientific-contract amendment or architectural redesign is
part of admission.

## Exact inputs and moving-base reassessment

| Role | Exact identity |
| --- | --- |
| Canonical base / intended first parent | `9f42d348298d76c5d5145aaf0c3eace1f3e154c1` |
| Canonical base tree | `e51f22760c64454ce7233c45dd740aa710777bae` |
| Accepted D2 closure / intended second parent | `54c3254aaf8a6639d3fe1e56e23709e29cd34b3b` |
| Closure tree | `2a4addce4458bb079c9e7c6539ce0f6b55e7fedd` |
| Closure parent | `d5d4b91fa97f86f2fb995546e6cc7846a0b28165` |
| Accepted D2 implementation | `bb060947175523d6fc6a777ae4ad4606693e9e5f` |
| Implementation tree | `87c9e3aee52e5e4119a484ba46dcd6fe493216b2` |
| Literal implementation base / common ancestor | `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` |

Read-only GitHub `git ls-remote` checks on 2026-09-04, including the proposal
checkpoint at 19:22 UTC and the post-approval pre-construction check, resolved
`refs/heads/codex/refactor-mainline` and
`refs/heads/codex/timestream-successor-d2-native-measurement` to the exact
canonical and closure SHAs above. The cached remote-tracking canonical ref
remains an older checkpoint and was not used as live authority.

The owner supplied the closure disposition: independent exact-SHA review
PASS with no findings, accepted and pushed. The closure handoff's historical
pending-review language remains byte-identical; this later record captures
the subsequent owner disposition without rewriting accepted evidence.

Trigger: canonical advanced independently after the literal D2 base. The
verified tips diverge by 207 canonical-only reachable commits and three
D2-only commits. Their common ancestor is exactly the original D2 base.
Canonical's net difference comprises 1,271 documentation paths: 1,268 under
`doc/`, plus `AGENTS.md`, `README.rst`, and `validation/README.md`. None
overlaps the five accepted D2 paths. Application source, build inputs,
configuration, tests, tools, and executable validation contracts did not
change on that canonical interval.

Disposition: continue within the approved integration scope using one
explicit merge with canonical first and accepted D2 closure second. The
merge applied without conflicts. No rebase, cherry-pick, squash, history
rewrite, source repair, or historical divergent control stack is imported.
The original three D2 commits remain intact and reachable through the second
parent. The integration candidate is a descendant of both authorities; a
direct fast-forward from canonical to the feature closure was not possible.

## Branch, paths, and preservation

Owned branch: `codex/timestream-successor-d2-canonical-admission`.

Owned worktree:
`/private/tmp/citlali-timestream-successor-d2-canonical-admission`.

The worktree was newly created at the exact canonical base with no staged,
unstaged, or untracked source changes. The canonical and accepted D2
worktrees were clean. The older dirty SCI-ALIGN checkout and all unrelated
worktrees, refs, bundles, and untracked artifacts were left untouched.

The five imported paths retain these exact accepted-closure SHA-256 digests:

| Path | SHA-256 |
| --- | --- |
| `include/citlali/core/pipeline/timestream_d2_native_measurement.h` | `c4426d5bda7b020aebdf27443446510d1ec1d38ed5eebf57a924cce2338431b7` |
| `tests/timestream_d2_native_measurement_header.cpp` | `a86d20350a790a9150b4bb3fc710743cd2a085ac6996952eadb1fed9702826fd` |
| `tests/test_timestream_d2_native_measurement.cpp` | `e60c113f83d6730455438190d5956c78d8d39af2d892b69b6e6a1813a7047680` |
| `tests/CMakeLists.txt` | `698ed1f1f2ac56b21517fd5b2f7785bae76736b31910395bb47ec3f49371c0e8` |
| `handoff/TIMESTREAM_SUCCESSOR_D2_NATIVE_MEASUREMENT_001_2026-09-03.md` | `b43eab8947345f08e5dd64926cfccb30d6b8fb180bb871a65137889dddaa9fcf` |

Only three additional paths are authored: this handoff,
`doc/REFACTOR_STATUS.md`, and `doc/INTEGRATION_LEDGER.md`. All other
canonical content is preserved exactly. The complete candidate SHA, tree,
parent order, changed-path digests, review disposition, and final clean state
are reported externally after commit, avoiding a circular self-SHA claim.

## Preserved product and lifecycle

D2 owns move-only derived residual `x/r` storage and mechanical
presence/completeness only. Its immutable per-network realization retains
the exact Paired-D1 parent, native detector/occurrence axes and physical-run
identity, route/profile, realization identity, and realization-time
`ValSnapshot` handle. Parent prefilter numerics remain zero-copy references.
Publication receives residual payload and the applicable immutable snapshot
explicitly and requires exact snapshot-handle equality.

VAL remains the sole semantic authority for sample and coordinate
validation. Non-finite residual values remain mechanically present; D2 adds
no validity, cause, or usability plane or consumer inference from the bound
snapshot. Source-mask and already-applied line-operator evidence remain
separate processing evidence, with no translation into invalidity. No
producer, consumer, lifetime owner, numerical operation, allocation model,
or persistent identity changes during admission.

## Gates and environment disposition

Pre-commit validation of the assembled candidate passed:

| Gate | Result |
| --- | --- |
| Fresh local configure | PASS, 44.06 s |
| CLI plus D2 behavioral and isolated-header target build, eight jobs | PASS, 159.96 s |
| Focused D2 CTests | PASS, 7/7 |
| Explicit safety-target build | PASS, 5.98 s |
| Full `check` target | PASS, 891/891 runnable CTests; build plus check 106.97 s, CTest 24.63 s |
| Configuration preflight with `--require-all` | PASS, 130 tests, four mode kits, 8/8 compact cases, 100% surface coverage, all authority/boundary audits |
| Baseline tools | PASS, 207/207 |
| Validation ledger | PASS, 60 records |
| Science-change ledger | PASS, three changes and five integration commits; unchanged |
| Merged-index preservation, D2 digests, admission links, whitespace | PASS |

The sole non-run test among 892 registered tests remains the established
disabled `citlali::MapFitterLifecycle.ExactProductSequence`. Required data
were not skipped. The pre-existing `check` dependency omission for
`citlali_safety_test` was handled by explicitly building that target first.

The local build is supplemental AppleClang 21.0.0 / CMake 4.3.0 / Homebrew
fallback evidence, with local Python 3.13.2 from `$HOME/tolteca/bin/python`.
It reuses the existing dependency source paths and fallback settings from the
D2 implementation worktree in a newly configured `build/` tree. No package
upgrade, dependency patch, Conan execution, or Spack-policy change occurred.
The inert local fallback bootstrap include was copied from that existing
build with all Conan-install options disabled. The external Kidscpp checkout
remains at `04088da182622c3e879f04314974a7c0d60ee2d6` with its three
pre-existing dirty files unchanged; this is not a clean Spack dependency
realization. The new build emitted CMake/dependency configuration warnings
and two glog deprecated-`syscall` compiler warnings. No unexpected
error-level output occurred in the passing gates.

Pre-commit CLI provenance truthfully reported the canonical parent
`9f42d3482-dirty` while the merge was uncommitted. A post-commit build must
refresh the CLI to the final clean candidate SHA; its exact version check
and gate results are reported externally with the review subject.

The mechanical preservation proof compares Git path, mode, and blob
identity. Before the three record edits, the merged index equals canonical
plus exactly the five D2 paths. Relative to accepted implementation
`bb0609471...`, all 1,688 entries outside `doc/**`, `AGENTS.md`, `README.rst`,
`validation/README.md`, and the two named D2 handoff paths are identical.
The sorted `mode type blob<TAB>path<LF>` manifest of those entries has SHA-256
`4487507d991febcfdd31aab7a974a678c0e895071e54a9bd172578a4f93ebb77`.
The final exact-commit proof must also verify both-parent order, all eight
changed paths, unchanged canonical content outside the five imports and
three records, original D2 ancestry, and a clean worktree. This establishes
application/build/test-input equivalence, not equality of generated Git
provenance or of compiled executable bytes across commits/environments.

Task-local commands, logs, configuration reports, content proof, and gate
JSON records are retained at
`/private/tmp/citlali-d2-canonical-admission-evidence-2026-09-04`. This path
is a local evidence locator, not an immutable identity or a Unity archive.
The exact final proof and log digests are reported externally after commit.
The gate commands, run from the owned worktree, are:

```bash
cmake --build build --target citlali_cli citlali_timestream_successor_d2_native_measurement_test -j 8
ctest --test-dir build -R d2_native_measurement --output-on-failure
cmake --build build --target citlali_safety_test -j 8
cmake --build build --target check -j 8
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all --work-dir /private/tmp/citlali-d2-canonical-admission-evidence-2026-09-04/config
$HOME/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'
$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py
$HOME/tolteca/bin/python tools/baseline/validate_science_change_ledger.py
```

The exact local configure arguments are retained in
`configure-command.json` at the evidence locator. Python gate subprocesses
use `MPLBACKEND=Agg` and task-specific writable cache directories.

The authoritative owner-run Unity gate remains Slurm job `63987273` at exact
implementation `bb060947175523d6fc6a777ae4ad4606693e9e5f`, with all 891
runnable CTests passing. Its preserved evidence root is:

`/work/toltec/citlali_dev/citlali_acceptance_evidence/TIMESTREAM-SUCCESSOR-D2-NATIVE-MEASUREMENT-001/bb060947175523d6fc6a777ae4ad4606693e9e5f/unity-gcc13-sbatch-63987273`

The accepted `final-status.txt` SHA-256 is
`c36ea6953350875fc7adb28e4f06870061d6e64e3f827355a94d78cc6d699a28`;
`SHA256SUMS` SHA-256 is
`01158279b8dd62a88f2f808c9a5b737b5d0a0bb1a65463f1375254f957464dbe`.
The accepted environment used `unity-gcc13` Spack compiler wrappers, six
jobs/six CPUs, a job-local ext4 build tree, and a 64-GiB allocation. Its
approximately 21.6-GiB peak RSS and 9m09s elapsed time remain historical
implementation-gate evidence, not new candidate measurements.

The pre-commit preservation proof establishes application/build/test-input
equivalence; the final exact-commit proof must reconfirm it. Reuse is not
execution of the admission merge SHA or a
claim that its Git version string or binary bytes equal the old executable.
No new Unity run or affected-mode reduction is required for the unchanged,
unactivated carrier under the approved admission plan. No Unity access,
representative-science campaign, performance campaign, or reduction occurs
in this task. The accepted V2 application-generation and production axes
remain unchanged.

## Writer conformance assessment

Scientific and behavioral conformance: PASS within the integration scope;
the accepted D2 implementation and current canonical scientific content are
preserved. There is no intentional scientific change or new science claim.

Architectural conformity and ownership: PASS within the integration scope;
the D2/VAL boundary, native identities, explicit snapshot binding, lifecycle,
memory ownership, and all inactive route boundaries are unchanged. This
operation introduces no new performance-sensitive path or memory owner.

Repository, branch, and evidence hygiene: PASS at preparation; one isolated
candidate retains both histories and confines authored content to the three
admission records. Final commit identities and clean-state proof follow
externally. These are writer assessments, not independent approval.

Independent review: pending at record preparation. Canonical integration:
not performed. Push: not performed. Activation and production authorization:
none. Cleanup: none. Next owner decision: dispose the independently reviewed
exact candidate before any canonical movement or user-controlled push.

## Review triggers and stop boundary

Recheck live canonical before later admission. A moving base, content
conflict, failed equivalence, unexpected gate/error-level output, new
dependency, scientific contradiction, or need to change accepted D2 bytes
requires recorded reassessment. A consequential scientific, architectural,
scope, or ancestry choice requires owner disposition; it cannot be repaired
by redesign inside this work order. A changed source/environment may require
fresh owner-run Unity evidence. No earlier review automatically transfers
to a changed candidate SHA.

Independent review must start from fresh context, operate read-only, and
bind the complete exact candidate SHA and tree. It must separately assess
scientific/behavioral conformance, architecture/ownership, and repository/
evidence hygiene, including current canonical authority, both-parent
preservation, unchanged D2 blobs, evidence reuse, and scope exclusions. The
writer does not approve the candidate.

Canonical movement and push remain pending a later exact-candidate owner
decision after review. Only then may canonical fast-forward to the accepted
candidate, subject to another live-base check; the user performs the push.
Any later committed closure record likewise requires its own exact-SHA review.

Explicit exclusions: filtering, filter design, factor selection,
downsampling, resampling, substantive RTC/PTC/AST, common-grid projection,
CAL, MAP, route activation, ordinary-route changes, serialization,
production, release, legacy retirement, cleanup, governance amendments,
and any subsequent Timestream Successor implementation.
