# POINT / NOI / FLT-FIXED Integration Candidate

Date: `2026-09-03`

Status: **historical preflight record; owner accepted and canonically integrated**

Candidate branch: `codex/integrate-point-flt-fixed-2026-09-03`

Candidate worktree:
`/private/tmp/citlali-integrate-point-flt-fixed-2026-09-03`

This is a Tier 2 integration preflight record. It binds the source refs,
bounded merge resolutions, verification results, and exclusions used to
prepare the candidate. At its creation it was not an accepted integration,
package-science change, implementation-conformity finding, validation result,
performance claim, readiness claim, production authorization, or Unity
record.

## 2026-09-04 Closure Update

The independent exact-SHA review and scientific-owner acceptance required by
this preflight were completed. The accepted integration line advanced
`codex/refactor-mainline` to exact corrected integration/audit base
`5f0fc20042b88fb6cd883c92d1b59b7f22832901`. The subsequent map-space
horizontal audit, owner disposition, independently reviewed shared-
conventions repair, and literal canonical fast-forward through
`a983e3e31ca6422ade8f081585f5ef6babcfe5d0` are closed in the
[2026-09-04 scientific-owner acceptance and integration record](audits/MAP_SPACE_SHARED_CONVENTIONS_REPAIR_001/SCIENTIFIC_OWNER_ACCEPTANCE_AND_INTEGRATION_2026-09-04.md).
The owner reports the resulting canonical ref pushed.

The execution-time stop statements below remain historical descriptions of
the preflight, not current gates. FRUIT remains independent and the protected
historical ALIGN worktree remains outside this integration.

## Owner-Approved Sequence And Stop Boundary

The approved sequence is:

1. preserve refs and record exact SHAs;
2. create a temporary branch from `codex/refactor-mainline`;
3. merge the complete POINT branch;
4. resolve and verify the four shared files;
5. confirm obsolete `src/` paths are not resurrected;
6. run scientific-contract and repository checks;
7. integrate frozen FLT-FIXED and the subsequently owner-approved frozen NOI
   closure;
8. reconcile INDEX, registry, roadmap, and status records;
9. stop for owner review before advancing or pushing
   `codex/refactor-mainline`;
10. leave FRUIT independent;
11. inventory and protect the old ALIGN dirty worktree; and
12. launch the map-space horizontal audit only after owner acceptance and
    canonical integration.

Integration mutation stops after step 8; the independent-state protections in
steps 10 and 11 are also in force. Steps 9 and 12 remain unperformed. No branch
was pushed, rebased, deleted, or cleaned.

## Exact Inputs And Safety Refs

| Role | Exact ref at integration start | Local safety ref |
| --- | --- | --- |
| Canonical accepted application/integration line | `codex/refactor-mainline@4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/refactor-mainline` |
| Authoritative POINT production/review candidate | `codex/sci-point-v0.1-stage-b@c7582052d48c991e0caec6f2b56ab63d2d44afcd` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/sci-point-v0.1-stage-b` |
| Frozen FLT-FIXED branch | `codex/sci-flt-fixed-v0.1-stage-b@7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/sci-flt-fixed-v0.1-stage-b` |
| Frozen NOI approval branch | `codex/sci-noi-v0.1-stage-b@f28d7a2617160febca85c1c40e6f7ba7494e266e` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/sci-noi-v0.1-stage-b` |
| Active FRUIT line at integration start | `codex/sci-fruit-v0.1-empirical-lane@dafa8fbd47af5934706fc1b123c2d4139b92acd0` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/sci-fruit-v0.1-empirical-lane` |
| Historical ALIGN line | `codex/sci-align-001-lissajous-timestream-fit@353b11887ff04dfd7bca12915917495f81a587fa` | `refs/codex/integration-snapshots/2026-09-03/pre-point-flt-fixed/sci-align-001-lissajous-timestream-fit` |

During verification, the active FRUIT branch advanced to
`ccb67a99257fc9fba82d25346e85503363673651`. That later committed state is
protected separately by
`refs/codex/integration-snapshots/2026-09-03/post-merge-review/sci-fruit-v0.1-empirical-lane`.
Neither FRUIT snapshot is merged here.

## Candidate Merge Topology

The temporary branch was created from exact mainline commit
`4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538`. Its bounded integration merges
are:

| Merge commit | First parent | Merged parent | Purpose |
| --- | --- | --- | --- |
| `4bff928e89c2ba2664d26707d2650698b1bbed01` | `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` | `c7582052d48c991e0caec6f2b56ab63d2d44afcd` | complete POINT branch |
| `1b0cecf90d4bed4c051c3ac840450d3cc228da34` | `4bff928e89c2ba2664d26707d2650698b1bbed01` | `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` | complete frozen FLT-FIXED branch |
| `aed08007f86b7416f7630266394f9c4fc4b7d1ed` | `1b0cecf90d4bed4c051c3ac840450d3cc228da34` | `f28d7a2617160febca85c1c40e6f7ba7494e266e` | complete frozen NOI approval branch |

POINT descends from its approved Stage A closure commit
`2c3269f29b1661948794aaa4ff9a81924bb1fb42`. Relative to the mainline merge
base, the complete POINT lineage contains 188 commits and 1,140 changed paths:
1,137 under `doc/`, plus `AGENTS.md`, `README.rst`, and `validation/README.md`.
No runtime `src/`, `include/`, `tests/`, configuration, or build-system change
was introduced by the POINT side of this merge. The non-package files are
scientific-contract governance, status, documentation routing, and validation
documentation; no unrelated application change was identified.

After the POINT merge, FLT-FIXED contributed six dedicated closure commits and
48 files under `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/`.
NOI then contributed the three r0.4/r0.5 closure commits absent from the POINT
lineage and 71 files under
`doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/`. Neither merge had a
content conflict.

## Shared-File Resolution

The four shared review files were handled as follows:

- `AGENTS.md`: retained current mainline governance and added the 17-point
  scientific-package protocol from the POINT line;
- `doc/REFACTOR_STATUS.md`: retained all mainline status and added the complete
  scientific-contract history; this candidate adds a current consolidation
  checkpoint rather than rewriting historical entries;
- `doc/SCIENTIFIC_CONVENTIONS.md`: the only content conflict; resolved
  additively by preserving all mainline APT/WP-7 authority text, adding the
  POINT documentation-routing paragraph before it, and retaining POINT's
  accepted significance terminology; and
- `validation/README.md`: retained mainline content and added the four POINT
  validation-documentation lines.

`README.rst` also received the POINT documentation-guide link through a clean
automatic merge. No conflict marker remains. The reconciliation changes to
INDEX, prior-work registry, downstream roadmap, package entry points, and this
status record change no frozen normative module, estimator, response,
uncertainty, support, lifecycle, or numerical-route meaning.

## Obsolete Source-Path Guard

The following obsolete source paths are absent from the candidate:

- `src/citlali/core/engine/beammap.cpp`
- `src/citlali/core/engine/engine.cpp`
- `src/citlali/core/engine/kidsproc.cpp`
- `src/citlali/core/engine/lali.cpp`
- `src/citlali/core/engine/pointing.cpp`
- `src/citlali/core/engine/todproc.cpp`
- `src/citlali/core/mapmaking/wiener_filter.cpp`
- `src/citlali/core/utils/utils.cpp`
- `src/citlali/dummy.cpp`
- `src/citlali/kids_main.cpp`
- `src/citlali/lali_main.cpp`
- `src/citlali/main_old.cpp`
- `src/citlali/mpi_main.cpp`

## Scientific-Contract Verification

The following current checks pass on the candidate:

- global scientific-contract layout;
- SCI-CAL, SCI-MAP, SCI-RTC, and SCI-PTC contract verifiers;
- SCI-ALIGN document verification;
- both canonical SCI-AST PDF-role checks;
- SCI-POINT r0.4 Stage B and exact r0.3 author-packet verification;
- SCI-NOI r0.5 Stage B verification: 51 requirements, 26 predictions, and
  three deterministic PDFs;
- SCI-FLT-FIXED Stage B verification with both exact visual-QA and authority-
  manifest gates: 53 requirements, 30 predictions, one shared normative core,
  three deterministic PDFs, and every bound page rendered; and
- SCI-FLT-MATCHED Stage A packet, r0.6 frozen-authority, and source/PDF
  consistency checks: 50 requirements, 25 predictions, 17 disposed SODL IDs,
  and 45/41-page scientist/engineering views.

The JINC tag resolves exactly to
`a9f43877e01a661db13bd85b2e7f34ea5ac82fb7`; its r0.3 freeze manifest has
SHA-256 `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2`.
The NOI r0.5, FLT-FIXED, and FLT-MATCHED manifest hashes independently match
their recorded
`b6915186424dd52d7c94fb0df47db91654d3c20cf4b3fa6ab98c3554626d8bfc`,
`69e6766f26396ba843ee29cfb89a48efd91b7e1b517ed90d3d93c87a63e55778`,
and `6b0231a7e9d34f028eda9cce48f62de1fc9e594348aa1448a2d182d732f78688`
identities.

Two historical entry-point checks are intentionally superseded by later
authority:

- the original SCI-VAL Core r0.3 verifier still requires the reserved MAP
  profile to be unbound, while the immutable successor Registry correctly
  binds MAP `@2`; the same failure occurs on the unmodified authoritative
  POINT branch and the current SCI-MAP verifier passes; and
- archived Stage A verifiers that assert absence of Stage B are not current
  package gates after verified Stage B publication.

These inherited historical assertions were not repaired by editing frozen
packages. They should be retired or wrapped by a separately authorized
manager-level verifier in a later maintenance change.

## Repository Checks

- `tools/config/run_config_preflight.py --require-all`: PASS, including 130
  unit tests, eight compact-compatibility cases, four mode kits, 100% compact
  surface coverage, and all typed-boundary drift gates;
- `tools/baseline/validate_validation_ledger.py`: PASS, 60 records;
- reconciliation working-tree `git diff --check`: PASS; and
- obsolete-source-path guard: PASS.

The candidate worktree has no configured `build/` directory, and this bounded
integration introduces no runtime source change, so CMake/CTest were not run.
No build or implementation claim follows.

`git diff --check codex/refactor-mainline..aed08007f...` reports inherited
Markdown hard-break whitespace and blank final lines in frozen or historical
scientific artifacts. The new reconciliation diff is clean. Those bound bytes
were not normalized during integration.

## Protected Independent State

### SCI-FRUIT

FRUIT remains outside this candidate. At the later verification snapshot, 53
commits reachable from
`codex/sci-fruit-v0.1-empirical-lane@ccb67a99257fc9fba82d25346e85503363673651`
were absent from the candidate. They include the SCI-FRUIT Stage A package,
empirical-lane records, FRUIT implementation/tooling changes, and associated
validation evidence. The worktree also contained three untracked artifacts:

- `SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz`
- `SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz`
- `validation/fruit_loop_point_123424_el_f8_penalty_placement_2026-09-03/`

No FRUIT file, commit, or untracked artifact was copied, edited, cleaned, or
integrated.

### Historical ALIGN Worktree

The older ALIGN worktree remains at
`/Users/gwilson/GitHub/citlali-refactor`, branch
`codex/sci-align-001-lissajous-timestream-fit`, HEAD
`353b11887ff04dfd7bca12915917495f81a587fa`. It has local tracked and
untracked material and was not used as an authoritative project checkout.

The branch retains 77 commits not reachable from the compared mainline,
POINT, FLT-FIXED, NOI, or FRUIT refs. Those commits include the historical
SCI-ALIGN timing, pointing-fit, Beammap, replay, and evidence work. The branch,
worktree, and dirty files therefore must remain protected until a separately
authorized preservation/disposition audit is complete. Nothing was switched,
merged, rebased, deleted, cleaned, or committed there.

## Historical Owner-Review Gate — Completed

This preflight correctly required independent fresh-context review and
scientific-owner acceptance before canonical movement. That gate was later
completed, canonical advanced, and the map-space horizontal audit and bounded
shared-conventions repair were completed and accepted as recorded in the
closure update above. FRUIT remains separate.
