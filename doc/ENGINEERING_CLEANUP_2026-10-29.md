# Collaborator engineering cleanup — 2026-10-29

## Work order and preflight

Owner: Grant Wilson; this chat owns engineering. Scientific product assessment
remains in the original chat. Tier 2: build environment, resource lifetime,
application ownership, deployment, and evidence preservation.

Read: AGENTS, engineering/successor/review governance, current status,
architecture, scientific conventions, retained debt, application baselines,
and the owner-supplied October 29 review. Canonical's integration ledger records
accepted governance `06a3ade51c1b3f38887295433d913811bf25cd14` and effectiveness
`77507836325eff9f469062d5884481ea37599594`. The candidate also records the
September 30 standing owner development amendment. This work does not amend
governance or scientific contracts.

- Canonical integration: `codex/refactor-mainline` at
  `e32c4584dec07617d6caa857d39bf13047b6c4b0` (verified locally).
- Reviewed candidate and engineering base:
  `77063575c4fac2bb207ca462435009358be396f6`, clean when inspected.
  Canonical is its ancestor, 262 commits behind. Ancestry is not acceptance.
- Frozen weighted POINT trial source:
  `331ff29ea1720b1454e671cbbab93814d8df8321`.
- Owned branch: `codex/citlali-engineering-20261029`.
- Owned managed worktree:
  `/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor`.
  Initial staged/unstaged/untracked state: clean.
- Shared store: `/Users/gwilson/GitHub/citlali/.git`.
- WIP: owner-authorized engineering lane; original chat owns scientific review.
  Build changes remain in separate commits from application changes.
- Grant confirmed October 29 that trials finished; local builds may proceed.

## Scope, sequence, and gates

| Stage | Owned interface / scope | Gates |
| --- | --- | --- |
| 1 | CMake sources, strict FP, ordinary native tests, current CI | Fresh build, complete executable aggregate, native/Python/config checks |
| 2 | Native publication, FITS and FFTW lifetimes | Synthetic exceptions, allocation failure, competing writers, no replacement |
| 3 | Shared input/product services for runtime and experiment callers | Both callers compile/test; exact identity and experiment restrictions preserved |
| 4 | Maintained Python package and boundary types | Explicit modules/dependencies, packaging tests, targeted lint/typing |
| 5 | Front door, current state, integration proposal | Exact identities and links; accepted historical records retained |
| 6 | Inventory and cleanup proposal | Enumerate protected state; preserve administrative records; exact approval batch |

Target collaborator scope: POINT and SCIENCE on all three arrays; familiar
TolProj configuration, validation and guarded submission; TolTECA preparation
and launch; established pointing calibration, real multi-observation coadds,
and downstream consumers. Grant supplies beammap/APT products. This is a
release target, not a claim that every route is qualified.

Accepted scientific contracts, owner decisions, scientific conventions,
product/config contracts and retained limitations govern semantics. No changes
to thresholds, rejection, noise estimation, fitting, weighting, calibration,
FRUIT mathematics or admission. Preserve unavailable measurements, exact input
associations and required failure propagation. No Engine growth, generic
framework, observation launches, Unity access, trial modification, frozen
executable replacement or sibling-repository edits. Normal diagnostics remain
compact. Local fallback checks are supplemental; supported Spack qualification
remains owner-run. Independently review substantive final source SHAs with
separate behavioral, architectural and repository dispositions.

Stop for unresolved scientific intent, required sibling changes, frozen-input
changes or consequential compatibility uncertainty; continue bounded repairs
and routine implementation choices. Grant controls reductions, canonical
integration, pushes and exact destructive cleanup approval. The initial work order authorized no deletion, pruning, remote-ref removal,
reflog expiration, garbage collection or history rewrite. Grant subsequently
approved only the exact 33-registration metadata batch, whose completed
execution is recorded below and in the cleanup proposal.

## Issue and disposition table

| Finding | Confirmation | Disposition |
| --- | --- | --- |
| Source/strict-FP drift | 26 fallback sources vs 20 Spack; same six missing from strict FP | Fixed; shared source/FP owner, aggregate and current CI; fresh local gates pass |
| Ordinary tests | Safety missing from check; Spack lists only 11 targets | Fixed; shared source/FP owner, aggregate and current CI; fresh local gates pass |
| Historical CI | Conan-era workflow misses current gates | Fixed; shared source/FP owner, aggregate and current CI; fresh local gates pass |
| Output race | POINT/MAP check absence then share unchecked staging creation | Fixed; exclusive/no-replace native publication and resource owners; synthetic gates pass |
| Resource leaks | FITS exceptional paths and unchecked FFTW creation | Fixed; exclusive/no-replace native publication and resource owners; synthetic gates pass |
| Acceptance inclusion | Production includes acceptance source with renamed main | Fixed; compiled shared service and explicit unchanged historical profiles; both callers compile |
| Python deployment | No portable maintained runtime declaration | Implemented; curated wheel/frozen closure, hash locks and focused quality gates; target qualification open |
| Documentation | Current and historical summaries overlap | Reconciled; current-state front door and explicit historical checkpoints; owner integration pending |
| Repository cleanup | Protected originals verified intact before and after execution | Exact 33-registration batch approved and pruned; 20 existing worktrees retained; independent verification passed |

The stage records below bind changes, checks, unresolved compatibility,
owner decisions and release blockers. Engineering checks do not promote a
release or establish scientific acceptance.

## Stage 1 — build and test consistency

Both dependency entry points now consume one 26-source/13-strict-FP declaration.
Dependency resolution stays separate. `citlali_test_binaries` builds every
ordinary registered executable, including safety and the synthetic JINC
command; truth-only and sanitizer targets remain explicitly separate. Spack's
developer runner builds that aggregate and checks executable/source identity.
CI targets canonical integration, partitions source-only versus native-dependent
checks explicitly, and replaces the old Conan job with an owner-dispatched
prepared Spack runner. No observational launch is included.

Verification on a wholly fresh `build/` binary directory: AppleClang 17,
C++20 Release, existing read-only dependency sources and Homebrew libraries.
No existing binary/object tree was copied. CLI and ordinary aggregate built;
1,366 enabled CTests passed (18.37 s), with the inherited
`MapFitterLifecycle.ExactProductSequence` explicitly disabled. All 207 baseline
unit tests passed after building their required CLI. Full `--require-all`
configuration preflight passed. Focused build tooling, config and configured
POINT checks are recorded in the final handoff. First baseline attempt before
CLI completion failed its native prerequisite; the successful rerun establishes
the required order. Compiler/dependency deprecation and duplicate-library
warnings remain visible; no unexpected runtime errors were accepted.

Intended behavior changes: complete ordinary builds, stricter source identity
verification and current CI routing. No scientific algorithm or parameter
change. Spack/Unity qualification and runner provisioning remain open. Source
and binary revision agreement is checked again after this stage's commit.

## Stage 2 — native output and C-resource ownership

Stage 1 source/binary agreement passed at
`1bb75f1a9b337485e197b659144d9e8b6394e5bd` before applying this stage.

`StagedOutputDirectory` owns atomic creation of a direct invocation's `.pending`
directory and uses atomic no-replace rename on macOS/Linux. POINT weighted and
saved-MAP callers use it; unsuccessful attempts retain their staging and failure
evidence. Unsupported no-replace filesystems/platforms fail closed. This does
not promise fsync crash durability or change configured POINT's existing atomic
`reduNN` reservation.

`FitsHandle` closes on exceptions, with independent explicit close status for
required writes. The FRUIT, probe, saved-MAP and weighted feedback call sites
use it. `FftwComplexBuffer` and `FftwPlan` check creation and release in reverse
lifetime order at all six retained Wiener sites. Transform operations, buffer
relations, signs, flags, thresholds and planning order are unchanged.

CLI and complete ordinary test aggregate built; 1,375 enabled CTests passed
(11.51 s), with the same inherited disabled test. Nine new tests cover competing
writers, failed staging, existing/intervening/dangling destinations, FITS cleanup
and close failure, FFTW allocation/plan failure and exact tiny FFT roundtrip.
Three new public headers separately passed isolation compilation. Supplemental
standalone synthetic checks also passed. No observations were processed.
Intended failure change: reject competing publication/resource-creation failure
before unsafe use, and propagate required FITS close failures. Linux's syscall
branch and supported Spack/Unity environment still require qualification.

## Stage 3 — reusable application input services

Stage 2's clean executable identifies
`e1490b25735854005a96a29a1c9bdee9bb53bac4`. Independent fresh-context review
of stages 1–2 passed all three categories with recorded local/environment
limitations and no actionable findings.

`citlali_successor_input_bindings` now owns concrete runtime-config, NetCDF,
APT/raw identity, Tune normalization, native mapping/occurrence and member-state
helpers. The production executor and acceptance program link this internal
compiled service. Production no longer textually includes the acceptance
executable or renames its main. Acceptance argument parsing, comparisons,
reporting and process policy stay in the tool. The six existing acceptance/tool
callers retain explicit linkage. Historical telescope/mapping and saved-MAP
masked-covariance restrictions have named profiles; their domains are unchanged.
No new scientific admission or Engine state is introduced.

The extraction audit found 12 moved helper blocks byte-identical, including
the telescope guard/loading body. Mapping's literal is unchanged in its named
profile; existing production overrides retain ownership. No compile-speed
improvement is claimed without a controlled measurement.

Verification: production CLI, ordinary aggregate, acceptance executable and all
five historical includers compile/link. All 1,382 enabled CTests pass (14.05 s),
including seven new boundary tests and isolated service-header compilation;
the inherited disabled test is unchanged. All 207 baseline tests and full
configuration preflight pass. Build-tool tests: 75 passed. Validation ledger
(60 records), scientific-change ledger (3 changes/5 integration commits), and
session exit audit (zero library/CLI exits or growth) pass. No observational
processing was performed; Spack/Unity qualification remains separate.

## Stage 4 — curated Python deployment

The installable configured POINT application declares CPython 3.11–3.13 as
compatibility targets, runtime/optional-diagnostic/development dependencies,
and a tested 28-module runtime plus its packager and seven data files. Frozen
TolProj launchers retain exact source/native checks and byte verification.
Historical tool imports required by existing measurement helpers remain in the
explicit closure; copying every successor script is no longer necessary.
Typed file/input/prepared-observation boundaries and a focused ten-file Ruff
baseline improve maintenance without changing scientific operations. Pyright
checks four config/input/resource/type boundaries without blanket suppressions.

A fresh CPython 3.13.2 Mac arm64 environment was recreated offline from 31
full-wheel SHA-256 locks; wheel build/install, pip check, all runtime imports
and installed config/resource lookup passed. The applied synthetic suite passes
261 tests; Ruff lint/format pass; Pyright reports zero errors and warnings.
Equivalent Linux x86_64 CPython 3.13 artifact locks are prepared, with glibc
2.28 minimum, but Linux execution and other Python versions are unqualified.
The wheel version 0.0.0 is a packaging candidate identifier, not a release.

Intended deployment change: installable explicit module/resource inventory and
narrower frozen bundles. Existing CLI/scientific behavior is preserved. Locks
record installation artifacts; the launcher still honestly reports mutable
external packages as unpinned at launch. Native shared-library identity,
Spack/Unity qualification, Linux execution and scientific acceptance remain
separate. See requirements/README.md for reproducible deployment instructions.
CI wiring to these declarations is recorded in a separate build/CI commit.

### Stage 4 CI follow-through (separate build commit)

Hosted CI now installs the application and runs its focused lint, format, type,
installed-import/resource and synthetic contracts on Python 3.11 and 3.13.
Linux 3.13 consumes the full hash lock. The 3.11 compatibility lane resolves
the declared ranges and uses isolated build requirements; it is not claimed
to reproduce the locked 3.13 environment. The earlier duplicate CI dependency
file is superseded by project metadata/locks. Workflow YAML, embedded shell
syntax, filenames and package metadata were checked locally. GitHub jobs have
not been dispatched; hosted CI success remains unverified until owner push.

### Stage 4 independent-review repair

Independent review found a real cross-revision packaging defect: an installed
packager could use its own inventory when targeting a checkout that declared a
new module, create a bundle without that module, and fail at launch. The bounded
repair validates the target's committed literal inventory without executing
its code and rejects incompatible/malformed membership before creating the
destination. It directs the caller to the matching source packager. Added,
removed, duplicate, missing, malformed and nonliteral declarations are tested,
with compatible success and no code execution. All 273 configured POINT/package
tests pass, along with Ruff/Pyright and full config preflight. This intended
failure behavior prevents creation of an incomplete cross-revision runtime;
scientific execution is unchanged. The initial review finding and repaired
exact-source disposition are both retained in the delivery artifacts.

## Stage 5 — documentation and integration disposition

README and CURRENT_STATE now route the supported workflow, exact canonical,
reviewed and frozen identities, implementation snapshot, qualified evidence and
open collaborator gates. Stale August held/no-work-order summaries are labeled
as historical in AGENTS, architecture and application landmarks, with original
acceptance/history retained. Scientific conventions gain a routing note only.
Architecture describes the real shared source manifest, concrete input service
and publication/resource owners. No governance or scientific acceptance is
amended. Owner-run Unity instructions use an isolated testing checkout and
fresh build/environment; no remote access or reduction was performed here.

Canonical remains unchanged, 262 commits behind the reviewed engineering base.
The proposal is owner review of that bounded application continuation and its
scientific dispositions before integration of an exact complete candidate.
Preserve accepted ancestry when the whole continuation is accepted; otherwise
prepare a separately reviewed selective integration plan. Engineering commits
must not be blindly cherry-picked onto the older canonical application. No
canonical update or GitHub push was performed.

## Stage 6 — preservation and exact cleanup proposal

The refreshed inventory enumerates branches and provisional dispositions,
worktrees/dirty paths/ignored roots, detached state, three stashes, recovery
bundles, preservation refs and bundle prerequisites. The exact 33 missing
registrations are listed in REPOSITORY_CLEANUP_PROPOSAL_2026-10-29.md. Their
administrative records are archived with per-file verification. Two otherwise
unreferenced reflog commits are in a verified thin recovery bundle, with exact
prerequisites retained. Restoration in an isolated bare repository without
alternates, commit/tree comparison and strict fsck passed; every administrative
index object remains reachable from named refs. Copies and hashes are retained
outside temporary storage in this task's artifact directory.

All branches, three original dirty checkouts, detached checkout, stashes,
ignored dependencies and original recovery bundles remain protected. The
initial delivery proposed cleanup only. Grant later approved exactly the
33 missing registrations; the verified execution below closes that batch.
No other deletion, shared reflog expiration, garbage collection or history
rewrite was performed. Future large evidence belongs in retrievable project
storage with compact summaries/manifests/digests in Git; existing approximately
1.70 GB of historical validation evidence remains intact.

## Exact implementation commits and open gates

| Stage | Commit | Local result / remaining gate |
| --- | --- | --- |
| Build/test/CI parity | `1bb75f1a9b337485e197b659144d9e8b6394e5bd` | Fresh CLI/ordinary aggregate; 1,366 enabled tests; supported Spack runner still needs execution |
| Native publication/FITS/FFTW ownership | `e1490b25735854005a96a29a1c9bdee9bb53bac4` | 1,375 enabled tests; independent review passes; Linux syscall qualification open |
| Shared input service / historical profiles | `00b24dee6ecc484a7130d41bdb6a0c6bd6e41ce0` | 1,382 enabled tests; all six tool callers compile; independent extraction review passes |
| Curated Python application | `357b256e1642378d951d03a5880209b8737688b4` | 261 synthetic tests; offline Mac recreation, installed smoke, lint/types pass; Linux/other Python execution open |
| Packaging review repair | `8c3b676ec9c3f5d8c7ccbb0f0f2ede97a959c983` | 273 synthetic tests; incompatible selected-source closure rejected before publication; static gates pass |
| Python CI wiring (separate build change) | `cd35fa6e9db7fc30c4b7be1f5a6be737868d04ef` | YAML/shell parse; fresh-env 75 build, 130 config, 203 source-baseline tests pass; actual hosted dispatch open |

All 207 native-capable baseline tests and full config preflight passed after
Stage 3. One inherited disabled CTest remains disclosed throughout:
`MapFitterLifecycle.ExactProductSequence`. The local AppleClang 17/Homebrew
build uses cached dependency sources, including a dirty KIDs dependency;
its evidence is supplemental and cannot replace supported Spack qualification.
The native executable was refreshed and version-matched after each C++ stage;
the final delivered commit gets a separate version refresh and artifact record.

Independent exact-source reviews and final delivery evidence are retained in
the task artifact directory. Review limits, final source/binary hashes and the
post-commit inventory are bound there, avoiding a self-referential commit hash
inside this document. A clean engineering branch does not imply the entire
shared repository is clean or safe to prune.

Remaining owner decisions: acceptance of POINT scientific products in the
original chat; pointing-derived calibration compatibility; all-array SCIENCE
and true multi-observation coadds; downstream consumers; supported Spack/Unity
and Linux/Python deployment qualification; canonical integration of an exact
reviewed continuation; and GitHub push. The exact 33-registration metadata
cleanup has been approved, executed and independently verified. Beammap production and OOF remain outside the collaborator
release. No engineering test grants scientific or production acceptance.

## Approved cleanup execution — 2026-10-29

Grant subsequently approved only the exact 33 missing registrations and
required post-operation verification. At source
`941ccf476da011fa72043488fa566f7aafe45785`, repeated archive, object-reachability,
identity/absence and exact dry-run checks passed, then Git pruned exactly that
batch. All 20 existing worktrees, 104 branches, three stashes, shared reflogs,
dirty-file contents, 41 original recovery bundles, two preservation artifacts
and checked frozen runtime files remained unchanged. Independent preflight and
post-operation checks passed; a fresh prune preview is empty.

The [cleanup execution record](REPOSITORY_CLEANUP_PROPOSAL_2026-10-29.md) binds
the owner approval, operation, preserved artifacts and retained verification
records. No application behavior changed or observational gate was triggered.
Architecture is unchanged; repository/evidence hygiene passes independent
verification. This documentation follow-up retains the reviewed source as
ancestry. No further cleanup, canonical integration or push is authorized.
