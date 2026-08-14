# SCI-MAP-002-OWNERSHIP-001 Role-Separated Re-Audit

Date: 2026-08-14

Role: independent repair re-auditor

Scope: exact pushed SCI-MAP-002-OWNERSHIP-001 candidate only

## Result

The exact candidate is accepted by this re-audit for the bounded ownership
contract. It rejects ambiguous ownership or destination state before diagnostic
allocation, destination mutation, output side effects, or parallel launch. Its
valid path preserves the established worker arithmetic and accumulation order,
and the independent thread-count, focused, retained-product, full-suite,
configuration, and ledger gates pass.

```yaml
contract_status: proposed
implementation_status: conformant
validation_status: complete
production_status: existing_use_only
verdict: accept
```

This is a proposed audit disposition pending coordinator acceptance. It does
not update a canonical ledger or handoff, authorize integration, or authorize
production.

## Exact identities and scope

- Candidate remote: `origin/codex/repair-sci-map-002-jinc-ownership-invariant`
- Candidate commit: `e6c8d126157674a9990abc8d1e96ce2dd69f9374`
- Parent/base: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
- Candidate tree: `c69aa1718178d892a4c9d71fbeb931b55ed2d607`
- Candidate standard binary patch SHA-256:
  `d20cf7cde7e6397e9876e048f9401217dcc7592414e1dc94cbd669043ecb55aa`
- Audit branch: `codex/reaudit-sci-map-002-ownership-001`
- Audit worktree:
  `/private/tmp/citlali-reaudit-sci-map-002-ownership-001`
- Audit branch HEAD: the exact candidate commit above; no upstream.

The repair remains exactly three paths:

1. `include/citlali/core/mapmaking/jinc_mm.h`
2. `tests/test_jinc_parallel_ownership.cpp`
3. `tests/CMakeLists.txt`

Candidate and base file identities are:

| Path | Candidate SHA-256 | Base SHA-256 |
|---|---|---|
| `include/citlali/core/mapmaking/jinc_mm.h` | `459f83747ebd82c5be7de9d570a9c6a015876dcb74f8d925cf9a7385a31c48a4` | `65fd41b97e06048d2b4b2b0a6d6ecd175a22f1b415d1b52d925a7ec0da4c9e5b` |
| `tests/test_jinc_parallel_ownership.cpp` | `715608ef51c868d5f096c25c47a0921368b0ad82423713098c672eafdbfc2846` | absent |
| `tests/CMakeLists.txt` | `5b64d9b32db97c9948f9a70c8fcb875a410285fb0939b628c20683c39174ec42` | `08992e9af7b005339d9ba1c723db9d15b46615bda2f9def34afa75d51d99074a` |

This report is the sole audit artifact. No repair source, test, CMake,
dependency, preset, frozen audit/core/disposition, validation, ledger, or
handoff file was edited.

## Frozen authorities

The scientific-audit history remains immutable:

- SCI-MAP-003 audit commit:
  `65d1abb88b6d8fa0c3235a3d62ef9a2ab3122839`
- Frozen report:
  `doc/audits/packages/SCI-MAP-003_SCIENTIFIC_CONTRACT_AUDIT.tex`
- Frozen report SHA-256:
  `2cbad1fd11ff8851b66202197f881c48f8c0f5cc5d08f376256d6dfdb8eb1764`

The current-disposition authority is:

- Branch: `origin/codex/register-sci-map-003-audit-disposition`
- Commit: `8c581bfb26f01b187f4f1e0565f4457bcc25f099`
- Parent: `f2ba74bd62f3d88c328935683c7908e5fb327aa2`
- Tree: `8051ac955e76a32335dbeb69c04a340023638bbe`
- Standard patch SHA-256:
  `ad389405627c3fb914d1f34566a324d9a442449c0e51405da6f2f1d7324d9979`
- Handoff:
  `doc/audits/handoffs/SCI-MAP-002/SCI-MAP-002-XAUD-002.yaml`
- Handoff SHA-256:
  `78784b31ab550d227b60e7f2b210dcadac1dd06d48419d4e2aa7ecdaeffc49d8`

The disposition establishes unique per-detector map ownership for the current
detector grouping and sequential invocation across scans. It does not establish
the original frozen report's shared-array collision premise as current
production truth.

## Exposure chronology

The first content-open events were ordered as follows:

1. `2026-08-14T18:51:13.704Z` — canonical disposition handoff.
2. `2026-08-14T18:51:13.835Z` — frozen SCI-MAP-003 F003 context.
3. `2026-08-14T18:51:13.939Z` — exact JINC repair diff.
4. `2026-08-14T18:51:28.868Z` — exact ownership test body.
5. `2026-08-14T18:51:28.969Z` — exact `tests/CMakeLists.txt` diff.

Later read-only exposure was limited to the accepted build caches, generated
CTest registration files, the exact candidate CMake registration needed for
inventory reconciliation, and repository validators. No raw TSan evidence was
opened or regenerated for this mission, and no unrelated implementation body
was used as audit authority.

## Audit question and acceptance matrix

The bounded audit question is whether this exact candidate enforces unique
worker-to-destination ownership and complete destination compatibility before
any diagnostic allocation, mutation, output side effect, or parallel launch,
while preserving the valid-path arithmetic and product behavior.

| Requirement | Independent result |
|---|---|
| Duplicate destination or ownership ambiguity rejects eagerly | Conformant |
| Cardinality, index, alias, and destination-shape ambiguity rejects eagerly | Conformant |
| Rejection leaves every observable accumulator and diagnostic plane unchanged | Conformant |
| No output or parallel launch occurs before successful preflight | Conformant |
| Valid mappings preserve arithmetic, accumulation order, outputs, and tolerances | Conformant |
| Behavior is invariant at 1, 2, 4, and 8 OpenMP threads | Conformant |
| No lock, atomic, private reduction, serial fallback, or grouping change | Conformant |
| No broader race-free claim is introduced | Conformant |
| TSan remains superseded/non-applicable | Conformant; not run |
| Repair scope remains three paths and audit scope one report | Conformant |

## Static conformity

The candidate order is:

1. existing global input preflight;
2. new ownership and destination preflight;
3. diagnostic allocation;
4. `grppi::map` parallel launch.

The worker suffix beginning at `grppi::map(...)` is byte-for-byte identical
between the parent and candidate. Therefore the candidate changes the admitted
domain, not the valid worker arithmetic or accumulation order.

No added lock or mutex, atomic, thread-private reduction, thread-local
accumulator, serial fallback, scientific grouping change, destination-policy
change, or TSan configuration was found. The new check is an eager serial
precondition only.

### Failing-first counterexamples

The focused test target is `citlali_jinc_parallel_ownership_test`; its CTest
prefix is `citlali::jinc_parallel_ownership::`. The six registered cases and
their counterexamples are:

1. `DuplicateMapIsRejectedBeforeMutation`
   - duplicate worker-to-map destination ownership;
   - rejection before diagnostics or mutation.
2. `InvalidSizesAndIndicesFailBeforeMutation`
   - undersized worker/map-index cardinality;
   - oversized worker/map-index cardinality;
   - negative map index;
   - out-of-range map index.
3. `InconsistentDestinationsFailBeforeMutation`
   - inconsistent signal destination cardinality;
   - inconsistent grid-weight destination cardinality;
   - inconsistent weight destination cardinality;
   - inconsistent coverage destination cardinality;
   - inconsistent kernel destination cardinality;
   - inconsistent noise destination cardinality;
   - incompatible destination shape.
4. `ValidUniqueMappingsMatchSerialExactly`
   - every valid map and plane equals the serial reference exactly.
5. `ValidUniqueParallelPoliciesPreserveExactRepeatedResults`
   - repeated OpenMP-policy results equal the sequential-policy result exactly.
6. `ValidUniqueMappingAllowsExtraMapSlots`
   - valid unused destination slots are permitted and remain zero.

The invalid-path harness snapshots all signal, grid-weight, weight, coverage,
kernel, noise, and contribution-diagnostic planes before the call and compares
them after rejection. Static ordering additionally establishes that output and
parallel work cannot begin before these cases reject.

## Independent build provenance

All compilation and execution used the fresh external build:

`/private/tmp/citlali-reaudit-sci-map-002-ownership-001-build-recovery-3`

No object, generated package file, or other build output was copied from the
repair-author build. The earlier external directories were left untouched:

- `/private/tmp/citlali-reaudit-sci-map-002-ownership-001-build`
- `/private/tmp/citlali-reaudit-sci-map-002-ownership-001-build-recovery`
- `/private/tmp/citlali-reaudit-sci-map-002-ownership-001-build-recovery-2`

Their stops were, respectively, an obsolete preset compiler path, an omitted
package-selection surface that allowed a Conan-helper download attempt, and an
omitted accepted CMake policy value. None supplied audit test evidence.

The successful recovery used:

- CMake `4.3.0`;
- generator `Unix Makefiles`;
- build type `Release`;
- C compiler `/Library/Developer/CommandLineTools/usr/bin/clang`;
- C++ compiler `/Library/Developer/CommandLineTools/usr/bin/clang++`;
- Apple clang `21.0.0 (clang-2100.1.1.101)`;
- target `arm64-apple-darwin25.6.0`;
- `CMAKE_POLICY_VERSION_MINIMUM=3.5`;
- `FETCHCONTENT_FULLY_DISCONNECTED=ON`;
- all 17 `CONAN_INSTALL_*` switches explicitly `OFF`.

The accepted repair-author cache is
`/Users/gwilson/.codex/worktrees/d2d6/citlali-refactor/build/CMakeCache.txt`,
SHA-256
`ffc834c412cc62dd16074f0e92ef9aa1bddafe5be6f46d77193f5d80ebaf8dbf`.
It records CMake 4.3.0 and the same policy value. The accepted dependency-source
cache is
`/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/CMakeCache.txt`,
SHA-256
`8d1eb1cc0601f309c6b064ca73be762ed91472dbe1ea3f51aa1162355f256839`;
it also records CMake 4.3.0 and the same policy value.

The external initial-cache control file is
`/private/tmp/citlali-reaudit-sci-map-002-ownership-001-build-recovery-2-initial-cache.cmake`,
SHA-256
`7c4a9c1c044c001f764248948ca4e5a3a7f781ab7c147ea95348d0c4154eb94b`.
Recovery 3 reused it byte-for-byte and added only the accepted policy value on
the command line. The selected compiler, generator, policy, prefix,
`CONAN_INSTALL_*`, `USE_INSTALLED_*`, `FETCH_*`, `FETCHCONTENT_SOURCE_DIR_*`,
OpenMP, and relevant threading key/value surface matches the author cache with
an empty value-level diff.

Installed package selection was Boost, CCfits, Eigen3, FFTW, and NetCDF.
Local FetchContent selection was Ceres, Clipp, CSV, Enum, GramSavgol, GRPPI,
logging libraries, NetCDF C++4, RE2, Spectra, testing libraries, and YAML.
OpenMP 5.1 resolved through `/opt/homebrew/opt/libomp` with
`-Xclang -fopenmp`. Configure output displayed declared upstream URLs, but
every populated dependency resolved to its verified local source directory;
no download, Conan invocation, or network request occurred.

### Local dependency identities

All paths below existed before configure. Dirty state is a limitation, not
audit authority; no dependency source was inspected for scientific claims or
modified by this audit.

| Dependency source path | Git HEAD | Branch | State |
|---|---|---|---|
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/benchmark-src` | `f91b6b42b1b9854772a90ae9501464a161707d1e` | detached | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/bitmask-src` | `0454f32733d4fc910ac0c3c85e61b45b9ae7eee9` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/ceres-src` | `399cda773035d99eaf1f4a129a666b3c4df9d1b1` | detached | dirty: 4 entries |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/clipp-src` | `ddf69f70eaaefe318cc8aa0d018ff523111410bb` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/csv_parser-src` | `bc3bebcc16fb74144e9d94035346b3d9150b39c5` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/fmt-src` | `6c285ba88a22e287f8d33a4e15b43c0095160181` | `main` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/glog-src` | `8f9ccfe770add9e4c64e9b25c102658e3c763b73` | detached | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/googletest-src` | `963dd8ea18b5490674048e9dbe344c6435caf9d3` | `main` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/gramsavgol-src` | `00cb1ca7bd4c009ed21678429b5b6e630f4b2290` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/grppi-src` | `12f5c11b1a5ad4a283e313bd1966bfd6446b9e24` | `cpp20` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/kidscpp-src` | `04088da182622c3e879f04314974a7c0d60ee2d6` | `v1.x` | dirty: 3 entries |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/meta_enum-src` | `f940f15bc3f4321f5ef458742b7731c7f03543ff` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/netcdfcxx4-src` | `a43d6d4d415d407712c246faca553bd951730dc1` | `main` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/re2-src` | `8e08f47b11b413302749c0d8b17a1c94777495d5` | detached | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/spdlog-src` | `f5f173a1a57d0e2e0115f2ed71ee7ea316516853` | `v1.x` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/spectra-src` | `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95` | `master` | clean |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/tula-src` | `f30f81d97c44bd79618273bb842302ef839c6ab1` | `main` | dirty: 6 entries |
| `/Users/gwilson/.codex/worktrees/1e8b/citlali-refactor/build/_deps/yaml-src` | `2f64971599ed05b15e2945949e834e96b6288afb` | `master` | clean |

## CTest inventory reconciliation

Before the `EXCLUDE_FROM_ALL` safety target was built, fresh `ctest -N`
reported 617 registrations: 616 discovered tests plus the single
`citlali_safety_test_NOT_BUILT` placeholder. Building the existing target did
not reconfigure or edit anything. PRE_TEST discovery then replaced the one
placeholder with 14 current safety tests, producing 630 fresh registrations.
The mechanical delta is therefore `-1 + 14 = +13`.

The fresh generated inventory contains:

- main `citlali_test`: 578 names;
- science-map FITS: 32 names;
- ownership: 6 names;
- safety: 14 names;
- total: 630 names.

The readable generated discovery files in the accepted author build have the
same four counts. The sorted fresh and author name sets are byte-identical and
have SHA-256
`a40589b690f5e3e32deb56b3ce9b541e5a6c984fde5468c7d641fde0c53b539e`.
There are zero author-only names and zero fresh-only names. Consequently no
author-only registration exists to classify as stale; the earlier 617 count
was solely the unbuilt safety placeholder state.

## Executed evidence

| Gate | Exact result |
|---|---|
| Fresh configure | Passed; exact accepted selected cache surface |
| Direct ownership, `OMP_NUM_THREADS=1` | 6/6 passed |
| Direct ownership, `OMP_NUM_THREADS=2` | 6/6 passed |
| Direct ownership, `OMP_NUM_THREADS=4` | 6/6 passed |
| Direct ownership, `OMP_NUM_THREADS=8` | 6/6 passed |
| Direct ownership matrix | 24/24 passed |
| Focused ownership CTest | 6/6 passed |
| Retained science-map contract/provenance selection | 22/22 passed |
| Complete parent science-map executable | 31/31 passed |
| Science-map FITS products | 32/32 passed |
| Fresh full CTest | 629/629 runnable passed; 1 established disabled |
| Config preflight unit tests | 127/127 passed |
| Compact compatibility | 8 passed, 0 failed, 0 skipped |
| Compact surface coverage | 261 covered, 17 profile-owned, 0 gaps, 100% |
| Config authority inventory | 15 domains valid |
| Validation ledger | valid, 60 records |
| Science-change ledger | valid, 3 changes and 5 integration commits |

The sole disabled full-suite registration was test 473,
`citlali::MapFitterLifecycle.ExactProductSequence`. There was no other skip,
not-run, failure, or unexpected error-level result.

## F003 disposition

For this exact candidate and corrected production trace, the F003 closure gate
can be accepted through its alternative of proved disjoint ownership:

- production grouping assigns a unique map destination per detector group;
- scans invoke the map operation sequentially;
- the candidate validates that ownership and every destination invariant before
  diagnostics, mutation, output, or parallel launch;
- all valid mappings preserve the pre-existing worker arithmetic and order;
- the independent 1/2/4/8, focused, retained, full-suite, config, and ledger
  evidence passes.

This conclusion does **not** claim that the frozen SCI-MAP-003 report's
original shared-array race premise was demonstrated in production. That claim
remains immutable post-core evidence and history. The present result is a
bounded contract-hardening acceptance for the exact candidate, not a broader
race-freedom theorem.

TSan is superseded and non-applicable to this corrected pre-parallel mission.
It was not passed, waived, rerun, or reinterpreted. The independent Apple TSan
history of cross-region stack/lifetime reports remains outside this re-audit's
acceptance basis.

The proposed finding disposition is:

```yaml
finding: SCI-MAP-003-F003
closure_basis: proved_disjoint_ownership_alternative
status: proposed_closed_pending_coordinator_acceptance
original_race_premise_demonstrated: false
tsan: superseded_non_applicable_not_run
production_authorization: false
```

## Limitations and stop boundary

- Ceres, kidscpp, and tula local dependency source trees were dirty as recorded
  above. Their state is a reproducibility limitation, not scientific or audit
  authority.
- The report accepts only the exact candidate SHA and exact current production
  ownership trace.
- No Unity access, reduction, telescope work, successful network access,
  network-derived input, external contact, TSan, repair, source edit,
  dependency edit, preset edit, reconfigure after the successful configure,
  downstream launch, integration, push, merge, rebase, or production action
  occurred. The earlier Conan-helper download attempt was stopped immediately
  and supplied no audit evidence.
- No canonical disposition, ledger, handoff, frozen audit, or frozen core was
  modified.
- Successful local validation does not authorize broader consumers or
  production. Existing use remains the ceiling until separate owner action.

The real Git index remains intentionally empty. This report remains
uncommitted for coordinator review.
