# Citlali Integration Ledger

This is the compact authority for concurrent Citlali workstreams. Update it
when a workstream starts, changes authority, reaches a gate, or integrates.
Detailed scientific and build evidence remains in the linked plans, handoffs,
ADRs, and validation records.

The application landmarks and their five independent status axes are routed
through [`APPLICATION_BASELINES.md`](APPLICATION_BASELINES.md). The active
WP-7 work order is governed by
[`WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`](WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md)
and `validation/wp7_timestream_successor_authority.json`.

## Current Work Orders — 2026-08-31

This table governs current work when an older row later in this historical
ledger uses stale language.

| Work order | Scope and authority | WIP rule | State | Exit |
| --- | --- | --- | --- | --- |
| `APP-CANON-001` | Canonical application integration on `codex/refactor-mainline`; the executable application tree remains exact `cb3d568c701217ee0248c77f6dccd0bab7deef31`, followed by accepted governance-only descendants | Normal accepted application work lands only through reviewed canonical ancestry | Active authority; administrative descendants do not imply a new tested executable identity | Ordinary affected-behavior gates and owner-controlled integration |
| `BUILD-ENV-001` | Current build/dependency authority is Spack under ADR 0014; exact V2 binding is `validation/citlali_v2_spack_validation_authority.json` | Build-environment work retains separate scope/ownership and may not silently change application science; non-Spack tests are supplemental; material Conan work requires an owner-authorized compatibility work order | Spack accepted for the most recent end-to-end V2 application generation; portable release bundle and exact V2 lock-byte retention remain open | Close only the bounded ADR 0015 release/reproducibility gaps or record a new owner build decision |
| `WP7-GOV-001` | WP-7.1 governance reconciliation on `codex/wp7-governance-reconciliation`, based exactly on `cb3d568c...`; accepted package commits are `e874044c4...` and `28e9e559b...` | Historical closed work order; later corrections require their own bounded authority | **Closed 2026-08-31:** owner reviewed and accepted G0--G3 and explicitly resumed G4 | Complete |
| `WP7-REPLAY-001` | First G4 canonical replay unit on `codex/wp7-g4-replay-001`; exact divergent source is `49fe73e757daa1885cd23127e8441cba47e648d2` | Only D2 PSD/line evidence tooling and its reconciled records are in scope; no dirty producer prototype, factor selection, filter design, nonidentity RTC route, or production activation | Locally verified candidate; all ten source paths are dispositioned and canonical gates pass | Owner-controlled review/integration of this single unit; stop before any producer or filter-design unit |
| `WP7-DIV-001` | Preserved divergent implementation/evidence lane `codex/wp7-rtc-fixed-decimation-authority` at immutable head `49fe73e757daa1885cd23127e8441cba47e648d2` | Preserve without discard, rebase, push, or new commit during `WP7-REPLAY-001`; uncommitted work remains outside immutable identity | Source-only lane with all committed and dirty work preserved | Supply exact source material to bounded replay units; never become a second application mainline |
| `SCI-CONTRACTS-001` | Scientific-contract library on `codex/scientific-contract-library`; last committed head `54475956f6aefb839d43b2f0fb019a142cb64310`; package status owned by `doc/scientific_contracts/INDEX.md` in that lane | Contract authoring has its own package-level WIP and does not count as WP-7 application implementation; staged/uncommitted authoring is not an immutable identity | Active independently | Frozen package/owner gate under the contract-library charter; application impact enters a separate integration work order |
| `MAPSPACE-CONTRACTS-001` | MAP, FLT-FIXED, FLT-INF, NOI, JINC, SRC/MODE, and related mapspace contract packages under the scientific-contract library | Packages may be authored in parallel when their scopes do not edit the same frozen authority; no package silently changes application code | Active independently of the WP-7 application hold | Package-specific Stage A/Stage B, owner, and freeze gates in the contract index |

Application-integration WIP ceiling: one active WP-7 application work order.
Scientific-contract authoring is a separate authority lane, but every proposed
application consequence receives a new bounded integration work order before
code changes begin.

## Detailed Branch Authorities And History

| Workstream | Authority | Purpose | Integration rule | State |
| --- | --- | --- | --- | --- |
| Refactored application | `codex/refactor-mainline` at `cb3d568c701217ee0248c77f6dccd0bab7deef31` | Canonical source, tests, configuration, operational behavior, and validation history | Normal application changes land here after their affected gates | Active canonical authority |
| WP-7 timestream integration checkpoint | `codex/refactor-mainline` and `codex/wp7-timestream-integration-candidate` at `a36abaebfb82d503b113de0cf4c1c6e0f6dcffc3`, preserving exact Unity-tested application commit `3ebc2a67fc32bad69759ff45638484efabf91773`; bounded cleanup `aa85a2287`; audit-tool repair `ff7899668` | Carry completed Stage 7 152390 evidence, retire only pre-censused inactive sources, admit current campaign evidence to the reduction auditor, and preserve a classified ref/worktree retirement record | Retain `3ebc2a67f` as the exact Stage 7 science identity and `a36abaeb` as the pre-build-modernization point-smoke identity; do not infer a general successor baseline or science-ledger disposition | Integrated and pushed by the owner; both forensic tags are published; the bounded point smoke passed and its reports are preserved outside Git |
| SCI-MAP-001 application integration | `codex/refactor-mainline` at `d5015fe716971bf8ea617e8a187311bf5af05185`, containing exact application source `af0c849ce59a5f80e5efc8db435bb6662863052f` followed only by its documentation-only integration record | Accepted bounded implementation, product/provenance contract, truth suites, owner-amended evidence, and frozen campaign/closeout history | Preserve as the application base for subsequent audited integrations; do not merge MAP audit or coordination branches | Integrated; bounded MAP contract accepted; production remains `existing_use_only` |
| SCI-NOI-002 application integration | `codex/refactor-mainline` at `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, containing exact audited application source `5b29e13548a6fec884c67b192dec20c92f0bbb62` followed only by its documentation-only integration record | Accepted bounded conditional-stack, package-provenance, truthful-labeling/count, writer/finalizer, and validator contracts | Preserve as the application base; do not merge repair, audit, or coordination branches | Integrated; production remains `existing_use_only`; F005/F006 remain external |
| APT-PROD-001 canonical baseline APT v1 | `codex/repair-apt-prod-001-canonical-baseline-v1` at pushed commit `d4a808c59f383a5f77059b83083af2a69802a12a`, parent `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree `f77150abe863de73585d37a91485ea0e8a1951d0`; frozen audit authority `6cf83a21169516303db1fa30d26f4be32a813844` | Typed Citlali-only Beammap baseline APT producer, embedded raw relation, semantic/envelope/transport identities, executable artifact contract, and receipt-last publication | Preserve as the exact APT-PROD-002 base; integration or downstream admission requires a separate owner decision | Pushed bounded producer; artifact contract unactivated; no application-mainline or production/downstream admission |
| APT-PROD-002 observation-specific APT v1 | `codex/repair-apt-prod-002-observation-contract`, created from exact pushed APT-PROD-001 base `d4a808c59f383a5f77059b83083af2a69802a12a`; frozen audit authority `6cf83a21169516303db1fa30d26f4be32a813844`; accepted Phase-B full-index patch SHA-256 `8f452e9775a5a74b688ef3766ec31ae327e80ba50359953eebb436a006114cb8` | Citlali-owned observation APT contract, canonical ECSV codec, strict JSON protocol, complete embedded target/relation provenance, and reusable receipt-last publisher | Admit only one coherent exact candidate after all broad gates; owner controls push and any later integration; never infer TolProj/downstream or production activation | Active bounded Phase-C candidate; all contracts unactivated; no downstream launch or production admission |
| APT-PROD-003 compact canonical APT v2 | `codex/repair-apt-prod-003-compact-v2` at base `20feebc26f5ab36f3db04d05835de6ac907fd2e6` | Owner-directed normalized ECSV contract, v2 producer/guardian, TolAPT occurrence-scoped matcher seam, TolProj orchestration, migration-only v1 reader, and root-receipt-last publication | Exact 148669/148670 equivalence, <20 MiB portable bundle, determinism, relocation, tamper, parser-count, and all consumer gates; no Unity before owner runbook | Active uncommitted repair; all APT-dependent validation and new baseline issuance suspended; no activation |
| APT / ALIGN / JINC convergence | `codex/converge-apt-align-jinc`, created from exact owner-run Unity-tested implementation `e77460cffad49387795009539d6abc7e370e8b58`; documentation-only validation parent `91f42ccdc8ce9a4e6811f2f03857180d50d21345` | Preserve the accepted compact-v2/MAP/JINC spine, account for omitted and historical commits, and reconstruct only independently dispositioned application contracts | Do not merge the 76-commit SCI-ALIGN research line or cherry-pick old APT-v1 consumers; reconstruct JINC ownership first, review a separate compact-v2 native-consumer candidate, and keep conditional PTC metadata repair isolated | Active preparation; no post-`e77460cf` application replay, no production expansion, and no push authorized |
| Spack build-modernization integration | `codex/integrate-spack-build-modernization`, based on exact mainline checkpoint `a36abaebfb82d503b113de0cf4c1c6e0f6dcffc3` and merging exact build-lane head `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046` through merge `4cf8db223cdfc7163bbac91972528d8c0c2dbe78` | Historical source lane for the Tula CMake/Spack build, package-consumer, provenance, release-contract, timing, and V2 build environment | Preserve the source-lane evidence; Spack is the accepted realization for the V2 generation, while release composition and same-SHA platform closure follow the canonical status document | Integrated and operationally demonstrated by owner-accepted Spack-backed V2; source lane preserved, remaining Phase 5 release gates open |
| Successor build adaptation source | `codex/build-adaptation` at `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046`, five commits ahead of `origin/codex/build-adaptation` | Port the full application into the accepted Tula CMake/Spack architecture | Preserve until the integration candidate passes local and Unity gates; do not delete or rewrite its evidence | Frozen as the exact merge input in `/Users/gwilson/GitHub/citlali-refactor-build` |
| Historical Conan 2 adaptation | `codex/conan2-adaptation` at `9aae0e669` | Preserved pointer to the superseded package-manager lane name | Legacy reproduction or bounded compatibility only; do not resume, add dependencies, extend recipes, or repair opportunistically without an explicit owner-authorized compatibility work order | Frozen remotely; not a current build authority |
| Historical structural refactor | `codex/structural-refactor` at `171487196` | Forensic pointer to the pre-follow-on integration tree | Do not resume as an application authority | Frozen |
| Historical fruit-loop/raw-IQ topic | `codex/fruit-loop-calibration-reference` at `b02fef613` | Forensic pointer to the topic-named branch before mainline normalization | Do not resume as an application authority | Frozen locally; its existing remote remains historical |

## External Build Inputs

Latest isolated review completed 2026-08-10:

| Repository | Branch | Reviewed commit | Disposition |
| --- | --- | --- | --- |
| `tula_cmake` | `v3.x_spack` | `6433cdabe7010d0af2d0ba69da7af27510391b80` | Retains the normalized dependency adapters and separates artifact provenance from deployment provenance |
| `tula` | `v3.x_spack` | `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` | Exports source, compiler, package, and concrete build identity through its installed interface |
| `kidscpp` | `v3.x_spack` | `498ece1113001ae2d42d96d9fc29152aea3eaaef` | Exports equivalent artifact provenance and retains portable versus optional real-data test separation |
| `citlali` | `v3.x_spack` | `ceb4335c3f00b52af58ce2c09093b863040434b6` | Reference implementation only; its provenance design is reviewed without replacing the refactored application |
| `tolteca_deploy` | `main` | `0a6b896` (`v0.1.1`) | Read-only deployment design evidence; not consumed or modified by this lane |

Branch movement does not update this table automatically. Re-review and record
new exact commits before importing subsequent upstream work. The three consumed
dependency revisions are also enforced by `spack/upstream-revisions.json`;
`tolteca_deploy` is neither modified nor used as an input to this lane.

The accepted `tula-netcdf-cxx4` adapter does not yet export NetCDF-C's include
directory to installed consumers. Citlali therefore retains a direct
`netcdf-c` dependency and target until that upstream interface is corrected;
this is a build-contract workaround, not a fork of the upstream package.
The 2026-08-10 upstream revision leaves that adapter interface unchanged, so
the workaround remains required despite the other release fixes in this round.
Tula's generated version header also still reports `cxx_standard=0` while the
Kidscpp header correctly reports C++23. Citlali's own build provenance records
the effective C++23 standard; the Tula metadata defect remains upstream debt.

## Current Gates

### Application Mainline

- Default reproduction and extension of the most recently validated application
  generation to the accepted Spack environment. Preserve the existing local
  build only as fallback compatibility and supplemental smoke-test machinery.
- Run the focused tests, complete CTest/config gates, and affected-mode Unity
  validation required by the behavior touched.
- Record intentional scientific changes separately from refactoring and build
  integration.
- Treat the owner disposition that restored Stage 7 busy-row suppression is
  not overly aggressive as bounded correctness-repair evidence only. The
  current intended-science-change schema still requires an accepted-run
  record, and the one-observation campaign has no accepted same-input
  successor comparison; do not alter either ledger or infer a general
  successor baseline from this result.

### APT-PROD-001 Canonical Baseline APT v1 Candidate

- Accepted identity is artifact-local only: `uid` is a unique, sparse-
  permitted, nonnegative exact `int64` in `0..2^53-1`, never persistent
  detector identity. The artifact embeds a complete bijection from UID rows to
  the declared `(network, channel)` raw inventory; persistent detector and tune
  identities are omitted.
- The fixed standalone contract admits five protected structural fields, the
  exact 27-field baseline registry, and only the exact 20-field optional
  extension allowlist. A general C++ strict-extension seam does not activate
  custom artifact fields. `fg`, `pg`, `ori`, and `loc` remain nonidentity
  semantic content under nullable unavailable authority.
- Optional `kids_flag` is copied-declared from `kids:fit-report-v1` as an exact
  signed, potentially nonbinary `int64`. It is distinct from `flag` and
  `flag2`, and is absent for simulation without a fit report.
- Semantic content, publication occurrence/envelope, and byte transport have
  separate versioned SHA-256 scopes. The adjacent envelope-bound receipt is
  published last after staged and final reread/recomputation and never replaces
  the embedded semantics or raw relation.
- Required candidate gates include focused model/codec/adapter/writer tests,
  independent fixed-vector recomputation, executable artifact-contract tests,
  full retained CTest and baseline-tool gates, config preflight, CLI build,
  exact changed-path and patch/commit/parent/tree verification, clean state,
  and proof of unchanged science/detector set/order. Results are not recorded
  here until the exact coherent candidate is supplied and independently
  verified.
- The artifact contract remains `unactivated`: no validation-profile change,
  downstream ingestion, historical migration/repair, CAL closure, production
  expansion, external-repository change, or audit/coordination branch movement
  follows from this candidate. Owner-controlled push/integration and any later
  downstream admission are separate decisions.

### APT-PROD-002 Observation-Specific APT v1 Candidate

- Persist exactly one observation-specific canonical APT-family ECSV and its
  adjacent envelope-bound receipt. The complete target manifest and generalized
  match relation are embedded logical records, not separately published
  artifacts or a public bundle chain. JSON is protocol representation only.
- Every baseline, target, relation-pair, and output reference is occurrence-
  scoped and artifact-local. UID or sequence equality never creates persistent
  detector identity. Source, application, seed-source, and presentation
  sequences remain explicit nonidentity permutations.
- Relation coverage remains complete and generalized: every target is matched
  or unmatched, every seed is matched or unused, pair sets are reciprocal, and
  per-field source selection may name different valid pairs. An unmatched
  target has no fabricated seed endpoint and carries typed missing state for
  unavailable seed-derived values.
- Citlali owns the schemas, field registry, canonical framing and ECSV bytes,
  semantic/envelope/transport identities, validation, opaque output issuance,
  output-local UIDs, receipt, and no-replace receipt-last publication. TolProj
  remains the legitimate issuer of observation-specific target values and
  realized matcher provenance; the protocol implements no matcher policy.
- The closed KMP value authority admits only required `kids_fr`, `kids_f_out`,
  and `kids_Qr`, plus artifact-optional exact signed `kids_flag`. Unknown
  diagnostics remain covered only by selected-source SHA-256/count and cannot
  be requested for canonical identity, matching, transformation, output,
  units, or authority.
- Required candidate gates are the four named local build targets, complete
  CTest with only the established disabled test, full baseline-tool discovery,
  config preflight, both validation ledgers, retained focused C++/Python/
  protocol/vector/header gates, exact 20-path and patch/commit/tree identity,
  `git diff --check`, and a clean post-commit worktree/index. A required skip,
  identity drift, science/detector/order drift, or unexpected error result
  stops the package.
- The accepted limitations are not repaired here: publication is not
  fsync/crash-durable before receipt; a post-publication stdout failure can
  cause a false-negative acknowledgement recoverable through validation; and
  protocol stdin has no owner-specified absolute byte quota.
- All successor contracts remain `unactivated`. No validation profile,
  accepted run, ingestion, CAL, ALIGN, TolProj, sibling repository, Unity,
  downstream, or production state changes in this candidate. The containing
  coherent commit and owner push remain distinct from later integration or
  admission decisions.

### SCI-MAP-001 Application Integration

- Final independent re-audit
  `8fc716557ca78b0d220200a92be46fa3545797e9` and final canonical coordination
  candidate `c7bb0214edfd57fddf31165923f08784dfd1b8c9` establish the bounded
  package axes `approved`, `conformant`, `complete`, and
  `existing_use_only`, with bounded-contract verdict `accept`.
- The coordinator-directed 2026-08-05 task separately authorizes application
  integration. Exact application source remains
  `af0c849ce59a5f80e5efc8db435bb6662863052f`; its application tree is
  `47aa745554e47514398e72d579625484abdcb79e`. The branch-tip child is only the
  documentation/status commit containing the linked
  [integration handoff](../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md).
- F001--F011 are closed. F012 is
  `closed_bounded_owner_accepted`, retaining the absent raw/sample ledger,
  pre-normalization/commit-order, operational-chain, and historical same-case
  S-X observation-realization lanes as explicit limitations. F013 remains
  `open_conditioned` on ALIGN, CAL, AST, PTC, and VAL.
- Preserve ordinary compatible numerical behavior, the frozen raw-parent
  snapshot and `RAWPDGST` carriage, explicit unsupported-profile absence, and
  the exact ADR 0009 F009/F010 contract. Do not broaden mapmaking,
  noise-generation, coadd, WCS, threshold, output-routing, configuration
  defaults, or any other mature algorithm.
- The completed seven-case external campaign remains frozen historical
  evidence; do not repeat it. This application integration did not access
  Unity or the external corpus and closes no upstream dependency.
- The owner fast-forwarded the verified integration candidate to application
  mainline at `d5015fe716971bf8ea617e8a187311bf5af05185`. Production expansion
  and Conan adaptation import remain separate decisions; production stays
  `existing_use_only`.

### SCI-NOI-002 Application Integration

- Final independent Cycle 4 re-audit
  `6de648f5ae2b37f5bc65162feae221f19bb84a5a` and canonical coordinator
  closeout `d03ef80b31f704859ef836e368801dc17d92e76e` accept exact application
  source `5b29e13548a6fec884c67b192dec20c92f0bbb62` as `approved`,
  `conformant`, `complete`, and `existing_use_only`, with controlled verdict
  `retain`.
- The integration branch starts at exact current mainline `d5015fe...` and
  advances by an exact six-commit fast-forward to application tree
  `641c724f40a9fa9f322f09c703705239439d2374`. No audit, coordination,
  conflict-resolution, or patch-reconstruction commit enters the application
  ancestry.
- F001/F002/F003/F004/F007/F008 and the recorded Cycle 3/package repair
  findings close within their bounded contracts. F005/RA-B004 remain
  SCI-FLT-001-owned; F006 remains SCI-FRUIT-001-owned. No physical-noise,
  calibrated-significance, count-adequacy, default, or production claim is
  added.
- The exact candidate passed all required build, focused/full product, Python,
  CTest, and configuration gates. The later integration record is
  documentation-only, so complete executable gates are not repeated.
- The intended-science-change ledger remains unchanged because its current
  policy requires an accepted reduction and the bounded audit required no
  astronomical reduction. Do not weaken that gate or cite an unrelated run;
  the exact product contracts and re-audit/closeout artifacts remain the
  authority for this schema and labeling correction.
- The owner fast-forwarded `codex/refactor-mainline` to verified integration
  tip `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`. Production expansion and
  Conan-lane import remain separate decisions.

### Spack Adaptation Entry

- The native macOS foundation environment is reproduced with exact Homebrew
  LLVM 20.1.8 and Spack 1.2.2. Its source-built Tula closure passes 14 package
  tests and an independent installed consumer. Local containers remain
  optional.
- The native Kidscpp environment now includes an explicit Spack
  `llvm-openmp@20.1.8` edge. Kidscpp installs from source, its independent
  installed consumer passes, and a separate reader consumer opens a current
  raw TolTEC file and reads a two-sample I/Q slice. The external payload for
  pointing observation 152389, network 0, scan 2 is bound to a checked
  path-independent manifest by basename, byte size, and SHA-256. The
  historical upstream fixture is not used as acceptance evidence.
- Citlali now consumes upstream-owned `tula-ccfits`, `tula-netcdf-cxx4`, and
  `tula-perflibs` targets. Its temporary package and CMake adapters have been
  removed, and general pipeline OpenMP is distinct from Wiener OpenMP.
- The full refactored Citlali library, production CLI, 539 enabled CTests,
  complete config preflight, installed CLI, and independent installed package
  consumer now pass natively under exact Homebrew LLVM 20. The CLI records its
  source dirty state and concrete Spack DAG identity. Managed launches also
  require their runtime environment lock to contain that exact root DAG and
  record the deployment profile and lock digest in product metadata.
- Add a user-owned Unity environment based on user-supplied
  Spack/compiler/module facts. The first inventory identifies Ubuntu 24.04,
  GCC/G++/GFortran 13.3, CMake 3.30, Python 3.12, no Ninja, and no
  user-callable Spack.
  A user-owned Spack 1.2.2 `unity-gcc13` profile and prerequisite gate are now
  prepared. Its earlier lock is superseded by the upstream-adapter graph;
  fresh concretization passed at lock SHA-256 `d2204524bc170cf9e9458a9f83f2730f4e44c78fe628eb01f7d9f8dee0c52f72`.
  Unity job `62888690` installed the full graph and passed all 539 enabled
  developer CTests at exact Citlali `0add18c24`, but stopped before installed
  CLI/consumer acceptance because the expected Linux Spack development build
  directory was not ignored and the clean-source gate rejected it. Job
  `62889350` at exact Citlali `34b83df51` then passed that hygiene gate and all
  539 enabled CTests; its installed CLI reported the correct source, package,
  compiler, DAG, profile, lock, and managed-binding identities. A checker-only
  assumption that Git abbreviations are always nine characters stopped the
  job before the independent installed consumer. Dynamic abbreviation
  matching was corrected in `7ee2c4f7`. Unity job `62890572` then closed the
  build gate at that exact commit: both executions of all 539 enabled CTests
  and the independent installed consumer passed, and the complete manifest
  binds executable SHA-256 `89a159c508f3f51bd0556105486fb4dadccf6bce525a6d0ceff99f53da04145e`
  to the accepted source, dependency, compiler, DAG, profile, and lock
  identities. An unmanaged point smoke completed with an exact effective
  config, complete product contract, unchanged common pointing-fit columns,
  and negligible map differences, but could not close the managed provenance
  gate. Its managed rerun exposed concurrent NetCDF-C/HDF5 access across the
  independently ordered timestream output streams and segfaulted during file
  close. A shared output-I/O serialization repair passes the full local build,
  CTest, config, build-tool, and baseline-tool gates. Exact repair commit
  `b8e80fb15` then passed the interactive Unity acceptance workflow, and two
  completed point runs reproduced all 1,930 compared scientific records
  exactly without another HDF5 failure. Direct Citlali invocation accepts the
  managed profile, lock, and DAG binding, but TolTECA-launched products remain
  marked unmanaged. That propagation contract is assigned to the external
  deployment owner; this repository will not add a competing local wrapper.
  One deployment-owned fix and managed point rerun remain pending.
- The macOS graph builds CFITSIO 4.3.0 from source and declares Homebrew FFTW
  plus GCC 15 Fortran as checked host externals; all C/C++ application code is
  LLVM 20.
- The formal clean native build campaign at exact source `3a4defda5` records
  5.93 seconds configure, 161.60 seconds clean build, 0.94 seconds no-op, and
  173.22 seconds for the header-dominant CLI translation unit. The Unity GCC
  13 campaign at clean source `4f9c7e55a` records 20.16, 315.56, 4.98, and
  277.38 seconds respectively. This closes the formal build-timing gate and
  confirms the CLI translation unit as the dominant development cost on both
  hosts.
- The Citlali-owned release schema, validator, bundle layout, source-build
  fallback, and signed-buildcache policy are accepted in ADR 0015. Proposed
  repository-owned metadata commits now bind all ten canonical recipes to the
  exact accepted sources, and the local manifest-bound audit passes 10/10.
  Release production remains open: review and publish those recipe revisions,
  publish immutable source and recipe archives/checksums, and only then
  generate one source-based, host-path-free lock for each supported profile.

### Spack Adaptation Exit

The exit gates remain those in the build integration review:

1. Full refactored library and CLI targets build through Spack.
2. Installed library and CLI package consumers pass.
3. Complete local CTest, configuration, baseline, ledger, and exit-policy gates
   pass without required-data skips.
4. Source, dependency, compiler, configuration, dirty-state, and managed
   deployment provenance are truthful and the runtime lock root matches the
   executable's compiled DAG.
5. The exact commit builds on Unity without source edits and passes a point
   smoke reduction.
6. The frozen same-SHA point, OOF, science, and Beammap matrix passes.
7. The existing build remains available until a separate retirement decision.

## Import Policy

- Do not merge `citlali/v3.x_spack` wholesale.
- Import build definitions and infrastructure in coherent, reviewable changes.
- Consume Tula and Kidscpp through explicit versioned package identities.
- Review upstream Citlali source changes individually against mainline; port
  only changes that remain applicable.
- Do not resolve build conflicts by deleting mainline tests, generated config,
  provenance, CLI behavior, or validated application sources.
- Do not combine numerical algorithm changes with build integration commits.
- Do not import the isolated SCI-MAP-001 repair directly into the Conan 2 lane.
  It becomes eligible for ordinary synchronization only after the owner
  fast-forwards the verified application-integration candidate into
  `codex/refactor-mainline`; MAP closure alone does not update the Conan lane.
- Do not import the isolated SCI-NOI-002 repair directly into the Conan 2 lane.
  It becomes eligible for ordinary synchronization only after its verified
  application-integration candidate reaches `codex/refactor-mainline`.
- Do not import APT-PROD-001 into the Conan 2 lane or any downstream consumer
  from its repair branch. It becomes eligible for ordinary synchronization
  only after exact candidate verification and owner-controlled integration into
  `codex/refactor-mainline`; integration still does not imply downstream
  artifact admission.

## Repository Hygiene

Existing committed evidence is retained without history rewriting. For new
studies, keep reusable tools, compact reports, manifests, checksums, and
essential fixtures in this repository. Prefer a versioned external evidence
archive for large generated matrices, plots, and reduction products. A later
housekeeping change may improve navigation, but it is not part of Conan 2 or
Spack adaptation.
