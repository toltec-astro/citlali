# Timestream Successor Program Reset — Owner Package

Date: 2026-08-31

Status: **Milestones 0–3 complete; Milestone 4 owner checkpoint; no Milestone 5
operation authorized or performed**

## Recommended owner disposition

1. Accept the governance candidate at exact commit
   `06a3ade51c1b3f38887295433d913811bf25cd14` after its independent PASS.
2. Preserve and integrate the paired-native D1 implementation commit
   `d7d19bc90d7c994fa767ec2a9fd35e4d8599f032`; preserve its historical
   closure record `2f1d836c1db122d22015853582133abf3611bc30`, but do not merge that control
   record mechanically across the new governance and naming documents.
3. Preserve the rejected D2 source and review at
   `916fa07600cf6c5e9ea7317a396fdce160a6c419` and
   `34f609e4b1dc9a04f8157063c7a1662b707d96a7` as evidence. Do not resume it.
4. Approve the staged integration sequence and exact existing-object GitHub
   packet below. A later canonical push for commits that do not yet exist MUST
   wait until those commits have exact SHAs; this package does not fabricate
   them.
5. Approve no remote deletions. Close only the clean, proven-redundant
   worktrees after integration, remote preservation, and exact post-operation
   verification.
6. After closure, make the first new Timestream Successor work order the
   concrete native paired-ingress adapter, not the rejected D2 seam.

The consequential decisions are therefore governance acceptance, paired-D1
integration, the exact preservation push packet, and the proposed next work
order. Existing dirty owner checkouts, contract lanes, stashes, bundles, and
special evidence refs remain untouched.

## 1. Current paired-D1 disposition

Disposition: **candidate preserved at its bounded validation/documentation
boundary**

| Identity | Exact value |
| --- | --- |
| Canonical base | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` |
| Implementation commit | `d7d19bc90d7c994fa767ec2a9fd35e4d8599f032` |
| Implementation tree | `af8d4d7c6e8f855845590e63d59ae4a3d43d00f5` |
| Documentation/control record | `2f1d836c1db122d22015853582133abf3611bc30` |
| Control-record tree | `a24e698cff294466964cd27b6efd7f30cbac019f` |
| Historical work-order alias | `WP7-REPLAY-002A` |
| Branch locator | `codex/wp7-g4-replay-002a` |
| Worktree | `/private/tmp/citlali-wp7-g4-replay-002a` |
| Final worktree state | clean |

The implementation commit changes exactly:

- `include/citlali/core/pipeline/timestream_native_paired_readout.h`;
- `tests/CMakeLists.txt`;
- `tests/test_timestream_native_paired_readout.cpp`; and
- `tests/timestream_native_paired_readout_header.cpp`.

It was reviewed against the WP-7.1 Timestream Contract Baseline source
`170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure
`20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, the canonical authority router,
canonical native timing/identity decisions, `doc/ARCHITECTURE.md`, and
`doc/SCIENTIFIC_CONVENTIONS.md`.

Focused results on the exact implementation were:

- isolated C++23 public-header syntax compilation: pass;
- native-paired-readout contract tests: 6/6 pass; and
- exact identity, shape, finiteness, run, occurrence, cause, and bounded
  logical-memory cases: pass.

Broader results were:

- 839 CTests discovered; 838 runnable passed; the established disabled
  `MapFitterLifecycle.ExactProductSequence` was not run;
- `citlali_cli` built and reported short identity `d7d19bc90`;
- configuration preflight: 130 tests, four mode kits, 8/8 compact cases,
  100% compact-surface coverage, all authority audits: pass;
- baseline tools: 207/207 pass;
- build tools: 62/62 pass;
- Timestream Successor/WP-7 historical tooling: 26/26 pass;
- validation ledger: 60 valid records; and
- intended-science-change ledger: 3 changes and 5 integration commits valid.

The build reused already-present fallback dependency sources after a fresh
configure could not reach GitHub. It is supplemental local
compilation/regression evidence. **No Spack or Unity validation was performed,
and these results do not reproduce the accepted Spack-backed V2 campaign.** No
affected-mode reduction was triggered because the increment activates no
route, configuration, numerical operation, or product publication.

Excluded and still unauthorized: a producer adapter, D2 observation,
prefilter/residual planes, RTC/PTC wiring, common-grid projection, AST change,
filter design, factor selection, downsampling, production routing, activation,
and a new abstraction framework.

Integration: **NOT PERFORMED**. Push: **NOT PERFORMED**. Production
authorization: **NOT ESTABLISHED**.

## 2. Authoritative application-generation naming map

| Durable human-readable name | Historical aliases/locator | Exact representative identity | Authority/evidence role | Owner status |
| --- | --- | --- | --- | --- |
| Legacy Citlali — fork comparator | original refactor fork point | `376e002238b1f49aeced8a3f33e8742db141634b` | legacy source comparator | accepted naming interpretation |
| Legacy Citlali — preserved head | `gw_dev`, informal “Classic” | branch `gw_dev`; `ffc6b9070f4744f9778f3db71cdc468846d1da89` | later preserved legacy source | accepted naming interpretation |
| Structural Refactor Baseline | behavior-preserving architectural refactor | application `cee74ecbdfb4187756183879163a22ca2b8518f6`; validation landmark `71b3fd3d33b5b8ff236ea5ceff616ffa199d9208` | byte-identical architectural baseline and validation evidence | accepted |
| Native Integration Baseline — pre-WP-7 | “Citlali Native”; not ALIGN and not V2 | final integration `f0f423827ab321640e0cbcb003f7bf015368f694`; V2 application identities `c31a60…` and `187df04…` | completed integration baseline; mixed-SHA Spack-backed V2 acceptance evidence | accepted with explicit non-claims |
| WP-7.1 Timestream Contract Baseline | precise historical/scientific qualifier retained | source `170ecea9de1ee810da7d7e45a489a4545ccd623d`; closure `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa` | frozen scientific and engineering authority | accepted |
| Timestream Successor | historical `WP-7.1 Timestream Successor Program`, `WP7-REPLAY-*`, and `codex/wp7-*` aliases | current paired-D1 evidence `d7d19bc90d7c994fa767ec2a9fd35e4d8599f032`; closure `2f1d836c1db122d22015853582133abf3611bc30` | enduring implementation program; candidate evidence only | enduring name explicitly directed; no production generation yet |
| Canonical application integration line | `codex/refactor-mainline` | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | sole local integration authority | verified from canonical records |
| Spack Build Adaptation | separate build lane | exact merge input `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046`; canonical merge `4cf8db223cdfc7163bbac91972528d8c0c2dbe78` | accepted build adaptation and Spack evidence; not science authority | accepted; cached remote branch is stale |
| Divergent successor predecessor | `codex/wp7-rtc-fixed-decimation-authority` | `49fe73e757daa1885cd23127e8441cba47e648d2` | forensic implementation/design evidence only | explicitly noncanonical |

The V2 corpus is not an application name. `citlali-validation/v2` is a corpus
revision, and its accepted runs bind application, executable, configuration,
and Spack environment identities separately. ALIGN is likewise a pipeline
stage; SCI-ALIGN is its contract/audit program; `codex/sci-align-*` names branch
lineage.

No owner-accepted permanent branch spelling for the future Timestream
Successor spine was found. This package therefore does not represent a new
branch spelling as already accepted. The governance branch name is temporary,
and old `WP7-*` names remain immutable searchable provenance.

## 3. Canonical branch and subject-specific authority

Repository common Git directory:
`/Users/gwilson/GitHub/citlali/.git`

Audit invocation/current directory:
`/Users/gwilson/GitHub/citlali-refactor`

Current checkout during census:
`codex/sci-align-001-lissajous-timestream-fit` at
`353b11887ff04dfd7bca12915917495f81a587fa`

Canonical local branch and tip:
`codex/refactor-mainline` at
`4dc7844e59e03cf2d18a9262fe5b75d3ff078681`

Cached, not fetched, remote-tracking tip:
`origin/codex/refactor-mainline` at
`cb3d568c701217ee0248c77f6dccd0bab7deef31`

Local canonical is five commits ahead of that cached tip:

1. `e874044c4c562fe672890495a3f4d5064e789d8f` — canonical governance
   reconciliation;
2. `28e9e559b7e74d13e05427c54b13c89e9a6c6f1b` — Spack-backed V2 validation
   authority;
3. `f6c9033f80810da255a9bfa987e0fba8a082b785` — prior governance/G4 closure;
4. `f8ba732bc4072e918c2521a013305be354ed7b53` — accepted bounded D2 evidence
   tooling; and
5. `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` — its canonical integration
   record.

| Subject | Current authority | Does not establish |
| --- | --- | --- |
| sequencing/status | reset directive for this sequence; canonical `doc/REFACTOR_STATUS.md` and `doc/INTEGRATION_LEDGER.md` otherwise | science, production |
| architecture/ownership | canonical `doc/ARCHITECTURE.md`; accepted canonical ADR index and ADRs | new scientific method |
| scientific meaning | WP-7.1 source `170ecea9…`, closure `20ba6ae5…`, canonical authority manifest, scientific conventions and owner dispositions | implementation conformance |
| executable acceptance | exact validation manifests and accepted-run records, including Spack environment identity | canonical integration or production |
| bounded work order | exact owner-approved scope and base; historical alias is provenance | work outside the increment |
| implementation/tests | exact source/test commits and reproducible results | authority merely from code or passing tests |
| integration | owner disposition recorded on `codex/refactor-mainline` ancestry | production authorization |
| production | explicit owner operational disposition | broader science or release claims |

Reconciliation findings:

- Canonical ADR 0014 is the Spack build foundation. Divergent successor ADR
  numbers 0014–0020 were reconciled as canonical ADRs 0017–0023; the old
  numbers remain historical only.
- The scientific source and closure are not canonical ancestors. Canonical
  manifests bind them by exact identity, and cached remote scientific-contract
  refs contain both. This is durable reference, not application integration.
- `codex/wp7-g4-replay-001` is an exact local duplicate of canonical tip
  `4dc7844e…`; it is not a second mainline.
- The paired-D1 base exactly matches canonical tip. Its two commits are
  descendants, not divergent implementation ancestry.
- `49fe73e…` and the earlier D2 capture are divergent evidence and cannot be
  renamed into a canonical successor branch.
- `codex/scientific-contract-library` is both staged-dirty and
  ahead 1/behind 20 relative to its cached upstream. It cannot be reconciled
  by a simple rename, push, rebase, or merge without its owner.
- The accepted V2 application validation is Spack-backed. Local fallback,
  Conan, Homebrew, or syntax-only evidence remains supplemental and must not
  be promoted to equivalent representative evidence.

## 4. Governance candidate

| Identity | Exact value |
| --- | --- |
| Branch | `codex/timestream-successor-governance` |
| Clean base | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` |
| Initial candidate | `bf649411d0760ee456a270e89884f66f39b63a62` |
| Reviewed/repaired candidate | `06a3ade51c1b3f38887295433d913811bf25cd14` |
| Candidate tree | `56d652bc23364761bacb805b04975e331114515e` |
| Worktree | `/private/tmp/citlali-timestream-successor-governance` |
| Final state before this package | clean |

The cumulative candidate changes exactly:

- `AGENTS.md` — SHA-256
  `27e21353a9578fa2f2cc01fbf266499e62ed6f99b3cd1de6d56fc7f1d691ed22`;
- `doc/governance/ENGINEERING_GOVERNANCE.md` — SHA-256
  `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md` — SHA-256
  `29fae6f789bb6133c1f5bcdaf0f15437f2eb8c4110f338d3a8de9a4d98ba88dc`;
  and
- `doc/governance/REVIEW_AND_CONFORMANCE.md` — SHA-256
  `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1`.

Independent fresh-context exact-SHA review of `bf649411…` found two minor
repository-policy omissions: governance documents were missing from the
durable bare-WP7 naming prohibition, and prose guidance was present where the
directive required a mechanical-check template. The bounded repair commit
`06a3ade51…` changed only the two affected documents. Independent re-review of
that exact commit passed:

- scientific/behavioral: **PASS, no findings**;
- architecture/ownership: **PASS, no findings**; and
- repository/evidence: **PASS, no findings**.

Validation of the documentation-only candidate:

- `git diff --check`: pass;
- configuration preflight: 130 tests, four mode kits, 8/8 compact cases,
  100% surface, all authority audits: pass;
- validation ledger: 60 records valid;
- science-change ledger: 3 changes/5 integration commits valid; and
- baseline-tool suite: 207/207 pass using the locally available executable
  from `187df04…` solely to satisfy the test harness's worktree-local
  executable precondition. There is no application/build/test-system diff
  from that source through the governance candidate. The temporary binding was
  removed and the candidate tree remained unchanged.

No CTest, Spack, or Unity run was warranted or claimed for the
documentation-only governance candidate. It is not effective until owner
acceptance, canonical incorporation, and a canonical ledger record naming
`06a3ade51…` and the four digests above.

Canonical integration: **NOT PERFORMED**. Push: **NOT PERFORMED**.

## 5. Exact repository census and reconciliation

The exhaustive machine-readable census is
`handoff/TIMESTREAM_SUCCESSOR_REPOSITORY_CENSUS_2026-08-31.json`, captured at
`2026-08-31T18:26:55.012556+00:00`, SHA-256
`b288b225dcaf823392b6aeb87ce0ae67f068f4d5e495997cbbd746ef4c34f34e`.
It contains every exact ref/object identity and canonical graph relation, all
worktree states and fully enumerated paths, and every stash, tag, bundle head,
bundle digest, snapshot ref, and turn-diff ref. Cached remote refs are labeled
cached; no network synchronization occurred.

Counts at the census boundary:

| Category | Count |
| --- | ---: |
| worktrees | 20 |
| local branches | 23 |
| cached remote-tracking refs | 115 |
| tags | 21 |
| snapshot refs | 74 |
| turn-diff refs | 30 |
| stashes | 3 |
| untracked bundles | 41 |

### Worktrees

The dirt column is staged/unstaged/untracked path count. Every path is listed
in the census appendix.

| Worktree | Branch/detached HEAD | Exact HEAD | Dirt | Recommendation |
| --- | --- | --- | ---: | --- |
| `/Users/gwilson/GitHub/citlali` | `gw_dev` | `ffc6b9070f4744f9778f3db71cdc468846d1da89` | 0/0/0 | retain Legacy Citlali comparator worktree |
| `/private/tmp/citlali-contracts-consolidated` | `codex/scientific-contract-library` | `54475956f6aefb839d43b2f0fb019a142cb64310` | 56/0/0 | retain active; unresolved divergence; do not touch |
| `/private/tmp/citlali-timestream-successor-governance` | `codex/timestream-successor-governance` | `06a3ade51c1b3f38887295433d913811bf25cd14` | 0/0/0 | integrate after owner approval; later close while preserving ref |
| `/private/tmp/citlali-wp7-d2-producer-source-capture` | `codex/wp7-d2-producer-source-capture` | `34f609e4b1dc9a04f8157063c7a1662b707d96a7` | 0/0/0 | preserve failed-source/review evidence |
| `/private/tmp/citlali-wp7-g4-replay-001` | `codex/wp7-g4-replay-001` | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | 0/0/0 | later close; exact duplicate ref candidate for later local deletion |
| `/private/tmp/citlali-wp7-g4-replay-002a` | `codex/wp7-g4-replay-002a` | `2f1d836c1db122d22015853582133abf3611bc30` | 0/0/0 | retain until D1 and closure evidence are integrated/preserved |
| `/private/tmp/citlali-wp7-governance-reconciliation` | `codex/wp7-governance-reconciliation` | `f6c9033f80810da255a9bfa987e0fba8a082b785` | 0/0/0 | later close; branch tip already canonical ancestor |
| `/private/tmp/citlali-wp7-network-timed-rtc-repair` | `codex/wp7-rtc-fixed-decimation-authority` | `49fe73e757daa1885cd23127e8441cba47e648d2` | 0/1/3 | preserve dirty divergent evidence; do not touch |
| `/Users/gwilson/.codex/worktrees/0473/citlali-refactor` | detached | `abb33fdb9e45352190d2e55592cc5eba967993f2` | 0/0/0 | later close; exact commit proven reachable from `49fe73e…` |
| `/Users/gwilson/.codex/worktrees/166d/citlali-refactor` | detached | `37c2adfa84762cb8cef5dc66d5b1fbc6753331f6` | 0/0/0 | later close; exact commit proven reachable from `49fe73e…` |
| `/Users/gwilson/.codex/worktrees/4c31/citlali-refactor` | `codex/sci-fruit-v0.1-stage-a` | `fb704457bf237503127ffa9c8ed29b7b0041f101` | 0/0/0 | retain active contract-authoring lane; moving during census |
| `/Users/gwilson/.codex/worktrees/6130/citlali-refactor` | detached | `93de2cd9ca37f3740ceab98bf994ed684e9281ee` | 0/0/0 | later close; exact commit proven reachable from `49fe73e…` |
| `/Users/gwilson/.codex/worktrees/9448/citlali-refactor` | `codex/sci-noi-v0.1-stage-b` | `f28d7a2617160febca85c1c40e6f7ba7494e266e` | 0/0/0 | retain contract-authoring/evidence lane |
| `/Users/gwilson/.codex/worktrees/c3e1/citlali-refactor` | detached | `b0e5dde2ac532a7a36e141bf22c7560e0fbbc8a1` | 0/0/0 | later close; exact commit proven reachable from `49fe73e…` |
| `/Users/gwilson/.codex/worktrees/dc94/citlali-refactor` | `codex/sci-flt-v0.1-stage-a` | `cd55752e716051383da54356833ef0fac20b083a` | 0/0/0 | retain contract-authoring/evidence lane |
| `/Users/gwilson/.codex/worktrees/f797/citlali-refactor` | `codex/sci-flt-fixed-v0.1-stage-b` | `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` | 0/0/0 | retain contract-authoring/evidence lane |
| `/Users/gwilson/.codex/worktrees/fab4/citlali-refactor` | `codex/sci-flt-inf-stage-a` | `bbbef9fe2c111c8c1ddeec059027626e652c9b79` | 0/0/0 | retain; local is two commits ahead of cached upstream |
| `/Users/gwilson/.codex/worktrees/fddf/citlali-refactor` | `codex/refactor-mainline` | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | 0/0/0 | retain canonical worktree |
| `/Users/gwilson/GitHub/citlali-perf` | `codex/perf-map-accumulation-noise-lifecycle` | `d01ffa5981551345eeb1c765c24125200b896847` | 0/0/1 | retain owner evidence; do not touch untracked file |
| `/Users/gwilson/GitHub/citlali-refactor` | `codex/sci-align-001-lissajous-timestream-fit` | `353b11887ff04dfd7bca12915917495f81a587fa` | 0/0/464 | retain owner evidence; all 464 paths and 41 bundles enumerated; do not touch |

All four detached candidate-close worktrees are clean, and each exact HEAD is
an ancestor of the preserved divergent ref `49fe73e…`; removing those
worktrees later would not delete their commits. No removal is authorized now.

### Local branches

`Canonical only/ref only` is the exact commit-count result for
`canonical...branch`; full graphs remain in Git and the census records the
merge bases.

| Local branch | Exact tip | Canonical only/ref only | Role and recommendation |
| --- | --- | ---: | --- |
| `codex/refactor-mainline` | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | 0/0 | retain canonical |
| `codex/timestream-successor-governance` | `06a3ade51c1b3f38887295433d913811bf25cd14` | 0/2 | integrate; temporary candidate locator |
| `codex/wp7-g4-replay-002a` | `2f1d836c1db122d22015853582133abf3611bc30` | 0/2 | integrate `d7d19bc…`; preserve `2f1d836…` as historical closure |
| `codex/wp7-g4-replay-001` | `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | 0/0 | exact duplicate; later close and local-delete candidate |
| `codex/wp7-governance-reconciliation` | `f6c9033f80810da255a9bfa987e0fba8a082b785` | 2/0 | ancestor with zero unique commits; later close/local-delete candidate |
| `codex/wp7-d2-producer-source-capture` | `34f609e4b1dc9a04f8157063c7a1662b707d96a7` | 46/61 | preserve failed D2 source/review evidence; not a rename candidate |
| `codex/wp7-rtc-fixed-decimation-authority` | `49fe73e757daa1885cd23127e8441cba47e648d2` | 46/59 | preserve divergent forensic line; not canonical |
| `codex/scientific-contract-library` | `54475956f6aefb839d43b2f0fb019a142cb64310` | 129/118 | active/staged; upstream +1/-20; unresolved and owner-controlled |
| `codex/sci-fruit-v0.1-stage-a` | `fb704457bf237503127ffa9c8ed29b7b0041f101` | 129/174 | active contract lane; preserve; no cached upstream |
| `codex/sci-noi-v0.1-stage-b` | `f28d7a2617160febca85c1c40e6f7ba7494e266e` | 129/164 | contract evidence; exact cached remote twin; retain |
| `codex/sci-flt-v0.1-stage-a` | `cd55752e716051383da54356833ef0fac20b083a` | 129/163 | contract evidence; no cached upstream; retain |
| `codex/sci-flt-fixed-v0.1-stage-b` | `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` | 129/169 | contract evidence; exact cached remote twin; retain |
| `codex/sci-flt-inf-stage-a` | `bbbef9fe2c111c8c1ddeec059027626e652c9b79` | 129/177 | active/evidence; cached upstream +2; retain pending contract-owner push |
| `codex/sci-align-001-lissajous-timestream-fit` | `353b11887ff04dfd7bca12915917495f81a587fa` | 135/77 | exact cached remote twin, but 464 untracked paths; preserve |
| `codex/sci-cal-001-atmosphere-operator` | `7156881bd1a47e8cece97b8c541a013c93ac03e1` | 135/23 | exact cached remote twin; preserve historical contract evidence |
| `codex/coherent-iq-sidecar-validation` | `9e739db6d17410497aceaf33a4fd22e9d62ff793` | 135/6 | cached upstream +2; outside reset integration; unresolved |
| `codex/convolve-contract-implementation` | `2d1fbb4897e1fa416a587847895266abecb43100` | 135/6 | exact cached remote twin; preserve historical evidence |
| `codex/implement-sci-map-003-slice-001` | `d0512362ba8666621f7622062fa51542ae76bfa1` | 121/1 | exact cached remote twin; preserve historical evidence |
| `codex/perf-map-accumulation-noise-lifecycle` | `d01ffa5981551345eeb1c765c24125200b896847` | 3400/29 | exact cached remote twin plus untracked local evidence; preserve |
| `codex/repair-cal-apt-repair-001` | `0ff2de27992d862db3d606bd9d57a4dc176da6e8` | 121/9 | exact cached remote twin; preserve historical evidence |
| `codex/repair-rtc-learned-sampling-stage-a-successor-3` | `1d5beec4ae49c2bda131228486d67a21263a464e` | 121/4 | exact cached remote twin; preserve historical evidence |
| `codex/unity-update-hardening` | `1919626c0ee006bd52bf546f9216512a71c9982a` | 135/2 | local-only build/operations evidence; unresolved; retain |
| `gw_dev` | `ffc6b9070f4744f9778f3db71cdc468846d1da89` | 3345/6 | Legacy Citlali preserved head; exact cached remote twin; retain |

The governance branch's two unique commits are `bf649411…` and `06a3ade51…`.
The paired branch's two unique commits are `d7d19bc90…` and `2f1d836c…`.
The failed-source branch ends with `916fa076…` and `34f609e4…`. All exact
unique histories for large divergent branches remain preserved in Git; the
census gives their exact counts and merge bases rather than duplicating
thousands of commit records.

### Remote-tracking refs, tags, special refs, stashes, and bundles

- All 115 remote-tracking refs are cached observations, not live GitHub state.
  Thirty-three are canonical ancestors and 82 diverge. Each exact name, tip,
  subject, merge base, and ahead/behind count is in the census. Default
  disposition for all not specifically named in this package: **preserve on
  remote; unresolved; no deletion**.
- `origin/codex/refactor-mainline` is the known stale cached exception addressed
  by the exact push packet.
- `origin/codex/build-adaptation` is cached at
  `4f9c7e55ab41c2d7ca88ffee86bc9e60e8c5727b`, exactly five commits behind the
  accepted tagged build input `d9843e85…`. Recommend an ordinary fast-forward
  push from the accepted exact commit; never force.
- `origin/codex/scientific-contract-library` is not a safe synchronization
  source for the staged local checkout because the branch is ahead 1/behind 20.
- The scientific source and closure are already contained in multiple cached
  remote scientific-contract refs. Do not manufacture application ancestry
  from that fact.
- All 21 tags are enumerated. Retain release and historical science/audit tags.
  Retain and synchronize the meaningful annotated build tag
  `spack-build-adaptation-d9843e85` (tag object
  `428865859f330768d4f712077341e1a98c644795`, peeled commit `d9843e85…`). The
  three `wp7-*` tags remain historical aliases, not new durable naming.
- Preserve all 74 snapshot refs and all 30 turn-diff refs. Turn-diff refs point
  to non-commit evidence objects; ancestry-based deletion logic does not apply.
- Preserve all three stashes:
  - `stash@{0}` `75cc620256a460d6002ad367cecf17f7df1dbd1b` — perf `naive_mm` work;
  - `stash@{1}` `1c98f93376776b4dd51494f230c559cf1e031a5b` — `gw_dev` RTC work; and
  - `stash@{2}` `b27271fc76ffea063341465a5a62112167f50e01` — pre-reset JINC/pointing WIP.
- Preserve all 41 bundles. They are fully enumerated by absolute path, byte
  size, SHA-256, validity, and exact bundle heads in the census. All are
  currently untracked in the 464-path owner checkout and MUST NOT be cleaned.

### Branches observed moving during this reset

- This reset created `codex/timestream-successor-governance` from
  `4dc7844e…` to `bf649411…` and then `06a3ade51…` in its dedicated worktree.
- M0 closed `codex/wp7-g4-replay-002a` at implementation `d7d19bc90…` and
  closure `2f1d836c…`.
- The independent SCI-FRUIT authoring worktree was initially observed at
  `d44863f3b18e6d351dbd5fd1ba95b03045375cd7` with 7 modified and 1 untracked
  paths, then became clean at `a210bfb4f9c480e325ed9f8a1fc82449127c55e0`,
  and advanced again through `431f02dc751a1931e645ebe32c98c4ee13f8a9bc`
  to census tip `fb704457bf237503127ffa9c8ed29b7b0041f101`. It is an active independent
  contract lane and was not touched by this reset.

## 6. Exact proposed integration sequence — not executed

### Phase A — verify and preserve existing objects

1. Confirm the canonical worktree is clean at exact `4dc7844e…`; governance is
   clean at `06a3ade51…`; paired evidence is clean at `2f1d836c…`; and source
   capture is clean at `34f609e4…`.
2. Recheck branch/ref tips immediately before mutation. A mismatch stops the
   sequence; it is not silently rebased or overwritten.
3. Run the owner-approved ordinary push packet in section 7. Normal push
   rejection is a stop signal. No force option is permitted.

### Phase B — incorporate governance and make it effective

4. Fast-forward `codex/refactor-mainline` from exact `4dc7844e…` to exact
   governance candidate `06a3ade51…`. The known graph permits a fast-forward;
   no merge commit, rebase, or cherry-pick is required.
5. On that canonical ancestry, create the smallest governance-effectiveness
   record in `doc/INTEGRATION_LEDGER.md` naming full candidate SHA
   `06a3ade51…` and all four normative document digests. Reconcile current
   status/naming routing only where required. This future commit has no exact
   SHA yet and therefore is not in the push packet.
6. Validate and independently review the exact effectiveness-record SHA. Only
   then describe governance as effective.

### Phase C — integrate the paired-D1 implementation without importing stale control text

7. From the accepted post-governance canonical SHA, create a bounded
   integration candidate and merge exact implementation commit
   `d7d19bc90d7c994fa767ec2a9fd35e4d8599f032` so that the accepted
   implementation remains an ancestor. Do not merge `2f1d836…` wholesale:
   its `AGENTS.md`, status, ledger, and historical program text predate the new
   governance and naming disposition.
8. Create one canonical reconciliation record that cites `2f1d836…` as the
   immutable closure evidence, uses **Timestream Successor** as the enduring
   name, preserves `WP7-REPLAY-002A` only as historical alias, and states all
   non-authorizations.
9. Re-run the focused product tests, full config preflight, baseline/build
   tools, validation ledgers, relevant CTest surface, and exact identity gates.
   Report the actual local environment. Obtain the owner/Unity disposition for
   a representative Spack compile/CTest gate before treating C++ integration
   as representative; no observational reduction is implied by a route-inert
   carrier.
10. Obtain fresh-context exact-SHA review with separate science, architecture,
    and repository verdicts. Present the exact integration candidate SHA for
    owner acceptance before moving canonical.
11. Fast-forward canonical only to that accepted exact integration candidate.
    Generate a second exact, ordinary canonical push command after the SHA
    exists. Do not use a placeholder as an executable push command.

### Phase D — worktree closure and local cleanup

12. After canonical and GitHub verification, close the clean replay-001 and
    old-governance worktrees. Their tips are respectively equal to or ancestors
    of canonical.
13. The four clean detached worktrees may then close because their exact HEADs
    are reachable from retained `49fe73e…`.
14. Close the governance worktree after `06a3ade51…` is canonical and remotely
    preserved. Keep the ref through verification; local branch deletion is a
    separate exact owner-approved action.
15. Retain the paired and source-capture worktrees/refs until the implementation
    and both closure/rejection records are remotely preserved and their final
    dispositions verified.
16. Do not close or clean any dirty or moving owner/contract worktree. Do not
    drop stashes, remove bundles, or delete special evidence refs.

### Phase E — local and remote deletion

- No local branch deletion occurs during integration. Later candidates, after
  exact post-operation reachability proof: `codex/wp7-g4-replay-001` and
  `codex/wp7-governance-reconciliation`; later, the temporary governance branch.
- No remote deletion is proposed. Historical remote sprawl requires its own
  fresh live-remote audit, preservation proof, and exact owner-approved packet.

## 7. Exact GitHub push packet — existing objects only; not executed

All commands are ordinary non-force pushes. They use immutable local object
IDs as sources so the proposed content is exact. Cached remote observations
must be rechecked at execution time; any non-fast-forward rejection stops the
operation.

| Purpose | Local source / expected SHA | Destination | Exact proposed command | Expected post-push state |
| --- | --- | --- | --- | --- |
| synchronize accepted current canonical before new integration | commit `4dc7844e59e03cf2d18a9262fe5b75d3ff078681` | `refs/heads/codex/refactor-mainline` | `git push origin 4dc7844e59e03cf2d18a9262fe5b75d3ff078681:refs/heads/codex/refactor-mainline` | remote canonical fast-forwards from cached `cb3d568c…` to exact `4dc7844e…`, or rejects safely if live state differs |
| preserve reviewed governance candidate | commit `06a3ade51c1b3f38887295433d913811bf25cd14` | `refs/heads/codex/timestream-successor-governance` | `git push origin 06a3ade51c1b3f38887295433d913811bf25cd14:refs/heads/codex/timestream-successor-governance` | remote candidate ref points exactly to reviewed SHA |
| preserve paired-D1 implementation and closure | commit `2f1d836c1db122d22015853582133abf3611bc30` | `refs/heads/codex/wp7-g4-replay-002a` | `git push origin 2f1d836c1db122d22015853582133abf3611bc30:refs/heads/codex/wp7-g4-replay-002a` | remote historical alias contains exact `d7d19bc…` implementation and `2f1d836…` closure |
| preserve rejected D2 capture/review | commit `34f609e4b1dc9a04f8157063c7a1662b707d96a7` | `refs/heads/codex/wp7-d2-producer-source-capture` | `git push origin 34f609e4b1dc9a04f8157063c7a1662b707d96a7:refs/heads/codex/wp7-d2-producer-source-capture` | remote evidence ref contains `916fa076…` capture and `34f609e4…` review |
| synchronize accepted Spack build lane | commit `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046` | `refs/heads/codex/build-adaptation` | `git push origin d9843e85ed87ba9ac8c42d8cc21f997dacbe1046:refs/heads/codex/build-adaptation` | remote branch ordinarily fast-forwards five commits from cached `4f9c7e55…` to exact accepted input |
| preserve meaningful annotated build tag | local tag ref; tag object `428865859f330768d4f712077341e1a98c644795`, peeled `d9843e85…` | `refs/tags/spack-build-adaptation-d9843e85` | `git push origin refs/tags/spack-build-adaptation-d9843e85:refs/tags/spack-build-adaptation-d9843e85` | remote tag exists with the exact annotated tag object, or the push rejects rather than overwrites |

This packet intentionally excludes:

- the future governance-effectiveness commit;
- the future governance-plus-D1 merge/reconciliation candidate;
- SCI-FRUIT, because it advanced during the census and has no accepted remote
  synchronization disposition;
- `codex/scientific-contract-library`, because it is staged-dirty and
  diverged from cached upstream;
- the two-commit-ahead SCI-FLT-INF and coherent-IQ branches, because their
  contract/validation owners, not this reset, must authorize those pushes; and
- all deletion refspecs.

After each future commit exists and is accepted, its canonical push must be
presented with its literal full SHA in the same form. No command containing a
symbolic placeholder is approved for execution.

## 8. Remote-deletion proposals

**None.**

No live remote census was performed because fetch and remote mutation were
forbidden. Cached ref state is insufficient evidence for deletion, and several
remote branches carry scientific, audit, repair, or historical evidence not
fully represented by canonical ancestry. No `git push --delete`, deletion
refspec, force-push, or rename command is included.

## 9. Recommended surviving topology

Long-lived authority/evidence shape:

- one canonical application integration branch:
  `codex/refactor-mainline`;
- during implementation, one bounded Timestream Successor spine worktree and
  at most one explicitly approved module-probe worktree;
- the separately governed Spack build-adaptation ref/tag;
- only genuinely active scientific-contract authoring branches/worktrees,
  governed per package rather than counted as application WIP;
- Legacy Citlali and meaningful accepted milestone tags as comparators; and
- evidence branches, stashes, bundles, snapshot refs, and turn-diff refs kept
  until an explicit archival/deletion review proves preservation.

Temporary replay/governance locators should not survive as pseudo-mainlines.
After integration and remote verification, their worktrees may close and exact
duplicate/ancestor local refs may be deleted under a separate packet. Historical
names remain in commits, records, and any retained archive/evidence refs.

## 10. Proposed first post-closure work order

**Timestream Successor Native Paired Ingress Adapter**

Purpose: populate the accepted paired-native D1 carrier at the authoritative
Tune/readout boundary through one concrete typed adapter, making the existing
carrier a real producer boundary before any D2 observer is reconsidered.

| Work-order field | Proposal |
| --- | --- |
| WIP slot | one integration/spine increment; no module probe |
| Exact base | the future owner-accepted canonical SHA containing effective governance and integrated `d7d19bc…`; must be filled literally before start |
| Proposed branch locator | `codex/timestream-successor-native-paired-ingress` — proposal, not previously accepted durable authority |
| Scientific authority | WP-7.1 source `170ecea9…`, closure `20ba6ae5…`, accepted native timing/identity and x/r coordinate authority |
| Architectural boundary | concrete producer adapter -> immutable paired D1 product; orchestration owns invocation; no `Engine` growth |
| Included | exact Tune/readout producer identity; native network/run/occurrence and detector association; x/r payload and coordinate-local validity/cause transfer; fail-closed construction; focused adapter/product tests; bounded allocation evidence |
| Excluded | D2 observation, prefilter/residual, common-grid ALIGN, AST changes, RTC/PTC/CAL/MAP wiring, filter/factor/downsampling, persistent output, route activation, generic producer framework |
| Stop conditions | source boundary cannot supply required paired identity without a new scientific choice; cross-stage reach-through; need to mutate `Engine`; hidden common-grid or D2 semantics; prerequisite contract contradiction |
| Gates | isolated header/adapter compilation, focused identity/validity/failure/memory tests, full local regression gates, actual-environment reporting, independent exact-SHA review, representative Spack compile/CTest disposition |

This is recommended ahead of the rejected D2 seam because D2 should observe an
accepted, populated D1 producer product—not recreate or reach around it. The
work order remains unstarted until the owner approves the exact post-integration
base, bounded branch/worktree, and scope.

## Operation confirmation

No fetch, pull, push, merge, rebase, cherry-pick into canonical, canonical ref
movement, branch rename, tag creation, deletion, prune, clean, stash drop, or
worktree removal was performed. No dirty owner checkout was modified. No new
scientific or application implementation increment was begun after M0.
