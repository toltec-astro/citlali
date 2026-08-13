# INTEG-CONVERGENCE-001 Preparation Record — 2026-08-13

## Status and authority boundary

This is a preparation record, not an integration or acceptance record. It
defines exact convergence identity, lane-handoff requirements, overlap
ownership, combined gates, and a future one-lane-at-a-time protocol. It does
not merge, reconstruct, repair, rebase, squash, accept, promote, push, or move
any application ref.

Fact labels in this record have fixed meanings:

- **Observed** — read directly from a named local Git object/ref or repository
  authority at the recorded snapshot.
- **Derived** — follows mechanically from observed identities, ancestry, diffs,
  or an explicit owner ruling.
- **Provisional** — recommended sequencing that is not integration authority.
- **Deferred** — requires a later exact-SHA candidate, independent disposition,
  repository owner, scientific owner, or human-operated evidence.

Explicit owner direction outranks repository prose. TolProj may receive only a
separately reviewed, bounded owner-repository repair. TolTECA v2 and v3 are
strictly read-only for this work; their confirmed APT selection/transport
defects remain named external blockers. No other repository may compensate by
weakening identity, admission, provenance, or fail-closed contracts.

## Exact convergence and lane inventory

The Phase-2 start snapshot was `2026-08-13T15:19:51Z`; the local end-tip check
used here was `2026-08-13T17:07:18Z`. No live network query was made at either
snapshot. `origin/...` below means the locally stored remote-tracking ref.

| Authority | Start tip | End tip | Tree / disposition |
|---|---|---|---|
| Authorized convergence base and current checkout | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | same | tree `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`; parent `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4` |
| Local `codex/refactor-mainline` | `9aae0e669384c5c0c0dda93debc194d6b8dac787` | same | exact ancestor, 0 ahead/14 behind the authorized base; stale pointer only |
| `origin/codex/refactor-mainline` | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | same | owner-selected canonical base fact |
| CAL `origin/codex/repair-sci-cal-001-successor-8` | `7cdc4152931f3d42f11f26ef62649fb755e1b553` | same | tree `ebc32f152b9400e00c994c0a78076953145d83a2`; merge base `46ad23888...`; 8 ahead/0 behind |
| CAL independent audit | `9437d1a4d18f310cbef7c6542c1be46cd893098c` | same | tree `48cff72928e9cb5c122c302bbd0427c4e45f19e0`; audit-only direct child of CAL candidate |
| ALIGN foundation `origin/codex/repair-sci-align-native-cohorts` | `c87d5693dbcf185b2e76d15b41ac55ff3d71f1ef` | same | tree `d308ed9eb2bf94be45a72a296f618e64999e7256`; merge base `46ad23888...`; 7 ahead/0 behind, of which six commits are excluded MAP-002 ancestry |
| Cross-repository APT-order audit | `716ac36d06a4b7ff05573a7a24c2973f40d194ab` | same | tree `307a6a16d261b1a3577cc58eba01033551741518`; audit-only evidence |

**Observed:** canonical ancestry contains accepted MAP-001 application tip
`af0c849ce59a5f80e5efc8db435bb6662863052f`, MAP integration record
`d5015fe716971bf8ea617e8a187311bf5af05185`, accepted NOI-002 application tip
`5b29e13548a6fec884c67b192dec20c92f0bbb62`, NOI integration record
`4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, then documentation confirmation
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`. It is a single-parent
application line and contains no audit or coordination merge.

**Derived:** the local mainline discrepancy is a pointer lag, not competing
history. A later owner could update it with
`git branch -f codex/refactor-mainline 46ad23888a40f5102cdfd50c06e49a549bdf8a20`
only after verifying no worktree has that branch checked out. Consequence: the
local name moves forward 14 commits without changing any object, remote ref, or
worktree. Rollback would be
`git branch -f codex/refactor-mainline 9aae0e669384c5c0c0dda93debc194d6b8dac787`.
Neither command is authorized or executed here.

### CAL ancestry and disposition

The eight CAL commits are, in order:

1. `7894346a91fa78ceb2a8b3d625335f466e5e1756`
2. `8b1534807f5abe4d80be2fbd45ed3838ed351509`
3. `3af6faf996fa002b2647adca8f33991002d49ff1`
4. `693f1b107855e3ae9b36617323ca14aac868f304`
5. `5dfc414a13fe69e6b063608906d87e3b30491ec7`
6. `211e2f16f6354609de3ce6c6ee526d8aa4c6c59c`
7. `9037314fd84241fa535c486d4ffb28966bb0394d`
8. `7cdc4152931f3d42f11f26ef62649fb755e1b553`

They mix production application, tests, validation tooling, generated inputs,
generated evidence, and documentation. A future packet must classify each
commit and path rather than treating the entire branch as one semantic kind.
Audit `9437d1a4...` returns this exact candidate for bounded repair: native
CFITSIO close failures can falsely publish/complete required FITS products, and
the Python provenance validator mishandles valid physical APT permutations and
typed/optional identity contradictions. No accepted run or promoted successor
epoch closes those findings. CAL is not handoff-ready.

### ALIGN ancestry contamination and reconstructable input

ALIGN currently carries this excluded MAP-002 sequence:

`854a04b124e083e64706fd043e105182fee568af` →
`6c74d214a49af5520f02ca071b5d513b14b58b03` →
`02f443bfeb85f3b2e12a6eff60f3a77e77fe342c` →
`86f1582fad92bdd0453bca3264ce39478b00c227`, followed by merge
`214484e21a00e0c11d86c2b0460ec98b969469f2` and documentation child
`2917c5210cb131f0e7f952d9b04295e87e22718d`. None is an accepted convergence
prerequisite.

The sole ALIGN foundation delta is
`2917c5210cb131f0e7f952d9b04295e87e22718d..c87d5693dbcf185b2e76d15b41ac55ff3d71f1ef`:
two new pipeline
headers, three new focused tests, and `tests/CMakeLists.txt`. The five new blobs
are prospective provenance inputs only. The CMake hunk is anchored on a
MAP-002-only test target and does not apply cleanly to `46ad23888...`; it must
be independently composed and reviewed on the authorized base. The active
production-consumer bridge is uncommitted working evidence in another
worktree, not a ref-bound candidate. It is excluded from all conclusions.
ALIGN has no independent exact-SHA disposition and is not handoff-ready.

## Overlap and ownership

Git overlap is only the first screen. A scientific or lifecycle conflict may
exist without a shared path and must never be resolved by `ours`, `theirs`, or
textual convenience.

| Boundary | Path/interface overlap | Classification and priority | Architectural owner | Scientific owner(s) | Future stage owner / evidence |
|---|---|---|---|---|---|
| Test registration | `tests/CMakeLists.txt` | textual/additive; reconstruct ALIGN registration after accepted first-lane tree | build/test owner | CAL and ALIGN test owners | second import stage; clean-base discovery of every focused test |
| RTC lifecycle | CAL applied-response state and ALIGN native cohort/alignment state in RTC/observation flow | coupled semantic/lifecycle; CAL truthfulness first, then coexistence | RTC and observation/scan lifecycle owners | CAL calibration owner and ALIGN timestream owner | combined stage; reset, multi-observation, stale-state and both-focus regression tests |
| PTC typed identity | ALIGN consumer needs typed detector identity; current working evidence casts legacy double `calib.apt["uid"]` | scientific identity conflict; no lossy round trip | PTC/calibration-provenance owner | APT identity owner plus CAL and ALIGN owners | combined stage; consume exact admitted typed member or stop for authority |
| Requested/effective/observation/realized state | calibration selection, cohort mapping, flags, application and output provenance | coupled; one-way authority only | pipeline/provenance owner | CAL and ALIGN owners | combined stage; explicit state transitions and truthful products |
| APT cross-repository chain | Citlali, `toltec_beammap`, TolAPT, TolProj, TolTECA | coupled scientific/product contract, not a Git merge conflict | each producer/consumer repository owner | APT identity/product owners | A--G interface gates plus later sample/library phases |

TolProj defects remain `repairable_only_in_separately_reviewed_repository_lane`.
TolTECA defects remain `blocked_deferred_at_tolteca`; no complete production
APT claim is possible while its configured-path transport and exactly-one
selection are unresolved.

## Frozen handoff contract

A conforming lane packet uses
[`FROZEN_LANE_HANDOFF_PACKET_TEMPLATE.md`](FROZEN_LANE_HANDOFF_PACKET_TEMPLATE.md),
the machine schema
[`frozen_lane_handoff_packet.schema.json`](../validation/frozen_lane_handoff_packet.schema.json),
and the read-only checker
[`validate_frozen_lane_handoff.py`](../tools/baseline/validate_frozen_lane_handoff.py).

It binds one full implementation SHA/tree/base, start and end snapshots,
parents, merge base, divergence, standard binary-patch and name-status
digests, exact changed paths/blobs, application ancestry, excluded side
history, independent review, findings, scientific-change/epoch state, every
gate result, local evidence, human-mediated Unity evidence, external blockers,
and candidate-bound approvals. Audit, failed/abandoned repair, generated
evidence, coordination, and contaminating history remain distinct from
application history. A later packet-container commit is never the tested
implementation identity.

Readiness is derived. A packet may be structurally valid and ready for a
bounded lane handoff while later APT interface or library rows remain
conditioned, but it must keep cross-repository, real-chain, scientific,
production, and refactor-library claims false.

## Mechanically checkable combined gate matrix

Every instantiated row has these required fields: stable ID/version; domain
and scope; required flag; timing; blocking stage; exact candidate SHA, tree,
and base; hashed inputs; argv or human/review procedure; hashed outputs;
mechanical criteria; result; omission authority/reason; execution,
architectural, scientific and evidence owners; evidence reference; claim
constraints; start/end times; exit/log/output/skip counts; APT interface
contract; and post-software APT-phase contract. Passing evidence from another
SHA or tree cannot satisfy a row.

### Core and lane gates

| Gate(s) | Exact criterion | Timing / blocking stage | Owners |
|---|---|---|---|
| `AUTH-BASE-001`, `IDENT-FREEZE-001` | base is exact `46ad23888...`; candidate/ref snapshot, parents, tree, merge base, divergence, embedded version, patch/name-status digests and clean freeze agree | every freeze / all later stages | coordinator and application-mainline owner |
| `AUTH-PREREQ-001`, `AUTH-ANCESTRY-001` | every prerequisite classified; exact application whitelist; no audit, failed repair, coordination, generated evidence, or excluded MAP-002 ancestry | order selection and each import / import | coordinator, integration owner, independent auditor |
| `AUTH-DISPOSITION-001` | independent exact-SHA/tree disposition; all findings closed, accepted, or explicitly later-conditioned | lane freeze / lane handoff | independent auditor and lane science owner |
| `CAL-FITS-CLOSE-001` | injected native close failure prevents required FITS publication/completion, preserves old final, and reports failure for all affected Pointing/Science paths | CAL freeze, post-import, final / CAL handoff | output/atomic-publication and CAL/product owners |
| `CAL-APT-LINEAGE-001` | valid same-network permutation passes; UID type and retained optional-field contradictions fail; exact recursive type/value closure remains | CAL freeze, post-import, final / CAL handoff | calibration/provenance and APT identity owners |
| `CAL-FOCUS-001` | CAL atmosphere, admission, lifecycle, publication, product and validator suites pass without required skip | CAL freeze, intermediate, final / freeze | CAL implementation/science owners |
| `ALIGN-FOUNDATION-001` | clean-base target discovers all foundation tests; headers compile alone; no MAP-002 symbol or CMake anchor | ALIGN freeze, post-import, final / ALIGN handoff | pipeline, ALIGN science, build/test owners |
| `ALIGN-CONSUMER-001`, `ALIGN-PCA-001`, `ALIGN-LIFECYCLE-001` | accepted production RTC/PTC path invokes bridge; cohort identities/flags and finite placeholders stay truthful; scatter is atomic; revisions reset by scope | ALIGN freeze, post-import, final / ALIGN handoff | RTC/PTC/lifecycle and ALIGN owners |
| `OVERLAP-RTC-001`, `OVERLAP-APT-001`, `OVERLAP-PROVENANCE-001` | CAL and ALIGN states coexist; typed admitted identity has no row/double shortcut; requested/effective/observation/realized provenance is one-way and truthful | after second import / combined freeze | named architectural owners and both science owners |
| `COMBINED-REGRESSION-001` | both lane focus suites and all overlap counterexamples pass on one combined SHA/tree | after second import and final / combined freeze | integration, lane and validation owners |
| `ARCH-BOUNDARY-001` | dependency direction, lifecycle ownership, isolated headers, no broad `Engine` state, no library `exit()` | each lane and final / freeze | architecture and subsystem owners |
| `BUILD-CLI-001`, `TEST-CTEST-001` | supported clean build and full runnable CTest pass; disabled tests enumerated and proven unrelated | intermediate and final / freeze | build/test owners |
| `CONFIG-PREFLIGHT-001` | required config preflight passes; requested, effective, observation-resolved and realized state remain distinct | intermediate and final / freeze | config/provenance owner |
| `VALIDATION-TOOLS-001` | baseline tests, accepted-run/profile/science-change/product/readiness validators pass | intermediate and final / freeze | validation owner |
| `PRODUCT-CONTRACT-001`, `REQUIRED-OUTPUTS-001` | requested products present, disabled absent, cardinality/provenance valid; injected required failure reaches CLI | intermediate and final / freeze | product/provenance owners |
| `PROVENANCE-APT-LINEAGE-001` | exact artifact, component, membership, mapping, transformation and application identities join; no basename/row inference | lane and final / affected handoff | APT/provenance owners |
| `FAILURE-LOG-001` | zero unexpected error-or-higher records and zero unexplained/missing required outputs | every run and aggregate / acceptance | session/output and validation owners |
| `SCIENCE-EPOCH-001` | reviewed `none`, or accepted intended-change IDs with predecessor comparison and accepted successor epoch/profile; no loosened profile | lane and final / handoff | affected science and validation owners |
| `MODE-ROUTING-001` | touched paths/interfaces select every required mode; any non-applicability has owner proof | lane and combined freeze / validation | validation, subsystem and science owners |
| `MODE-POINT-001`, `MODE-OOF-001`, `MODE-SCIENCE-001`, `MODE-BEAMMAP-001` | human-operated run on exact SHA/config/input/profile with required products, no skipped comparison or unexpected errors | human Unity / combined acceptance | authorized human and mode owners |
| `LOCAL-SAME-SHA-001`, `UNITY-SAME-SHA-001` | all local and human Unity rows bind one exact candidate SHA/tree/version/dependency environment | final local and human Unity / combined acceptance | coordinator, human operator, independent auditor |
| `EXTERNAL-APT-001` | TolAPT/`toltec_beammap` owner dependencies, separate TolProj lane, read-only TolTECA blocker and prohibited E2E claim are explicit | every packet / production end-to-end | repository owners and coordinator |
| `PACKET-CONFORMANCE-001` | schema/checker pass, no unresolved placeholder, exact ancestry/scope/evidence/approval accounting | pre-acceptance / integration authorization | checker, coordinator, independent auditor |

### APT interfaces A--G

The exact producer/consumer sequence is:

`Raw/KMP → Citlali Beammap axis → Citlali Beammap product → toltec_beammap →
TolAPT and/or TolProj matching/application → TolProj selected refactor package →
TolTECA v2 transport → Citlali admission/application`.

The seven required recorded gate IDs and detailed contracts are defined in the
packet template. Each mode route records whether TolAPT is an offline
downstream/package step, an already-produced input lineage, or not in the
path. No route may imply inline TolAPT execution. Overall conformance is false
until every applicable route passes with joined exact digests.

### Post-software sample and dedicated library gates

| Gate | Criterion | Timing / blocker |
|---|---|---|
| `APT-LIB-SOFTWARE-FREEZE-001` | clean, independently accepted exact SHA/tree for every participating repository, hashed as one revision set | after software acceptance / refactor APT generation |
| `APT-LIB-COHORT-MANIFEST-001` | owner-approved observations, exact BEAM config and raw-data manifest | post-software / refactor APT generation |
| `APT-SAMPLE-NEW-CONTRACT-FIXTURES-001` | bounded human-produced sample with ≥2 networks, ≥2 artifact scopes, complete/permutation/rejection cases, and joined component/artifact/membership/mapping/transformation/application digests | before full campaign / refactor APT generation |
| `APT-LIB-BEAM-CAMPAIGN-001` | human-run new-contract BEAM campaign from frozen software/config/raw identities | later human Unity / refactor APT generation |
| `APT-LIB-CANDIDATE-CONFORMANCE-001` | each Beammap/APT/fit-QC candidate gets producer, consumer and science disposition | curation / refactor APT generation |
| `APT-LIB-IMMUTABLE-GENERATION-001` | new immutable/versioned dedicated refactor root/generation; never an in-place legacy change | curation / refactor APT generation |
| `APT-LIB-COMPLETENESS-QUARANTINE-001` | every candidate is accepted or quarantined/rejected with exact reason; zero unexplained candidate | curation / refactor APT generation |
| `APT-LIB-PROVENANCE-001` | exact artifact/component/membership/mapping/transformation/application manifests join | curation / refactor APT generation |
| `APT-LIB-NO-MIXED-LINEAGE-001` | legacy-input count and mixed-generation count are zero | curation / refactor activation |
| `APT-LIB-HISTORICAL-IMMUTABILITY-001` | legacy libraries/runs remain immutable, pinned comparison evidence only | curation / refactor activation |
| `APT-LIB-SHADOW-COMPARISON-001` | optional old/new scientific comparison with explicit cross-generation labels; no equivalence requirement | comparison evidence / never a compatibility blocker |
| `APT-LIB-ACTIVATION-ROLLBACK-001` | atomic selected-generation reference change and exact rollback; roots immutable | owner activation / refactor activation |
| `APT-LIB-SELECTED-CONTRACT-001` | all selected artifacts were created and accepted under frozen A--G | activation / refactor activation |
| `APT-REFACTOR-REDUCTIONS-001` | regenerate Pointing, applicable OOF, and Science from scratch using only selected refactor generation | post-activation / production end-to-end |

All these operational rows are currently blocked. This task runs no reductions
and mutates no APT library.

## Provisional integration order and future protocol

**Provisional:** use an independently accepted CAL repair first and an
independently accepted clean-base ALIGN reconstruction second. CAL's current
blocking repairs do not depend on ALIGN. ALIGN's production consumer crosses
the unresolved typed APT/PTC seam; current evidence does not permit
ALIGN-first. If an accepted CAL repair still does not provide that seam, order
becomes unresolved and the coordinator must stop for the APT, PTC, CAL and
ALIGN scientific owners.

1. Freeze accepted CAL and ALIGN packets and an exact prerequisite graph.
   Classify every dependency as base-present, lane-local, cross-lane,
   separately promoted, or externally blocked.
2. Create a new clean integration branch at exact `46ad23888...`. Preserve
   rollback point **R0** at that SHA/tree.
3. Import only the first lane's independently accepted application ancestry.
   Exclude packet-container, audit, failed/abandoned repair, generated-evidence,
   coordination, and contaminating history. A separately promoted prerequisite
   must already have its own authority; MAP-002 cannot enter by implication.
4. Resolve no scientific overlap automatically. Run first-lane focus,
   architecture, build, full CTest, config, validation, product, log, mode and
   epoch gates. Freeze clean intermediate SHA/tree **R1/I** and its packet.
5. Import only the second lane's accepted clean application ancestry. The
   second stage owns combined RTC/APT edits under the named architectural and
   both scientific owners.
6. Rerun both lane focus suites immediately, every overlap counterexample, and
   all broad local gates. Any edit creates a new candidate. Freeze clean final
   combined SHA/tree **R2/C** only after all local rows pass.
7. An authorized human builds and runs the same **R2/C** on Unity for every
   routed mode without source/config/dependency drift. Returned evidence is
   human-mediated and hash-bound. Codex never accesses Unity.
8. Produce the conformance packet without changing **R2/C**. An independent
   reviewer accepts or rejects that exact combined SHA/tree before any
   mainline movement. Push and mainline advancement remain later owner actions.

If the first lane passes at **I** but regresses at **C**, preserve the failed
tip, return by a new branch from **I**, and assign the failure to the second or
overlap stage. Rerun second-lane focus, every first-lane regression gate,
overlaps, broad local gates, and the complete Unity matrix. If it reproduces
at **I**, invalidate the intermediate packet and return to **R0** for first-lane
repair and re-acceptance. Never reset or rewrite a shared ref. A scientific
difference stops until an intended-change entry and successor epoch are
approved.

## Future post-software refactor APT protocol

After software/interface conformance is independently frozen at exact SHAs:

1. Meet the bounded new-contract sample milestone; this is independent of the
   full campaign and is the first possible real-chain integration fixture.
2. Approve an observation cohort and BEAM campaign manifest.
3. Have an authorized human run exact-identity BEAM reductions on Unity and
   return scoped evidence.
4. Apply A--G producer/consumer and scientific acceptance to every candidate.
5. Curate accepted candidates into a new immutable/versioned
   `citlali-refactor` library root/generation. The TolProj `_apt_library` is the
   publication/selection layer, never the workspace that generated candidates.
6. Keep legacy libraries and reductions immutable and addressable as historical
   comparison evidence. Do not admit them as refactor inputs and do not require
   legacy selection equivalence. Any comparison is explicitly cross-generation.
7. Prove no mixed-generation selection, complete candidate/quarantine
   accounting, exact provenance, and that every selected artifact was produced
   under the accepted contract.
8. Atomically change only an explicit selected-generation reference after owner
   approval. Preserve immutable old/new roots and an exact rollback reference.
9. Regenerate Pointing, OOF where applicable, and Science from scratch using
   only that selected refactor generation and exact frozen software/config/raw
   identities. Historical accepted runs are not rewritten or promoted.

Failure before activation quarantines the candidate/generation and leaves the
selected reference unchanged. Failure after activation switches the explicit
reference back using the rollback manifest, retains both immutable roots, and
invalidates all downstream refactor reductions from the failed generation.

## Guarantees not enforced by ordinary CI

Ordinary CI does not presently prove application-only ancestry; exclude
MAP-002/audit history; bind local and human Unity evidence to one SHA; inject
native CFITSIO close failures; test physical APT permutation/type identity;
require scientific-owner overlap disposition; route every touched mode; bind
generated evidence to commands and hashes; prove zero unexpected errors in
human reductions; enforce A--G cross-repository joins; suppress claims while
TolTECA is blocked; create bounded real sample artifacts; or govern immutable
dedicated refactor-library generation, selection and rollback. The packet
checker closes only the documented preparation subset and never performs a
scientific acceptance.

## Current stop conditions and deferred owners

- CAL stays returned for bounded repair and a new independent exact-SHA audit.
- ALIGN needs a clean-base reconstruction, frozen production consumer, typed
  APT/PTC authority, and independent exact-SHA disposition.
- Any MAP-002 prerequisite needs independent MAP-002 promotion or explicit
  bounded prerequisite authority; it cannot arrive through ALIGN ancestry.
- The APT scientific owner must resolve the typed detector-member seam.
- TolProj repair requires its owner and a separate reviewed lane.
- TolTECA selection/transport requires its owner or an owner-approved external
  replacement contract; this repository remains read-only.
- New-contract sample generation, full BEAM campaign, library creation,
  activation, and downstream regeneration require human operational authority
  and are not authorized now.
- An ambiguous base, moved/unbound candidate, unknown scientific owner,
  unexplained required-output failure, unexpected error log, mixed SHA, mixed
  library generation, or failed required gate stops advancement.

At this preparation stage, neither active lane is ready for handoff and no
production end-to-end APT claim is supported.
