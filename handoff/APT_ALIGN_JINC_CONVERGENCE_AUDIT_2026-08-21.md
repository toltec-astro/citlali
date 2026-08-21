# APT / ALIGN / JINC Convergence Audit — 2026-08-21

## Verdict

The owner-run Unity-tested JINC repair line is the correct convergence spine,
but it is not yet a complete application-integration candidate.

The exact tested implementation is
`e77460cffad49387795009539d6abc7e370e8b58`, tree
`94ad86ed66429993297a26b0fd2f3009afac9509`. The local packet-container tip
`91f42ccdc8ce9a4e6811f2f03857180d50d21345`, tree
`43254ff37def9a923a58537fc8b586061113ea09`, adds only the accepted targeted
Unity validation record. It is 17 commits ahead and zero behind canonical
remote base `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

No commit from the 76-commit historical SCI-ALIGN research line is eligible
for direct replay. Ten commits touch application sources, but each is either
an unaccepted old-base repair, a diagnostic-only mode, or a narrow product
repair whose recorded gate is incomplete. The other 66 commits are diagnostic
tools, generated evidence, campaign/runbook mechanics, or an unrelated
handoff.

Three application changes outside that 76-commit line require explicit
disposition:

1. `e6c8d126157674a9990abc8d1e96ce2dd69f9374` is an independently re-audited
   JINC ownership-preflight repair. It is not in the convergence tree and is
   the first reconstruction candidate.
2. `fd3627fc70060a78e65b47b3f798825fd3238514` and
   `9d9d55a54fb16cd3964af79522d0d37de253dce2` are the missing native-cohort
   consumers implied by the subject of merge `a71fce419`. They are absent from
   that merge, patch-unique, based on canonical APT v1 rather than compact v2,
   and have no independent exact-SHA acceptance record. They must not be
   cherry-picked. Their intended consumer behavior requires a clean compact-v2
   reconstruction and fresh review.
3. The older PTC metadata fixes `7fc59344c4e9f861aaa534315e1644415f380b1e`
   and `5c6309125fef15f7c98e70a62b591c663944b130` are real bounded engineering
   candidates, but the latter still records a required fresh-root Unity replay.
   They remain a separate reconstruction lane.

This audit prepares dispositions only. It does not replay application code,
alter the tested JINC implementation identity, move a remote ref, push, or
claim general JINC, ALIGN, CAL, APT, or production readiness.

## Frozen identities

Snapshot time: `2026-08-21T20:17:50Z`.

| Role | Ref / commit | Tree | Ancestry disposition |
| --- | --- | --- | --- |
| Canonical remote application base | `origin/codex/refactor-mainline` at `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408` | Authorized base; local branch name remains a stale pointer and is not moved here |
| Unity-tested implementation | `origin/codex/jinc-redu04-validated` at `e77460cffad49387795009539d6abc7e370e8b58` | `94ad86ed66429993297a26b0fd2f3009afac9509` | Application authority for the targeted JINC repair |
| Convergence packet container | `codex/converge-apt-align-jinc` at `91f42ccdc8ce9a4e6811f2f03857180d50d21345` | `43254ff37def9a923a58537fc8b586061113ea09` | Tested implementation plus documentation-only validation record |
| Historical split-direction tip | `92cfa670a33255250895d68aaf26e8b01aa057bd` | `908825af674e3ea19c03cbb54441680dd4d6ad12` | Diagnostic research authority only |
| Historical Lissajous tip | `2094d730280555d49d61000eeb1e0fb42dc6595b` | `0f999db014102f0c080cfca2e42ea5ca7ec66695` | Diagnostic research authority only |

The historical Lissajous line merges at stale application base
`9aae0e669384c5c0c0dda93debc194d6b8dac787`; relative to the convergence
packet it is 76 commits ahead and 31 behind. Its standard binary range-patch
SHA-256 is
`85b19e551f0704c1fda0a19671056ddf538dd99cbadad5ca7c3928112f5bbb0c`;
its name-status SHA-256 is
`31dcadf1a3825a56a789e27c84760712e20c38b4ac4ea27fb9ad2b4adc14ec35`.
`git cherry` reports all 76 patches as absent from the convergence line.

The Unity-tested `e77460cf` tree is byte-identical to local repair tree
`59a142b334a4d7882f85f031ba090cdd74171839`. The differently named local
cherry-picks remain equivalence backups and are not replay inputs.

## Misleading merge subject and missing native consumers

Merge `a71fce4198769a88c6c0c85fc035ec3496ccbe03` is titled
“Integrate native per-network ALIGN consumers with compact APT v2,” but its
parents are compact-v2 tip `89f7a91e362cd22b66929ec70b51fb69b8cae8fd`
and foundation-only commit
`c87d5693dbcf185b2e76d15b41ac55ff3d71f1ef`. The actual consumer commits are
later children of `c87d5693d`:

| Commit | Tree | Disposition |
| --- | --- | --- |
| `fd3627fc70060a78e65b47b3f798825fd3238514` | `e45286cc15b3d448fb19dcfffa0d2a90bdb23edb` | Missing application consumer implementation; reconstruct, do not cherry-pick |
| `9d9d55a54fb16cd3964af79522d0d37de253dce2` | `9e87f2f733adefe6fca6d07ce0791e3ab7e430ed` | Required Beammap lineage correction for that implementation; reconstruct with the consumer contract |
| `5ef2d011660d7d7d3e17e4a30874003f713746b5` | `86b18d783126627161617fcf016ff998203e132e` | Patch-equivalent to Unity cherry-pick `952b7f766`; already present, do not replay |

`fd3627fc7` changes 48 paths and adds 20,689 lines, including canonical APT v1
implementations duplicated from its old lineage. A three-way comparison with
the compact-v2 parent has conflicts or independent authority edits in
`doc/REFACTOR_STATUS.md`, `doc/RETAINED_DEBT.md`,
`doc/SCIENTIFIC_CONVENTIONS.md`, `include/citlali/core/engine/calib.h`,
`src/citlali/core/engine/calib.cpp`, `tests/CMakeLists.txt`,
`tools/baseline/test_validate_product_contract.py`, and
`validation/product_contracts.json`, plus competing additions of canonical APT
headers. Preserving the commit would reintroduce the superseded v1 authority
and defeat the compact-v2 boundary.

The correct salvage unit is therefore the intended native cohort consumer
contract, not either historical patch. A reconstructed candidate must consume
the admitted compact-v2 detector relation, preserve Beammap's raw-only
producer lineage, and rerun native scatter, partial-cohort, PCA-placeholder,
production-consumer, lifecycle, APT-identity, and affected-mode gates.

## Other omitted application candidates

| Commit / lane | Evidence-backed status | Convergence disposition |
| --- | --- | --- |
| `e6c8d126157674a9990abc8d1e96ce2dd69f9374` — JINC parallel map ownership | Independent re-audit `f541d81a266fce0f7baed58e9ec275dadba260ee` reports conformant, complete, `existing_use_only`, verdict `accept`, pending coordinator acceptance | Reconstruct first against current JINC; retain its eager pre-mutation ownership and destination preflight and focused tests; do not cherry-pick the old `jinc_mm.h` hunk |
| `110d36fe432e6475599607ea12bd60d14b64ff94` — omit observation KMP fields from baseline APT v1 | Bounded Stage-D v1 repair | Superseded for new issuance by compact v2 and its later real-KMP compatibility commits; historical evidence only |
| `0ff2de27992d862db3d606bd9d57a4dc176da6e8` — CAL APT identity/FITS publication | Ninth CAL successor on a separate eight-commit CAL ancestry; no later independent exact-SHA disposition found in this audit | Keep in the CAL lane; never import through APT/ALIGN/JINC convergence |
| `08f0a6733d1cb523ae78ccf9348ac6832b834e52` and `1d682ee78ca5d85bd30673783a978265bd01048c` | Acquisition-event semantic audit and producer request on a two-commit side fork after `92cfa670` | Audit/handoff evidence only; no application replay |
| `5bcaa3700`, `aeeac7f36`, `9e234eada` | Frozen SCI-ALIGN independent core/audit identity | Authority evidence only; remain outside application ancestry |

## Complete disposition of the 76-commit SCI-ALIGN line

Every exact commit in
`9aae0e669384c5c0c0dda93debc194d6b8dac787..2094d730280555d49d61000eeb1e0fb42dc6595b`
is accounted for below. “Retain” means keep reachable on the historical branch,
not replay into the convergence ancestry.

### Phase-0 and bounded old-base repair

| Commits | Classification | Disposition |
| --- | --- | --- |
| `53c7154a3`, `5a0d64b8f`, `bfffe0e60` | Preregistration and generated phase-0 evidence | Retain as historical evidence |
| `4eabf604a`, `0280cc079`, `cb56f238e`, `c77105b9b` | Application-bearing bounded timing/Hold/gap/provenance repair | No direct replay. The branch status records application tip `c77105b9b` as not accepted, merged, pushed, or production-authorized; reconstruct only after a new owner-scoped ALIGN contract decision |
| `dbdad8899`, `52f72288a`, `1bc91045f`, `b9787afa6` | Phase-one fixture and runner binding | Retain with the old-base validation package |
| `3e12c0dcb` | Local phase-one evidence/status | Retain as historical evidence |

### Left/right and sample-lineage diagnostics

| Commits | Classification | Disposition |
| --- | --- | --- |
| `ccf705c67`, `a89b665ad`, `323d83773`, `00b114d65`, `c468ffc58` | Left/right Beammap diagnostic tooling and evidence | Retain on the diagnostic branch; no application replay |
| `e1b29ab6d`, `5176a46bc`, `a2b37924d` | Frozen sample-lineage diagnostic and results | Retain as diagnostic tooling/evidence |

### Corpus, replay, and transport mechanics

| Commits | Classification | Disposition |
| --- | --- | --- |
| `6776931e7`, `8e4fcae2f` | Reusable 3C273 corpus diagnostic tooling plus frozen evidence | Retain; extract later only under a separately reviewed validation-tool change |
| `5f3627990`, `33dbb84b3`, `078510900`, `0b6e3a824`, `e33166452` | Unity runbook and bundle-transfer mechanics | Exclude from application ancestry |
| `c8ec1049b`, `49ff7bb54`, `d9a2dadce`, `afcb896e5`, `537e0ddfb`, `d37aa216a`, `0da62bba8` | Replay/aggregation/campaign tooling | Retain on the diagnostic branch |
| `c9598fc10` | Application/config support-cropping diagnostic | Diagnostic-only numerical/config surface; no direct replay or default change |
| `9e0290722`, `083e60c24`, `b510a7e4d`, `a77ba45d2`, `ac1409d1f` | Counter, failure, retry, and heldout-block diagnostic handling | Retain with the corpus tooling |

### Split-direction Beammap and PTC product investigation

| Commits | Classification | Disposition |
| --- | --- | --- |
| `c492ca761`, `9730f0e2e`, `d5078d2c6` | Optional split-direction Beammap runtime/config/product implementation | Diagnostic-only mode; do not merge while repairing mature mapmaking contracts. Preserve as a separate candidate requiring a current-base rebuild and independent disposition if revived |
| `5e91bf0ea`, `5cf106826`, `c79dc3d09` | Split-direction rendering/frame/kernel diagnostics | Retain as diagnostic tooling/evidence |
| `7fc59344c` | Early creation of mutable Beammap PTC `FRUITLOOPS_ITER` metadata | Real narrow product-lifecycle repair candidate; reconstruct separately on the current tree and rerun output-schema/full-product gates |
| `5c6309125` | Variable-length PTC scan-bound metadata repair | Real narrow repair candidate, but recorded status still requires a fresh-root Unity replay; reconstruct together with `7fc59344c`, not by historical cherry-pick |
| `fa1405d91`, `1f52b1b69`, `9ec62b2f9`, `2ee6d5116`, `9e617bef7`, `d16d0be73`, `6363d396f`, `f20280f5c`, `77c8a1a71`, `92cfa670a` | PTC sampling/join/audit tooling and same-T0 evidence | Retain on the diagnostic branch; no application replay |

### Lissajous and pointing-fit diagnostics

| Commits | Classification | Disposition |
| --- | --- | --- |
| `6ec08656f`, `90cc4caf9`, `74e9bf8b3`, `a5f40f718`, `9469cf611`, `14de70ffc`, `a7443d5e8`, `f80d3889e`, `e20ce4ffd`, `e5a309674`, `f00202daf`, `91d532070`, `174fb18fe`, `30de7d50e`, `9eab0f725`, `9ce828b77`, `7e5881087` | Diagnostic algorithms, tests, generated evidence, and campaign mechanics only; no production paths | Retain as the terminal SCI-ALIGN research lineage; no replay into application ancestry |
| `2094d7302` | Unrelated SCI-MAP-003 OOF handoff committed on the SCI-ALIGN branch | Exclude as branch contamination; the handoff remains reachable historically |

## Path overlap and conflict surface

The Unity-tested convergence range changes 83 paths. The 76-commit historical
line changes 450 paths. Only nine final-tree paths overlap:

- `doc/REFACTOR_STATUS.md`;
- `include/citlali/core/engine/calib.h`;
- `include/citlali/core/engine/detail/beammap_apt_table_output_impl.h`;
- `include/citlali/core/engine/detail/beammap_map_population_impl.h`;
- `include/citlali/core/engine/detail/beammap_timestream_pipeline_impl.h`;
- `include/citlali/core/pipeline/reduction_observation_pipeline.h`;
- `tests/CMakeLists.txt`;
- `tests/test_config_scaffold.cpp`; and
- `tools/config/audit_raw_timestream_execution_reads.py`.

The low textual overlap does not make the old line merge-safe. The native
consumer and old timing repairs cross detector identity, APT admission,
requested/effective/realized provenance, RTC/PTC lifecycle, telescope timing,
and map ownership without necessarily sharing paths. Those semantic interfaces
require reconstruction against the compact-v2 and accepted JINC authorities.

## Proposed integration order

1. Preserve rollback point `91f42ccdc` and keep the exact tested
   implementation `e77460cff` identifiable beneath its documentation child.
2. Reconstruct the accepted `e6c8d1261` ownership preflight and focused test
   matrix against current `jinc_mm.h`. This is contract hardening and should
   leave valid-path arithmetic and `redu04` results unchanged.
3. Run focused JINC ownership, all existing JINC contract, build, complete
   CTest, baseline-tool, ledger, and config-preflight gates on that exact
   candidate. A Unity rerun is required only if the reviewed mode-routing
   decision finds the eager preflight capable of changing the valid `redu04`
   path; otherwise retain the existing targeted science result and state the
   bounded non-applicability.
4. Freeze a separate compact-v2 native-consumer reconstruction plan from the
   behavior of `fd3627fc7` plus `9d9d55a54`. Do not copy their canonical APT
   v1 implementation or historical source state. Require independent review
   before implementation integration.
5. Keep the PTC metadata pair `7fc59344c`/`5c6309125` in its own repair lane;
   reconstruct and validate only if full PTC output is an active requirement.
6. Keep the old timing repair, support-cropping option, split-direction mode,
   and all research/campaign history outside application ancestry.
7. After every admitted reconstruction has an exact-SHA disposition and
   affected gates, prepare a final application-integration candidate for owner
   push toward `codex/refactor-mainline`. Do not move the stale local mainline
   pointer as part of this audit.

## Claim boundary

This record supports only the following claims:

- the targeted JINC working-support incident is repaired at exact Unity-tested
  implementation `e77460cff` for observation 148670;
- the current convergence ancestry is complete through the 17 recorded
  APT/MAP/ALIGN-foundation/JINC commits;
- omitted and historical commits are now enumerated and dispositioned; and
- three bounded reconstruction questions remain: JINC ownership, compact-v2
  native consumers, and optional full-PTC metadata.

It does not claim complete native ALIGN consumer integration, acceptance of the
old SCI-ALIGN timing repair, split-direction production support, full-product
JINC validation without the omitted FITS cubes, CAL closure, general APT
production admission, or production readiness.
