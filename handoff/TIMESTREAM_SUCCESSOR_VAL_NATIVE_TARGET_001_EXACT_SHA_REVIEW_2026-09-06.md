**INDEPENDENT EXACT-SHA REVIEW — 2026-09-06**

**Verdict: PASS WITH RECORDED LIMITATIONS.** No actionable scientific/behavioral, architectural/ownership, or repository/evidence findings. Fresh owner-run Unity GCC13/Spack validation remains required before source admission.

This review began from fresh context and remained read-only. I made no source, evidence-file, branch, or ref changes; performed no rebuild, push, cleanup, or Unity access; and created no additional agent.

**Exact subject and ancestry**

- Candidate: `37e6fb6ce50acba644f04b6b6b546762bf07f384`
- Tree: `32dea6ffe86b5855ad75a21cd56e6817559fb9f9`
- Sole parent and approved implementation base: `080df2a1431487e8cabc255beb9ddc59c0721b59`
- Branch: `codex/timestream-successor-val-native-target-001`
- Worktree: `/private/tmp/citlali-timestream-successor-val-native-target-001`
- Initial and final candidate worktree state: clean.

I independently verified those identities and confirmed local canonical remains at the approved parent. Accepted governance, its effectiveness record, pushed D2 closure `5244e04638db5aa92180feab52acd782229cfa32`, original D2 implementation `bb060947175523d6fc6a777ae4ad4606693e9e5f`, repair `7d57a5acf893ae0f34c3639499484b3f5976768a`, and admission `aeea0eef04ec70d8142c8a20fd7b09dfb64725ed` remain ancestors. No history was replaced.

The owner's acceptance of the prerequisite and approval of this source unit are durably recorded in the candidate's status, ledger, and work order. Live GitHub state was not queried during this review; the recorded live/local difference is the expected accepted prerequisite documentation commit.

**Authority sources read**

I read applicable `AGENTS.md`; the TolTEC context skill and its repository, authority, and boundary routing; all three effective governance documents; current status and integration-ledger entries; the candidate work order and accepted prerequisite; relevant architecture and scientific conventions; the WP-7.1 authority router; SCI-VAL's r0.3 freeze and relevant normative requirements; and accepted D2 owner corrections with their unchanged implementation boundaries.

Subject-specific authority remains:

- Sequencing and bounded authorization: current status/ledger and accepted prerequisite Section 5.
- Governance: accepted `06a3ade51c1b3f38887295433d913811bf25cd14`, effective through `77507836325eff9f469062d5884481ea37599594`.
- Science: WP-7.1 source `170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, canonical authority router, and frozen SCI-VAL Core.
- Architecture: VAL owns fact-target and snapshot mechanics; actual producers own numerical subjects and facts; named consumers retain scientific-use policy.
- Integration, publication, activation, and production: separate owner dispositions.

I independently verified the three normative governance documents remain byte-identical to the accepted governance commit, and checked the recorded frozen-Core and prerequisite-authority digests. Historical status text does not override the current work order.

**Changed-path inventory**

Exactly six paths differ from the sole parent:

| Path | Candidate SHA-256 |
|---|---|
| [doc/INTEGRATION_LEDGER.md](/private/tmp/citlali-timestream-successor-val-native-target-001/doc/INTEGRATION_LEDGER.md) | `7b5f6c4519adbc59807a4383486360df126df39f9ae3775545b06822abb08aa5` |
| [doc/REFACTOR_STATUS.md](/private/tmp/citlali-timestream-successor-val-native-target-001/doc/REFACTOR_STATUS.md) | `401cefbe0819e771e75e2c5430fd8e52ac8ccdad55f739972b3f132f82242551` |
| [handoff/TIMESTREAM_SUCCESSOR_VAL_NATIVE_TARGET_001_2026-09-06.md](/private/tmp/citlali-timestream-successor-val-native-target-001/handoff/TIMESTREAM_SUCCESSOR_VAL_NATIVE_TARGET_001_2026-09-06.md) | `b64946e5a131f9fdb27d77fc364356a3dbbb1e93d2f465b936ecc7bc2e90a327` |
| [include/citlali/core/pipeline/timestream_val_state.h](/private/tmp/citlali-timestream-successor-val-native-target-001/include/citlali/core/pipeline/timestream_val_state.h) | `a3b3d93d133cd39bbd852902d6d6a0fc714e706186ecc8351dd147cf43ab1a4c` |
| [tests/test_timestream_val_state.cpp](/private/tmp/citlali-timestream-successor-val-native-target-001/tests/test_timestream_val_state.cpp) | `e843199ffbcc0e2ad86e186ba124bf328bc361d193d352690e7de8cdea7fa29f` |
| [tests/timestream_val_state_header.cpp](/private/tmp/citlali-timestream-successor-val-native-target-001/tests/timestream_val_state_header.cpp) | `ff9b72020ad4a8555026c27e5ffc1b94537e19bbc4af176ce7f3e2f4eef84667` |

All six digests were independently computed. Prior status/ledger bodies are preserved, and `git diff --check HEAD^ HEAD` passes.

**Scientific and behavioral conformance — findings: NONE; disposition: PASS within scope**

Source inspection establishes that qualified targets require an exact Paired-D1 parent, matching network, explicit valid detector address, and valid x/r coordinate. Factory checks reject null or foreign realizations, invalid roles/producers, nonexistent networks, and incomplete coordinate targets. The builder revalidates targets before accepting findings, including rejection of moved-from targets.

The variant keeps existing unqualified findings and qualified targets in distinct key domains. There is no lookup fallback, coordinate broadcast, pair-wide promotion, or reinterpretation of existing fact codes. Fact-author identity remains separate from the numerical subject's producer identity.

Target comparison and equality agree: address, coordinate, and retained descriptor instance determine target identity. The address timestamp is finite by its existing construction contract; descriptor ordering uses `std::less` on retained instances. Equal descriptive product/realization numbers in separately created descriptors do not alias. No persisted pointer order or reproducibility across fresh allocations is claimed.

Fact states and causes remain opaque. Missing findings and nonfinite numerical payloads trigger no inferred validity, permission, cause, or coercion. Existing immutable overlay and snapshot-branch behavior is preserved. D2's exact realization-time/publication snapshot-handle requirement is unchanged.

These conclusions establish the approved mechanical extension only. They do not establish a full SCI-VAL evaluator, registered PSD-use policy, or producer/consumer adoption.

**Architectural conformity and ownership — findings: NONE; disposition: PASS**

`ValNativeRealization` is noncopyable and nonmovable, created through its checked factory, and exposed through a const shared handle. It stores scalar identity and one exact Paired-D1 handle. It has no numerical payload or snapshot reference.

Ownership is acyclic: snapshots retain findings and their prior snapshot; qualified findings retain descriptors; descriptors retain Paired-D1. No descriptor points back to a snapshot. Comparison and lookup introduce no allocations, global counters, registry, or cross-stage reach-through.

The descriptor's support is explicitly the complete unchanged native network. It does not claim transformed, selected, resampled, or common-grid support. Actual producers must supply and retain the exact descriptor when this interface is adopted later.

Memory evidence is proportionately limited. On the disclosed Mac realization, `ValAddress` remains 56 bytes, the descriptor is 48 bytes, the key grows from 80 to 112 bytes, and each finding grows from 88 to 120 bytes. The additional 32 bytes also apply to unqualified findings because of variant storage. Reference counts describe handles in a delta, not unique descriptors. No RSS, allocator-overhead, or throughput claim is established.

**Repository and evidence hygiene — findings: NONE; disposition: PASS with recorded limitations**

The three authorized source/test paths contain the complete implementation change. D2, Paired-D1, identity-route implementation, CMake registration, configurations, scientific authority, registries, and numerical tooling are unchanged. No producer, policy evaluator, persistence, filter/factor/downsampling operation, substantive RTC/PTC/AST/CAL/MAP work, activation, or cleanup entered the candidate.

I independently verified all 1,600 recorded precommit input digests against the committed candidate. Precommit executions retain their actual dirty-source checkpoint identity; they are not relabeled as executions at the later commit.

**Gate reproduction and environment**

Reviewer-reproduced checks:

- Exact commit/tree/parent, changed scope and digests, ancestry, authority preservation, prior status preservation, and clean state: PASS.
- Existing dedicated VAL executable: 10/10 PASS after inspecting its compile command and checking source-input equivalence.
- Test inventory: independently verified 900 unique registered tests, 899 runnable, 10 VAL tests, eight D2 tests, and sole disabled `citlali::MapFitterLifecycle.ExactProductSequence`.
- Final gate-log digest bindings: PASS.
- Cached Kids revision, three-file dirty inventory, and all three dirty-file digests: unchanged.
- Both owner-run scripts: `bash -n` PASS.
- Bundle verification, advertised candidate ref, prerequisite ancestry, package digests, and upload checksums: PASS.

The dedicated focused executable reviewed and run has SHA-256 `5366e8b1bd165232bd61fbab0a9f4795818f5c9d11eecfb81c6ccf0377f41e4a`.

Writer executions inspected and bound to evidence:

- Final clean-candidate `check`: 899/899 runnable CTests PASS; test runtime 28.08 seconds.
- Final clean-candidate focused successor selection: 50/50 PASS.
- Final CLI revision: matches `37e6fb6ce`, with clean Citlali source.
- Precommit CLI/safety/identity/D2 build, full config preflight, and 207 baseline-tool tests: PASS, with input equivalence verified.
- Both validation ledgers and scientific-contract layout: PASS.

Actual local environment is AppleClang 21 on arm64 macOS, CMake 4.3.0, Homebrew/cached dependencies, and the required local Python 3.13.2 environment. Kids remains at `04088da182622c3e879f04314974a7c0d60ee2d6` with its three disclosed pre-existing dirty headers. This is supplemental evidence.

The initial baseline attempt's 11 missing-executable assertion failures remain preserved; all 207 tests subsequently passed without source/test repair. The task-local overbroad dirty-token assertion and its corrected Citlali-specific binding check are separately recorded.

I independently reproduced the unmodified SCI-VAL verifier's exit 1 and identical `MAP reserved profile must remain explicitly unbound` failure on both this candidate and the clean accepted canonical base. This remains a historical verifier limitation, not PASS, a candidate regression, or certification of its unexecuted later checks. No unexpected application error-level output was found in the final check/focused logs.

**Owner-run Unity package**

The reviewed runner is:

`/private/tmp/citlali-val-native-target-evidence-2026-09-06/run_unity_gate_37e6fb6ce.sbatch`

SHA-256: `a0bfbefeea38e994053b3b3d34f9b1f5b44d2da50994141185364ecbc9c8d3f6`.

Its differences from the independently reviewed D2 v2 runner are bounded to unit/source/tree identity, corresponding locations, test inventory, and expanded focused selection. The unchanged CMake identity-suite registration includes VAL. The runner checks source/tree/cleanliness, pinned dependencies before and after execution, CLI revision, full `check`, and the expected inventory.

Additional reviewed artifact identities:

- Bundle: `26b89b0b2cb8b42e52c18f50d94db09b118d05b885957ab4e26abb04e6e5d4a8`
- Submission script: `8adeab228d3e946d70570cbeaf8bafb4842c281637cf5013aaf0c7014b5868e4`
- Upload checksum file: `d7ff60097b05d23b1dc15f016803832eac244758443f1cc8ea2e731fba363bd8`
- Owner commands: `8294161390b602cbd8a438bc5cb33c58caf9706f1f8fb80ebbb94455298d12b7`

The submission boundary verifies uploads, imports the bundle into `FETCH_HEAD`, creates or verifies the exact detached source worktree, preserves named branches, verifies the dependency symlink and cleanliness, and refuses an existing job receipt. Transfer commands use `unity_toltec` and the required testing directory. Interactive strict options are confined to a subshell. No Unity operation has been executed by this review.

**Acceptance boundary**

The candidate and owner-run package pass independent review with the limitations above. Fresh Unity GCC13/Spack execution at this changed source is still unperformed and required before source admission; earlier D2 job `64018093` does not cover it. No scientific reduction is triggered by this unactivated identity-only change. Canonical admission, GitHub publication, activation, production, cleanup, and subsequent numerical work remain separate owner dispositions.
