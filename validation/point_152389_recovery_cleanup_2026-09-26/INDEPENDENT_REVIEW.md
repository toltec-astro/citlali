# Independent cleanup review — 2026-09-26

Verdict: PASS WITH RECORDED LIMITATIONS for the exact proposed three-target cleanup package, pending the owner's exact-operation approval. No deletion, science, Git mutation, network access, or Unity access performed by this reviewer.

Reviewed source: 11b4d2c07a81cc5876bd3da833c4168731e34aea; tree 55f909c0e25ad08c31faec374fecdbca7978f69e; worktree /private/tmp/citlali-point-additive-20260923. Initial dirt was the point_corr_admission.py logging modification and untracked validation/point_152389_a2000_m1_cut3_2026-09-25. The concurrent writer's recovery evidence directory appeared during review and was not mistaken for committed source. This review binds the cleanup plan and evidence hashes below, not a future report commit.

Read authority: AGENTS.md; TolTEC context skill and routing maps; doc/governance/ENGINEERING_GOVERNANCE.md; TIMESTREAM_SUCCESSOR_GOVERNANCE.md; REVIEW_AND_CONFORMANCE.md; relevant REFACTOR_STATUS.md and INTEGRATION_LEDGER.md sections; handoff/TIMESTREAM_SUCCESSOR_RTC_PERFORMANCE_001_2026-09-18.md and TIMESTREAM_SUCCESSOR_PARTIAL_COMPLETION_001_2026-09-18.md; first-forward MAP manifests. Governance file hashes match ledger accepted identities 70769787..., 29fae6f7..., 691e6d62.... Tier2 evidence cleanup requires preservation and exact-operation owner authorization.

## Exact package

Base directory D = /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs

1. Target D/successor-full-network-ready-ae4842e-20260919/unity-packet/assets. Preserve D/successor-full-network-checkpoint-98380090e-20260919/unity-packet/assets. 38 regular files; 6,809,587,215 bytes.
2. Target D/successor-full-network-ready-ae4842e-20260919/unity-packet/local-reference. Preserve D/successor-full-network-checkpoint-98380090e-20260919/unity-packet/local-reference. 5,985 regular files; 4,603,256,446 bytes.
3. Target /private/tmp/point-fruit-one-cycle-20260923. Preserve D/successor-point-152389-fruit-one-cycle-20260923/cycle. 1,016 regular files; 1,391,237,811 bytes.

Total 7,039 files, 12,804,081,472 logical bytes (11.924730122 GiB). Zero target/retained same-inode/device pairs and zero target hardlink counts above one. APFS extent sharing, snapshots and storage accounting mean actual physical space recovered remains unmeasured.

Targets 1–2 were freshly SHA256 verified by this independent reviewer, all 6,023 pairs identical, matching complete relative file inventories. Exact ledger /private/tmp/cleanup-packet-pairs-20260926.json SHA256 719434e1a7f0552095b9631732da43eeaed108d7452cbb9c30274ffd82cd6786. Both target and preservation have no multiply linked files.

Target3 writer's full fresh-hash ledger was independently checked for internal consistency, all 1,016 equal pairs, byte total, path/stat existence, and inode distinction. Reviewer did not repeat those content hashes. Exact ledger validation/point_152389_recovery_cleanup_2026-09-26/SCRATCH_DUPLICATE_VERIFICATION.json SHA256 c441a2dc5b9eb7579bc322a027489aa0cf79bd159e6a6690fc02040ba560f883.

## Preservation and dependencies

Retain ALL ready-ae4842e packet-specific scripts, manifests, reviews and other metadata; retain ready/local-evidence and local-inputs. Retain entire checkpoint98380090e tree. Preserve packet-pair ledger and this review in durable recovery evidence before deleting targets. Do not alter old scientific manifests to pretend the historical original layout remains complete. Record packet retirement and a reconstruction map; restoration copies the retained counterpart subtrees into their original target paths and verifies every saved digest.

After cleanup the ready historical packet is deliberately NOT self-contained. Its historical PREPARATION.json still says ready_for_owner_submission=true; old README instructions must not be used on the retired layout. Keep an explicit visible retirement/restore note beside the historical packet or an equivalently discoverable recovery record. Owner approval must acknowledge this operational-layout consequence. This is not loss of scientific content: all exact bytes remain preserved at the recorded counterpart paths.

Source basis: doc/REFACTOR_STATUS.md:874–883 names the superseding 20260920 resident-raw packet and makes the old sealed packet historical-only. Lines969–972 distinguish the ready packet and explicitly preserve the earlier immutable checkpoint. The ready packet-specific metadata should remain as historical evidence rather than being replaced with the checkpoint metadata.

Direct dependencies prevent whole-run deletion: handoff/evidence/MAP_SUCCESSOR_FIRST_FORWARD_REAL_2026-09-22.json:194 and :213 bind ready/local-evidence network0 and network7 products; :232 binds array-lowpass/a2000-explicit/output. D/successor-point-152389-forward-map-20260922/inputs/inventory.json:9 and inputs/processing-request.json:79 bind checkpoint/unity-packet/assets/effective_config/citlali_o152390_0_2_c2.yaml. The proposed package leaves all those paths intact.

Searches over repository tools/validation/handoff/doc and bounded development-run input/request/completion/preparation manifests found no active external dependency on ready/unity-packet/assets or local-reference. This is a scoped search, not proof that no arbitrary external shell script mentions them. Historical packet scripts themselves expect the removed paths; that is the explicitly recorded retirement limitation.

## Six older runs

No entire older run is approved for deletion. Checkpoint and ready have explicit preservation obligations plus MAP/POINT dependencies. Array-lowpass a2000 products are MAP inputs and replay equality evidence. Observation-residency, RTC-performance and partial-completion retain measured baseline/candidate evidence, unique receipts, executables and scripts; age or later bitwise equality alone does not establish that these entire evidence trees are disposable. Logical sizes respectively: checkpoint28.3608GiB, ready24.1475GiB, residency9.5051GiB, RTC9.1426GiB, partial4.7386GiB, lowpass5.4641GiB.

A later bounded cleanup could target generated baseline__/candidate__*.yaml.json semantic-fingerprint expansions while retaining YAML sources and comparison results; current compare_successor_outputs.py:129–132 shows these are derived outputs. This is a separate candidate, not part of this approved review scope, because per-file regeneration and manifest-departure records are still needed.

## Three review dispositions

Scientific/behavioral: pass with limitations. No scientific change or calculation; all proposed discarded payload bytes have retained identical copies. Historical packet's self-contained execution layout is retired explicitly.

Architecture/ownership: pass. No source, contract, cross-stage, ownership or production changes.

Repository/evidence hygiene: pass with limitations, subject to durable exact ledgers, preserved packet-specific metadata, explicit retirement/reconstruction note and owner exact-operation approval. Preserve initial dirty work. Do not delete whole runs or other paths under this review. Before execution, recheck no writers/open jobs target the selected paths and verify the inventory still matches the sealed cleanup manifest.

No numerical tests, broad tests, or representative Spack/Unity runs were required or performed for this read-only duplicate audit. Hash comparison and inventory checks are the relevant gates. This does not authorize a new scientific run, integration, push, or production change.
