# Exact duplicate-only cleanup proposal — 2026-09-26

**Status: ready for owner review; no deletion executed.**

The owner requested cleanup. This first batch contains 7,039 regular files,
12,804,081,472 logical bytes (11.925 GiB), all freshly SHA256-matched to retained
copies. Physical reclaimed space may differ because of APFS allocation/snapshots.

## Exact operations

Delete only these three directory trees, after rechecking the complete pair
ledgers and confirming that no process is consuming them:

1. `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-ready-ae4842e-20260919/unity-packet/assets`
   — 38 files, 6,809,587,215 bytes. Preserve the identical tree at
   `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-checkpoint-98380090e-20260919/unity-packet/assets`.
2. `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-ready-ae4842e-20260919/unity-packet/local-reference`
   — 5,985 files, 4,603,256,446 bytes. Preserve the identical tree at
   `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-checkpoint-98380090e-20260919/unity-packet/local-reference`.
3. `/private/tmp/point-fruit-one-cycle-20260923`
   — 1,016 files, 1,391,237,811 bytes. Preserve the identical tree at
   `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-fruit-one-cycle-20260923/cycle`.

`CLEANUP_PROPOSAL.json` is the machine-readable allowlist.
`PACKET_DUPLICATE_VERIFICATION.json.gz` contains all 6,023 packet file pairs and
their fresh digests; `SCRATCH_DUPLICATE_VERIFICATION.json` contains the 1,016
scratch pairs. The packet pairs have distinct inode/device identities, and no
multiple hard links were observed. No symlinks or new files may be swept into
the operation; reject changes to the verified inventory before deleting.

## Why these copies can be retired

The September 20 status record supersedes the old raw-upload packet with the
resident-input submission design and explicitly keeps the older packet as
historical evidence. Its scientific payload remains byte-identically preserved
in the earlier checkpoint. No new science calculation is needed to restore it.

The ready run's **local-evidence** tree is retained: later first-forward MAP
records directly bind it. The checkpoint tree is retained, including the
effective configuration directly required by POINT forward-map input records.
A bounded reference search found no current external input binding to the two
ready packet subtrees proposed here; that search is not a proof about every
possible owner script. The exact old layout can be reconstructed from the
preserved counterparts and the pair ledger.

Keep every packet-specific script, review, preparation record, manifest and
the containing run directory. After approved removal, add a conspicuous
retirement notice explaining that this historical packet is no longer
self-contained and its old submission instructions must not be used as-is.
Preserve old manifests unchanged. A check of the old complete packet layout
will report missing copied payloads until restored; do not present it as still
passing its original full-payload seal.

The scratch tree duplicates its durable first-cycle archive exactly, including
all metadata and the manifest. The durable copy and subsequent POINT runs stay
in place. Recorded old scratch locators remain historical locators; use the
retained archive or restore the old layout if a historical script requires it.

## Execution and closure after approval

Reverify both sides of every pair and the full relative file inventory; stop on
any missing/different/new file, symlink or active consumer. Delete enumerated
targets only, then remove their empty directories. Retain the pair ledgers and
write a deletion/retirement receipt. Measure free space before and after. Recheck
the current checkpoint ledgers/native binary and preserve the unfinished CUT3
patch. No parent seal is rewritten and no scientific processing is launched.

All other development products, old FRUIT products, raw originals, build trees,
Git worktrees/refs, main-checkout changes and Unity/remote resources are outside
this deletion allowlist. Additional reclamation needs its own concrete evidence
and disposition; it is not included by implication.

## Approval basis

Applicable governance: `doc/governance/ENGINEERING_GOVERNANCE.md`,
“Evidence-preserving cleanup,” and the Tier 2 review requirement in
`doc/governance/REVIEW_AND_CONFORMANCE.md`. The general cleanup request supplies
scope; exact approval of these three targets is still pending. This is the
concrete operation for that final approval, not a request to restart planning.
