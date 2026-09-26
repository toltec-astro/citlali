# Approved duplicate cleanup completed — 2026-09-26

The owner approved the exact three-target proposal with “yes, go ahead and delete those files.” The operation completed successfully. The earlier proposal, planning manifest and independent planning reviews remain unchanged as historical records; their approval-pending labels have been superseded by this execution record.

- Deleted: **7,039 files**, **12,804,081,472 logical bytes (11.925 GiB)**, from exactly the three approved directories.
- Every target and retained counterpart was freshly SHA256-verified against the reviewed ledgers before removal. Complete inventories, regular-file/no-symlink checks, file stability and open-file checks passed before deletion.
- The three target directories are absent. Retained copies remain, and their file identities, sizes and modification/change times were unchanged after removal. The eight selected checkpoint ledger/review/assessment/seal/native-binary bindings passed before and after.
- Packet-specific metadata, current POINT products, ready/local-evidence, the immutable checkpoint and unfinished CUT3 edits were preserved. No science was run, and no Git ref/worktree or remote resource was deleted.
- The obsolete ready packet now has a visible retirement/restore notice. Its original full-payload verification will report the removed copied paths missing until restored. Old manifests were not rewritten.

## Measured space

Available space immediately before: 171.884 GiB. Immediately after: 173.180 GiB. **Measured increase: 1.296 GiB**, rather than 11.925 GiB. This is the observed filesystem accounting; shared APFS extents, snapshots or concurrent activity can affect physical recovery, and this operation does not determine their individual contributions. No claim of 11.925 GiB physically freed is made.

## Evidence

`EXECUTION_RECEIPT.json` records authorization, exact proposal/script hashes, pre/post checks, times and raw df output. `DELETED_PATHS.jsonl` records each planned unlink immediately before it was performed; successful completion and full target absence establish that all 7,039 listed unlinks completed. `execute_approved_cleanup.py` is the exact script used, preserved for audit; it refuses to run again when an execution receipt exists. `PACKET_RETIREMENT_NOTICE.md` copies the notice placed beside the retired historical packet. The original pair ledgers provide the reconstruction map and per-file digests.

Next scientific work remains the unfinished bounded m1 replay/CUT3 implementation and its pre-execution review. This cleanup does not start that experiment or authorize a larger deletion batch.
