# Independent publication check — 2026-09-26

PASS WITH RECORDED LIMITATIONS. Read-only check of the concrete recovery publication at source HEAD 11b4d2c07a81cc5876bd3da833c4168731e34aea; no scientific calculation, payload rehash, source edit, deletion, Git mutation, or network/Unity access.

Verified exact SHA256 identities:

- handoff/POINT_152389_RECOVERY_AND_CLEANUP_2026-09-26.md: 6de16823ce34deda7ccf54885fc1180056e0d313caff44ff14505bbe74924d92
- doc/REFACTOR_STATUS.md: fdf423bea87d8397e4d6593e21b8bae56b06500aaeeee5d2bc28dbae359c85cb
- validation/point_152389_recovery_cleanup_2026-09-26/CLEANUP_PLAN.md: ccf76a7f7f34e7a0f12d8f485205e1c81a101f96ad90dafd8843d8a31ffc54c0
- validation/point_152389_recovery_cleanup_2026-09-26/CLEANUP_PROPOSAL.json: 47d9ea6deeab053a907bfebf2144623aff032cccafeb32507a865b1235b7de8e

The publication matches the independently reviewed three-target allowlist, retained counterparts, 7,039 regular files and 12,804,081,472 logical bytes (11.924730122 GiB). It preserves explicit exact-operation approval-pending status and makes no claim of deletion or physical space already recovered. Packet retirement, incomplete original layout seals, restoration, pre-execution revalidation and current scientific dependency exclusions are clearly recorded. Whole older runs remain outside the allowlist.

The compressed packet-pair ledger decompresses byte-for-byte to the independent review ledger /private/tmp/cleanup-packet-pairs-20260926.json, SHA256 719434e1a7f0552095b9631732da43eeaed108d7452cbb9c30274ffd82cd6786. Its gzip digest and the scratch ledger digest match CLEANUP_PROPOSAL.json. All target and retained directories still exist.

UNFINISHED_CUT3.patch matches the current exact git diff for tools/timestream_successor/point_corr_admission.py. Both preserved draft documents match their original untracked counterparts byte-for-byte. Publication correctly identifies this unfinished work as preparation, with implementation/tests/review still required; preservation does not approve or execute it.

Scientific/behavioral disposition: pass for this recovery publication; no new scientific result claimed. Architecture/ownership disposition: pass; no implementation or authority change. Repository/evidence hygiene disposition: pass with the original cleanup review's limitations and owner exact-operation approval still required before deletion. This binds document hashes, not an as-yet-uncreated report commit. Selective report commitment must leave the original unfinished scientific modification and original untracked CUT3 directory outside the commit.
