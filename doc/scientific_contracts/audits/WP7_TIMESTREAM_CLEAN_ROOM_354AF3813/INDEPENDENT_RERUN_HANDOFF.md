# WP-7 Successor Independent Rerun Handoff

Status: **successor packet verified; fresh-context Codex and ChatGPT audits
required; no successor audit result recorded here**

Prepared: `2026-08-25`

## Handoff artifact

Archive:
`WP7_TIMESTREAM_CLEAN_ROOM_354AF3813.tar.gz`

SHA-256:
`fc322e4f07303352dad6d33484b16c6a75920808317fb3884b510ca3cfb858a0`

Size: `2,687,904` bytes

Immutable source commit:
`354af3813b98bc5e6abfcf97ee9e3b856804ce9c`

The archive has been extracted into a new temporary directory and its bundled
verifier passed without Git or repository access.

## Fresh Codex task

Start a new task with no inherited conversation context. Attach only the
archive above. Use the exact prompt in `AUDIT_THREAD_LAUNCH_PROMPT.md`.

Request these locked outputs before providing any previous audit or comparison
material:

1. `WP7_SUCCESSOR_INDEPENDENT_SCIENTIFIC_CONTRACT_AUDIT.md`;
2. `WP7_SUCCESSOR_INDEPENDENT_SCENARIO_SUITE.md`; and
3. `WP7_SUCCESSOR_INDEPENDENT_REPORT_SHA256SUMS.txt`.

The task must stop after locking those outputs and request the separate
comparison packet. It must not inspect this repository, Git history, the old
WP-7 reports, repair records, or chat history.

## Fresh ChatGPT task

Start a separate new task with no inherited conversation context. Attach only
the same archive. Use the same exact launch prompt from the archive.

Request these locked outputs before providing any previous audit or comparison
material:

1. `CHATGPT_WP7_SUCCESSOR_CLEAN_ROOM_AUDIT.md`;
2. `CHATGPT_WP7_SUCCESSOR_FINDINGS.csv`; and
3. exact SHA-256 identities for both files.

The ChatGPT task must also stop after locking its independent outputs and
request comparison material. It must not receive the earlier ChatGPT audit,
the earlier Codex audit, this repair history, or any finding crosswalk before
that lock.

## Coordinator return

Return both locked result sets to the coordinator only after each task has
published its own hashes. The coordinator may then perform a regression
comparison, map independently derived findings to the locked predecessor
audits, and recommend final closure dispositions. The comparison must not edit
either independent result set or either predecessor audit.

