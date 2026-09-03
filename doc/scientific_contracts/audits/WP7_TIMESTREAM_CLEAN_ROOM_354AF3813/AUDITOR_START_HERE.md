# Independent Auditor Start Here

This bundle is a self-contained, implementation-blind scientific-contract
successor audit packet pinned to source commit
`354af3813b98bc5e6abfcf97ee9e3b856804ce9c`.

## Required Order

1. Run:

   ```bash
   python3 verify_packet.py
   ```

2. Read `CLEAN_ROOM_CHARTER.md`.
3. Read `READABLE_SOURCE_ALLOWLIST.md` and
   `SANITIZED_COMPOSITION_NOTES.md`.
4. Inspect only the material under `sources/` that the allowlist admits.
5. Derive and lock the independent authority graph, interface matrix,
   invariants, scenarios, findings, and readiness results before accepting any
   comparison material from the coordinator.

`SOURCE_OBJECT_SHA256SUMS.txt` binds every extracted source object. The
top-level source manifest binds the frozen package, corrected RTC, composition,
native-interface, and WP-7 repair-authority generations used to assemble this
handoff.

The exact native-interface authority set contains incidental administrative
references to historical work. They are admitted only because their exact
approval and precedence bytes are required authority. Do not use an
administrative identifier to assign, map, suppress, or rename an independent
finding before the report and scenario suite are locked.

Do not consult a repository checkout, Git history, implementation, prior
audit, repair tracker, prior scenario set, web source, or mapping package.
Return any required-but-absent authority as unavailable rather than seeking an
unlisted substitute.
