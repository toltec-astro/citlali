# Independent Auditor Start Here

This bundle is a self-contained, implementation-blind scientific-contract
audit packet pinned to source commit
`f01e22f5f8d8d92e49ae70312bdc59a81c1540ec`.

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
top-level packet source manifest binds the frozen package and composition
manifests used by the coordinator to assemble this handoff.

Do not consult a repository checkout, Git history, implementation, prior
audit, repair tracker, prior scenario set, web source, or mapping package.
Return any required-but-absent authority as unavailable rather than seeking an
unlisted substitute.
