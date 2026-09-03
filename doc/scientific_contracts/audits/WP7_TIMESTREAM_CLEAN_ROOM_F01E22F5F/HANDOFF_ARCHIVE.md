# Independent-Auditor Handoff Archive

Archive:
`WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F.tar.gz`

SHA-256:
`7affa6927754ba968b79b7f3adf2c6d6100025f6127fa8daf77042067aee8fdf`

Size: `2,087,929` bytes

Contents: one top-level directory containing eight packet files, one extracted
source checksum list, and 99 exact readable source objects from commit
`f01e22f5f8d8d92e49ae70312bdc59a81c1540ec`.

Verification performed:

1. repository-side packet verifier passed all 13 nested authority bindings,
   99 readable objects, packet hashes, firewall rules, and mapping exclusion;
2. archive was extracted into a new temporary directory;
3. the bundled `verify_packet.py` passed without repository or Git access;
4. the archive inventory contains 108 entries and no history directory,
   mapping-package source, or prior wide-audit source; and
5. all source-object hashes match `SOURCE_OBJECT_SHA256SUMS.txt`.

The archive is the single handoff artifact. This adjacent record supplies its
external digest and is intentionally not included inside the archive, avoiding
a self-hash cycle.
