# Independent-Auditor Successor Handoff Archive

Archive:
`WP7_TIMESTREAM_CLEAN_ROOM_354AF3813.tar.gz`

SHA-256:
`fc322e4f07303352dad6d33484b16c6a75920808317fb3884b510ca3cfb858a0`

Size: `2,687,904` bytes

Source commit:
`354af3813b98bc5e6abfcf97ee9e3b856804ce9c`

Contents: one top-level directory containing eight packet files, two checksum
lists, 117 exact readable source objects, and 18 exact integrity-only binding
objects.

Verification performed:

1. all six package verifiers passed, including the corrected SCI-RTC source,
   two rebuilt PDFs, and corrected source manifest;
2. the repair-authority verifier passed all 21 publication objects, the
   1,368-row atmosphere table, and the canonical four-member TolTECA-v1
   passband aggregation;
3. the repository-side successor verifier passed all 18 nested authority
   bindings, 117 readable objects, packet hashes, firewall rules, and SCI-MAP
   exclusion;
4. two consecutive deterministic archive builds produced the identical
   SHA-256 above;
5. the archive was extracted into a new temporary directory and the bundled
   `verify_packet.py` passed without repository or Git access; and
6. all 74 corrected RTC PDF pages were Poppler-rendered and inspected with no
   clipping, overlap, malformed glyph, blank spill, table truncation, or page
   geometry defect.

The archive is the single independent-auditor handoff artifact. This adjacent
record supplies its external digest and is intentionally not included inside
the archive, avoiding a self-hash cycle.

