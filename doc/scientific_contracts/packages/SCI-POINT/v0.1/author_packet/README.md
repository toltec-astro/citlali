# SCI-POINT v0.1 Author Packet Archive

Status: `r0.3` review candidate active; `r0.1` and `r0.2` retained as
superseded review candidates; none owner-approved; Stage B not launched

Archive:
`SCI-POINT-v0.1-r0.3-stage-b-author-packet.tar.gz`

The archive is deterministic and contains only the root
`AUTHOR_PACKET_MANIFEST.md`, its SHA-256 sidecar, and the 37 exact admitted
author objects. The archive's own digest is recorded in the adjacent
`.sha256` sidecar and its exact byte count in the adjacent `.bytes` sidecar.

Rebuild with:

```text
$HOME/tolteca/bin/python author_packet/create_author_packet_r0_3.py
```

Verify with:

```text
$HOME/tolteca/bin/python author_packet/verify_author_packet_r0_3.py
```

Any author-object edit requires a successor packet revision, recomputed object
and manifest hashes, and renewed owner review.

The immutable `r0.1` and `r0.2` archives and digest sidecars remain beside the
active candidate to preserve prior exact review bytes. Their source objects
have been superseded; active build and verification scripts target `r0.3`.
