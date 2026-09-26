# Detached final seal review — POINT 152389

**Disposition: pass. No archive closeout blocker remains.**

Archive:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-importance-and-archive-compaction-20260926`

Verified final FILES_SHA256SUMS SHA256:
`737e72ad60db225a6036839c54c53b24c580b0aeb4625569dd699388d9be5083`.
Verified final SEAL.json SHA256:
`0bafac7a4d10ab94bc27f75ff377c0e83a211d05594b0b8141d33fa32a93f3bf`.

Independently freshly hashed all 51 listed closeout payloads; every digest
matches. Paths resolve inside the archive, all are regular nonsymlink files,
entries are unique, and the complete current file set equals the manifest plus
its two explicit exclusions, FILES_SHA256SUMS and SEAL.json. No scratch replacement
payload or unlisted file is hidden in the closeout.

The payload total is 49,971,845 bytes. Manifest size is 4,940 bytes; seal size is
4,464 bytes; the complete closeout is 49,981,249 bytes. Adding the separately
accounted 27 retained historical roots including notices (120,581,041,144 bytes)
gives a final combined logical footprint of **120,631,022,393 bytes**. This agrees
with the seal's explicit final-total rule and distinguishes snapshot accounting
from final metadata sizes. It does not replace the separately measured
110,390,718,464-byte filesystem free-space increase or the
110,330,864,284-byte logical payload removal.

Freshly authenticated both external compaction notice hashes and sizes. The
applied archive index references the exact reviewed action manifest and both
retained representations; all referenced representation hashes match. The seal
truthfully describes historical full-payload layouts as incomplete until explicit
restoration, rather than claiming their original presence checks pass.

Checked the archived exact execution source file hashes against SOURCE_AND_COUNTS.
Checked the final publication source and patch digests against both that record
and the seal; their bindings agree. The copied POST_REMOVAL_REVIEW is byte-identical
to this reviewer's issued report. The bundle includes independent scientific and
publication reviews, pre-removal manifest review, post-removal review, restoration
proof records, retained representations, reproduction commands, and accounting.
Execution remains bound to `8f214e06c4e01722194e7bc8ea9a1d991974ddf4`; the small
later publication correction is represented explicitly by source and patch bytes.

Integrity scope is accurately limited: fresh hashes cover this new approximately
50 MB closeout, while external original payload identities come from preserved
accepted ledgers, checked unchanged file identities, the fresh pre-removal
duplicate verification, and bounded post-removal checks. No claim of a new full
120.6 GB hash pass or numerical replay is made. Exact-byte restoration remains
explicit, original seals remain historical, unique producer-specific state and
shared parent dependencies remain retained, and the missed 25 GB target is clear.

This detached review changed no archive/source file, restored no deleted file,
and performed no scientific processing, fit, candidate evaluation, covariance
fit, synthetic trial or full retained-archive rehash. The review itself is kept
outside the sealed archive and may be bound by repository completion records
without invalidating the completed seal.
