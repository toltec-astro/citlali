# Independent post-removal review — POINT 152389

**Disposition: pass for completed removal, retained integrity and published archive access. Final new-seal verification remains pending.**

The destructive operation ran at exact reviewed clean source
`8f214e06c4e01722194e7bc8ea9a1d991974ddf4` and action-manifest SHA256
`facb8174c5aeeeae9f0104e7a88509bc5b01b6e7f6777c20a40d9c9659faa763`.
The operation receipt binds the independent review JSON by digest and reports
successful completion. This review performed no deletion, restoration,
numerical inference, or full retained-archive rehash.

## Independently verified

- Receipt source/review/manifest digests match the approved artifacts. The
  removal log matches the 48,151 actions one-for-one and in order; exact removed
  logical bytes are 110,330,864,284.
- Every target is absent, including no replacement symlink. All 6,287 direct
  retained canonical files still match their frozen stat identities.
- Both retained-representation file hashes pass. All 81,382 representation rows
  match the original historical ledger digests, with correct counts and byte
  sums; every surviving original ledger entry is represented. All retained
  canonical replacements are present in those representations; no deleted target
  is misclassified as retained.
- Original guarded and C+D FILES_SHA256SUMS bytes remain unchanged, with digests
  `5f6bf979f4e4ee2f4e2d249512c9107e73ed7f8fab8c58ccc257db27c65265a9`
  and `d18b0de1585baa96bdf1c047b695b3ec6f971a546ad340e3457116478207d061`.
- Both new external per-root notices correctly identify intentional incomplete
  historical layouts and direct readers to the compact index/restoration recipe.
  Their 2,162 total bytes and current digests match POST_STORAGE_ACCOUNTING.
  The guarded notice reports 35,059 files/86,001,140,024 bytes; C+D reports
  13,092 files/24,329,724,260 bytes. Their sum matches the operation.
- Freshly hashed all 24 named scientific report inputs after removal; all match
  their recorded identities and sizes. Current IMPORTANCE.json has SHA256
  `002eae2a26deb263916555285a0deccd5de32a96f2c5f34344b632a734958ad5`,
  exactly matching the successful cached-only post-removal re-extraction record.
  Assessment, measurement table and reproduction instructions remain readable.
- Read-only explicit status probes for all six tested bulk classes resolve the
  correct canonical file/hash and leave the old target absent. No automatic
  restoration or generation occurs. Six verification scratch copies remain absent.
- ARCHIVE_INDEX is applied and carries the two retained representations exactly
  as recorded by POST_REMOVAL_INTEGRITY. The narrow publisher correction that
  preserves these records across cached report reruns was reviewed; it performs
  no scientific processing. Its other inspected changes add whitespace and
  truthful metadata-snapshot accounting only.

## Storage accounting

Original inventory: 230,911,903,266 logical bytes. Removed original payload:
110,330,864,284 bytes. Retained original payload: 120,581,038,982 bytes.
Adding the two 1,081-byte notices gives 120,581,041,144 bytes in the 27 historical
roots, exactly as the post-removal inventory reports. The timestamped closeout
snapshot adds 49,912,290 bytes for a combined 120,630,953,434 bytes. Shared parent
roots remain separately identified at 7,303,308,491 bytes, leaving
113,327,644,943 derived bytes including that metadata snapshot. The 25 GB target
is explicitly missed, with unique producer-specific exceptions retained.

The receipt's df values independently imply 110,390,718,464 bytes observed
free-space increase. This is separate from logical removal and file st_blocks;
APFS clone/snapshot effects and unrelated concurrent filesystem activity are
not assigned false precision. POST_STORAGE_ACCOUNTING explicitly labels its
pre-seal metadata snapshot and gives the final-footprint accounting rule.

## Scope and remaining finalization

POST_REMOVAL_INTEGRITY records all 198,590 surviving inventory files unchanged;
this reviewer independently checked the canonical set, representation/ledger
closure and named scientific sources rather than repeating a 120 GB hash pass.
Upstream cache, original observations/calibration, unique derived arrays, science
maps/candidates/tables, source history and original seals were outside the
approved deletion set. Numerical replay and synthetic allowances remain unused.

No removal, restoration, integrity, report-access or accounting blocker remains.
The final archive must now include this review and receive its new compact
representation seal. A detached final check must authenticate that seal and its
listed payload bytes and external-notice bindings. The original two full-payload
seals remain historical and must never be described as currently passing.
Scientific interpretation has a separate independent review; this disposition
addresses the archive operation and its reproducibility/access contract.
