# MAP-UNITY-ED2 local resource report

This report is package-preparation evidence, not Unity capacity or reduction
evidence. The machine-readable authority is `resource-report.json`;
`scripts/ed2-capture.py` records the owner-operated Unity-root measurements.
Neither is a full/all-PTC serialization bound.

The measured local reference metadata contains 1,717,082,860 source terms and
projects 1,688,839,080 post-RTC detector/sample terms. The full-PTC core-array
lower bound is 94,254,679,616 bytes (87.7815 GiB). The prior preliminary full
capture estimate is 127,064,131,640 bytes (118.3377 GiB). Allocating that
estimate between the Point and Science capture stages in proportion to their
measured core arrays and reserving 16 GiB for compact evidence, logs,
manifests, and any authorized focused expansion gives a planning envelope of
144,244,000,824 bytes (134.3377 GiB). This leaves 70,504,363,976 bytes
(65.6623 GiB) beneath the owner-approved 214,748,364,800-byte ceiling.

That headroom is not permission to run or a guarantee. The owner first records
the frozen planning estimate and the live Unity-root measurements, reviews the
record, then separately decides whether to perform the next stage. Before and
after every stage the package tool inventories every root, directory, regular
file, and symlink under all five governed roots (the two fresh source projects,
two capture roots, and compact root) without following symlink payloads. It
records logical and allocated cumulative usage, the selected filesystem, and
available capacity. The owner stops before the next stage if observed use
approaches or exceeds the cap, or capacity is inadequate. There is no
operator-supplied byte count and no automatic cleanup or continuation.

The digest-bound planning values are: the 144,244,000,824-byte preparation
envelope; the two frozen capture estimates;
64 MiB plus two bytes per producer-authority primitive term for compact
production; 64 MiB plus 256 or 2,048 bytes per discrepancy-request
`max_terms` for expansion planning or emission; 4 GiB plus the result
collection size for analysis; and 64 MiB plus three times the final-inventory
return-member size after exact retained-PTC exclusions for the final bundle.
Resource records, analysis, manifests, evidence, and return construction live
below the compact root's `_campaign` directory, so subsequent measurements
include them in the governed inventory. Failure of a measurement or owner
review is a stop, never authority to delete evidence or increase the ceiling.

Canonical raw targets are excluded only because the staging lane creates
individual symlinks to pre-existing files. CAP-POINT and CAP-SCIENCE remain
retained through the fresh MAP re-audit and any requested focused expansion.
No package script automatically deletes them.
