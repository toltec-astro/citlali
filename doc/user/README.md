# User And Product Guides

This directory is the team-facing entry point for physicists and astronomers
who configure Citlali or interpret its products. Guides should be concise,
scientifically candid, and organized around an actual workflow or product
rather than an internal class or audit finding.

## Guide Index

Add guides as user-visible audit repairs reach a stable contract; do not write
ahead of unsettled behavior.

| Guide ID | Scope | Status |
| --- | --- | --- |
| [`DOC-MAP-001`](DOC-MAP-001_ORDINARY_NAIVE_MAPS.md) | What each emitted map measures | Phase A validated for the ordinary naive Stokes-I bundle; NOI/JINC/filter sections remain queued |

`DOC-MAP-001` has begun with the bounded, completed SCI-MAP-001 contract. Its
later static dictionary will grow only as additional product contracts
stabilize. A future compact per-reduction rendering will be driven by the
effective plan and realized observation state, list only emitted products and
meaning-changing settings, and link to the provenance package rather than
reproduce it. Citlali owns the semantics; TolTECA and TolProj may surface the
guide but do not independently redefine the products.

## What A Guide Must Answer

A guide should answer only the questions applicable to its subject:

1. What is this workflow or product for, and why would an astronomer use it?
2. What does each reported quantity or map measure?
3. Which configuration choices materially change that meaning?
4. What inputs, calibration state, units, normalization, support, and validity
   conditions are assumed?
5. Which products belong together and must travel or be interpreted as a
   package?
6. What is not scientifically supported, even if a file or option exists?
7. Where is the reusable mathematical method explained and where is the
   behavior validated?

Use [`PRODUCT_GUIDE_TEMPLATE.md`](PRODUCT_GUIDE_TEMPLATE.md) when a new guide
is warranted. Delete unused template sections. A guide should normally link to
the relevant machine-readable product contract and validation profile rather
than reproduce their field inventories.

## Separation From Scientific Notes

A guide may show a compact identity or normalization needed for correct use,
but it should not carry a full estimator derivation. Link the applicable stable
method ID from [`../science/README.md`](../science/README.md). If several
products use the same method, all of their guides link to the same note.
