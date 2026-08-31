# SCI-FLT-MATCHED v0.1 Stage B Build Verification

Date: 2026-08-31

Scope: source/render/build consistency only. This is not scientific validation,
implementation conformity, achieved performance, readiness, production, or a
numerical-route authorization.

## Result

`PASS`

- All eight approved author-packet objects retain the SHA-256 values in
  `AUTHOR_PACKET_MANIFEST.md`; the manifest itself retains SHA-256
  `255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.
- Both LaTeX views import the same six shared modules in the same order.
- Shared source contains 39 stable requirements, 18 falsifiable predictions,
  six authored-option families, and 21 exact option alternatives.
- `CROSSWALK.md` contains all 78 stable requirement, prediction, and option-
  alternative identities.
- `SCIENTIFIC_OWNER_DECISION_LEDGER.md` contains 17 open owner questions.
- Both rendered PDFs contain all 39 requirement IDs, all 18 prediction IDs,
  all 21 option-alternative IDs, and the required draft/nonclaim language.
- Tectonic completed both builds. Mechanical log inspection found zero compile
  errors, zero undefined-control-sequence/reference/citation warnings, zero
  missing-character warnings, and zero overfull boxes. Underfull boxes occur
  only in deliberately narrow table columns and produced no visible defect.
- Poppler reopened both PDFs successfully. Neither is encrypted or contains a
  form or JavaScript. Both use US Letter pages with zero rotation.
- All 70 pages were rendered at 120 dpi. Seven contact sheets covering every page
  plus representative full-page inspections showed no clipping, overlap,
  broken glyphs, malformed equations, unreadable table flow, missing headers or
  footers, or defective section transitions.

## Final PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 37 | `0681ada98d57b00fbe9abef23699cf6eee1334ab9420ff84cd862ee0964897c0` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 33 | `f7f40b8823f7c98a49a69be2db44092c21329f259ca2ac83a554aa4029586145` |

The machine-readable details, including every shared-module and packet-object
digest, are in `build/consistency-report.json`.
