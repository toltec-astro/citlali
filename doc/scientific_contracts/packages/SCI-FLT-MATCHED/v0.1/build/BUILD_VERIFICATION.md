# SCI-FLT-MATCHED v0.1 Stage B r0.5 Build Verification

Date: `2026-09-01`

Scope: source/render/build consistency only. This is not scientific approval,
implementation conformity, response/covariance fidelity, observational
validation, performance, readiness, production, or route authorization.

## Result

`PASS`

- All eight author-packet objects retain their approved SHA-256 values.
- `AUTHOR_PACKET_MANIFEST.md` retains SHA-256
  `255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.
- Both views import the same six shared modules in identical order.
- Shared source contains 50 stable requirements, 25 contract-consequence IDs, six AO
  families, and all 21 r0.1-stable AO alternative IDs.
- `CROSSWALK.md` contains all 96 stable requirement, consequence, and AO IDs.
- The owner ledger retains all 17 SODL IDs; all four r0.5 boundary drafts and
  the route-status record are present.
- Superseded shared-core forms `Q_x`, `DeclareOrLearnOnce`, `10^-3`, and
  `10^-2` are absent.
- Both PDFs contain all 50 requirements, 25 contract consequences, and 21 AO alternatives
  plus the required draft/nonclaim language.
- Tectonic completed both builds. The logs contain zero compile errors,
  undefined controls/references, missing characters, or overfull boxes.
- Poppler reopens both PDFs. They are unencrypted, form-free, JavaScript-free,
  unrotated US Letter documents.
- The final targeted closure vocabulary (`S_parent_fact`, `D_m`, `ell_p^star`,
  `c_p`, `S_apply(p)`, `h_pre`, role-specific validity, full lifecycle,
  reference/operational covariance separation, fixed-template authority, and
  request-qualified downstream state) is mechanically present.
- All 80 pages were rendered after the final build and visually reviewed as
  recorded in `PDF_QA_R0.5.md`.

## Final PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 42 | `95007fb16de1eeb5a6efaa77e7af8b64981e1d5ff572e9c53e5254a0b7b81876` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 38 | `cee6476e664af1c47c89e06fc95b279f0cfeb6cea8e0553f0d9343047b338496` |

Machine-readable details are in `build/consistency-report.json`. Exact source
and PDF bytes are listed in `SOURCE_BYTE_REPORT_R0.5.md` and bound by the active
Stage B r0.5 authority manifest.
