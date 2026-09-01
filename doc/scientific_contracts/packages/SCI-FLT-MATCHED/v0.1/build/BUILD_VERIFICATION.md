# SCI-FLT-MATCHED v0.1 Stage B r0.4 Build Verification

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
- Shared source contains 50 stable requirements, 24 contract-consequence IDs, six AO
  families, and all 21 r0.1-stable AO alternative IDs.
- `CROSSWALK.md` contains all 95 stable requirement, consequence, and AO IDs.
- The owner ledger retains all 17 SODL IDs; all four r0.4 boundary drafts and
  the route-status record are present.
- Superseded shared-core forms `Q_x`, `DeclareOrLearnOnce`, `10^-3`, and
  `10^-2` are absent.
- Both PDFs contain all 50 requirements, 24 contract consequences, and 21 AO alternatives
  plus the required draft/nonclaim language.
- Tectonic completed both builds. The logs contain zero compile errors,
  undefined controls/references, missing characters, or overfull boxes.
- Poppler reopens both PDFs. They are unencrypted, form-free, JavaScript-free,
  unrotated US Letter documents.
- The second-review repair vocabulary (`h=(g,theta)`, `P_C`, fixed response and
  covariance domains, optional `tilde d_p`, mandatory diagnostics, qualified
  amplitude-coordinate rescaling, and general operational response) is
  mechanically present.
- All 74 pages were rendered and visually reviewed as recorded in
  `PDF_QA_R0.4.md`.

## Final PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 39 | `d39f754eadb6a9f19f0231d786b3d7acd7f5dc408ce2fa3cda21d0f532f5b87b` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 35 | `e06bfed81c7be5fcc8883ca5027057aa636ac083fbe9ac65940b67e55fa13f2c` |

Machine-readable details are in `build/consistency-report.json`. Exact source
and PDF bytes are listed in `SOURCE_BYTE_REPORT_R0.4.md` and bound by the active
Stage B r0.4 manifest.
