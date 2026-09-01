# SCI-FLT-MATCHED v0.1 Stage B r0.3 Build Verification

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
- The owner ledger retains all 17 SODL IDs; all four r0.3 boundary drafts and
  the route-status record are present.
- Superseded shared-core forms `Q_x`, `DeclareOrLearnOnce`, `10^-3`, and
  `10^-2` are absent.
- Both PDFs contain all 50 requirements, 24 contract consequences, and 21 AO alternatives
  plus the required draft/nonclaim language.
- Tectonic completed both builds. The logs contain zero compile errors,
  undefined controls/references, missing characters, or overfull boxes.
- Poppler reopens both PDFs. They are unencrypted, form-free, JavaScript-free,
  unrotated US Letter documents.
- The directed-review repair vocabulary (`D_loc`, `F_g`, `K_NOI`,
  `Q_FLT^0.1`, and `W_p=A_p^dagger D_p A_p`) is mechanically present.
- All 69 pages were rendered and visually reviewed as recorded in
  `PDF_QA_R0.3.md`.

## Final PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 36 | `2b623da8ce85445f7f4db18bab8d719269842658ecbfba75d8de00fe4445f8a2` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 33 | `079c96b6a044aaa7ab2b2ea62434c8173b3723928b3a0eb50e2a42584b191775` |

Machine-readable details are in `build/consistency-report.json`. Exact source
and PDF bytes are listed in `SOURCE_BYTE_REPORT_R0.3.md` and bound by the active
Stage B r0.3 manifest.
