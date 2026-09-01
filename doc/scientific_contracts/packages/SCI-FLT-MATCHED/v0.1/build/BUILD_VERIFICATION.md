# SCI-FLT-MATCHED v0.1 Stage B r0.6 Build Verification

Date: `2026-09-01`

Scope: source/render/build/bundle consistency only. This is not scientific
approval, implementation conformity, response/covariance fidelity,
observational validation, performance, readiness, production, route
authorization, or scientific freeze.

## Result

`PASS`

- All eight Stage A author-packet objects retain their approved SHA-256 values;
  `AUTHOR_PACKET_MANIFEST.md` remains
  `255c66da880fc7664a57635b28a98d874fc024490d04528f802635c0382a57c8`.
- The exact r0.6 directive is byte-identical to the supplied attachment at
  SHA-256 `5758640064918b2d3021afc7ea63ffba063ba7b1abbb66dc6d43d945ed73ebd3`.
- The repository-context Stage A verifier passes. It is not copied into the
  authority-only standalone bundle.
- Both views import the same six shared modules in identical order.
- Shared source retains exactly 50 requirements, 25 prediction IDs, six AO
  families, all 21 stable AO alternatives, and 17 SODL IDs. No REQ-051 or
  PRED-026 was introduced; the crosswalk retains all 96 stable IDs.
- The stochastic random object `M:D_model->R`, observed numerical payload
  `m_obs:D_m->R`, covariance authority, payload-free `h_pre`, three-case
  PRED-025 behavior, corrected lifecycle order, and package-versus-realization
  AO authorization distinctions are mechanically present.
- All four r0.6 boundaries, all seven honestly labeled role-semantics drafts,
  the owner disposition packets, route status, conditional-freeze proposal,
  semantic map, and parity record are present.
- Both PDFs contain all stable IDs and required conditional-freeze/nonclaim
  language. Tectonic logs contain zero compile errors, undefined controls or
  references, missing characters, and overfull boxes.
- Poppler and pypdf reopen both PDFs. They are unencrypted, form-free,
  JavaScript-free, unrotated US Letter documents.
- All 86 final-build pages were rendered and visually inspected as recorded in
  `PDF_QA_R0.6.md`.
- The active manifest set contains no repository-context-only local links. The
  deterministic archive verifies from its SHA-256 sidecar; its extracted
  `verify_stage_b_draft.py` and complete Markdown-link audit both pass.

## Final PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 45 | `a931c946ab83aab36278056106b525638d9f017ce461a1359b5ce95efc3deb3d` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 41 | `e14b5dce8559311e57217f927496bf809042d492fd76a42ac3839ed18f69d861` |

Machine-readable source/PDF details are in `build/consistency-report.json`.
Exact active-object bytes and link closure are in
`SOURCE_BYTE_AND_LINK_CLOSURE_R0.6.md` and are bound by the active r0.6
conditional-freeze preflight manifest.
