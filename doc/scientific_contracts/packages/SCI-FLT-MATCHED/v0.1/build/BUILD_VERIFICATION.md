# SCI-FLT-MATCHED v0.1 Stage B r0.6 Build Verification

Date: `2026-09-01`

Scope: source/render/build/bundle consistency for the frozen scientific
authority. This is not implementation conformity, response/covariance fidelity,
observational validation, performance, readiness, production, route
realization, or Unity evidence.

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
- All four r0.6 boundaries, all seven frozen but SCI-VAL-unregistered role-
  semantics definitions, the final owner dispositions, route status, adopted
  freeze proposal, semantic map, and parity record are present.
- Both PDFs contain all stable IDs and required frozen-authority/nonclaim
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
| `pdf/SCI-FLT-MATCHED-SCIENTIFIC-RATIONALE-v0.1.pdf` | 45 | `aacb61f0f197fcfdd95613e286a5f84dafc15356acd735bfbbfe445e5ac797ff` |
| `pdf/SCI-FLT-MATCHED-ENGINEERING-CONFORMANCE-v0.1.pdf` | 41 | `c7c50a116faffb152d5035e063454597c11a507a3915385942f778c8d0cd02ab` |

Machine-readable source/PDF details are in `build/consistency-report.json`.
Exact active-object bytes and link closure are in
`SOURCE_BYTE_AND_LINK_CLOSURE_R0.6.md` and are bound by the active r0.6
frozen scientific-authority manifest.
