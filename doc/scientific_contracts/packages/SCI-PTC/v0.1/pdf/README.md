# SCI-PTC v0.1 PDF Outputs

Status: SCI-PTC v0.1/r0.5 scientific authority frozen by Grant Wilson on
`2026-08-23`; implementation conformity not yet assessed under this contract.

The canonical frozen outputs are:

- `SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf`: 13 letter pages; SHA-256
  `3c927dbcb631b2033f04d933fa4d69911698b840b7b4642e759ad8c715c16ab6`;
  and
- `SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf`: 26 letter pages; SHA-256
  `f0b3bdc28c0997e4960b8231d6113339fe391d331e84dc7c0638912b0d3f7adb`.

Both PDFs were generated from the frozen r0.5 audience views under `../src/`.
The scientific rationale is the standalone science-team document with compact
traceability. The engineering view imports the six shared normative modules
exactly once and is the complete formal contract view. Both are US Letter,
unencrypted, contain no forms or JavaScript, and completed Poppler all-page
visual inspection.

The verified content-bound candidate at commit
`8f0ecccfacbdce0543141c4289ec06c702065f5e` was promoted
without scientific change. Its candidate-status PDFs remain under
`r0.5-candidate/` as review history; they are not the canonical entry points.
The superseded canonical r0.4 PDF identities remain recorded in
`SCIENTIFIC_OWNER_FREEZE_R0.4.md` and immutable Git history.

These canonical files establish frozen scientific authority only. They do not
claim implementation conformity, representation/response fidelity,
validation, achieved performance, science qualification, production
readiness, MAP availability, or audit-finding closure. Future substantive
edits require explicit owner authority and a versioned successor or formally
reopened revision.
