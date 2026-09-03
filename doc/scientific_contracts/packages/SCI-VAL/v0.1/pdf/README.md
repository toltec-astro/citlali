# SCI-VAL v0.1/r0.3 PDF Outputs

Status: SCI-VAL v0.1/r0.3 scientific authority frozen by Grant Wilson on
`2026-08-24`; implementation conformity and validation not assessed.

The canonical frozen outputs are:

- `SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf`: 8 US Letter pages; SHA-256
  `53e32a12ad4b60b4cccaaf05e1c0f9ad248d7e31637fbcfe2b4344992b81359c`;
  and
- `SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf`: 20 US Letter pages; SHA-256
  `e5b353d52303e7f9fd3d10abcd35a4a15eb24021eab4e0663244d414052232fa`.

Both PDFs were rebuilt from the status-clean r0.3 audience views under
`../src/`. The engineering view imports the unchanged six-module VAL Core;
the science-team rationale explains the same architecture in a standalone
narrative. The continuing `PROFILE_REGISTRY.md` and
`SOURCE_BINDING_REGISTER.md` update exact policy/source bindings without
rewriting Core.

All 28 pages were rendered with Poppler at 140 dpi and inspected. The PDFs
are US Letter, unencrypted, contain no forms or JavaScript, and show no
clipping, overlap, broken glyph, missing content, or unreadable layout.

The owner-approved candidate at commit `3ad018e97`, candidate-manifest
SHA-256
`314823249917e09d36ba76557699c1fbd1ba29171b3604a9b6d74cea8ca5d7f1`,
was promoted without scientific change. These files establish frozen
scientific authority only. They do not establish implementation conformity,
validation, performance, production readiness, MAP availability, or
clean-room finding closure.
