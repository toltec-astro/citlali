# SCI-MAP v0.1 r0.7.1 Byte Equality And Shared-Authority Report

Status: deterministic author-artifact comparison; no implementation,
validation, performance, readiness, production, or freeze claim

Prepared: `2026-08-28`

## PTC-to-MAP boundary copies

The MAP and PTC copies of `SCI-PTC_TO_SCI-MAP v0.1/r0.1` are byte-identical.
Each has SHA-256
`a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7`.
Their canonical identifier contains one space before `v0.1/r0.1`.

## Shared authority

The ordered byte concatenation of the r0.7.1 wrapper followed by notation,
definitions, equations, assumptions, requirements, and edge cases has SHA-256
`649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b`.

| Source | SHA-256 |
| --- | --- |
| `src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex` | `08fcc9782cfba806d33dc07652a2363c8bd6540084f54e752e1fa91a5336b6bb` |
| `src/common/notation.tex` | `2b132704dd1ee8da7a56e5bafdc998df98422fe512736ac4f904fad8a693e569` |
| `src/common/definitions.tex` | `740f4a6f1ef0bbb12f721f192b7883144c247d039ab0c9dfa0ffae53cd711b65` |
| `src/common/equations.tex` | `36329f4cd1a103c78fcdcc5ff247a850f40aba92a09156d1aa55a1411a430c04` |
| `src/common/assumptions.tex` | `bba33b92c4189fe5886ef849caebeeb5400bdc7f2572f58a74de53ca578881de` |
| `src/common/requirements.tex` | `68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7` |
| `src/common/edge_cases.tex` | `47022012e79173a1778a4e5bdc6743b4691bfecb27faa3090bcf03458d87e123` |

All three views import the exact r0.7.1 wrapper once. The formal and
engineering views expand the same 52 requirements and 25 predictions; the
scientist-facing rationale uses the same shared definitions/equations while
retaining its compact narrative form.
