# SCI-RTC v0.1/r0.10 content-bound candidate review snapshot

Date: 2026-08-21

Owner: Grant Wilson

Status: candidate snapshot sealed for scientific-owner review. This record
freezes the exact review content below; it is not a scientific-authority freeze
and does not supersede frozen v0.1/r0.9 without explicit owner approval.

Source baseline:
`2ad12caeabc4a1f84b6748cd7a4cf5683202c51d`, descendant of
`9564bcca0323dacb8bea13a5ec4bbbf3b908de8f` and ancestor of governing line
`codex/scientific-contract-library`.

## Review disposition requested

The owner is asked to review this exact bounded candidate and either:

1. approve and explicitly freeze v0.1/r0.10 as successor scientific authority;
2. request a bounded correction while retaining this snapshot for comparison;
   or
3. reject the candidate, leaving frozen v0.1/r0.9 authoritative.

No implementation, validation, performance, science-qualification, production,
PTC-policy, or SCI-VAL-policy disposition is requested.

## Exact candidate content

All paths are relative to `doc/scientific_contracts/packages/SCI-RTC/v0.1/`.

| Path | SHA-256 |
| --- | --- |
| `SCIENTIFIC_OWNER_REOPENING_DIRECTIVE_R0.10.md` | `6e1c215dda40e8b716b8274f9bbf6fd42c67335b9cd8574aaa4f207b5f47f4cb` |
| `src/common/notation.tex` | `a346a1c01df853776f05b0e0db5b3c962bb4b8c92a972c26686284ae91f7be48` |
| `src/common/definitions.tex` | `7e93a1fee7853425406efdd31cc9c2642c7399cd026159a57781a7f4cd22abbb` |
| `src/common/equations.tex` | `1c0dc832b102ac5d11c4da14a7b4c9a456a2bb69f5922557c083379722d0086c` |
| `src/common/assumptions.tex` | `6b18a5d93e9bf8631208195389138f27bce75345ecfac32e459563e299ae7b71` |
| `src/common/requirements.tex` | `0c9674dc6dd037474b06ef0c6877abf8e6cf292d50527e0a0efa24846dfdae3d` |
| `src/common/edge_cases.tex` | `de0d36bead91394a5e717ff9b51120f474fdc0cebee34ee9b6a8ec28b7d0c74a` |
| `src/scientific-rationale.tex` | `dced79c90c769582bb588e3fa1317657e0af0ba26ac867c27f5498d72bc5cff3` |
| `src/engineering-conformance.tex` | `169d0ee05b2d7eb808fe3e5325549f03fa221ab448fd4f14db1cc2b3e953e6cf` |
| `src/verify_contract.py` | `2aa015ac57feba1428426d3a4bcfca65b9419f9879eabfd8f7d48100149815e4` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `c8cae2d88272a475a9c031b37941172f8b687f2398d9e5c48f5b61afc3255412` |
| `CROSSWALK.md` | `8f0ed924ffaa60926d7803c67678afc2d9bde78bf179c2ebf106e9c2fba9b4b1` |
| `README.md` | `1ce5c76dfa8b9826b31505bcb0799bffc2e6d068b5555a2772a686142661fb1d` |
| `CHANGE_LOG_R0.10.md` | `a11949e79aa4963d9763b96c49b071186989cf3cb3a76918f273734e4ae76b32` |
| `RATIONALE_TO_CONTRACT_CROSSWALK_R0.10.md` | `6eb34171d13a8764457a9e9bdea792932e8614989126c99c41d3b46e1d080c74` |
| `CONSISTENCY_REPORT_R0.10.md` | `a091d2bf101b1d4bfd530a3e946bece5c4fc20d27494dc53beeb178e83da0296` |
| `pdf/README.md` | `4c7f999ed22ef6cd0ae2c919473e123f4a4e0195ce0b2094787a04df9072b401` |
| `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `b09efeb698c736917c159bf5295e0281b21d7ee90f0deea81aca2737ea042e87` |
| `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `ce474dd5f9aa64ddcd664ef21a509fa3de2d53b9c7a6055b1ab0596813dfed49` |

## Verification state at sealing

- The mechanical verifier passed the exact 39-definition, 38-equation,
  12-assumption, 114-requirement, 77-prediction, and 89-owner-entry inventory.
- Frozen r0.9 `SCI-RTC-EQ-005` and `SCI-RTC-EQ-006` were confirmed
  byte-identical in the candidate.
- Both PDFs compiled without TeX warnings.
- All 62 final PDF pages were Poppler-rendered and visually inspected.
- `git diff --check` passed.

Any change to a path or hash in this table creates a different candidate and
requires an updated review snapshot before owner freeze.
