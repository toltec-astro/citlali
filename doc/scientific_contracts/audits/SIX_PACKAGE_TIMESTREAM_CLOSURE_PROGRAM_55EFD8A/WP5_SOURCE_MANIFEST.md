# WP-5 SCI-VAL Frozen Source Manifest

Date: `2026-08-24`

Status: SCI-VAL v0.1/r0.3 scientific authority frozen under
`WP5-OWNER-D013`; WP-7 clean-room re-audit pending

Scope: SCI-VAL v0.1/r0.3 Core and canonical views plus the exact continuing
source/profile registry generation approved through `WP5-OWNER-D012`. MAP and
coadd remain deferred and unbound.

Scientific owner: Grant Wilson

This manifest does not hash itself.

## Bound Authority

| Authority | Bound identity |
| --- | --- |
| Consolidated clean-room audit baseline | `55efd8a54464636a24e621f6d1b60486d235b20e` |
| Approved content-bound candidate | commit `3ad018e97e134a0b0324d3fa2674ef96d5a680d4`; candidate-manifest SHA-256 `314823249917e09d36ba76557699c1fbd1ba29171b3604a9b6d74cea8ca5d7f1` |
| Scientific-owner freeze | SCI-VAL v0.1/r0.3 under `WP5-OWNER-D013`, `2026-08-24` |
| VAL Core | unchanged six canonical r0.3 modules; 49 requirements and 24 predictions |
| Continuing registries | exact source/profile records approved through `WP5-OWNER-D012`; no retroactive evaluation change |
| MAP/coadd | deferred and unbound |

The approved authority intentionally retains two layers:

1. **VAL Core r0.3** — the six canonical formal modules, standalone
   scientist-facing explanation, and engineering conformance view; and
2. **continuing registries** — exact adjacent-source and owner-bound profile
   records that update bindings without rewriting VAL Core or retroactively
   changing an earlier evaluation.

Profile-availability examples and adjacent-source tables embedded in the r0.3
Core remain labeled historical r0.3 snapshots. Current evaluation uses the
continuing registers below.

## Final Frozen Hashes

### Package, owner, and review records

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/README.md` | `f29e27972505e7f01e74e0c6b577299dc640fc32c01c5228f65755dc4a1edbbe` |
| `packages/SCI-VAL/v0.1/CROSSWALK.md` | `2fcbb48ed152b325065b81bd48c4ee13528e982f5c7316d385e3b16ef6b28ed0` |
| `packages/SCI-VAL/v0.1/DECISION_LOG.md` | `29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60` |
| `packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `249685df554c2879f8ebc4737c81f9cf37dfcc3e4e8a00e2a9e99d54c0788d49` |
| `packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.2.md` | `5b8f36288917bb12c342ada192d2dee0b87bb40f8f9868acdcc11eff489d8ef0` |
| `packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.3.md` | `c33e07121dcb2979a28463eecbfe61025e4bd4b9c310b733f8f8e5ebe5c9da0e` |
| `packages/SCI-VAL/v0.1/MANAGER_REVIEW_R0.3.md` | `8902276659274c076d2f5c43615cfd49785076619a2c41f1c05f036d9dc19e89` |
| `packages/SCI-VAL/v0.1/R0.3_FREEZE_CANDIDATE_REVIEW_COVER.md` | `f16f489648919e57f65a0e0625565c7a701e8ba2616ad6736235b8cf804995dc` |
| `packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md` | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` |
| `packages/SCI-VAL/v0.1/FREEZE_VERIFICATION_R0.3.md` | `cfda29e50d34dd5ab07c4a0227948e9c48d6ca043fd298c05ea90279669f067b` |
| `packages/SCI-VAL/v0.1/pdf/README.md` | `f7a09d9954d1c4d34b8f2815dc8ec9d3c463039fdc3f1b4e2d537dfda921473f` |

### Continuing registers and exact PTC policy authority

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md` | `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_PTC_PROFILE_REGISTRY_REVIEW_R0.1.md` | `87a034c5d60999eaac7321302dc612e6406f2376e80e4f031050c93e0054ce48` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_VAL_SCIENTIFIC_OWNER_DECISION_PACKET.md` | `40559198d6a7c3a55cb46a338efc61c6abc518b36b843c9b7d31006e70bfc047` |

### Canonical VAL Core sources

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/src/common/notation.tex` | `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44` |
| `packages/SCI-VAL/v0.1/src/common/definitions.tex` | `30b86d7fc888b21d21794799f2ef5c77f869ca1a9506cb49b6349af27feabdd1` |
| `packages/SCI-VAL/v0.1/src/common/equations.tex` | `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e` |
| `packages/SCI-VAL/v0.1/src/common/assumptions.tex` | `e616b2c28ea2052b7a0af39a0ca5a320e0b2be95cbe65393d51408e696c16b9b` |
| `packages/SCI-VAL/v0.1/src/common/requirements.tex` | `8c518ea1ffba9142d70a5982ce6f403dcb462ecf7de772047bffc7d24bad99d6` |
| `packages/SCI-VAL/v0.1/src/common/edge_cases.tex` | `13381205d78f7b69b6e80f3705c9c74351ad2028bffa010dac4f2ae6ea7bb579` |
| `packages/SCI-VAL/v0.1/src/scientific-rationale.tex` | `a56d79357ee9758c74c7fc692646ba2d2bdb5ecef12bd3499410b06318a03b65` |
| `packages/SCI-VAL/v0.1/src/engineering-conformance.tex` | `6f5e5f6bd8aec2a2577b426b54dad1deb8bd761858e8f124d1f7f234199768fd` |
| `packages/SCI-VAL/v0.1/src/verify_contract.py` | `b11342d962ed2fd01e881f48cb36824ef0ca971b55107f80e333aca113270fdb` |

### Canonical frozen PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `packages/SCI-VAL/v0.1/pdf/SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | 8 | `53e32a12ad4b60b4cccaaf05e1c0f9ad248d7e31637fbcfe2b4344992b81359c` |
| `packages/SCI-VAL/v0.1/pdf/SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | 20 | `e5b353d52303e7f9fd3d10abcd35a4a15eb24021eab4e0663244d414052232fa` |

## Mechanical And Visual Verification

The package verifier passes original/revision authority hashes, the canonical
profile and aggregate schema, all four response/uncertainty roles, continuing
source bindings, 49 sequential requirements, 24 sequential predictions,
exact 73-row crosswalk coverage, dual-view separation, formal PDF text
coverage, and expected page counts.

Both PDFs are unencrypted US Letter throughout and contain no forms, widgets,
or JavaScript. All 28 pages were rendered with Poppler at 140 dpi and
inspected, with additional full-resolution inspection of both title pages. No
clipping, overlap, broken table, bad glyph, missing content, or unreadable
layout was found.

## Claim Boundary

This manifest establishes the exact frozen SCI-VAL document and continuing
registry authority needed by WP-5. It does not establish implementation
conformity, representation fidelity, observational validation, achieved
performance, production readiness, MAP or coadd availability, or clean-room
finding closure.

`F-003`, `F-004`, `F-005`, `F-016`, and `F-020` remain open until WP-7 binds
this final manifest and performs the authorized clean-room re-audit.
