# WP-5 SCI-VAL Freeze-Candidate Source Manifest

Date: `2026-08-24`

Status: content-bound candidate; scientific-owner freeze approval pending

Scope: SCI-VAL v0.1/r0.3 Core and canonical views plus the exact continuing
source/profile registry generation approved through `WP5-OWNER-D012`. MAP and
coadd remain deferred.

## Authority layers

The candidate intentionally binds two layers:

1. **VAL Core r0.3** — the six canonical formal modules, the standalone
   scientist-facing explanation, and the engineering conformance view; and
2. **continuing registries** — exact adjacent-source and owner-bound profile
   records that update bindings without rewriting VAL Core or retroactively
   changing an earlier evaluation.

Profile-availability examples and adjacent-source tables embedded in the r0.3
formal Core remain labeled historical r0.3 snapshots. Current evaluation uses
the continuing registers below.

## Exact candidate hashes

### Package, decision, and review records

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/README.md` | `2ab4d5aee5caa394065a4d44ba7d4f6286cc141587fd136abcf721954373e41c` |
| `packages/SCI-VAL/v0.1/CROSSWALK.md` | `2fcbb48ed152b325065b81bd48c4ee13528e982f5c7316d385e3b16ef6b28ed0` |
| `packages/SCI-VAL/v0.1/DECISION_LOG.md` | `29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60` |
| `packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `249685df554c2879f8ebc4737c81f9cf37dfcc3e4e8a00e2a9e99d54c0788d49` |
| `packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.2.md` | `5b8f36288917bb12c342ada192d2dee0b87bb40f8f9868acdcc11eff489d8ef0` |
| `packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.3.md` | `c33e07121dcb2979a28463eecbfe61025e4bd4b9c310b733f8f8e5ebe5c9da0e` |
| `packages/SCI-VAL/v0.1/MANAGER_REVIEW_R0.3.md` | `8902276659274c076d2f5c43615cfd49785076619a2c41f1c05f036d9dc19e89` |

### Continuing registers and exact PTC policy authority

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md` | `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_PTC_PROFILE_REGISTRY_REVIEW_R0.1.md` | `87a034c5d60999eaac7321302dc612e6406f2376e80e4f031050c93e0054ce48` |
| `audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/WP5_VAL_SCIENTIFIC_OWNER_DECISION_PACKET.md` | `83100ae1fb83f5e86556eed302b0e2554c299a6deff02bc607c36be4ef999749` |

### Canonical VAL Core sources

| Artifact | SHA-256 |
| --- | --- |
| `packages/SCI-VAL/v0.1/src/common/notation.tex` | `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44` |
| `packages/SCI-VAL/v0.1/src/common/definitions.tex` | `30b86d7fc888b21d21794799f2ef5c77f869ca1a9506cb49b6349af27feabdd1` |
| `packages/SCI-VAL/v0.1/src/common/equations.tex` | `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e` |
| `packages/SCI-VAL/v0.1/src/common/assumptions.tex` | `e616b2c28ea2052b7a0af39a0ca5a320e0b2be95cbe65393d51408e696c16b9b` |
| `packages/SCI-VAL/v0.1/src/common/requirements.tex` | `8c518ea1ffba9142d70a5982ce6f403dcb462ecf7de772047bffc7d24bad99d6` |
| `packages/SCI-VAL/v0.1/src/common/edge_cases.tex` | `13381205d78f7b69b6e80f3705c9c74351ad2028bffa010dac4f2ae6ea7bb579` |
| `packages/SCI-VAL/v0.1/src/scientific-rationale.tex` | `5c954a30658643942e4fb18c27958599648ca0015918361586bc867e4e777bd7` |
| `packages/SCI-VAL/v0.1/src/engineering-conformance.tex` | `f209da2982331c5531141c20ac93c1738f981fe0f34644ddc2b2f9324eadf137` |
| `packages/SCI-VAL/v0.1/src/verify_contract.py` | `b11342d962ed2fd01e881f48cb36824ef0ca971b55107f80e333aca113270fdb` |

### Canonical candidate PDFs

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `packages/SCI-VAL/v0.1/pdf/SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | 8 | `6966718182fe4a424696973197807ff8e8a0763b89e563df654f513a06467cfa` |
| `packages/SCI-VAL/v0.1/pdf/SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | 20 | `30f51062733c6dc1cd27cdd50aa8c2b42459944b69427536482151817767a526` |

Both PDFs are US Letter, unencrypted, contain no forms, and expose the
v0.1/r0.3 axes. Their title pages identify them as content-bound freeze
candidates pending scientific-owner approval.

## Mechanical and visual verification

The package verifier passes:

- six original/revision authority hashes;
- canonical profile, aggregate schema, four response/uncertainty roles, and
  current source bindings;
- 49 sequential requirements and 24 sequential predictions;
- exact 73-row crosswalk coverage and dual-view genre separation;
- all formal identifiers in the engineering PDF; and
- expected 8-page and 20-page PDF counts.

All 28 pages were rendered with the bundled Poppler runtime at 140 dpi and
inspected. No clipping, overlap, broken table, bad glyph, missing content, or
unreadable layout was found.

## Candidate claim boundary

This candidate establishes an exact document and registry identity for owner
review. It does not itself freeze scientific authority or establish
implementation conformity, representation fidelity, observational
validation, achieved performance, production readiness, MAP availability, or
clean-room finding closure.
