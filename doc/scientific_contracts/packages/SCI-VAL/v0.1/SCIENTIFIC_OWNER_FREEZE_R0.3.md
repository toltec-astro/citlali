# SCI-VAL v0.1/r0.3 Scientific-Owner Freeze

Date: `2026-08-24`

Owner: Grant Wilson

Approval record:

> The scientific owner answered “and yes, let's approve this as frozen
> authority” to `WP5-OWNER-D013`, which asked whether the exact content-bound
> SCI-VAL v0.1/r0.3 candidate should be promoted to frozen scientific
> authority while leaving all evidence-dependent claims unestablished.

Normalized freeze directive:

> Freeze SCI-VAL v0.1/r0.3 as the exact scientific authority. Preserve the
> unchanged VAL Core, continuing source/profile registries, deferred MAP and
> coadd state, and separate implementation, validation, performance,
> production-readiness, and clean-room re-audit gates.

Status: Scientific authority frozen; implementation conformity and validation
not assessed under this contract.

## Frozen Authority

This owner statement promotes the content-bound candidate at commit
`3ad018e97e134a0b0324d3fa2674ef96d5a680d4`, candidate-manifest SHA-256
`314823249917e09d36ba76557699c1fbd1ba29171b3604a9b6d74cea8ca5d7f1`,
without scientific change. It freezes SCI-VAL v0.1/r0.3 as the active
scientific authority for:

- the unchanged six-module VAL Core with 49 requirements and 24 predictions;
- the standalone science-team rationale and complete engineering conformance
  view;
- the continuing exact source and profile registries;
- the canonical `SCI-VAL:independent_exposure@1` proposition;
- the five owner-approved PTC named-use profiles and two explicit unsupported
  PTC use identities approved through `WP5-OWNER-D012`; and
- the two canonical status-clean PDF renderings recorded in `pdf/README.md`.

MAP and coadd remain deferred and unbound. Embedded r0.3 source/profile
examples remain historical snapshots; current evaluations bind the continuing
registries and never retroactively alter an earlier decision identity.

## Exact Frozen Hashes

| Artifact | SHA-256 |
| --- | --- |
| `README.md` | `f29e27972505e7f01e74e0c6b577299dc640fc32c01c5228f65755dc4a1edbbe` |
| `CROSSWALK.md` | `2fcbb48ed152b325065b81bd48c4ee13528e982f5c7316d385e3b16ef6b28ed0` |
| `DECISION_LOG.md` | `29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `249685df554c2879f8ebc4737c81f9cf37dfcc3e4e8a00e2a9e99d54c0788d49` |
| `R0.3_FREEZE_CANDIDATE_REVIEW_COVER.md` | `f16f489648919e57f65a0e0625565c7a701e8ba2616ad6736235b8cf804995dc` |
| `PROFILE_REGISTRY.md` | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| `SOURCE_BINDING_REGISTER.md` | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| `pdf/README.md` | `f7a09d9954d1c4d34b8f2815dc8ec9d3c463039fdc3f1b4e2d537dfda921473f` |
| `src/common/notation.tex` | `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44` |
| `src/common/definitions.tex` | `30b86d7fc888b21d21794799f2ef5c77f869ca1a9506cb49b6349af27feabdd1` |
| `src/common/equations.tex` | `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e` |
| `src/common/assumptions.tex` | `e616b2c28ea2052b7a0af39a0ca5a320e0b2be95cbe65393d51408e696c16b9b` |
| `src/common/requirements.tex` | `8c518ea1ffba9142d70a5982ce6f403dcb462ecf7de772047bffc7d24bad99d6` |
| `src/common/edge_cases.tex` | `13381205d78f7b69b6e80f3705c9c74351ad2028bffa010dac4f2ae6ea7bb579` |
| `src/scientific-rationale.tex` | `a56d79357ee9758c74c7fc692646ba2d2bdb5ecef12bd3499410b06318a03b65` |
| `src/engineering-conformance.tex` | `6f5e5f6bd8aec2a2577b426b54dad1deb8bd761858e8f124d1f7f234199768fd` |
| `src/verify_contract.py` | `b11342d962ed2fd01e881f48cb36824ef0ca971b55107f80e333aca113270fdb` |
| `pdf/SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | `53e32a12ad4b60b4cccaaf05e1c0f9ad248d7e31637fbcfe2b4344992b81359c` |
| `pdf/SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | `e5b353d52303e7f9fd3d10abcd35a4a15eb24021eab4e0663244d414052232fa` |

## Claim Boundary

This freeze establishes document identity and scientific authority. It does
not establish implementation conformity, representation fidelity,
observational validation, achieved performance, production readiness, MAP
availability, coadd availability, or clean-room audit-finding closure.

`F-003`, `F-004`, `F-005`, `F-016`, and `F-020` remain open until WP-7 binds
the exact frozen sources and performs the authorized clean-room re-audit. MAP
remains deferred rather than silently completed by the VAL freeze.

## Change Control

The freeze promotion is status-only. Any future change to a Core definition,
equation, assumption, requirement, prediction, profile predicate, exception,
missing/conflict rule, source binding, policy owner, or claim boundary
requires explicit owner authority and a versioned successor or formally
reopened revision. Later implementation or validation evidence may attach
under its own identity without silently editing frozen r0.3 authority.
