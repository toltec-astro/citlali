# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.8`

Status: exact 19-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `b004195cbd42617673755243fad0384a2f944b4b63c977eb86cdf285d13af3ca` |
| `SCOPE_BRIEF.md` | `3f0b512d810decd48fa3a127f7600866eba728bfa87bde19b2978bee1073ef67` |
| `PRIOR_WORK.md` | `65c1bbec568d4fe08d05863326959f04b44f17361342daa65e3dc48b4f534ab5` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `e85f3673dd3ba2b5a04947e14baa5c3522f9db2001829d532dda0a5836a66125` |
| `FAMILY_SPLIT_MATRIX.md` | `5ecd70a847760bb04fb03f4bdd479335b18406c5d61edc53953bdd72b0b149bc` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `cbe99bbd99eec25228e9aba2cfa551be376181a82fa5b48769a526293edced27` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `cf37ba246a24400e29e0b7661093e2aebb1782002b1c422a79dde4920717bc75` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `3615af4ed93622b9abcc2c6eab018513112c90f85e96ab528c24aa1904841f67` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `4a9e71218a9fd315f3ef6c00b7d9294358d865eb7b16fb4d5ae55508794941ea` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `f228b28947c31fd5343e513d8f7dff77b8deb7f3c301bb101201ac50dca0b8cc` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md` | `41532b9803c81c2cdf6a8c4bbe39ea31a1cddaa53a3a5d7ad121d3ac051e5cd9` |
| `SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md` | `d3315e82aabb5aa8abb497284a4f3a3181e490df5604d1e4e4de83215a5385b6` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `4db1d2fd81a7f74db3efb876cec92ca1edbc242354b7e4c4e390edabc09f7624` |
| `verify_stage_a.py` | `7dc9508eb32361e91a9074630b5885e039656bcfe13fbf1673fa65d0ce7f6880` |

This table is exhaustive for the manager study packet. The external pointer in
`STAGE_A_SOURCE_MANIFEST.sha256` binds this manifest. Manager index/status
updates are not study-packet objects.

## Firewall

The manifest intentionally includes the quarantined dossier and therefore
cannot be supplied wholesale to an implementation-blind author. A future
package requires a new exclusive author manifest containing only owner-
approved sanitized inputs. Nothing in this manifest authorizes inspecting any
active SCI-FLT-FIXED Stage B material.

## Nonclaims

Content binding establishes only which study bytes were returned for owner
walkthrough. It makes no implementation, conformity, validation, calibration,
uncertainty, significance, performance, readiness, production, Unity,
scientific-authority, or freeze claim.
