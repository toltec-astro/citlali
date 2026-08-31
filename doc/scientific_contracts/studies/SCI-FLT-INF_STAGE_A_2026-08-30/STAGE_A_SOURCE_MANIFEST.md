# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.7`

Status: exact 18-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `c0d66174d11d3495d3bb2d20c4211bea2cf5449f142df2d0ee921d84b56a905f` |
| `SCOPE_BRIEF.md` | `7a31a54dffdf77cc2b908af3d69014474453a9ef2620c9a9281d8670e691ded3` |
| `PRIOR_WORK.md` | `3198983ec09e9ad0f4239643db38dd79144c39650ea83e3f81ee3c98a7424b4e` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `d3da22a9b88255389dbb5f0bc8c1bcde9b77406b1d295add42ecd730a7f20c22` |
| `FAMILY_SPLIT_MATRIX.md` | `417182880fa5c43ff54a3ad828abe64b6e923bef4d7715f463eed558dffe3f07` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `30d6f1307dd90a35a3a0a3d97efff862532bb058678b90b48a6325a904b188a3` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `c1f88bfb7afeef5d3b129369ed1d8993c6065119549072d67029ccb220ec57b5` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `68bc8111536a7b4592ded4d23f3c6dc643c703d357e56736e8cecc4fa6ad6b6b` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `6309559f4db604241446f661671ac918e86b0b698c9dab8788a9a4e32c0d4095` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `dbcc8dec9d56afc796a97f9509af2fb059928b23ffc2826ce337a6b60debf1d8` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md` | `41532b9803c81c2cdf6a8c4bbe39ea31a1cddaa53a3a5d7ad121d3ac051e5cd9` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `5033e5dd86a7e07d736f68976237bb812f2b07b0c68cf42d539fbdadb8a5100c` |
| `verify_stage_a.py` | `dcd4ed6119a08979b2777c7b4962fc55b4127f6c0f4d540b9f1b663a661ba03a` |

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
