# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.5`

Status: exact 16-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `fc50f8570564ea618656120852dbcca2b22d1aa0c54de7dd3c1961268bf642f4` |
| `SCOPE_BRIEF.md` | `3d4830459f772e74133f0f8fa49de767496640b5462e0bd222fbcf460b3bcdd0` |
| `PRIOR_WORK.md` | `f359ea7724d61a0ac9bcab1e9ef4f8214434921d70f59d915bdd87a36c1f8181` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `167a023a7871415aa2e663b117ec02e687d2bcc5fefbbec81c5811109c9c3dc0` |
| `FAMILY_SPLIT_MATRIX.md` | `f0b7b1bc63c657e91280caaaf7c98677157bc2a795051217fcf81ea393061172` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `b155e73c3bba315a7db8bda9e5d4ee10ee8178092c0cb056a7dab9b3d0e7c222` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `2cbd49dd9147cd8a14b0b98f631ef2bf592329f935e7d9efbf9ecf173cc285e4` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `0cbc330c63fb9e630f0fe233dbf9fdc9e52716b1ee3682a00b774f0dc797a0e7` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `321a5bb42b4a4330c1ec4a890a01efb5592b0081ae058a936353716d1940451c` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `00953ed882d992904fdc9ae653a4265fbc7e58945235f77d7cd017316ad80459` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `25ffd2f7b5c14ecc2cb5fd06ffbbd80e9d9f9ece800d95a8dd4726c0bcb05c2f` |
| `verify_stage_a.py` | `2e021d0749c0f73cff5d1efa2f4450639bcc7a8fd5c7bf27ea28748a81813bad` |

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
