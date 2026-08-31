# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.4`

Status: exact 15-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `5176027e4fdcfef77e83ef4c6b68d9e6d36400fab05e8553a67157c620d4d390` |
| `SCOPE_BRIEF.md` | `ce20edc871dab783fdcdd050d256b7bfbdca4162adf0ae701e2e5e198d5a5d20` |
| `PRIOR_WORK.md` | `6f5b0f183ae0b7110b3ac2ddf2650198ebc82b177ef6d6d2f569233a5674464c` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `2ec0d7b80e569bef1f9c3e9bb559762d0aad3b883d137d6e24a4f501d5f1b2e2` |
| `FAMILY_SPLIT_MATRIX.md` | `afdf238bfda5aaccf0b188c3de03fa185997ea3f5d77fa3514955c7d534d79f7` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `00e39499c174476a54c9a0552d0ac602e6b788b91a74e2dc4593f569f1c1480b` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `1f7303354da03848367ca08bdd33580dca48e2ffea6399b58591138b5a2b80ba` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `9818ac4d6b19ef1475eae094b0afb8908201fff690976fda17f934b9fea07e9c` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `470a15ba41efc42bf9ca324b068b354e45407c94fe089dac0fcfdaf278a200d9` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `2b2e242b628b28fc70f2e9cc3ffef5d10e320394d8ad28c6b76989cc97531c85` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `e65669c2af6960c168400b70a5eccd7d3ec3e5f1818873b0352b1a8df6fbe259` |
| `verify_stage_a.py` | `2730c006a6ca72fbe5191a1c8372eb79749d508601c91237aa54507e31047147` |

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
