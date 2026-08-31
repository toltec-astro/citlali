# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.9`

Status: exact 20-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `d4beb5ba6b568db9ecc489ba1995a4158f4c5c768d8e1e51d50294ce2a807702` |
| `SCOPE_BRIEF.md` | `7e6503ae699258d3b77390c71210dbea2e93c4c7966e756824d0fa301513bc3c` |
| `PRIOR_WORK.md` | `982f9a436d1b831c3434038ceba9b0b338ce721f0bef52fdccbd466e6fefa5b7` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `4f179130308cf96111087c3a2973315ae0bd8fe86ac62dbdfd98ccc679d806fb` |
| `FAMILY_SPLIT_MATRIX.md` | `e4a5e9a80946c6d75cbf0db97f73f85ffe541abe23a0c645f00b4a1f1f8edbf9` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `2b4e00693259219a75847ce8e7bf1174d8cc5b6c8b02924e633e728e8caa0b34` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `ba28ad99f680788611431d522c052ce86174e8eac62142994219b6d368f2973c` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `a0e5825fa564db88974802be1e42b220f829163745d87460872f5403051aef85` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `7f1042a2696e7001bd2cf86727dc71c5d6e54d69999edc55188d7bcda01ca390` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `a2a24992198ba6579ee8b63e3c0e38664fe73ad8c19a04f11cf8e82a48bd8e74` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md` | `41532b9803c81c2cdf6a8c4bbe39ea31a1cddaa53a3a5d7ad121d3ac051e5cd9` |
| `SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md` | `d3315e82aabb5aa8abb497284a4f3a3181e490df5604d1e4e4de83215a5385b6` |
| `SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-08-31.md` | `bf66140f9b0aeb0a6f35e61698f21dcdc7a1a1401fc5e49229bda6a29f02661b` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `a41e21c0bbfac980514b7e1aff5389ed7e60cb9563063cd046f65e3cd326285a` |
| `verify_stage_a.py` | `6a2a2c9f926458ec2bfc5ddf9e82f2823e9fa206b59500d4514435d435d6b445` |

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
