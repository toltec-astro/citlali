# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.11`

Status: exact 26-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `136de7be7392e78ca09147a425a30a6ae60ce9878d97746c8730cca4401b351b` |
| `SCOPE_BRIEF.md` | `9fb81d3fe5d460610f58bedaaa2d298080abfacdba572f844879851416b5d426` |
| `PRIOR_WORK.md` | `bf7911fe3b49999af95663d2d10eef80fe9e787cb6471134fab1f44b8e001e24` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `2d4267b8c06ca531feed2ba5b781375bc2d0e091352b1fff50abf1de7b3d24e0` |
| `FAMILY_SPLIT_MATRIX.md` | `fcf00f4c6c29b35e4c49e424cdf0884eedc777875d1f62ce1231c858e3bbf594` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `afa9816cd281e18f07e18d5a7dcfca77af50f1d9e7dde119f7ad97ddfcd6b522` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `5495835cacaaf3a77773ab05790913c4b3c813ccf5083b62e1d84b9d6fabd829` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `18cdf4c0b1a2698bb6219f0f5ad10370a670dde9f00f6575b27f791a4c7d3006` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `237e68f17af348b7bc43856444c39238529f97d3727be1fe021d662bb59f88f9` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `85f5df6a6c22196260368913c30cfc9d77382cd22e8420771945ac2f49a70ee0` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md` | `41532b9803c81c2cdf6a8c4bbe39ea31a1cddaa53a3a5d7ad121d3ac051e5cd9` |
| `SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md` | `d3315e82aabb5aa8abb497284a4f3a3181e490df5604d1e4e4de83215a5385b6` |
| `SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-08-31.md` | `bf66140f9b0aeb0a6f35e61698f21dcdc7a1a1401fc5e49229bda6a29f02661b` |
| `SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-08-31.md` | `5c151f87601b52798bf90453aa02184d95f1d2585657f62d8527799649b53959` |
| `SCIENTIFIC_OWNER_ODQ_010_APPROVAL_2026-08-31.md` | `62b30c7ea7af54c6755261a7a2605e401764253328ec0b690825698c1ca7597c` |
| `SCIENTIFIC_OWNER_ODQ_011_APPROVAL_2026-08-31.md` | `825fcefd6fbfa517db292091ec6a57f992081059c98a3245fe9daff4613afc80` |
| `SCIENTIFIC_OWNER_ODQ_012_APPROVAL_2026-08-31.md` | `cac1dfb5446802a46c616f0ecc2a65cb5a97dbaacd2ba69f36119ec8817420c2` |
| `SCIENTIFIC_OWNER_ODQ_013_APPROVAL_2026-08-31.md` | `b045611cabc5b0301104a9804f587ea9f3fc12efd7cd2df0fec7a5947f277b9d` |
| `SCIENTIFIC_OWNER_PACKAGE_IDENTITY_APPROVAL_2026-08-31.md` | `94c97c34752c072d76fb599d0f62a8172ea186c02308b5c548fbcf14d7039655` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `26590fc928a558f548c97c833add95fa1474bd6f9d8c1d1cc1e058e33724db82` |
| `verify_stage_a.py` | `3551df1f4a0634310eaae371bfdfa56551542ee69ffc9a84c52efc3e08c70c1c` |

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
