# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.3`

Status: exact 14-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `1c5f7efbd09f53d6b18b138693723f277ce43589f896949fb935034b523da647` |
| `SCOPE_BRIEF.md` | `9b8b6d50a2a794d72a3e9e421a78ffb4aeecb9a4c3239e4d34a0107b035418bb` |
| `PRIOR_WORK.md` | `08a6cfa1b2fd9add7b9b18bafbf655dd80840f6a029ff58acdeb1732c09d29a9` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `933ac041f051d3d5afaefe9092d16cc0489919e7245c910c4debe56d6550e460` |
| `FAMILY_SPLIT_MATRIX.md` | `8ef20598a34ad8b9e24d43b873629753eb6ff0fc4eaa61fd5b95cd446f7134be` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `b104ebfd54ffc477e23350fc559933e50bc8bf96c5f51d2a93490d287879603c` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `692ec55057cc86bbf3dea8da801794af5e52aec6fd9d5d23268ce9af82f049c8` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `0ef669fdcf473f913e4c50a10f3d430a6e8603620416f6e5a52def6db649db3d` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `7420a367788b1b8493386af893617029731c709d18b4a8df07a3681dedc78085` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `79f2aaef5019e8187def12637fcffcddf7c86c868f3a39c8832562d43f2aa8af` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `5aa4d08ee658b94a772eaa6a380680e26f8be36917a82d7e08decfb65af62856` |
| `verify_stage_a.py` | `e3bee9ecb76a5c59300ec4a60c83c6e45e71183e85cbffce6e822dc006e8705b` |

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
