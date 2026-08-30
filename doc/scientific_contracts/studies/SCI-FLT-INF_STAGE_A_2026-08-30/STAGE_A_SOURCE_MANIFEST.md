# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.1`

Status: exact 12-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `7a3fd41bd7de727feb55adff14b85b39f0bc5ca4250485db7099cfc8f21874f4` |
| `SCOPE_BRIEF.md` | `3fb5ae2af310ad48e9a74a38575099032d68b2238941af6f1268b647fab97591` |
| `PRIOR_WORK.md` | `9292e80d6645ab3913bb6acbef373aa3e7fc94a749b73f6672de34e4aeaf40ab` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `53c72c131f8bb6052957fe4b966279d4bbc09effee15bc0b17bc647df60a324c` |
| `FAMILY_SPLIT_MATRIX.md` | `869b94c08e439c1506e8ca3a3d45b8f2a5a9d05c0326a7d03051c9f0f7d835b7` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `98361f841de1c96884c4eb2bfc19b0cc8648fc05eb7d87b01cf3dc445711c049` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `6340ce46c5a6816023f55d802ea96319692285e163cc875edd19c8bd8c6a1d2c` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `db1c25579cd31cdfc5710c714981308a6bde19e93de0d888db8ec6986d2e054f` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `bcf087cf0071d931afe671d49f9a7c49626c7866964d9b75953be290fee1cfc4` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `18f32bd504cc71dea57d0329d088d913ea490b45983b5ba89355cda55fca7582` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `8a0455119ead0d0cfc44540d7acb9acdb585a4c90c7306996f381c1c5ac112b1` |
| `verify_stage_a.py` | `d05bfb32e3da1c3c9a05b00c9acc7c7658823f4d3fecac5f11098bfd43229cf7` |

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
