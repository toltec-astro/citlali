# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.6`

Status: exact 17-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `b1fa3c3d092973967eb45d5f8064c1ed8069a96ee350748fa2f11994d9df8fd8` |
| `SCOPE_BRIEF.md` | `521f7052336d60afe493d106ec85c9596b571b5a07e05e4178ae141e38d5a742` |
| `PRIOR_WORK.md` | `361eac527ff8c56b61cd85cf0115b0dfb40a46ecb2ea7772f742bbbcadb1df01` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `c09525c54520cd2395cdcc0de306438aaef84927a5199f0edec275215b282205` |
| `FAMILY_SPLIT_MATRIX.md` | `eac1e11a4abecf32845440a7e00642584fd98324afec3e891cce1714d157a83d` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `0f7b7a3e63ef4bb4fd4ef5e22997bb1eded1b09636378992b18e37b7d8db4fbb` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `2de3ebad9cb0658f2938d5080d004e9a88234ae44089b65186dd8ad1bc2d889f` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `f0e7451bb0e155ddc263a54422e6471a5b6ad226a6602c4aef355a9f0f79f508` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `33a53bc70091031e9164f1d636ae1c19e28ceacfc278654751cc095365454bdc` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `19a2d08c331e62b4ea738dce227fc233fa18fc74748fa959ffa76a1aee5a5ee8` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `c6d05df09b490ec7ea8b28b453773180b108c9892d56e0225530664b3f4e26e0` |
| `verify_stage_a.py` | `51f09caf3c19202fa4a1e3562a102241a4098124fed66a4e6a437d14c67a748e` |

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
