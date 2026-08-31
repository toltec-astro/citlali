# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.10`

Status: exact 21-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `d70873561f5e6c408fd64dc0ddc6e92827e29a5b023b324efbf64c7b9a7dcd34` |
| `SCOPE_BRIEF.md` | `f5e3002bce3d0b635a40ccd995d19ab3c37d2b56b12012be0505e3dde13373f2` |
| `PRIOR_WORK.md` | `fb2a4322a937a5ee4abf59c4bd69d905b5001d5a2e8d78fc899cf5e787bf29cc` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `1f33c3e804994787f5d0e74018a9ca87de2dba02165f0831f8f08408db8552f6` |
| `FAMILY_SPLIT_MATRIX.md` | `a364fa5bd7f1831e7805d46306c32528abd99e8706fda81f6eeca3e5b9bd72b5` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `772633e2338c564372d4749c3df6067acb4657c053cdd93780ccbedb02211957` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `af862ffb29f690a94945fa6122e6858492a99aca5c8caae66e9963f5740a6929` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `e62319be466638e6eec61f14aed8a74e5f0ea072b3201d369f040a82221124a2` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `4882275ba8f9d52ce5a0acf08d99cb32ba46207334e902c8be80007cf60cbb3a` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `bbcae3582eb7db058d1681a6be85e895aff251a1f544d49cfacab1f33e70dc16` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md` | `be31439c9f1f4ab8335cc869cec7ffb379a19263f1ddddbc4bcec96bb71ad29f` |
| `SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md` | `865d6ac6113947598144a9f3e1c80ca24ca95f4aca98431387129d59d2e671ce` |
| `SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md` | `50babc956fba692562fb92e7177f48ca146d59266eb174b51d5b6412cc953a4b` |
| `SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md` | `bf372755a82d4d51f64b7d0b3ae3dfdf3bf13de23b6db1575f70958ed4092df8` |
| `SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md` | `41532b9803c81c2cdf6a8c4bbe39ea31a1cddaa53a3a5d7ad121d3ac051e5cd9` |
| `SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md` | `d3315e82aabb5aa8abb497284a4f3a3181e490df5604d1e4e4de83215a5385b6` |
| `SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-08-31.md` | `bf66140f9b0aeb0a6f35e61698f21dcdc7a1a1401fc5e49229bda6a29f02661b` |
| `SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-08-31.md` | `5c151f87601b52798bf90453aa02184d95f1d2585657f62d8527799649b53959` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `e74e0d7be084e69d4fda6df7c26e16811ff76dae5ff743cf89aaecd27a814bcf` |
| `verify_stage_a.py` | `7717d9138b72509eb619be16d622970ab58aabcc1bfb5e7a4379cf959bbc7f7c` |

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
