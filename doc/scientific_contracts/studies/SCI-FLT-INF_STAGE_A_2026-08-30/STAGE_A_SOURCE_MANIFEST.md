# SCI-FLT-INF Stage A study source manifest

Manifest identity: `SCI-FLT-INF-STAGE-A-SOURCE-MANIFEST v0.1/r0.2`

Status: exact 13-object manager study packet; not a scientific author packet,
scope approval, Stage B release, or scientific freeze

## Exact study objects

| Object | SHA-256 |
| --- | --- |
| `README.md` | `e0b8569a34d804999e6f4a8316b13bfe6aef42e8885dab45dcbc9cd509a68140` |
| `SCOPE_BRIEF.md` | `d171839c6759899d5a77668bc822e1548779c4c2c449975de81b592123cd1c27` |
| `PRIOR_WORK.md` | `156bf6e3e4ab480350122c4c7fc5ce1d07aa027712c726fa0eb87365d43622f1` |
| `IMPLEMENTATION_INFORMED_DOSSIER.md` | `560fb017407f1e2a2d5b6687c064b322efbb2ed9b60ee1d768b94905c5c2f387` |
| `FAMILY_SPLIT_MATRIX.md` | `ec7dcac37df932c5b2a628f95806dab13bc9adf5b78ec2f83e1395ca8bb220df` |
| `OPERATOR_STATE_PRODUCT_TAXONOMY.md` | `54090e473751dc6707856082cb856b5a0f3a3cf8de924a97da2cd1c171254954` |
| `CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `f59ad173c646ffc36f5516fee726c1c9827c07a67ce9139cf85af6f866e8bb00` |
| `CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md` | `7f1db0e13868f492ff0d56072518add157091380a51b28fcd92a0c68709de328` |
| `PROPOSED_SANITIZED_AUTHOR_INPUTS.md` | `896645186f4961254974df969beec7eaf405dee162f459c9108f3be2a0d52cea` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `4565fde3582dfec5017a2a3b8e9884ceb8ef8253e181a64914a8e81942c8a141` |
| `SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md` | `ccb404ab20f2118b373d57734f837bfebd571fb77c58e6056cfee8cc1f7afebb` |
| `FROZEN_AUTHORITY_AND_SOURCE_BINDING.md` | `65a849b24ab45b4f9a7b7a094fc49f037ebe65e09f44b7eaa098884bb8a399df` |
| `verify_stage_a.py` | `1fc8d14d2efa04f66e4fe883d973b3e726959ce5e00e69dc1dcb1a207e729319` |

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
