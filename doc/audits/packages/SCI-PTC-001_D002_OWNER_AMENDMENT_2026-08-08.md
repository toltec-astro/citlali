# SCI-PTC-001 D002 owner amendment — 2026-08-08

Record ID: `SCI-PTC-001-D002-OWNER-AMENDMENT-2026-08-08`

Status: owner-approved successor authority; documentation and coordination
only

## Exact authority and preserved audit identity

The project owner approves this amendment for `SCI-PTC-001` at governing
application SHA `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

The completed audit remains immutable at final commit
`01ee247461d6c19bc4db81ccac4fec21af162c88`, parent/core
`66e8d6f98c3e22da74de4eea84e568a0b4cc6310`, and tree
`e6685c920ff37f1d4e51d27ecf23b73ac16087b5`. This successor record does not
rewrite or relabel the frozen independent core, report, evidence, ledger
proposal, auditor brief, or original handoff submission:

| Frozen artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex` | `82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2` |
| `doc/audits/packages/SCI-PTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6` |
| `doc/audits/evidence/SCI-PTC-001_LOCAL_EVIDENCE_2026-08-08.yaml` | `091059abd088b8bca58ca5a885e12620972c1f75f75574e33bfff8b0eb90b195` |
| `doc/audits/proposals/SCI-PTC-001_LEDGER_PROPOSAL_2026-08-08.yaml` | `8daabce7d0d585e82d233dadb3f535bb993a28c62acb351c4474272c525eee63` |
| `doc/audits/packages/SCI-PTC-001_OWNER_DECISION_BRIEF_2026-08-08.md` | `eaaf7bc06988dcb4bc1ae2a7da235aab4ac5ebfd96850fb580d850d5c82a2752` |
| Original auditor-submitted `SCI-VAL-001-XAUD-008` bytes | `fdeaa3d18909a35b3caff85257f70e7f51ae6115ec07d172cbe96fd1b5007a32` |

## Approved D002 contract

> Inputs known to be invalid for estimating the PTC model must be excluded
> before fitted-state arithmetic. A later result requires refitting or
> fitted-product invalidation only when its typed cause explicitly
> reclassifies that input as fit-invalid. Residual-based post-PCA sample
> rejection, clean-state detector-quality selection, and coefficient-based
> preference are downstream eligibility/weighting decisions; they directly
> exclude or downweight the selected output only and do not automatically
> invalidate the PCA state or other detectors.

The durable interface must preserve at least cause and decision-stage semantics
sufficient to distinguish `fit_invalid`, `postfit_output_reject`, and
`weight_only`. Exact representation and the engineering choice of refit versus
fitted-product invalidation for `fit_invalid` causes remain engineering-owned.

## Finding and reopen disposition

- `SCI-PTC-001-F001` remains an unchanged P0 `implementation_defect`.
- The original blanket `SCI-PTC-001-F002` P0 `implementation_defect`
  classification is withdrawn from canonical coordination state. F002 is a P1
  `contract_gap`: the current binary flag channel does not durably distinguish
  fit-admission invalidity from post-fit output rejection or weight-only
  noncontribution.
- Post-fit quality selection does not itself require PCA recomputation. Only a
  typed cause explicitly classified `fit_invalid` invokes refit or
  fitted-product invalidation.
- F001 remains a narrow Tier-A reopen trigger if its invalid-input admission
  contradiction cannot be repaired at the admission/interface boundary
  without changing the estimator. F002 is not an independent Tier-A numerical
  reopen trigger; it routes to bounded cause/stage/provenance contract work.
  The audit's separate consumer-dependent F005 response routing is unchanged.

After reclassification, the 13 open findings comprise three P0 implementation
defects, two P0 dependency gaps, and eight P1 findings. The package remains
`contract_status: proposed`, `implementation_status: nonconformant`,
`validation_status: in_progress`, `production_status: existing_use_only`, with
verdict `amend`.

## Supersession and routing

This record supersedes only the original broad D002 recommendation, F002
classification, F002 closure wording, and F002 Tier-A routing in the frozen
audit report, frozen ledger proposal, and frozen auditor brief. Those artifacts
remain immutable historical audit evidence. Canonical coordination state and
the coordinator brief apply this amendment. `SCI-VAL-001-XAUD-009` supersedes
`SCI-VAL-001-XAUD-008` for recipient routing while preserving the original
submission identity.

`SCI-PTC-001-D001` remains exactly as previously approved. D003--D006 remain
unresolved. This amendment does not authorize application, test, or
configuration changes; repair; evidence execution; Unity access; re-audit;
VAL/MAP/NOI/BEAM launch; production expansion; merge; or push.
