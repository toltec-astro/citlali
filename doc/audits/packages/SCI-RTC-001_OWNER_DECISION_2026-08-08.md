# SCI-RTC-001 owner decision — 2026-08-08

Record ID: `SCI-RTC-001-OWNER-DECISION-2026-08-08`

Application authority: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`

Audit authority: `3319d7424c732c1c9fc300c336e4d428e6f91068`
(parent `3620434eb988662210b2466ee357ffc8f891aa58`, tree
`8b01ee4d8117816904d7f078682c0c62a2ea88ac`)

Disposition: owner decisions D001--D004 approved on 2026-08-08; audit package
accepted into coordination with `contract_status: proposed`,
`implementation_status: nonconformant`, `validation_status: in_progress`,
`production_status: existing_use_only`, and `verdict: amend`.

## Approved decisions

### `SCI-RTC-001-D001` — replacement, synthesis, and influence eligibility

Any signal corrupted by replacement or synthesis, and any downstream signal
influenced by it, is ineligible for scientific analysis. The implementation
must use compact influence/support bookkeeping sufficient to enforce that
rule and preserve causes; it is not required to emit bulky per-sample
provenance products.

### `SCI-RTC-001-D002` — complete conditioned response

Every response-changing RTC stage must be represented in the kernel or
realized local response. If the complete response cannot be represented
truthfully and economically, response is unavailable. A partial kernel must
never be presented as the complete conditioned response.

### `SCI-RTC-001-D003` — immutable stage identities

Outer, inner, full, mini, diagnostic, simulated, and processed RTC products
have distinct immutable stage identities and explicit parent/processing
links. This is metadata and provenance authority; it does not require
duplicate computation.

### `SCI-RTC-001-D004` — bounded filter and multirate semantics

Serialize exact filter and multirate semantics at full precision once per
coherent observation or processing segment, not per sample. Preserve the
existing timing lattice and supported approximately 8-ms sample-rate
relationships (`0.5x`, `2x`, and `4x`). This decision does not authorize DSP
redesign or expensive new computation.

## Effect on findings and axes

The four policy questions are resolved for future repair scope. Finding
`SCI-RTC-001-F012` remains open until the approved rules are implemented in an
exact successor and independently re-audited. Findings F001--F011 and the
ALIGN, CAL, and AST dependencies remain open. Nothing in this decision changes
the audit's four status axes, verdict, production allowlist, or fail-closed
restrictions, and it does not authorize repair, validation execution,
re-audit, or production change.

## PTC sequencing interpretation

The conservative gate sequence in the audit brief governs acceptance or
closure of the RTC-to-PTC handoff and any scientific or production use of its
claims. It does not prevent a fresh PTC auditor from freezing an independent
mathematical core against the exact application snapshot while all RTC and
ALIGN observations remain quarantined as `post_core_evidence`.

Accordingly, a frozen `SCI-PTC-001` independent-core audit dispatch may be
prepared and may later be launched by the owner/coordinator. Launch does not
acknowledge, accept, close, or dispose `SCI-PTC-001-XAUD-005`; it does not
authorize RTC or PTC repair; and it does not change any consumer or production
status. The PTC auditor must stop for the three scope checkpoints and for
coordinator/owner review after the audit.

## Exact audit artifacts accepted into coordination

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex` | `d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7` |
| `doc/audits/packages/SCI-RTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `169a0b6e013e727d5d04c25da5234e29def4ef79bc6f25a3f343bc675000301a` |
| `doc/audits/evidence/SCI-RTC-001_LOCAL_EVIDENCE_2026-08-08.yaml` | `1cb733544d6b9d6decb1794279d2e81249b00375071c01ee71f514135dcf6394` |
| `doc/audits/proposals/SCI-RTC-001_LEDGER_PROPOSAL_2026-08-08.yaml` | `3783fa2c61a9fd280ea3bc6ed2b4514c4a16418daebfab6474493b54f5616e88` |
| `doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001-XAUD-005.yaml` | `89ce799a6526f87025bc1d57cc0bf8a0234b149374920027beaa4cb18b94e289` |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-007.yaml` | `2c0a47c9f7f4eed4820db133304108f8f52d9702c22762b19e09cb851cce14b0` |
| `doc/audits/handoffs/SCI-BEAM-001/SCI-BEAM-001-XAUD-003.yaml` | `24cd271c5b27d9885c01002e063571a53aa7fd4517d34509ef8d2194e259e1a1` |
| `doc/audits/packages/SCI-RTC-001_OWNER_DECISION_BRIEF_2026-08-08.md` | `0565ad21ddf5113ee3fcee70ec8a6545d92c8b7073fb81d0b5a19d30a2f58e89` |

The listed handoff digests are the immutable auditor-submitted bytes at audit
commit `3319d742...`. The canonical registry may add coordinator-owned commit
identity and status/disposition fields; any such current canonical digest is
recorded separately in the ledger and dispatch manifest.

## Non-authorizations

This record does not authorize application changes, RTC/PTC repair,
re-audit, local Citlali reductions, Unity access, external contact, broad or
costly execution, another downstream audit, BEAM launch, or production change.
