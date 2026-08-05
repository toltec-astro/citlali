# Package provenance

## Immutable authorities

- Repair candidate: `ed28dafb37f9113c0d3c95297148157129a90886`
- Candidate tree: `cf75c36557178f351fb62781108a6f4b41b19225`
- Candidate parent: `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- Original SCI-MAP Unity protocol authority:
  `a9ce0da5ae54164c8f7dbe6062d13649259cc76c`
- Original protocol SHA-256:
  `6c8decef93f5607bc9e8dfc84e31aee67f45fa5c695fc80563c7e7064f78d556`
- Bounded repair handoff SHA-256:
  `c02d9ba0b8bb2d3d59c117affacb06016b34ed0b1f63c69f8e2b6f415f2019fd`
- SCI-ALIGN-001 owner-decision record:
  `4f905f4f353e91847a303f4f3959654f3f03c302`
- Canonical owner-decision identity-correction commit:
  `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`
- SCI-ALIGN-001 bounded handoff coordination commit:
  `0309fd48a973a6e7e136224906ac49c02f0171be`
- Clean coordination-ledger HEAD:
  `846128c8ee6dc27851bd6c71aeecbe4739e1d24a`
- Selected SCI-ALIGN-001 repair base:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- SCI-MAP-001 project-owner amendment commit:
  `6409a36d324072c9b29145c620d01a0686275870`
- SCI-MAP-001 project-owner amendment SHA-256:
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`

The audit protocol and repair handoff bytes were checked against the pinned
coordination commit while preparing this package. The application candidate
was not changed. No audit or coordination commit was merged or cherry-picked.

## Local source authorities

The package reads the repaired successor profiles and contracts from the exact
candidate checkout. `campaign.json` pins each governing file's SHA-256. It
also pins the active TolProj config kit `phase4.1-v2.1`, whose source marker is
`f59c663f83b6f087fc09762e11c3bc74bc713740`. The kit source marker and repair
binary SHA are intentionally distinct facts.

The local TolProj operational sources used to verify setup semantics were clean
at `74395c824860ca41410dde5cf2e0272e5535fc19`. The reviewed TolTECA operational
implementation was `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`; a deployed
campaign must independently record its installed TolProj/TolTECA identities
and prove the same runtime semantics.

## Original-protocol source-count correction

The original appendix listed eight reduction sources and said no ninth source
was permitted. Its list omitted TolTECA-owned `40_setup.yaml`. TolProj
documents that `40_setup.yaml` remains active, and TolTECA recognizes every
numbered YAML. A fresh native refactor directory therefore has nine recognized
sources: `40_setup.yaml` plus the eight post-setup mode/launcher sources.

This package does not delete or ignore native setup authority. It requires
nine total, exactly eight after `40_setup.yaml`, exact deployed RuntimeContext
order, deterministic merged-value agreement, and no tenth source.

## Preparation boundary

This package was prepared and tested locally on 2026-08-01. No Unity host was
accessed or queried, no job was submitted, no external output was read, no
re-audit was launched, and nothing was pushed. `SHA256SUMS` records the final
local package bytes. That digest manifest is package identity, not scientific
evidence.

## Coordination state

SCI-MAP-001 implementation remains nonconformant pending independent re-audit.
F009 and F010 remain `addressed_pending_reaudit`. The owner amendment accepts
F012 only for the bounded external product/execution/SEQ-OMP claims in the
returned seven-case corpus; missing operational, internal-reconstruction, and
same-case S-X observation-realization lanes remain explicit limitations. No
Unity rerun is required solely for those absences. The analyzer's prospective
typed-Stokes assertion is corrected to governing typed index `0`; this changes
no application bytes, FITS physical-code rule, or historical corpus.

F013 remains conditioned on the named upstream audits. ALIGN-OD1 through
ALIGN-OD8 and ALIGN-C001 are owner-approved, and the dedicated phase-0 repair
task is active, but no ALIGN application-repair commit or re-audit exists.
ALIGN implementation remains nonconformant, validation is in progress, and
production remains `existing_use_only`. MAP F013 remains conditioned until the
ALIGN repair, exact-repair-SHA evidence, and fresh re-audit succeed. No MAP
campaign result closes ALIGN, CAL, AST, PTC, or VAL.
