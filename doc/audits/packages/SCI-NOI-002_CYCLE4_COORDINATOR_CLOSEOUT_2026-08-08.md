# SCI-NOI-002 Cycle 4 coordinator closeout — 2026-08-08

## Decision

The coordinator accepts the independent Cycle 4 re-audit and applies its
machine-readable proposal within the exact bounded scope reviewed by the
project owner. The controlled package verdict is `retain`; exact application
candidate `5b29e13548a6fec884c67b192dec20c92f0bbb62` is `conformant`, and the
applicable deterministic validation is `complete`. Production remains
`existing_use_only`.

This closes the SCI-NOI-002 repair/re-audit chain, not the complete scientific
program surrounding every possible noise consumer. It does not integrate the
application candidate onto `codex/refactor-mainline`, recommend a realization
count or default, establish physical-noise variance, authorize significance or
catalog claims, close SCI-FLT-001 or SCI-FRUIT-001 work, or expand production.

## Verified identity

- application candidate: `5b29e13548a6fec884c67b192dec20c92f0bbb62`
- candidate parent: `390edf4f8c696551921c615f2439e956d240ec1d`
- candidate tree: `641c724f40a9fa9f322f09c703705239439d2374`
- repair branch: `codex/repair-sci-noi-002`
- independent re-audit commit: `6de648f5ae2b37f5bc65162feae221f19bb84a5a`
- independent re-audit parent: exact application candidate above
- independent re-audit tree: `102bdbbb01e6c3c1de3302a368e8567e0c07d91c`
- coordination integration commit for the three immutable audit artifacts:
  `823d0a1a42eca599a726f5d4b8b0bd03eb8c6e73`
- containing coordinator-closeout commit: returned out of band because this
  artifact cannot contain the hash of the commit that contains itself

The pushed remote-tracking audit ref and local audit worktree both resolved to
the exact independent re-audit commit before integration. The re-audit commit
contained exactly the three artifacts listed below and passed `git diff
--check`.

## Accepted immutable artifacts

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-NOI-002_CYCLE4_INDEPENDENT_REAUDIT_2026-08-07.md` | `aa03f3bc94fda349c4f1bcb16274ad9fb21195a3aef26a58383d12fce2af37e5` |
| `doc/audits/results/SCI-NOI-002_CYCLE4_REAUDIT_RESULT_2026-08-07.yaml` | `da6ed16e6a93fd10539bc001014818e99a45361cf0b2442ecc32738e3a2f6b8c` |
| `doc/audits/proposals/SCI-NOI-002_CYCLE4_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-07.yaml` | `a42180d881f4db56870a9ecd5221a4da92338686e9b2f1209486eba3fa2305f1` |

The two YAML artifacts parse as mappings and agree on the target, verdict,
status axes, finding dispositions, gate counts, and acyclic digest bindings.

## Applied finding disposition

Cycle 4 requirements C4-R001 through C4-R004 are satisfied. Cycle 3 findings
`SCI-NOI-002-C3RA-P1-001` through `SCI-NOI-002-C3RA-P1-003` are closed.
Original findings F003, F004, and F007 close within their bounded contracts;
F001, F002, and F008 retain their prior bounded closures. Repair findings
RA-B001 and RA-B003 close, while earlier RA-B002, RA-R001, and RA-R002 remain
closed.

F005 and RA-B004 remain `open_conditioned` with parity state
`scope_blocked_not_applicable_pending_FLT`, owned by SCI-FLT-001. F006 remains
open and `held_external`, owned by SCI-FRUIT-001. Decisions D001 through D008
remain settled without modification.

F004 closure establishes truthful descriptive/engineering identities and
restrictions only; it does not validate SCI-SRC-001 significance or catalog
claims. F007 closure establishes requested/effective/completed count
truthfulness for existing use only; it does not establish count adequacy or a
recommended default.

## Accepted validation scope

The accepted re-audit evidence includes four of four required Release build
targets; 40/40 focused core tests; 2/2 focused and 32/32 full science-product
tests; 88 baseline-auditor, 23 product-contract, 9 reduction-validator, and 7
science-change-ledger Python tests; 623/623 runnable CTests; and the complete
127-test configuration preflight with four mode kits, eight compact
compatibility cases, and no gap or required skip. The one disabled external
pointing-corpus replay is unrelated and is not an NOI failure or skip.

The paired exact C++/Python fixture matrix covers compact ECSV and ordered
NetCDF missingness, unscaled and scaled-coefficient-only successor coadds,
preserved standalone realization files, false empirical claims, split Beammap
zero/one/multiple logical maps, per-map identity and scopes, cardinality, and
negative shapes. No astronomical reduction was required or claimed for this
bounded Cycle 4 closure.

## Remaining gates

1. Any application-mainline integration of candidate `5b29e135...` requires a
   separate explicit authorization and integration review.
2. F005/RA-B004 remain with SCI-FLT-001; F006 remains with SCI-FRUIT-001.
3. Production expansion requires a separate owner/coordinator decision after
   applicable dependency and consumer work. Until then the package remains
   `existing_use_only`.
