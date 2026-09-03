# SCI-VAL v0.1 — Content-Bound Author Packet Manifest

Status: r0.2 owner-approved and content-bound; fresh implementation-blind
Stage B Ultra dispatch authorized

Scientific owner: Grant Wilson

Prepared: `2026-08-20`

## Proposed Admitted Files

| File | Role | SHA-256 |
| --- | --- | --- |
| `SCOPE_BRIEF.md` | Owner-approved problem, boundary, questions, and exclusions | `98510ed385164ee2f3339284a3b15434da4821b85a43b19de1f9f691186594f9` |
| `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | Sanitized stable conventions and adjacent ownership | `32dc62160dff5dcb15e4af83d0df3311024494f30de075784603d4b4bfb4a52c` |
| `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md` | Sanitized exact RTC/CAL/PTC/MAP meanings and shared use vocabulary | `7296112f48fd1edc8eb4b4527883aad86b3dbade19509ab8268e9c6f8b7e4964` |
| `DECISION_LOG.md` | Owner-approved package-scope decisions | `29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60` |

These hashes bind the exact owner-approved bytes admitted to Stage B. A change
to any admitted file invalidates this manifest and requires renewed scope
approval and content binding.

## Explicit Exclusions

The packet shall not contain:

- `PRIOR_WORK.md` or `INTERNAL_DOSSIER.md`;
- any source code or implementation path;
- `validation/product_contracts.json` or another current product/config
  schema;
- historical `SCI-VAL-001-XAUD-*` handoffs or the audit ledger;
- audit reports, findings, repairs, re-audits, tests, reductions, validation
  evidence, Unity records, or production status;
- current adjacent-package drafts beyond the sanitized approved statements in
  the admitted packet; or
- unbounded external references.

## Dispatch Gate

The Stage B author must be a fresh GPT-5.6 Ultra task with no inherited
Citlali implementation context. The owner approved the revised scope, all
scope-level choices, the exact four-file packet content, and Ultra dispatch on
`2026-08-20`. The manager verified the packet firewall and exact hashes before
dispatch.
