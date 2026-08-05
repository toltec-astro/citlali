# SCI-MAP-001 Application Integration Decision — 2026-08-05

## Decision

The coordinator-directed application-integration task authorizes the accepted
bounded `SCI-MAP-001` candidate for application-mainline integration. This is
an application integration and status-alignment decision, not a new audit,
repair, campaign, upstream-production decision, or production release.

The dedicated `codex/integrate-sci-map-001` branch was created from exact
`codex/refactor-mainline` source
`9aae0e669384c5c0c0dda93debc194d6b8dac787` and fast-forwarded only to
accepted repair tip `af0c849ce59a5f80e5efc8db435bb6662863052f`.
Before this record and the two live ledgers were edited, the branch tree was
exactly the accepted application tree
`47aa745554e47514398e72d579625484abdcb79e`. The commit containing this record
is a later documentation-only child of `af0c849`; it is the integration
candidate tip, not a different application-source revision.

## Verified authorities

- Base, local application mainline, origin-tracking mainline, and pushed
  mainline: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Accepted repair branch, origin-tracking branch, and pushed tip:
  `af0c849ce59a5f80e5efc8db435bb6662863052f`.
- Merge-base and direct ancestor: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Exact linear series: `ed28dafb3`, `1b824f138`, `02b9eb303`, `f84b9fd7d`,
  `af0c849ce`.
- Forbidden convolve/noise candidate
  `02a198cbfb379eaf6ab279c5a3d44ee73ff90435`: absent from repair ancestry.
- Final independent re-audit, including matching pushed branch:
  `8fc716557ca78b0d220200a92be46fa3545797e9`.
- Final canonical coordination candidate, matching its pushed continuation
  branch and pushed `codex/scientific-audit-framework`:
  `c7bb0214edfd57fddf31165923f08784dfd1b8c9`.
- Byte-identical owner amendment SHA-256:
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`.

The final audit report, decision, and ledger proposal are byte-identical to
their canonical coordination copies at SHA-256
`48c0e389ff0cee6003b9dba157839601d4d2abfa7e9e3f95ae4e6a617d753703`,
`96d06b8b9e918922ac0752d0f1ffad35586fde1631f499986ef03f64e36629f4`,
and `15d34205de19f67c9919f3db1be25c5d209286505abca11f1956ae23a0c15e57`.

## Accepted disposition and limits

The bounded MAP contract is approved, conformant, and validated within the
local plus owner-accepted evidence scope. F001--F011 are closed. F012 is
`closed_bounded_owner_accepted` for the named exact-`ed28dafb` external claims,
with these limitations retained:

- no independent raw manifest or sample ledger;
- no scan-farm pre-normalization planes or commit-order trace;
- no wrapper/Slurm/environment/collection/retrieval chain; and
- no same-case S-X observation-realization files in the historical corpus.

F013 remains `open_conditioned` on `SCI-ALIGN-001`, `SCI-CAL-001`,
`SCI-AST-001`, `SCI-PTC-001`, and `SCI-VAL-001`. Production remains
`existing_use_only`; MAP acceptance closes none of those dependencies and
does not establish upstream production eligibility.

## Integration boundary and owner handoff

The five-commit scope is accepted as already audited: MAP implementation and
tests, frozen campaign and closeout, owner amendment, and governing records.
After `af0c849`, this candidate changes only this handoff,
`doc/REFACTOR_STATUS.md`, and `doc/INTEGRATION_LEDGER.md`. It changes no
application source, test, configuration, numerical behavior, product
contract, validation registry, campaign artifact, or audit/coordination
record.

All proportionate local application-integration gates are required to pass at
the committed documentation-only candidate, including build targets, complete
enabled CTest, focused MAP truth/TSan/FITS, baseline and config suites,
validation/science-change ledgers, product/provenance/package checks, artifact
digests, and final Git identity/diff/cleanliness checks. The exact containing
commit and gate results are reported with the owner handoff because a commit
cannot embed its own Git object ID.

`codex/refactor-mainline` is deliberately not moved by this task. After review,
the owner may push `codex/integrate-sci-map-001` and fast-forward mainline to
that exact tip. No push, Unity action, production expansion, or Conan-lane
update is part of this candidate preparation.
