# WP-7.1 Timestream Successor Program

## Purpose

This program implements the scientifically closed WP-7.1 timestream contract
on the canonical Citlali application lineage. It replaces the accidental
equation of “the branch with the most WP-7 work” and “the application
authority” with an explicit, gated integration path.

The program is not a release and does not rename the whole future Citlali
application. It owns only the WP-7.1 timestream successor work.

## Exact Starting State

| Role | Exact identity | Disposition |
| --- | --- | --- |
| Canonical application base | `cb3d568c701217ee0248c77f6dccd0bab7deef31` on `codex/refactor-mainline` | Sole ancestry base for the repaired program |
| Native Integration Baseline | `f0f423827ab321640e0cbcb003f7bf015368f694` | Integrated predecessor; not WP-7.1-conformant |
| WP-7.1 successor source | `170ecea9de1ee810da7d7e45a489a4545ccd623d` | Exact scientific source packet |
| WP-7.1 scientific closure | `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa` | Contract-closed for the approved bounded scope |
| Divergent implementation/evidence head | `49fe73e757daa1885cd23127e8441cba47e648d2` on `codex/wp7-rtc-fixed-decimation-authority` | Preserved evidence and replay source; not application authority |
| Canonical governance lane | `codex/wp7-governance-reconciliation` | G0--G3 package accepted by the owner on 2026-08-31 and promoted through canonical ancestry |
| First G4 replay lane | `codex/wp7-g4-replay-001` | `WP7-REPLAY-001` accepted and integrated at exact `f8ba732bc4072e918c2521a013305be354ed7b53`; no active successor unit remains |

The divergent lane is not rebased, force-moved, or discarded. Its commits,
tests, tools, and evidence remain available for reviewed replay. Uncommitted
work present when the governance hold was issued remains in that worktree and
is not part of any immutable program identity.

## Authority Order

When sources disagree, use this order:

1. explicit scientific-owner decisions incorporated into the WP-7.1
   scientific contract;
2. the exact scientific source and closure commits above;
3. accepted post-closure owner corrections listed in
   `validation/wp7_timestream_successor_authority.json`;
4. canonical Citlali architecture, scientific conventions, and ADRs;
5. reviewed implementation evidence;
6. historical branch behavior.

Implementation convenience cannot reopen a scientific decision. Reopening
requires an actual contradiction, inability to perform the intended science,
or new scientific evidence that the approved decision is wrong.

## Reconciliation Gates

### G0 — Canonical ancestry

- The program branch descends from exact canonical
  `cb3d568c701217ee0248c77f6dccd0bab7deef31`.
- The divergent implementation branch remains preserved and is never treated
  as a second application mainline.
- Existing dirty worktrees, contract-authoring state, refs, and tags are left
  unchanged.

### G1 — Scientific authority binding

- The machine-readable authority manifest binds the exact source packet,
  closure record, locked independent outputs, retained scope limits, and every
  accepted post-closure owner correction needed by implementation.
- The binding distinguishes scientific closure, application conformance,
  validation evidence, and production authorization.
- No absolute workstation path is an artifact identity.

### G2 — ADR reconciliation

- Canonical ADR numbers are allocated prospectively after existing canonical
  ADR 0016.
- Historical divergent WP-7 ADRs 0014--0020 map to canonical ADRs 0017--0023.
- Each canonical record preserves its original path and source commit as
  provenance; the historical numbers are never reused as canonical numbers.

### G3 — Workstream and WIP control

- This governance lane is the only active WP-7 application integration lane.
- Further successor application implementation remains held until G0--G2 and
  the canonical status/ledger update are committed and reviewed.
- Scientific-contract authoring continues independently under its package
  index. Contract authoring does not silently modify the application tree.
- Mapspace contract work may proceed in its own packages while timestream
  application implementation is held.

G0--G3 were accepted by the project owner on 2026-08-31 at governance package
commits `e874044c4c562fe672890495a3f4d5064e789d8f` and
`28e9e559b7e74d13e05427c54b13c89e9a6c6f1b`. The owner explicitly released
the application hold and resumed the program at G4.

### G4 — Reviewed replay

After G0--G3 close, replay divergent work in bounded units. Each unit must:

1. name the exact source commit and changed paths;
2. state the scientific authority it implements;
3. preserve canonical build, application, and validation behavior not in its
   scope;
4. renumber or update references to canonical ADRs;
5. run focused, CTest, configuration, baseline-tool, and affected-mode gates
   required by the behavior touched; and
6. stop before the next unit if evidence or authority is incomplete.

The D2 PSD/line evidence tooling at divergent head `49fe73e757...` is preserved
as a candidate replay unit. D2 itself remains open until a reviewed
network-native in-memory prefilter/residual producer exists. Filter design is
not authorized merely because evidence tooling exists.

## Current Status

The program is in G4 reviewed replay. `WP7-REPLAY-001`, sourced from exact
divergent commit `49fe73e757...`, was owner-accepted and integrated at exact
canonical commit `f8ba732bc...`. No WP-7 application work order is now active.
The preserved uncommitted producer prototype is outside that source identity
and remains unauthorized. No WP-7.1 application-conformance, same-SHA
validation, readiness, release, or production claim is made. The next bounded
unit requires explicit owner review before work begins. The status of the
predecessor application is summarized in
[`APPLICATION_BASELINES.md`](APPLICATION_BASELINES.md); current actions remain
governed by [`REFACTOR_STATUS.md`](REFACTOR_STATUS.md) and
[`INTEGRATION_LEDGER.md`](INTEGRATION_LEDGER.md).
