# SCI-MAP-001 Unity evidence-design owner decision — 2026-08-02

Status: `MAP-UNITY-ED1` owner-approved; a bounded successor evidence protocol
and validation-only producer are authorized; Unity execution remains blocked

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Decision ID: `MAP-UNITY-ED1`

Authority: project owner

## Decision

After reviewing the fixed observation/reduction matrix and the resource cost of
the frozen campaign's exhaustive actual-data term ledgers, the project owner
approved setting up the coordinator-recommended bounded successor protocol.

The decision intentionally replaces exhaustive retention of every actual-data
primitive term with deterministic actual-data trace coverage, streaming
digests, and compact sufficient statistics. It retains exhaustive local F011
primitive-semantics coverage. This is an explicit evidence-coverage tradeoff,
not an additive claim and not a scientific-estimator change.

## Execution and acceptance gates that remain unchanged

- Exact application candidate:
  `ed28dafb37f9113c0d3c95297148157129a90886`.
- Exact observations: Point 152389 and ordered Science 152390, 152392.
- Exact arrays: a1100, a1400, and a2000.
- Exact cases: `P-SEQ`, `P-OMP`, `S-C-SEQ`, `S-C-OMP`, `S-E-SEQ`,
  `S-E-OMP`, and repaired-success `S-X-SEQ`.
- Complete output inventories, all F010 facts and aliases, full WCS and
  centered-coadd relationships, 64 pinned realizations, provenance,
  support-floor characterization, and seq/OpenMP comparisons remain required.
- The local F011 truth suite remains the exhaustive authority for primitive
  equations, finite-state behavior, identity, atomicity, and exact scan-farm
  policy.
- External numerical comparisons retain `atol=2e-8` and `rtol=1e-10` as
  regression bounds only, plus the registered WCS bound.

## Authorized successor work

The dedicated task may create a new sibling campaign revision and validation-
only evidence producer on a clean worktree starting from campaign-preparation
commit `1b824f138754eeb1856ae5f102027db4b31598be`. The existing frozen package
directory must remain byte-for-byte unchanged.

The successor may:

1. generate the Point and Science raw-input manifests automatically;
2. stream digest-bound primitive identities and compact per-scan/per-pixel
   sufficient statistics without retaining the full term population;
3. retain a deterministic, preregistered actual-data trace spanning every
   active network, preselected scans, and valid and flagged detector states
   where present;
4. provide focused trace expansion for a discrepancy or named re-auditor
   request;
5. update the successor schemas, analyzer, verifier, runbook, launch checklist,
   provenance, result collection, and synthetic fixtures; and
6. measure and report output cardinality, storage, and preparation/analysis
   cost before any human launch.

Stored evidence must be structurally bounded by scans, map pixels, and a fixed
trace budget rather than by the full detector-by-sample term population. The
owner must not hand-author manifests, ledgers, or primitive traces.

## Stop and non-authorization boundary

This decision does not authorize:

- editing the MAP estimator or any application source;
- modifying, deleting, or relabeling the existing frozen campaign;
- weakening the seven-case execution or product-level gates;
- deriving independent authority from final products alone;
- contacting or querying Unity, transferring files, building there, or
  submitting jobs;
- filling the 22-field owner-values record yet;
- integrating the repair candidate;
- supplying external evidence or closing any finding or dependency;
- launching the MAP re-audit; or
- expanding production use.

If the compact design cannot preserve the named claims without application
changes, unbounded storage, or another scientific/operational tradeoff, the
dedicated task must stop and return a bounded owner decision brief.

## Next gate

The successor task must return an exact commit and tree, package and manifest
digests, proof that application source remains identical to the repair
candidate, local verifier/self-check results, measured resource estimates, and
an independent review. The coordinator must review that handback before asking
the owner for Unity operational values or authorizing human launch.
