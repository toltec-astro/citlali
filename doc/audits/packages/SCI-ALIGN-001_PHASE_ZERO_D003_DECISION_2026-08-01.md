# SCI-ALIGN-001 phase-zero D003 owner decision — 2026-08-01

Status: `ALIGN-P0-D003` resolved for bounded existing-use-only repair with a
setup-only proportionality guard; `ALIGN-P0-D004` and `D005` pending; phase
one unauthorized

Package: `SCI-ALIGN-001`

## Authority and evidence boundary

The project owner approved the complete bounded `ALIGN-P0-D003` disposition,
while observing that parts may be overkill but are harmless. This record makes
that approval proportional: the required identity and lifecycle checks occur
at observation setup, and standard evidence is one compact record per
observation/interface. It does not authorize per-sample bookkeeping, a generic
timing framework, a drift model, or measurable hot-path cost.

This record is a separate amendment to, and does not rewrite:

- the immutable phase-zero evidence at repair/evidence commit
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`, whose exact application parent
  is `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- `REPORT.md` SHA-256
  `4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8`;
- `SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md`;
- the D001 restricted legacy timestamp decision at content commit
  `86434df2cfb5b85d0ccd306150cb428321abdbb9`; or
- the D002 detector sample-rate/lattice/support decision at content commit
  `10981b29c1870e745b7f3c9cabed3c634a46427f`.

D001 already governs native detector timestamp/counter construction,
malformed timing state, duplicate/decreasing/reset `PacketCount`, unexplained
counter reversal, collisions, and nonmonotonic reconstructed time. Ordinary
forward packet gaps route to `ALIGN-OD4`. D002 already governs the supported
native rate family, cross-interface rate agreement, cadence, lattice phase,
slot boundary, collision admission, and union support. D003 does not reopen
either decision.

The phase-zero evidence found that all 38 selected config/file associations
matched the raw scalar `Header.Toltec.RoachIndex`, but the runtime does not
enforce or persist that reconciliation. The current alignment path reads the
raw index and uses it for offset-map lookup; the current state is only a
`map<string,double>`. Pointing evidence contains requested/effective zero
offsets, while the accepted Beammap config requests zero but lacks complete
requested/effective provenance. No surveyed chain records
observation-resolved or realized application state. LMT offset state is not
modeled, and HWPR offset state can be effective while unused. Existing config
parsing already rejects unknown, duplicate, and nonfinite offset entries and
uses explicit zero for omitted authoring.

## ALIGN-P0-D003 — offset lifecycle and malformed boundaries

Questions: `Q05`, `Q17`

Decision: owner-approved with a compact four-stage offset lifecycle,
setup-time interface/header reconciliation, explicit optionality, and exact
preservation of conforming zero-offset timing.

### Approved bounded policy

1. Preserve the one-way offset authority chain already approved by
   `ALIGN-OD2`:

   - immutable supplied configuration owns `requested`;
   - typed Citlali configuration owns `effective`;
   - observation setup owns `observation_resolved`; and
   - ALIGN owns `realized/applied` evidence.

   Realized state never synchronizes back into effective or requested state.

2. Use one compact observation/interface record containing only the facts
   needed to prove the contract: interface identity and applicability, offset
   value in seconds, source, positive-add sign, detector-clock reference,
   application stage, uncertainty/bound availability, and whether it was
   applied exactly once. `unavailable` is recorded when no uncertainty or
   producer authority exists; it is not replaced by zero.

3. An omitted authoring value may resolve to typed zero with source
   `schema_default_zero`. For a present admitted interface, realize that value
   exactly once as a numerical no-op before ordering and slotting. An absent
   optional interface is `not_applicable`, not “zero applied.” An explicit
   nonzero value for an absent interface, or any nonzero value lacking the
   comparable clock/epoch authority required by OD2, fails closed rather than
   being ignored.

4. Represent the LMT/telescope timing relationship in the same compact
   observation lifecycle with `schema_default_zero` for the bounded repair.
   Do not add a new nonzero LMT authoring correction or claim a measured
   detector/LMT epoch relation. Nonzero LMT, TolTEC-network, or HWPR offsets
   remain unavailable until their required authority exists. HWPR that is
   absent or disabled remains `not_applicable`; a requested HWPR offset must be
   applied and evidenced or rejected.

5. At observation setup, reconcile every selected detector input's supplied
   `toltecN` identity with scalar raw `Header.Toltec.RoachIndex = N`, requiring
   `N` in `0..12`. Reject an unknown or conflicting identity, duplicate raw
   Roach ID, or multiple inputs claiming one interface within one alignment
   input set. A future ordered multi-file segmentation/concatenation contract
   may explicitly replace that last restriction; none is inferred here.

6. Bind the already approved D001/D002 malformed-input rules at ingestion.
   Each selected detector stream requires nonempty two-dimensional
   `Data.Toltec.Ts(time,6)`, the admitted D001 timing profile, and finite,
   positive scalar `FpgaFreq`, `SampleFreq`, and `AccumLen` selecting one common
   D002 rate-family member. A malformed selected stream fails observation
   setup; it is not silently dropped, relabeled, or assigned a default header.

7. Preserve optionality. An absent detector network and absent or disabled
   optional HWPR are nonfatal `not_applicable` states for admitted intensity
   processing. Missing or malformed required telescope pointing, or HWPR in a
   mode that explicitly requires it, fails the affected observation/product.
   D004 retains authority over exact telescope/HWPR field identities,
   topology, support, units, and output semantics.

8. Preserve conforming zero-offset behavior exactly: native reconstructed
   timestamps and D002 slot assignments must remain unchanged. These are
   ingestion checks and compact evidence, not a retiming, interpolation,
   correction, or drift model. No new rollover normalization is approved.
   Ordinary forward packet gaps continue through OD4.

### Proportionality and performance guard

The bounded implementation must be the smallest one that proves the policy:

- perform identity/header/lifecycle validation once during existing
  observation setup or input traversal;
- reuse existing config diagnostics and timestamp/rate checks rather than add
  duplicate full-data passes;
- persist at most one ordinary compact state record per observation/interface,
  with exception detail only when needed;
- add no standard per-sample/per-detector identity, validity, offset, or
  provenance arrays;
- add no hot-loop allocation, virtual dispatch, generalized clock service,
  synchronized-telemetry model, time-varying correction, or new public
  cross-cutting `Engine` state; and
- measure setup/runtime, I/O, and storage effects on representative Pointing
  and Beammap cases. There may be no repeatable measurable timing regression
  and no change to conforming source crossings, centroids, or PSF behavior. A
  regression, material burden, or dramatic timing departure is a stop
  condition for coordinator simplification under `ALIGN-C001`, without
  weakening fail-closed identity or authority semantics.

Focused fixtures need cover default zero, explicit zero, fractional nonzero
rejection without authority, absent optional interfaces, mismatched and
duplicate identities, malformed required shapes/headers, repeated-observation
reset, and exact zero-offset native-time/slot identity. They do not require a
new general timing subsystem or dense provenance product.

## Explicit non-approvals and remaining authority

This decision does not authorize a nonzero offset, determine a latency,
establish comparable producer epochs, add a drift/time-varying correction,
normalize a new rollover, define D004 telescope/HWPR field semantics, choose
D005 fixtures/tolerances, authorize phase one or application code, request
Unity evidence, launch re-audit, close an open finding, or expand production.

`ALIGN-P0-D003` is resolved only for design of the bounded existing-use-only
repair. `SCI-ALIGN-001-F001`, `F002`, `F006`, `F007`, `F009`, and `F012`
remain open pending complete implementation, validation, exact-repair-SHA
human evidence, and fresh re-audit.

## Remaining phase-zero decisions

- `ALIGN-P0-D004`: telescope/HWPR registry, aliases, topology, units, and
  output contract; and
- `ALIGN-P0-D005`: compatibility fixtures and preregistered tolerances,
  including the rate-stratified multi-Beammap study.

Until both are resolved and the active-field registry is reviewed, phase one,
application edits, Unity evidence, and re-audit remain unauthorized.
