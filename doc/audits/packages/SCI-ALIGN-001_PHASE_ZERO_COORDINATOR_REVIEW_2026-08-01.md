# SCI-ALIGN-001 Phase-Zero Coordinator Review

Date: 2026-08-01
Package: `SCI-ALIGN-001`
Disposition: verified evidence; owner authority required
Phase-one authorization: none

The project owner explicitly authorized this ordinary AST/ALIGN return
integration on 2026-08-01. That authorization does not approve the held
composition-framework decisions, a framework amendment, or the closure pilot.

## Reviewed identity and scope

- Evidence branch: `codex/repair-sci-align-001`.
- Evidence commit: `53c7154a3633dfe19dc036cfb5a6250f729a897d`.
- Exact parent and selected application base:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Evidence root: `validation/sci_align_001_phase0_2026-08-01`.
- Evidence verdict: `STOP_FOR_OWNER_AUTHORITY`.

The branch was clean. The commit contains 26 diagnostic and evidence files
and no application-source edit. This review records the evidence by exact
commit and digest; it does not import the large raw evidence bundle into the
coordination line.

## Integrity

- `REPORT.md` SHA-256:
  `4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8`.
- `owner_questions.json` SHA-256:
  `6744ae1310b69f454b42fd9c9472b7e772ca9cfa14b39d112333081246f78669`.
- `SHA256SUMS` SHA-256:
  `074aff9deddd062d13a055589714f5d1b52ee18753052286119a184d2dbc08a2`.

All 22 entries in the digest manifest verify. All 45 source-manifest files and
all five exact coordination-source identities independently match. The
evidence binds the corrected owner-decision commit
`4f905f4f353e91847a303f4f3959654f3f03c302` and rejects the earlier mistyped
identity.

## Evidence accepted for phase-zero use

- Every surveyed detector interface supports candidate cadence
  `0.008192 s` through the recorded FPGA/accumulation metadata.
- Under the current-compatible phase and zero offsets, all 4,305,356 ordinary
  accepted rows retain their current slots. Thirty-eight edge-only rows expose
  the need for an explicit support rule.
- The proposed `4.063 ms` tolerance is data-tuned: it is only
  `1.063 microseconds` above the observed maximum and is not accepted.
- A generated registry exposes the surveyed telescope/HWPR field identities,
  topology candidates, units, and missing authority instead of treating all
  variables as generic angular scalars.
- Current interpolation invents 153 Pointing and 7,664 Beammap `PpsTime`
  values.
- Treating `Hold` as step state rather than scalar interpolation changes
  3,329 of 383,699 Beammap rows, about 0.8676 percent, and can change scan
  boundaries.
- Historical Pointing and Beammap products provide compatibility baselines,
  but not repeatability distributions or a successor comparison.

## Explicit limitations

The evidence does not establish detector counter epochs, widths, rollover,
modulus, sentinels, or the present detector anchor rule. It does not establish
the physical event/time scale represented by `TelUtc`, integration start
versus midpoint, nonzero latency magnitudes, complete telescope-field support,
enabled-HWPR schema, individual `Hold` bit meanings, or numerical astrometric
compatibility thresholds. No successor reduction exists.

## Owner-decision groups

### ALIGN-P0-D001 — detector timestamp and anchor authority (`Q01`, `Q16`)

Obtain producer authority for the six `Data.Toltec.Ts` fields and
`PacketCount`, including epoch, logical widths, modulus/rollover, sentinel
policy, and the detector anchor. Do not bless the current subtract-one-half
and truncation construction or choose between `2^32` and `2^32-1` from the
observed corpus alone.

### ALIGN-P0-D002 — lattice, slot tolerance, and support (`Q02`-`Q04`)

Recommended owner policy: accept candidate cadence `0.008192 s`; use the
latest first realized detector timestamp after offsets as a compatibility
lattice phase while keeping phase distinct from support start; use union
detector support with per-interface unavailability; and require strict
`abs(residual) < dt/2`, with exact half ties failing closed. Reject the
data-tuned `4.063 ms` proposal.

### ALIGN-P0-D003 — offset lifecycle and malformed boundaries (`Q05`, `Q17`)

Recommended owner policy: immutable supplied config owns requested offsets,
typed Citlali config owns effective state, observation setup owns resolved
state, and ALIGN owns realized/application evidence. Default zero is explicit;
nonzero needs authority; absent optional interfaces are `not_applicable`.
Fail closed on malformed identity, timestamps, headers, unauthorized rollover,
collisions, or nonmonotonic reconstructed time; normalize only an authorized
rollover before monotonicity checks.

### ALIGN-P0-D004 — HWPR/telescope registry and output contract
(`Q06`-`Q12`, `Q14`)

Producer authority remains necessary for enabled HWPR, individual `Hold` bits,
and exact telescope time/field identities. Recommended bounded policy while
that authority is absent: nonpolarimetric intensity processing may proceed
with enabled HWPR unavailable; preserve the raw `Hold` word and derive
`hold_active = raw_word != 0` with provisional left-continuous step placement;
preserve accepted `SourceRaAct/SourceDecAct` and use legacy `TelRaAct/TelDecAct`
only as schema-versioned aliases; fail closed if both disagree; preserve true
units, shapes, bitmasks, and vectors; and use exact-only, zero-span behavior as
the fail-closed registry default until an owner-reviewed allowlist exists.

### ALIGN-P0-D005 — fixtures and preregistered tolerances (`Q13`, `Q15`)

Recommended owner policy: make local Pointing observation 152389 and Beammap
observation 148670 mandatory fixtures, followed later by human-run
exact-repair-SHA Unity evidence. Do not select numerical compatibility
tolerances yet; first run a read-only preregistration analysis using repeat
observations and fit uncertainties without inspecting candidate successor
results.

## Composition-framework consequence

The ALIGN and AST return trigger is satisfied, but the evidence supports only
a partial future disposition of `FRAMEWORK-COMP-D005`. Clock sign and once-only
application, union support, origin/validity/support/uncertainty propagation,
zero direct science weight for synthesized samples, fail-closed defaults, and
metric names are ready for owner consideration. `TelUtc` identity,
start/midpoint, anchor/rollover, latency magnitude, slot tolerance, and all
numerical sky/registration/source-crossing/centroid/PSF thresholds remain held.
`FRAMEWORK-COMP-D006` also remains unapproved in this review.

## State and next gate

The approved ALIGN contract, `nonconformant` implementation,
`in_progress` validation, `existing_use_only` production status, `amend`
verdict, and required re-audit are unchanged. Resolve `ALIGN-P0-D001` through
`ALIGN-P0-D005` and record the reviewed active-field registry before phase-one
fixtures or any application-code edit. Unity evidence and re-audit remain
unstarted.
