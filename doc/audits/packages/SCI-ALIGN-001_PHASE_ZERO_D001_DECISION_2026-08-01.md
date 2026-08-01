# SCI-ALIGN-001 phase-zero D001 owner decision — 2026-08-01

Status: `ALIGN-P0-D001` resolved for bounded existing-use-only repair;
producer semantics unresolved; `ALIGN-P0-D002` through `D005` pending; phase
one unauthorized

Package: `SCI-ALIGN-001`

## Authority and evidence boundary

The project owner explicitly approved the bounded `ALIGN-P0-D001`
disposition after the coordinator's read-only timing review. This record is a
separate amendment to, and does not rewrite, the immutable phase-zero evidence
at:

- repair/evidence commit
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`;
- exact application parent
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- `REPORT.md` SHA-256
  `4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8`;
  and
- `SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md`.

No versioned firmware, packet-format, or NetCDF-writer contract was found for
the six positional `Data.Toltec.Ts` fields or `PacketCount`. The raw NetCDF
schema proves `int32(time, 6)` storage and a positional `long_name`, but not
logical signedness or width, epoch/time scale, modulus, rollover, reset,
sentinel, or integration-event semantics. TolTECA simulation code, `kidscpp`,
Citlali 4.x, and the refactor are downstream interpretations, not independent
producer authorities.

The supplemental read-only check of local observation 152390 covered
1,666,908 nominal detector rows across 11 interfaces. Under the exact current
reconstruction, `RecvTime - detector_time` was always positive, with median
approximately `0.235 ms`; removing the subtract-half/truncation anchor placed
every reconstructed timestamp approximately `0.830` to `0.999936 s` after
recorded receipt. This strongly supports compatibility preservation but does
not identify integration start, midpoint, end, FPGA capture, packet formation,
or another physical event. Likewise, the measured one-tick `2^32` versus
`2^32-1` sensitivity is compatibility evidence, not producer authority.

## ALIGN-P0-D001 — detector timestamp, counter, and anchor authority

Questions: `Q01`, `Q16`

Decision: owner-resolved by restricted `legacy_inferred` compatibility for the
exact previously accepted nonpolarimetric profiles. The adapter is not a
producer-authoritative clock and cannot support a new absolute-timing claim.

### Approved bounded policy

1. Preserve the exact native detector-time construction for the admitted
   legacy profile:

   ```text
   anchor = int(Ts[0,0] + Ts[0,5] * 1e-9 - 0.5)
   detector_time[j] = anchor + Ts[j,1]
                    + legacy_delta(Ts[j,2], Ts[j,4]) / FpgaFreq
   ```

   `legacy_delta` preserves the current conditional, within-row
   `ClockCount < PpsTime` adjustment using `2^32 - 1`. C++ `int` truncation,
   the subtraction, and the delta are named/versioned compatibility
   arithmetic, not authoritative rounding, epoch conversion, or modulus.

2. Keep this native-coordinate adapter separate from the common-slot
   assignment contract. It does not choose or alter `ALIGN-OD1`'s shared
   round-half-up slot operator or the D002 tolerance/support decision.

3. Keep interface offsets separate. `legacy_inferred` does not authorize a
   nonzero offset or comparable-epoch claim. `ALIGN-OD2` still requires one
   typed positive-add offset applied exactly once after native-coordinate
   construction, with nonzero values requiring authority.

4. Persist the truth about the adapter:

   - semantics source: `legacy_inferred`;
   - producer clock/epoch authority: unavailable;
   - integration-event anchor: unavailable;
   - logical counter width/signedness/modulus: producer-unverified;
   - absolute-timing precision claim: unavailable; and
   - admitted production scope: exact previously accepted nonpolarimetric
     profiles under `existing_use_only`.

5. Do not replace `2^32 - 1` with `2^32` in this bounded repair. A future
   versioned producer contract may authorize an intentional one-tick
   correction, with a named science-change record and fresh validation. The
   observed absence of slot changes is not authority to make that change now.

6. Fail closed at the affected interface or product boundary for malformed or
   empty required detector timing state, incompatible shape/type, invalid
   required clock headers, duplicate or decreasing/reset `PacketCount`, other
   unexplained row-sequence counter reversal, collisions, or non-finite,
   non-unique, or nonmonotonic reconstructed time. The exact named within-row
   `legacy_delta` correction is not a row-sequence reset; no other undeclared
   rollover is silently normalized.

7. An ordinary forward `PacketCount` gap is a typed acquisition-gap fact and
   routes to the already approved `ALIGN-OD4` network/chunk policy. It is not
   globally rejected, silently repaired by the timestamp adapter, or treated
   as an absolute-timing claim.

8. New timing/schema profiles, new consumers or production expansion, changed
   arithmetic, or any stronger absolute/event/precision claim remain
   fail-closed until a versioned producer ICD and converter define the six
   fields and `PacketCount`, including time scale/epoch, logical widths and
   signedness, exact modulus, reset/rollover/sentinel behavior, packet-count
   scope, and sample event.

9. Phase-one fixtures must preserve ordinary native-row-to-slot identity and
   established Point/Beammap source-crossing, centroid, and PSF behavior. They
   must also cover the exact legacy anchor, malformed input, duplicate/reset/
   backward row-sequence counters, both rollover candidates as explicit test
   hypotheses, monotonicity failure, observation replacement, and provenance
   that refuses a producer or absolute-timing claim.

### Explicit non-approvals and remaining authority

This decision does not establish that the legacy anchor is physically correct;
select an integration start, midpoint, or end; approve `2^32 - 1` as the true
hardware modulus; authorize a new profile or timing precision claim; decide
D002's phase/tolerance/support; decide D003's offset lifecycle or broader
interface/Roach/header boundaries; close `SCI-ALIGN-001-F002`, `F007`, or
`F012`; approve phase one; modify application code; request Unity evidence;
launch re-audit; or expand production.

`ALIGN-P0-D001` is resolved only for design of the bounded existing-use-only
repair. The producer-authority gap remains an explicit limitation and future
trigger. `SCI-ALIGN-001-F007` and `F012` remain open pending scoped
implementation, complete validation, exact-repair-SHA human evidence, and
fresh re-audit. D001 settles only the detector-counter subset overlapping
`Q17`; `ALIGN-P0-D003` remains pending for offset lifecycle and all broader
malformed-boundary policy.

## Remaining phase-zero decisions

- `ALIGN-P0-D002`: lattice phase, slot tolerance, and union support;
- `ALIGN-P0-D003`: offset lifecycle and malformed-boundary policy;
- `ALIGN-P0-D004`: telescope/HWPR registry, aliases, topology, units, and
  output contract; and
- `ALIGN-P0-D005`: compatibility fixtures and preregistered tolerances.

Until all four are resolved and the active-field registry is reviewed, phase
one, application edits, Unity evidence, and re-audit remain unauthorized.
