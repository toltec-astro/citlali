# SCI-ALIGN-001 phase-zero D002 owner decision — 2026-08-01

Status: `ALIGN-P0-D002` resolved for bounded existing-use-only repair; a
combined-Beammap preregistration study is deferred to `ALIGN-P0-D005`;
`ALIGN-P0-D003` through `D005` pending; phase one unauthorized

Package: `SCI-ALIGN-001`

## Authority and evidence boundary

The project owner explicitly approved the coordinator's complete
`ALIGN-P0-D002` recommendation and directed that a combination of Beammap
reductions be used later to fine-tune the validation recommendation. This
record treats that later work as a preregistered, pre-candidate compatibility
study. It is not permission to tune an admission threshold after inspecting a
repair result.

The owner then corrected the cadence scope: `0.008192 s` is the cadence of the
surveyed 1x sample-rate profile, not the only supported detector lattice. The
approved legacy acquisition-rate family is 0.5x, 1x, 2x, and 4x the reference
rate `122.0703125 Hz`. D002 therefore derives cadence and the exclusive
half-cell boundary from the admitted observation's rate profile.

This record is a separate amendment to, and does not rewrite:

- the immutable phase-zero evidence at repair/evidence commit
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`, whose exact application parent
  is `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- `REPORT.md` SHA-256
  `4ac7c1bb9c67da3ce99ddfe4f96e42799a704bcb5acf89e3fa17cdfda1ef31c8`;
- `SCI-ALIGN-001_PHASE_ZERO_COORDINATOR_REVIEW_2026-08-01.md`; or
- the separately approved `ALIGN-P0-D001` legacy timestamp decision at
  content commit `86434df2cfb5b85d0ccd306150cb428321abdbb9`.

The phase-zero census surveyed 110 detector interfaces over eight Pointing and
two Beammap observations. Every surveyed interface in that evidence reported
`FpgaFreq = 256000000 Hz`, `AccumLen = 2097152`, and
`SampleFreq = 122.0703125 Hz`, consistently defining cadence
`dt = 0.008192 s`. The measured phase-zero corpus therefore establishes the 1x
compatibility case only. The 0.5x, 2x, and 4x rate profiles are project-owner
authority and require profile-specific fixtures/evidence under D005; they are
not inferred from the 1x residual distribution.

For accepted Pointing 152389 and Beammap 148670, all 4,305,356 ordinary rows
inside current support retain their current slot under the proposed common
operator. There are no exact or near-half ties, collisions, mask/numerical
placement disagreements, or current-test packet gaps. The maximum measured
absolute residual is `4.061937 ms`. Thirty-eight additional native edge rows
are excluded only by the current intersection support; union coverage adds
three grid positions to each accepted observation. Across the extended corpus,
all 9,001,641 ordinary rows retain their slots and 181 native edge rows expose
the same support issue.

For the measured 1x profile, the phase-zero `4.063 ms` proposal was derived
only `1.063 microseconds` above the observed maximum and is rejected as
data-tuned. Its half-cell boundary is `4.096 ms`, leaving
`34.063 microseconds` above the measured maximum. For every admitted rate,
strictly less than half that profile's cadence is the natural
unique-nearest-slot boundary.

## ALIGN-P0-D002 — cadence, phase, slot admission, and union support

Questions: `Q02`, `Q03`, `Q04`

Decision: owner-approved with a current-compatible phase, strict half-cell
admission, detector-support union, typed per-interface unavailability, and
compact ordinary representation.

### Approved bounded policy

1. Let the observation's native detector sample-rate factor be
   `r in {0.5, 1, 2, 4}` relative to `f_ref = 122.0703125 Hz`. Freeze the
   supported rate family and derived cadence as

   ```text
   SampleFreq = r * f_ref
   dt         = 1 / SampleFreq
              = AccumLen / FpgaFreq
              = 0.008192 s / r
   ```

   | Rate factor `r` | `SampleFreq` (Hz) | `dt` (ms) | Exclusive half-cell boundary (ms) |
   | ---: | ---: | ---: | ---: |
   | 0.5x | 61.03515625 | 16.384 | 8.192 |
   | 1x | 122.0703125 | 8.192 | 4.096 |
   | 2x | 244.140625 | 4.096 | 2.048 |
   | 4x | 488.28125 | 2.048 | 1.024 |

   The required `FpgaFreq`, `AccumLen`, and `SampleFreq` header facts must be
   present, finite, positive, mutually consistent, and select the same approved
   rate factor on every admitted detector interface in the observation. A
   mixed-rate observation, missing/conflicting headers, or a factor outside
   the approved set fails closed. At `FpgaFreq = 256000000 Hz`, the matching
   `AccumLen` values are respectively `4194304`, `2097152`, `1048576`, and
   `524288`. Absence of a separate multiplier is implicit 1x only when the
   required headers select the base profile; missing headers do not default to
   1x. No interpolation between profiles or arbitrary positive rate is
   admitted. Membership in this rate family does not by itself admit a new
   timestamp/schema or production profile; D001, D003, and `existing_use_only`
   restrictions still apply. This native ALIGN rate is distinct from later
   RTC processing/downsampling.

2. After D001 native timestamp reconstruction and exactly-once application of
   any D003-admitted interface offsets, define the compatibility lattice phase
   as the latest first valid detector timestamp among admitted interfaces.
   This preserves the current ordinary mapping. The phase is a numerical
   lattice anchor only; it is not the union-support start, a producer epoch, an
   integration-event selection, or a new absolute-timing claim.

3. Define the common lattice for integer `k`, including negative edge indices,
   by

   ```text
   grid_time(k) = phase + k * dt
   q             = (native_time_after_offset - phase) / dt
   slot          = floor(q + 0.5)
   residual      = native_time_after_offset - grid_time(slot)
   ```

   Use this one round-half-up operator for admission masks, value placement,
   identity, tests, and provenance. No endpoint clamp or second nearest-grid
   operator is permitted.

4. Admit a detector row only when

   ```text
   abs(residual) < dt / 2
   ```

   Exact half-cell ties, outside-boundary rows, and multiple native rows from
   one interface assigned to the same slot fail closed before conditioning.
   The exclusive boundary is respectively `8.192`, `4.096`, `2.048`, or
   `1.024 ms` for the approved 0.5x, 1x, 2x, or 4x profile. D001 and D003
   malformed/monotonicity policies remain additional gates. The rejected
   `4.063 ms` data-tuned threshold is not an implementation target for the 1x
   profile and is not scaled into the other profiles.

5. The common detector acquisition support is the union of valid admitted
   per-interface support, not the intersection of their start/end times.
   Preserve leading and trailing native rows on their integer slots. At a
   union edge where an interface has no acquired row, represent that
   interface/slot as typed `unavailable`; do not clamp an endpoint,
   extrapolate, synthesize, or grant science weight or exposure. Internal
   acquisition gaps remain governed separately by `ALIGN-OD4`.

6. Keep the standard representation compact. Persist the lattice identity,
   phase, cadence, integer extent/count, assignment rule and exclusive
   half-cell boundary, plus
   per-interface edge-unavailability intervals/counts and required aggregate
   support. Ordinary row-to-slot mapping is generative. Routine dense
   per-sample identifiers are not required; expanded mappings remain
   `as_requested` under `ALIGN-C001`.

7. Phase-one fixtures must cover all four approved rate factors, exact
   coincidences, ordinary jitter, strict inequality on both sides of each
   profile's half-cell tie, negative and late edge slots, per-interface
   unavailability, collisions, mixed-rate interfaces, inconsistent cadence
   headers, and a proof that every ordinary accepted 1x row keeps its current
   slot. Each additional admitted production profile requires its own
   ordinary-row identity evidence. Fixtures must distinguish native ALIGN
   acquisition rate from later RTC downsampling and detector support from
   telescope, HWPR, scan, chunk, and output support.

## Deferred combined-Beammap validation study

Test identifier: `ALIGN-D002-BEAMMAP-VALIDATION-001`

The owner directs a later test using a combination of Beammap reductions to
fine-tune the compatibility recommendation. This test is assigned as an input
to unresolved `ALIGN-P0-D005` and must be designed before candidate successor
results are inspected.

The D005 record must freeze the selected observations/reductions, exact run and
artifact identities/digests, native sample-rate factors, array coverage,
admissible detector/fit flags, metrics, aggregation, and thresholds. The cohort
should cover all four approved rate factors where suitable historical or
owner-supplied Beammap evidence exists; unavailable rate coverage must remain
explicit and cannot be treated as a successful required test. An
owner-approved native-rate alternate fixture may satisfy a missing stratum;
otherwise that profile remains evidence-pending. Where the required artifacts
exist, the study should combine multiple reductions to measure:

- the cross-observation lattice-residual distribution and edge-support
  behavior, in both seconds and normalized cell units
  `abs(residual) / dt`;
- source-crossing timing or closest-approach consistency;
- centroid repeatability; and
- per-array major/minor PSF-width repeatability.

A resampled 1x observation is not native-rate evidence for a missing 0.5x, 2x,
or 4x stratum. Synthetic family fixtures may establish operator algebra but
not observational source-crossing, centroid, or PSF compatibility.

Use accepted historical/current results and their fit uncertainties to set a
baseline without inspecting a repair candidate. The study may support a
proposal for a tighter rate-profile-specific compatibility guard or validation
tolerance. It cannot loosen any profile's unique-nearest-slot half-cell
boundary, silently move the approved phase, establish producer/absolute-time
semantics, or change D002 automatically. Any scientific-policy change requires
an explicit owner amendment recorded before candidate inspection, followed by
the corresponding validation and re-audit scope.

## Explicit non-approvals and remaining authority

This decision does not resolve D003 offset lifecycle or its broader malformed
boundaries; approve the telescope/HWPR registry; choose D005 fixtures or
numerical compatibility tolerances; authorize phase one or application code;
request Unity evidence; launch re-audit; close any open finding; or expand
production.

`ALIGN-P0-D002` is resolved only for design of the bounded existing-use-only
repair. `SCI-ALIGN-001-F002`, `F007`, and `F012` remain open pending complete
implementation, local validation, exact-repair-SHA human evidence, and fresh
re-audit; D002's union-support and availability consequences also constrain
open `SCI-ALIGN-001-F009`.

## Remaining phase-zero decisions

- `ALIGN-P0-D003`: offset lifecycle and malformed-boundary policy;
- `ALIGN-P0-D004`: telescope/HWPR registry, aliases, topology, units, and
  output contract; and
- `ALIGN-P0-D005`: compatibility fixtures and preregistered tolerances,
  including `ALIGN-D002-BEAMMAP-VALIDATION-001`.

Until all three are resolved and the active-field registry is reviewed, phase
one, application edits, Unity evidence, and re-audit remain unauthorized.
