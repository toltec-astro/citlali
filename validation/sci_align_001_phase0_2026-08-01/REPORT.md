# SCI-ALIGN-001 mandatory phase-0 evidence report

Date: 2026-08-01
Verdict: **STOP_FOR_OWNER_AUTHORITY**
Scope: evidence only; no application source, T01--T18 implementation fixture,
Unity state, production policy, audit, re-audit, merge, rebase, or push was changed.

## Identity and authority

- The app-supplied worktree is
  `/Users/gwilson/.codex/worktrees/9c82/citlali-refactor`, from the same repository
  as `/Users/gwilson/GitHub/citlali-refactor`.
- Gate-time state was detached, clean, and exactly
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`; `codex/refactor-mainline`
  resolved to the same commit.
- No `codex/repair-sci-align-001` branch or other ALIGN repair worktree existed.
  Only `codex/repair-sci-align-001` was created, in the supplied worktree, at the
  exact base. Audit, coordination, MAP, CAL, convolve, and noise worktrees/refs
  remained independent and were not modified.
- The coordination gate was clean at corrected frozen dispatch commit
  `846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. Record commit
  `0309fd48a973a6e7e136224906ac49c02f0171be` is its ancestor. The handoff blob
  at that commit has required SHA-256
  `2231e09c4310e8ddf73b6e25cd52c3c10671234667607b88d3723571dfa7a5f8`.
- The authoritative owner-decision commit is
  `4f905f4f353e91847a303f4f3959654f3f03c302`; reject the mistyped expansion
  `4f905f4f39461c8f9a86b0bf589880362d0a49f7`. Correction commit
  `35cc8ce246e8e70c569e650be6c1eae2c91b80ef` corrects that identity and records
  this phase-zero task active; it changes no SCI-ALIGN scientific policy. Later
  coordination activity is excluded from the frozen gate.

Exact authorities and digests are in `git_isolation_evidence.json`,
`identity_and_isolation.json`, `source_manifest.json`,
`coordination_source_manifest.json`, and `coordination_correction.json`.
The historical pre-branch observation is explicitly separated from current
reflog/live assertions and digest-bound by gate snapshot SHA-256
`75b085b8f7bfea3af7dbdc579a1efb8ce17423080ea873d74647c945d0519481`.

## Frozen local input census

The read-only census is a path-sorted snapshot under
`/Users/gwilson/work_toltec/local_data`: 67 non-recomputed telescope paths
(51 unique contents) and 31 HWPR paths (23 unique contents). Full-file, schema,
and variable-registry digests are in `raw_input_manifest.csv`. The exact path
set is part of the evidence; a later local-data addition is a new snapshot, not
the same phase-0 run.

`boundary_stream_inventory.csv` records each file's primary coordinate,
interface, units/epoch status, sample count, cadence, bounds, duration,
finite-value result, and policy status. The per-file/per-field
`telescope_hwpr_timing_field_inventory.csv` has 2,029 rows and separately
measures every recognized telescope/HWPR time, clock, counter, cadence, and
acquisition-header field. It does not flatten HWPR packed block matrices into a
fictitious one-dimensional cadence.

The detector corpus comprises eight Pointing observations plus owner-local
Beammap 152307 and accepted Beammap 148670, with 11 detector interfaces each.
`detector_timing_inventory.csv` has 5,170 rows: 47 exact boundary facts per
interface. These comprise two requested config facts, every one of the 37
exposed `Header.*` fields (4,070 per-file header rows), SampleType, six
`Data.Toltec.Ts` columns, and `RecvTime`. The common sorted header registry
SHA-256 is `4082642c7571af87cbcefbcfbe52cb64e3204e45d9d5ca78323f5ef010172c47`.
Header/config scope is kept distinct from sample-stream cadence/support.
Accepted observations route to
`application_input_identity_inventory.csv`; extended observations explicitly
record that no observation-specific config was supplied. The 22 accepted Pointing/Beammap
detector files have full-file hashes in `selected_detector_input_manifest.csv`;
every extended input has a content-sensitive timing projection hash.

Only 10/37 detector headers declare units, 12/37 declare a long name, and none
declares fill/missing/validity attributes. All measured numeric header elements
are finite, but epoch, logical width, enum/bitmask validity, array-shape support,
and missing-value policy remain unproved where the producer metadata is silent.

The three selected configs contribute 38 application-input rows: 27 resolve by
the exact owner-local mirror rule and 11 by a unique basename fallback. All 38
requested interfaces agree with raw `Header.Toltec.RoachIndex` identity; there
are no duplicate requested interfaces, conflicts, ambiguous contents, or
unavailable local inputs. The current runtime does not enforce or persist that
reconciliation, so malformed/duplicate handling remains unproved.

The telescope/HWPR registry has 383 stable rows: 337 LMT and 46 HWPR. Candidate
scientific classes are 18 continuous scalars, 20 circular angles, 3 declared
half-open step/state fields, and 342 exact-only fields. Because period/wrap,
frame, validity, transition, epoch, and support authority is incomplete, every
active operator is fail-closed exact-only with zero support. Exact native
validity-attribute values and variants—not merely attribute names—are retained.

The surveyed telescope schema contains 50 data and 284 header names. Twenty of
22 configured data names occur (`TelRaAct` and `TelDecAct` are absent); 184 of
185 configured headers occur (`Header.Sim.Jobkey` is absent). Thirty data fields
and 95 headers are raw-available but unconsumed.

All 31 local HWPR files report `Header.Toltec.HwpInstalled=0`. Their common raw
schema has 14 data and 28 header names, but none contains application-required
`Data.Hwp.`, `Data.Hwp.Ts`, or `Data.Hwp.Uts`. Raw event fields are packed block
matrices with no proved valid-count/order/angle conversion. Provisional packet
coordinate reconstructions are nonmonotonic or duplicate-bearing in all 31
files, so no HWPR epoch, cadence, support, or angle contract can be derived.

## Detector grid and realized mapping comparison

The current detector formula anchors
`int(Ts[0,0] + Ts[0,5]*1e-9 - 0.5)`, adds `PpsCount`, and adds
`(ClockCount-PpsTime)/FpgaFreq`; a negative difference is adjusted by
`(2^32-1)/FpgaFreq`. Headers consistently give `FpgaFreq=256 MHz`,
`AccumLen=2,097,152`, `SampleFreq=122.0703125 Hz`, and cadence `0.008192 s`.
Epoch, anchor rounding, logical widths, and modulus remain unproved.

Accepted Pointing and Beammap configs explicitly enable `interp_over_gaps`, so
the compared realized path is the gap grid. The evidence models both existing
operators: `std::round` mask placement and `lower_bound` nearest-grid numeric
placement (exact numeric ties choose left), on an explicit LinSpaced-equivalent
grid. It compares both with one shared `floor(q+0.5)` round-half-up operator.

| Corpus | Native rows | Current-support rows | Edge-only rows | Shared-operator changes | Mask/numeric disagreements | Max absolute residual |
|---|---:|---:|---:|---:|---:|---:|
| Accepted Pointing 152389 | 84,689 | 84,667 | 22 | 0 | 0 | 4.061937 ms |
| Accepted Beammap 148670 | 4,220,705 | 4,220,689 | 16 | 0 | 0 | 3.918648 ms |
| Accepted pair | 4,305,394 | 4,305,356 | 38 | 0 | 0 | 4.061937 ms |
| Extended local corpus | 9,001,822 | 9,001,641 | 181 | 0 | 0 | 4.061937 ms |

There are no exact-half ties, 1e-12 near-half ties, slot collisions, ordinary
current-support mask/numeric placement disagreements, numeric rows rejected by
the current mask, or packet gaps under the current test. Current and proposed
support counts are identical under both the current half-sample tolerance and
the provisional strict tolerance: no row changes support class. Under the
provisional modulo/PpsCount model, measured native jitter is at most one
256-MHz tick (3.90625 ns); this is not producer authority.
A provisional strict 4.063-ms tolerance is 33.000 us below half a sample and
only 1.063 us above the measured maximum. It is not approved.

If the intended modulus is `2^32`, current `2^32-1` handling is one tick low.
For the accepted pair, 128,752 rows are modeled affected, 2,126 binary64
timestamps change, and zero slots change. Across the extended corpus, the
counts are 267,035, 4,391, and zero. This is sensitivity evidence only.

`changed_rows.csv` contains 181 extended-corpus rows, including 38 accepted-pair
rows. Every record is outside both current and proposed support; it records the
out-of-range round-half-up slot versus the current numeric endpoint clamp, not a
realized ordinary-data movement. No ordinary current-support row moves and no
row changes support class. ALIGN-OD1 nevertheless requires detector acquisition
support retention: those edge rows expose a separate union-support/
unavailability-policy repair and are not silently accepted here.

## Offset, telescope, and HWPR findings

`offset_state_trace.csv` has 45 rows: LMT, `toltec0..toltec12`, and HWPR for each
of the comparison Pointing, accepted Pointing, and accepted Beammap state
chains. Exact config/provenance paths and hashes are included. Both Pointing v2
provenances prove requested/effective unit `s`, all 14 configured interface
values zero, and requested/effective equality. The accepted Beammap v1
provenance has no requested/effective offset nodes, although its matched config
requests all 14 values as zero. Present detector values are added once before
slotting, with positive-add sign and detector-clock reference consistent with
ALIGN-OD2. Observation-resolved/realized state is absent in all three chains.
Selected config identities reconcile with raw RoachIndex in phase-0 evidence,
but runtime enforcement/persistence is absent. LMT offset/clock conversion is
not modeled. HWPR offset has no runtime consumer; it is effective-but-ignored
in Pointing v2 and has no effective Beammap v1 state. Nonzero compatibility is
unproved.

Current telescope processing linearly interpolates every configured data field,
then overwrites `TelTime`/`TelUtc` with common time. It has no field-specific
topology/support validity enforcement, labels all telescope outputs radians,
and truncates vector headers to their first element.

- Pointing 152389 raw and configured-recomputed arrays are exactly equal for all
  eight selected timing fields (`AcuTime`, `BackendTime`, `PpsCount`, `PpsTime`,
  `TelLst`, `TelTime`, `TelUtDate`, `TelUtc`) and for `TelAzAct`, `TelElAct`,
  `SourceAz`, `SourceEl`, and `Hold`.
- Accepted Beammap 148670 uses the configured recomputed telescope input with
  SHA-256 `e39f5b9e3066fd20086105964dd915ff67709142d699e8a18bb58cfd9da6b7ae`.
  Its same selected timing/science/state arrays are exactly equal to the raw
  accepted input; per-field raw timing measurements therefore apply exactly.
- Beammap 148670 has 1,370 `Hold` transitions and native values
  `{0,2,8,10,64,66,72,74}`, conflicting with Boolean metadata. Provisional
  left-half-open state placement changes 3,329/383,699 rows (0.867607%), all
  currently non-native interpolated values; maximum numeric change is 65.9040.
- Current linear interpolation creates non-native `PpsTime` on 153 Pointing and
  7,664 accepted Beammap rows. Beammap `PpsCount`/`PpsTime` each have 154,074
  nonpositive steps. Beammap `AcuTime` is nonmonotonic (3 nonpositive steps,
  min −0.670707 s, max +0.705489 s). These are stop facts, not approved repairs.
- `TelLst` and `TelUtc` carry `sec` metadata but exhibit radian-like periodic
  values; `TelUtc` is also treated as radians by current code. Identity/unit
  authority conflicts.
- Nearest telescope-coordinate residual reaches 9.999 ms for Pointing and
  10.480 ms for accepted Beammap. Telescope support encloses both grids, but
  per-field maximum support and invalid-row rules are unavailable.

## Available compatibility baselines

Accepted Pointing record `point-152389-refactor-2a974e0d-redu66` has 19 common
products, zero changed/skipped records, zero numerical tolerance, and exact
science. Its parsed PPT SHA-256 is
`344c85500d367566b7a1b9463fc46a8cd4d8aef0671f9f7eb3891accfbb53763`;
per-array centroids and major/minor FWHM are in
`science_compatibility_evidence.json`.

Accepted Beammap record `beammap-148670-refactor-398d5127-redu18` has 12 common
products, six accepted inactive-config metadata changes, no skipped records,
and exact science at its frozen ledger tolerances. The separate APT contains
5,234 detector rows. Parsed flag-zero median major/minor FWHM is
5.082842/5.990610 arcsec (a1100), 6.495068/7.016367 (a1400), and
8.882777/9.117482 (a2000); APT SHA-256 is
`f1dcd7e7ea88eb47d1b494cdfac3d3b365d5a938d87b5393c97b5fcde9b5b25c`.

The parsed source-crossing artifact has SHA-256
`948df213ac88ce516f85cf177ed33495f95ca5d26f85a5afedb5b68f548255ed`,
5,234 detectors, 5,135 good fits, and closest-distance medians 0.8567, 0.8435,
and 0.8707 arcsec for a1100/a1400/a2000. It contains scan identities/bounds but
no native/common time coordinate or closest-approach sample index, so exact
crossing time cannot be compared.

These are historical baselines only. Phase 0 made no candidate application and
no successor reduction exists. Direct old/new telescope position/timestamp,
source-crossing, centroid, and PSF-width comparisons remain unavailable; no
claim of non-degradation is made.

## Owner decisions required before phase 1

The exact machine-readable questions are in `owner_questions.json`:

1. Supply producer authority for every detector timestamp/counter epoch, logical
   width, rollover, modulus, and sentinel policy.
2. Freeze the detector phase rule after offsets and confirm header cadence.
3. Approve 4.063 ms as strict tolerance or supply the replacement/guard.
4. Define OD1 union support/per-interface unavailability at edge rows.
5. Name LMT/TolTEC/HWPR requested, effective, observation-resolved, and realized
   offset authorities, including LMT need and absent-interface policy.
6. Supply the HWPR schema/angle transform, unit, epoch, counters, rollover,
   cadence, support, valid-count/order, and offset stage.
7. Define `Hold` state coding and half-open transition side.
8. Define `TelTime`, `TelUtc`, `TelLst`, and `PpsTime`
   identities/epochs/topologies.
9. Define every circular field's frame, period, and wrap rule.
10. Define every field's validity/missing/nonfinite rule and maximum support.
11. Resolve `TelRaAct`/`TelDecAct` versus source-coordinate alias precedence.
12. Supply exact output units, identities, and vector-shape rules.
13. Select the owner-approved local fixture or human-mediated Unity run for
    direct timing/source-crossing/centroid/PSF non-degradation evidence.
14. Approve or replace the fail-closed exact-only, zero-span registry baseline.
15. Preregister timing, source-crossing, centroid, and per-array major/minor PSF
    tolerances from repeatability/fit uncertainty before candidate results.
16. Authorize or replace the current detector anchor/`-0.5` rounding rule.
17. Define fail-closed handling for interface/Roach mismatch, invalid/duplicate
    IDs, malformed `Ts`, invalid clock headers, packet rollover/duplicates, and
    nonmonotonic reconstructed time.

Phase 1, T01--T18, application repair, Unity evidence, re-audit, merge, rebase,
push, and production expansion are intentionally not started.

## Artifact routing and reproduction

- Human report: `REPORT.md` (copied byte-for-byte from the frozen report source).
- Machine-readable package: this directory.
- Generator: `tools/diagnostics/generate_sci_align_001_phase0.py`.
- Frozen report source: `tools/diagnostics/sci_align_001_phase0_report.md`.
- Exact package digests: `SHA256SUMS`.
- Stop facts/questions: `stop_facts.json`, `owner_questions.json`.
- Frozen historical gate source:
  `tools/diagnostics/sci_align_001_phase0_gate_snapshot.json`.

Run the generator with the repository, frozen coordination repository, local-data
root, and output directory. It reads only the repository-fixed report source,
fails on changed frozen compatibility inputs or unexpected output files, and
regenerates the complete digest-covered package deterministically.

```sh
/Users/gwilson/tolteca/bin/python tools/diagnostics/generate_sci_align_001_phase0.py \
  --repo /Users/gwilson/.codex/worktrees/9c82/citlali-refactor \
  --coordination-repo /private/tmp/citlali-scientific-audit-framework \
  --local-root /Users/gwilson/work_toltec/local_data \
  --output /Users/gwilson/.codex/worktrees/9c82/citlali-refactor/validation/sci_align_001_phase0_2026-08-01
```
