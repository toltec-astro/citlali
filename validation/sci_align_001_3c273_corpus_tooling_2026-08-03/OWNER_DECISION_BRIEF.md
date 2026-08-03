# SCI-ALIGN-001 3C273 corpus decision brief

## Decision to be made after owner execution

Use the complete, provenance-selected 3C273 Beammap corpus to decide whether
the measured left/right timing residual is globally stable, network-stable,
T0-session-stable, predictable from retained native timing metadata,
time-variable, unpredictable, or insufficiently constrained.  Held-out
prediction, not constancy alone, governs whether a later bounded mitigation
design may be investigated.

This package does not choose a production correction. It creates a
preregistered experiment and compact evidence contract for the project owner
to run on Unity. The corpus-analysis lane remains read-only; the owner has
separately authorized an isolated 148670 diagnostic reproduction solely to
create the missing detector-resolved PTC TOD.

## Producer timing authority incorporated during tooling development

Kamal reports that NTP supplies the integer-second `T0` at ROACH
initialization.  All ROACH internal clocks derive from the same
Octo-distributed 10 MHz reference, and PPS is derived and distributed from
the same system.  PPS does not reset or restart detector integration cadence;
an incremental PPS counter is observed at detector-sample cadence and is
updated through an interrupt service routine.  Each UDP packet carries `T0`,
the PPS counter, and the internal-clock counter.  FPGA source is not presently
available.

Consequently:

- arbitrary millisecond NTP epoch errors and differential oscillator drift
  are strongly disfavored; an NTP error should primarily change the integer
  second label;
- common frequency and PPS do not imply common detector-frame phase, so every
  network may have a distinct but stable phase within the 8.192 ms cadence;
- a correctly timestamped network phase, frame-quantized PPS observation,
  non-atomic or adjacent-frame data/counter association, and start/end/centroid
  timestamp semantics remain separate hypotheses;
- the prior Stage-A proof starts at the delivered raw `D[n]`/`Ts[n]` pair.  It
  proves downstream preservation of that pair but cannot exclude an upstream
  FPGA metadata-to-integration association error.

The previously measured 1.856 ms first-half/second-half difference is called
within-observation timing variation here.  It is not evidence for clock drift
unless raw counters contradict the shared-reference account.

## Owner-frozen corpus and execution policy

The checksum-bound `authoritative_obsnums_2026-08-03.json` file is the sole
40-ObsNum corpus authority. The inventory reports every listed observation,
including missing retained reductions, and records other discovered 3C273
products only in a separate out-of-scope table. The versioned owner run
directory is excluded from discovery, so a rerun cannot discover its own
outputs. The inventory and corpus analysis do not invoke Citlali. The only
exception is the separately checksum-bound, owner-reviewed 148670 replay gate;
`CITLALI_BIN` is otherwise execution provenance only.

The canonical rule is timing-blind: use the sole eligible retained reduction;
when exactly one `redu00` and one `redu01` are eligible, use `redu01` and keep
`redu00` as a sensitivity duplicate. Multiple candidates in either location,
an unexpected location, or ambiguous provenance stops selection freeze for
owner review. A lone `redu00` remains eligible.

`nw10` is structural and never reported as missing. `nw6` is recorded as
present or intermittently absent without excluding a map. Other per-map
network absences and unreadable metadata remain explicit support limitations.

## nw9 PpsTime increment diagnostic

Every available network is tested with the same authenticated PpsTime
increment detector. The compact bundle reports denominator, count/rate,
signed/absolute tick residual, physical-time conversion, transition and packet
geometry, first/last and clustered occurrences, and unavailable fields as
unavailable rather than zero. It separately reports nw9 timing relative to
other networks and all-network versus leave-nw9-out estimates with covariance
aware uncertainty, plus an uncertainty-aware anomaly-rate association across
observations. This is not a causal or clock-drift claim; direct affected-row
masking/repair is not authorized while metadata semantics remain unresolved.

## Preserved Beammap 148670 evidence

The predecessor package remains unchanged.  Its frozen common-support results
are:

| Quantity | Frozen result |
| --- | ---: |
| Assigned-slot pooled residual | -12.2495 +/- 0.2748 ms |
| Assigned-slot `k=+1, phi=+0.5` comparison | -0.0752 +/- 0.2720 ms |
| Assigned network span | 7.351 ms |
| Assigned residual correlation | -0.963 |
| Raw-time `k=+1, phi=+0.5` comparison | -0.9541 +/- 0.2659 ms |
| Raw residual correlation | +0.443 |
| First-half minus second-half estimate | 1.856 ms |

The approximately 1.5-sample common displacement is an effective diagnostic
comparison.  It is not an off-by-one finding and does not authorize a timing
correction.

## Frozen interpretation of the corpus result

The aggregate report must choose exactly one of these categories under the
rules in `frozen_analysis_protocol.json`:

- **GLOBAL-STABLE** — a global model generalizes and residual network/map/time
  structure is not resolved;
- **NETWORK-STABLE** — fixed network terms generalize across sessions and
  held-out maps;
- **SESSION-STABLE** — offsets are repeatable within provenance-supported T0
  sessions but change after initialization; permanent network constants are
  not authorized;
- **SLOT-PREDICTABLE** — measured native phase or native-to-assigned-slot
  residual predicts held-out timing.  Favor later structural native-time or
  fractional-slot mitigation, not physical clock corrections;
- **TIME-VARIABLE** — significant within-observation or within-session timing
  variation remains.  Prioritize acquisition/timestamp understanding;
- **UNPREDICTABLE** — variability exceeds fit uncertainty and no
  preregistered model predicts held-out results; do not mitigate;
- **INSUFFICIENT** — provenance, retained products, independent groups, or
  uncertainty support cannot sustain a conclusion.

Every model is evaluated on frozen held-out groups.  A complete ordered
per-network integer-`T0` vector is the first candidate session identity.  It
is used only when complete and when at least three distinct sessions exist;
otherwise the frozen date or observation fallback applies.  Duplicate
reductions never count as independent observations.

## Counter and phase evidence required for enhanced maps

Where raw fields are retained and row linkage is provable, the evidence must
record exact integer `T0` values by network, preserve the separate nanosecond
field without interpreting it as phase, inventory every PPS transition row,
and retain the internal-clock/PPS-time/packet values immediately before and
after each transition.  It must test:

- 122/123 detector rows between consecutive PPS transitions;
- exactly 15,625 detector rows over 128 PPS intervals;
- internal-clock increments modulo `2^32` against retained accumulation
  length;
- same-row, adjacent-row, or variable PPS-counter/PPS-time transition
  association;
- native detector-frame phase stability within a T0 session and changes
  across initialization;
- prediction of network timing by native phase and separately by
  native-to-assigned-slot residual, including the preregistered slope `-1`
  comparison.

These are observations of delivered metadata.  Without FPGA source they do
not establish which integration event a counter or timestamp marks.

## Deferred science-facing scale and later confirmation

SCI-ALIGN-001 records timing, support, uncertainties, network structure, and
descriptive model classifications. It imposes no arcsecond, FWHM, timing, or
mitigation acceptability threshold. On-sky scientific-impact translation is a
separate downstream analysis.

Even a successful 3C273 result is discovery/validation evidence for a later
bounded design, not production authorization.  Before any production change,
the minimum expansion is an exact-SHA, preregistered, held-out confirmation on
at least one non-3C273 source spanning the relevant arrays/networks and at
least two independent ROACH-initialization T0 vectors.  That expansion is not
performed by this task.

## Authorized stopping point

First stop after the owner returns the isolated 148670 replay product evidence;
do not start corpus inventory or another reproduction before review. The later
stop is a compact, checksum-bound corpus report with unresolved duplicate/path
choices explicit. No Unity access by Codex, physical timestamp correction, row
reassociation, application/source edit, merge, rebase, or push is part of this
work. The sole authorized reduction is the one source-isolated 148670 replay.
