# SCI-CAL-001 copied-AM follow-up study preregistration

## Registration boundary

This plan was frozen on 2026-08-01 after inspecting copied directory names,
workflow text, configuration headers, file sizes, and whole-file digests, but
before loading numerical arrays from the newly supplied annual q5/q95 or
seasonal NPZ files. The already reviewed q25/q50/q75 arrays and repair-base
q25/q50/q75/q95 polynomial literals are not blinded evidence. The copied
artifacts remain read-only under
`/Users/gwilson/work_toltec/local_data/AM`; generated outputs and reports go
only into this task-specific validation package or a task-specific temporary
directory.

This is a numerical atmosphere-representation study. It does not test
Citlali application behavior, observational absolute flux accuracy, or
observation-to-observation repeatability, and it does not authorize a repair
or re-audit.

## Frozen source and command identity

- Model source release: copied `am-12.2` source tree; source-file digests will
  be recorded before any native build.
- Supplied production binary: copied x86-64 Linux ELF; SHA-256
  `3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c`.
  It is custody evidence and is not executable on the local macOS host.
- Historical command grid: `am PROFILE 0 GHz 500 GHz 10 MHz ZA deg 1.0`,
  with zenith angle `ZA = 10, 12, ..., 80` degrees. The NPZ packager maps
  these to elevation `EL = 90 - ZA`, yielding 36 nodes from 10 through 80
  degrees.
- Model output columns: frequency in GHz, line-of-sight optical depth,
  transmission, Rayleigh-Jeans temperature in K, and brightness temperature
  in K. The legacy fit uses transmission only.
- Model profiles: annual, DJF, MAM, JJA, and SON climatologies, each at H2O
  percentiles 5, 25, 50, 75, and 95 with median O3. Their configuration bytes
  and MERRA-2 provenance text will be digest-bound.

## Study A: custody and legacy-lineage closure

1. Verify all copied workflow, source, profile, raw-output, and NPZ inventory
   identities with SHA-256. Record file counts and a deterministic aggregate
   manifest rather than copying multi-gigabyte raw files into Git.
2. Verify that the annual q25/q50/q75 copied full-grid arrays numerically
   contain the previously recovered 20--80 degree legacy grids after the
   documented elevation selection, and determine the exact deterministic
   extraction/serialization step responsible for any whole-file digest
   difference.
3. Derive the annual q95 legacy-format grid by the same transformation and
   compare its MD5 to TolTECA datafile ID 461's registered identity
   `0ca7b331823237767d26016d19bffb3d`. A mismatch is reported; it is not
   repaired by changing data or serialization opportunistically.
4. Reproduce the degree-six elevation-radian monochromatic transmission-ratio
   fit at 272.73, 214.29, and 150.00 GHz relative to 225.00 GHz over elevations
   20--80 degrees in 2-degree steps, including eight-decimal coefficient
   rounding, for q25/q50/q75/q95.

## Study B: native regeneration check

Build the copied AM 12.2 source in a task-specific temporary directory,
without editing the supplied tree. Preserve the compiler identity, flags,
source manifest, executable SHA-256, and AM version output. Run the exact
historical argv first for annual q95 at zenith angles 10 and 70 degrees
(elevations 80 and 20 degrees). If both runs complete and their parsed
frequency, optical-depth, and transmission columns agree with the copied raw
outputs, extend the comparison to all annual q5/q25/q50/q75/q95 profiles and
all 36 zenith-angle nodes.

Report byte equality of data lines separately from numerical equality because
AM embeds build metadata and the native compiler/platform differs from the
copied production binary. Numerical acceptance requires identical frequency
nodes and, for every parsed value, either exact binary64 equality after text
parsing or absolute difference no larger than one half-unit of the copied
text's final represented decimal place. No looser ad hoc tolerance may be
introduced after results are seen.

## Study C: continuous-operator held-out evaluation

### Frozen conventions

- Bands are the legacy monochromatic frequencies a1100 = 272.73 GHz, a1400 =
  214.29 GHz, and a2000 = 150.00 GHz; reference frequency is 225.00 GHz. No
  TolTEC passband integration is part of this lineage study.
- The operator coordinate is zenith `tau225`. For every physical profile it is
  derived once as `-log(T225 at EL=80 deg) / X(80 deg)`, using Citlali's frozen
  modified-secant `X`. Full sample `X(EL)` and top-of-atmosphere pivot
  `X_ref=0` remain mandatory when reconstructing a sample transmission.
- Truth is raw monochromatic AM line-of-sight transmission at each evaluated
  profile and elevation. Fractional extinction-correction error is
  `abs(exp(tau_los_candidate - tau_los_truth) - 1)`.
- Evaluate elevations 20--80 degrees inclusive. Existing even-degree raw
  nodes are used for the principal study. If native regeneration succeeds,
  odd elevations 21--79 degrees are generated as elevation holdouts.
- No opacity or elevation extrapolation is permitted. A held-out profile whose
  derived zenith `tau225` lies outside `[0, annual_q95_tau225]` is reported as
  out of support and excluded from fidelity maxima.

### Training anchors and candidates

The annual q25/q50/q75/q95 profiles are exact anchors, with the analytic clear
origin `(tau225=0, tau_los=0)`. Compare:

1. `piecewise_linear_los_tau_v1`: linear from the clear origin to q25 and
   piecewise affine in zenith `tau225` between adjacent annual q anchors, at
   each elevation, in line-of-sight optical depth.
2. `pchip_los_tau_v1`: the same fixed clear-to-q25 segment, then a
   shape-preserving PCHIP through annual q25/q50/q75/q95 in line-of-sight
   optical depth.

Evaluate both in two lanes:

- `raw_node_lane`: annual raw line-of-sight optical-depth anchor values at
  each even-degree elevation. This isolates opacity interpolation.
- `continuous_elevation_lane`: the recovered degree-six elevation-radian
  fit at every annual q anchor, followed by the same opacity interpolation.
  This measures the combined implementable surface against raw truth.

### Held-out profile sets

- Annual q5: primary low-opacity held-out profile; it tests the approved
  clear-to-q25 segment.
- DJF, MAM, JJA, and SON q5/q25/q50/q75/q95: physical profile-family
  generalization tests. These test whether zenith `tau225` alone adequately
  indexes seasonally different temperature and H2O structures. They are
  pre-existing calculations, not newly generated intermediate-percentile
  profiles, and are labeled accordingly.
- Any seasonal profiles landing between adjacent annual anchors are also
  reported by annual opacity interval. Coverage is descriptive; it must not
  be relabeled as owner-approved synthetic midpoint construction.

### Fixed gates and diagnostics

- Exact annual anchor reproduction, finite positive transmission, continuity,
  opacity monotonicity, and fail-closed support are mandatory structural
  gates.
- Provisional representation fidelity requires maximum fractional
  extinction-correction error no greater than 1% in every band over all
  in-support held-out profile/elevation rows in the declared study domain.
- Report maxima, p95, median, profile/season/band/elevation locations, interval
  coverage, and results separately for the low-opacity annual-q5 test and the
  seasonal-profile test.
- Diagnose the q95/a2000 elevation feature against raw annual q95 nodes and,
  when generated, odd-degree runs. A sub-percent feature is recorded rather
  than automatically treated as release-blocking.

Passing these numerical gates is not evidence of the 5--10% observational
absolute-flux objective or provisional approximately 5% repeatability. Common
calibrator, Beammap-extinction, selector, aligned-elevation, timing, and
airmass systematics remain outside this study.

## Decision rule

Recommend the simplest candidate that passes all structural and one-percent
held-out gates over a support-backed domain. Prefer
`piecewise_linear_los_tau_v1` on a tie. If neither passes, or if the copied
profiles do not populate every proposed opacity interval, stop with the exact
remaining physical-profile or domain decision; do not invent an atmosphere.
Any operational use of aligned elevation remains contingent on open handoff
`SCI-CAL-001-XAUD-001` sample identity, timing-gap/interpolation origin,
duration, and original-versus-synthesized eligibility.
