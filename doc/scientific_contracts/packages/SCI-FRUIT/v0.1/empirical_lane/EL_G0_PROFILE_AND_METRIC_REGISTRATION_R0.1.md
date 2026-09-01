# SCI-FRUIT v0.1 — EL-G0 Profile And Metric Registration Candidates r0.1

Status: **candidate-neutral registration proposal; no profile, metric,
threshold, or claim is approved**

## Proposed Development Profiles

### `compact_high_snr_response_recovery`

FRUIT-owned estimand: recovery of a compact astronomical response, including
peak and integrated amplitude, core and wing response, centroid, morphology,
local false structure, and causal convergence.

Proposed application strata may include OOF, pointing, or beammap-like inputs,
but the profile does not own OOF wavefront inference, telescope correction,
source truth, or beammap calibration. Whether these acquisition modes share one
applicability domain is an open Gate-D decision.

### `extended_low_snr_mode_recovery`

FRUIT-owned estimand: recovery of declared two-dimensional astronomical modes
over angular scale, orientation, amplitude, morphology, and observing
conditions, while controlling atmospheric/other nuisance leakage and false
large-scale structure.

SZE-like injections may motivate a development stratum, but cluster physics,
SZE astrophysics, and source claims remain outside SCI-FRUIT.

## Pre-Output Profile Selection

A profile/domain assignment may use only frozen input metadata and an owner-
authorized target declaration: reduction/acquisition type, declared target
class, scan geometry, array coverage, calibration/APT identity, observing-
condition strata, and injection design. It may not use a candidate or control
map, fitted flux, measured FWHM, residual map, convergence trace, or best-
looking result.

## Candidate-Neutral Metric Registration

Each metric binds grid/WCS, response/kernel, units, truth convention, support,
masks, weights, pairing, independent unit, clustering, missing/failure rules,
and whether larger or smaller is better.

| Family | Required registered target |
| --- | --- |
| angular/mode transfer | recovered-to-injected amplitude over a declared two-dimensional mode family; near-zero inputs use absolute or externally scaled error |
| flux recovery | peak and integrated flux bias/dispersion on fixed apertures or estimators; no post-output aperture choice |
| compact response | radial/2D core and wing response, centroid, width/shape, local residual and false structure |
| nuisance leakage | positive/negative and multi-amplitude atmosphere/other nuisance coupling plus nuisance-only/null false astronomical structure |
| convergence | truth error and inter-iteration stability separately; actual terminal, oracle-only evaluation, hard cap, censoring, oscillation, drift, and time-to-quality |
| support/availability | common support, candidate/historical support, support gained/lost, failure, rescue, regression, and unavailable fractions |
| response/uncertainty | fixed-state and complete-procedure response and uncertainty as separate targets |
| computation | wall and CPU time, peak resident memory, read/write volume, output/storage volume, restart overhead, and time to comparable scientific quality |

The accepted ratio, nuisance-coupling, convergence, outcome, support, and
multiplicity safeguards remain controlling. Computational performance does not
compensate for a protected scientific regression unless a separate trade is
frozen before qualification.

## Truth And Null Construction Candidates

The proposed minimum design combines:

1. known astronomical injections into candidate-neutral real-noise inputs;
2. positive/negative amplitudes and multiple amplitudes for nonlinearity;
3. compact and two-dimensional extended mode families;
4. nuisance-only and zero-astronomical-signal inputs;
5. fixed-state and complete-procedure replay where applicable; and
6. exact candidate/control pairing on identical inputs and random seeds.

Injected truth supplies truth only for the injected component within the
declared construction. It does not make the background sky or historical
control true.

## Decisions Still Required Before Gate D

- admit one or both generic profiles for development;
- decide whether OOF, pointing, and beammap-like compact inputs share a domain;
- freeze the exact mode/injection families and amplitude range construction;
- mark protected versus prioritized metric families;
- freeze equations, units, support, pairing, and independent-unit rules that
  must precede tuning; and
- define how development may estimate later thresholds without opening
  qualification outcomes.
