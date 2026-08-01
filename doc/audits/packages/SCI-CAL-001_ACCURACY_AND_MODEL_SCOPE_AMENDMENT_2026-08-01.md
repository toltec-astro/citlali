# SCI-CAL-001 accuracy and atmospheric-model scope amendment — 2026-08-01

Status: owner approved; successor atmosphere operator still requires selection

Package: `SCI-CAL-001`

Decision ID: `CAL-D001-ACCURACY-001`

## Authority and purpose

The project owner approved this successor acceptance framework on 2026-08-01
after review of the phase-0 q-model continuity evidence and the calibration
error budget. This amendment answers the above-q25 scope stop required by
`CAL-D001-OPACITY-001` and the bounded repair handoff.

The frozen independent core, completed audit, original coordinator decision,
opacity amendment, and phase-0 evidence remain immutable evidence. This record
does not claim that the atmosphere, calibrator flux, Beammap fit, or an
individual detector sample is physically known to floating-point precision or
to one-percent absolute photometric accuracy.

## Approved accuracy hierarchy

Calibration acceptance has three distinct layers.

### 1. Software and contract correctness

- The selected extinction equation, zenith-opacity identity, full sample
  airmass, top-of-atmosphere reference plane, units, detector identity,
  factor decomposition, validity state, and provenance must be implemented
  exactly as declared.
- Deterministic fixtures retain their tight floating-point tolerances against
  the declared operator. These are implementation tests, not claims about the
  accuracy of the physical atmosphere.
- The successor representation must be continuous across its opacity and
  elevation support. A software selector must not introduce an artificial
  photometric jump.
- Transmission must remain finite and positive, and unsupported opacity or
  elevation state must fail closed rather than clamp or extrapolate silently.

### 2. Atmospheric-model representation fidelity

- Numerical interpolation or tabulation must add negligible error relative to
  the underlying atmospheric calculation.
- The provisional engineering target is no more than one-percent fractional
  error in the extinction correction factor relative to the regenerated raw
  atmosphere-model grid over the declared operational domain.
- This one-percent target measures representation fidelity only. It is not a
  requirement that each corrected sample be physically photometric to one
  percent.
- Exact anchor equality, continuous interpolation, positivity, and the
  expected monotonic physical behavior remain separate structural gates.

### 3. Observational calibration performance

- The scientific objective is approximately five-to-ten-percent absolute flux
  accuracy per TolTEC band in final products, not one-percent accuracy for
  every sample.
- Observation-to-observation relative repeatability has a provisional
  approximately five-percent target.
- Repeated calibrator and repeated-field evidence must show no statistically
  significant residual flux trend with opacity or airmass over the declared
  operational domain.
- The exact calibrator sample, estimators, uncertainty treatment, and pass/fail
  statistics must be preregistered before the human-run exact-repair-SHA
  campaign. The stated goals are not converted into an unevidenced guarantee.

## Correlation and averaging interpretation

Many samples per map pixel reduce stochastic detector noise and may reduce
independent detector-fit error. They do not automatically reduce common
calibrator-flux error, shared Beammap extinction error, band/beam error, a
hard opacity-selector bias, or an incorrect airmass operator. The validation
record must therefore keep sample precision, relative repeatability, and
absolute calibration accuracy separate.

## Disposition of the phase-0 findings

- The missing sample-airmass factor is a genuine correlated calibration defect
  and remains mandatory repair scope.
- Above-q25 models are now in successor design scope. The current hard q-model
  selection must be replaced by a continuous, versioned representation rather
  than accepted merely because final maps average many samples.
- The existing q-models are evidence inputs, not automatically approved
  interpolation anchors. Their raw atmosphere calculations and fit residuals
  must be recovered or regenerated before selecting the successor operator.
- A sub-percent internal polynomial feature, such as the observed
  `am_q95`/`a2000` elevation feature, is not by itself a release-blocking
  absolute-photometry failure. It remains a model-quality diagnostic and is
  acceptable only if the successor representation passes the structural,
  representation-fidelity, and observational gates above.
- The several-percent and larger hard-selector jumps remain material because
  they are observation-level systematics comparable to the desired final
  calibration budget.

## Next gate and current stop condition

Before editing CAL application code, prepare and review a reproducible
atmosphere-model regeneration package that records the model/profile inputs,
TolTEC bandpasses and spectral convention, frequency and elevation grids,
opacity anchors, raw transmissions, band integration, provenance, and artifact
digests. Evaluate continuous interpolation in line-of-sight optical depth and
validate it against withheld atmosphere-model calculations.

The coordinator may then authorize one exact, versioned atmosphere operator
and operational domain for the existing `codex/repair-sci-cal-001` repair
line. Until that operator is selected, do not modify application code, request
Unity evidence, or launch the CAL re-audit. Other already-approved CAL repair
requirements remain unchanged.
