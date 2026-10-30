# Saved POINT products and calibration audit — 2026-10-30

The temporal study is paused. The saved POINT maps warrant continued calibration
work, but they are not yet qualified calibration products. A lower a1100 peak is
not evidence by itself of detector failure. The comparisons mix processing effects,
and the observations also have different recorded telescope optical settings.
No reductions, source fits, detector decisions or calibration products were made
or changed for this audit.

## What changed in a1100

These are three distinct comparators. “Uniform” means the recent configurable
successor trial, not the older V2 reference. “Weighted” is the saved final M10
product after ten FRUIT feedback passes. Values are fitted peaks in the respective
products' mJy/beam convention; the successor explicitly uses a nominal beam.
Similar numeric units do not establish equivalent photometric response.

| Observation | Recent uniform peak | Weighted M10 peak | Change | Archived V2 PPT peak | Weighted versus V2 |
|---|---:|---:|---:|---:|---:|
| 152389 | 985.801 | 697.434 | −29.3% | 661.057 | +5.5% |
| 152418 | 502.147 | 388.735 | −22.6% | 408.830 | −4.9% |

The smaller V2 differences are context, not a pass criterion or proof of recovered
flux. The objective remains improved scientific performance, not reproducing V2.

Within each observation, all 11 network requests have identical recorded raw,
tune, APT manifest, filter-plan and telescope hashes, and identical requested
detector lists. Request-file hashes were verified; the large raw files were not
rehash-read. All saved telescope variable values also compare equal within each
old/new pair. Realized detector admission and weights differ downstream.

The 152389 comparison also changes pixel size **1″ to 2″**, cleaning ranks
**5/5/5 to 3/4/6**, and coverage cut **0.10 to 0.15**. For 152418 these remain
**2″, 3/4/6, 0.15** in both runs. Both comparisons change cleaning/map weighting,
detector admission and FRUIT behavior. This corrects any earlier interpretation
that the two old/new pairs had identical sampling/settings, or that both old maps
used 1″ pixels. Neither pair isolates one algorithm's effect.

### Beam shape and background provide more information than peak alone

| Observation/array | Uniform → weighted peak | Uniform → weighted FWHM (arcsec; sorted axes) | Gaussian brightness-area change | 20″ aperture brightness-area change |
|---|---:|---|---:|---:|
| 152389 a1100 | 985.8 → 697.4 | 5.13×5.62 → 5.50×5.95 | −19.8% | +48.2% |
| 152389 a1400 | 669.6 → 596.7 | 6.57×7.10 → 6.48×7.20 | −10.9% | −16.7% |
| 152389 a2000 | 698.7 → 782.2 | 8.67×9.09 → 9.21×9.82 | +28.5% | +54.2% |
| 152418 a1100 | 502.1 → 388.7 | 6.10×7.61 → 5.90×7.25 | −28.6% | −44.3% |
| 152418 a1400 | 509.9 → 499.8 | 6.73×8.42 → 6.88×8.08 | −3.9% | +28.2% |
| 152418 a2000 | 566.9 → 680.1 | 8.86×9.72 → 9.41×10.05 | +31.7% | +63.8% |

Gaussian brightness-area is `2π A σx σy` from the existing fit, not a new fit.
Apertures use the fixed final weighted centroid, 8×8 pixel-footprint quadrature,
and the median of supported pixels at radii 45–75″ as background. Values have
units mJy/nominal-beam × arcsec²; **they are not calibrated total source fluxes**.
All apertures through 45″ are fully supported to quadrature precision (the
estimated area fraction can differ from unity by about 0.1%). No uncertainty or
statistical significance is assigned. No support extrapolation is used.

For **152389 a1100**, the old uniform map has a broad negative background: its
annulus-subtracted aperture sum falls from 30,406 at 8″ to 19,653 at 20″ and
becomes negative at 45″. The final weighted curve instead approaches roughly
28,000–29,000 over 12–45″. This is evidence of a more stable local background
under this diagnostic, despite a lower peak. At 20″, subtracting each saved fit's
constant instead gives 29,222 versus 28,921—only about 1% apart. The +48% aperture
comparison is therefore strongly background-dependent and must not be reported
as a measured gain in recovered flux. The Gaussian-area loss is smaller than the
peak loss but is still present.

For **152418 a1100**, the final weighted map still has background sensitivity:
the annulus-subtracted sum falls from 19,294 at 12″ to 8,966 at 45″, whereas
using the saved fit constant gives approximately 18,200–20,000 over 8–45″.
The old uniform map has excess signal away from the core, including a feature
around 20″ in the radial mean. The two background choices both leave the weighted
20″ sum substantially lower than uniform (ratios 0.557 and 0.591). Thus its
amplitude change is not explained merely by a broader fitted beam: the final
weighted fitted core is slightly narrower. The saved evidence cannot decide how
much is removal of background contamination versus loss of astronomical signal.

FRUIT increases a1100 peak from M0 to M10 by 9.8% in 152389 and 7.7% in 152418;
brightness-area increases by 19.5% and 19.7%. These show recovery relative to the
initial weighted map, not absolute flux accuracy. Existing convergence evidence
does not establish unbiased calibration. The a2000 peak and area increase in both
observations; this is not a uniform loss affecting all arrays identically.

[Saved radial/aperture figure](/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/point-calibration-audit-20261030/a1100-profiles.png)

## Telescope optics versus detector health

Grant's warning about thermal M1 deformation is retained as a physical possibility,
not reclassified as a detector fault. Saved telescope metadata support treating
the optical state as a separate variable:

| Recorded quantity | 152389 | 152418 |
|---|---|---|
| UTC start in archived PPT | 2026-02-19 06:34:47 | 2026-02-19 09:00:45 |
| Mean elevation | 59.22° | 67.37° |
| First seven M1 Zernike coefficients (µm) | 46, 39, −2, −54, 5, −22, −60 | 164, 60, 42, −178, −23, −3, −86 |
| M2 Z request (mm) | −0.952980 | +0.676016 |
| Weather temperature (°C) | 2.833 | 2.400 |
| Radiometer tau header | 0.021 | 0.021 |

M1 model and Zernike corrections are enabled in both; actuator position/command
arrays differ between observations. The legacy PPT records only the first
Zernike coefficient, so the full vector was read from the saved telescope file.

These settings do **not** measure residual surface error, establish poor focus,
or prove that thermal deformation caused a particular loss. Temperature sensors
lack a validated location/missing-value interpretation here; no mirror thermal
gradient or surface RMS is inferred. Equal tau headers do not establish equal
transmission throughout both observations.

Telescope evolution cannot independently explain why two reductions of the same
raw observation differ. However, changed detector membership/weights could change
how a nonuniform optical beam is represented. Across observations, the different
optical state, elevation and atmosphere remain confounders. In the weighted maps,
152418's a1100 beam is broader and lower than 152389's; that is consistent with
optical redistribution among other explanations, not a diagnosis.

**Keep separate evidence paths:** relative detector response to shared atmospheric
variations; compact-source beam shape/peak/aperture behavior; and telescope optical
state. The 10% response cut remains unchanged. An array-relative atmospheric test
does not measure a common loss of compact-source coupling, and neither low map
peak nor different M1 commands should automatically reject detectors. Do not
restore or remove unresolved detectors on the basis of this audit.

## Existing calibration interface: actual compatibility findings

Read-only authority: TolProj `6309a5a852a0412cee867992aff01341a56b0ebf`,
`tolproj/steps/calibrate_flxscale.py` and `setup_science_reductions.py`;
operational TolTECA v2 source at local `origin/main`
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`,
`tolteca/reduce/engines/citlali.py`. The installed TolTECA v3 runtime was not run.

| Requirement | Existing consumer behavior | Saved successor status / required next work |
|---|---|---|
| Table discovery | Exactly one `ppt_*.ecsv` under `pointings/reduced/<reduction>/<obs>/raw`, preferring observation-specific names | Trial directories are separate and contain no PPT; standard project discovery fails. Establish publication through the familiar project interface. |
| Array/observation identity | Requires all three arrays; rejects duplicates; optional metadata obsnum must match | Native products identify arrays/obs. Future adapter must emit exact integer array IDs and complete identity; consumer currently rounds fractional IDs. |
| Amplitude | Finite positive `amp`; mJy/beam in column units or legacy metadata | Saved JSON amplitude is mJy/nominal beam. Qualify its photometric meaning and sampling/cleaning dependence before exporting it. |
| Error and unavailable data | Finite nonnegative `amp_err`; missing array/nonfinite amplitude rejected | Saved `sigma_A` is conditional frozen-C0 reference-scale uncertainty, explicitly not final empirical/adaptive uncertainty. It cannot simply become `amp_err`. Unavailable measurements must fail closed, not become zero-error valid rows. |
| Eligibility | Reader does not inspect `calibration_eligible` | Saved products explicitly say false. Scratch tests confirm that a PPT with false metadata would nevertheless be accepted. A metadata flag alone is insufficient protection. |
| Error units | Reader checks amp units, not amp_err units | Scratch table with Jy/beam error and mJy/beam amplitude is accepted without conversion. Bound producer/consumer verification must address this. |
| Pointing correction | TolTECA averages PPT `x_t,y_t`, adjusts telescope user/paddle offsets, then negates to form az/alt corrections; requires MJD and telescope metadata | Native `pointing_correction` is null. Qualify centroid frame/sign and metadata mapping before PPT publication; do not pre-negate then negate again. |
| Widths and map weight | Legacy PPT widths are FWHM; native fit widths are sigma. Native normalization is empirical coefficient normalization | Convert widths with explicit units if exported. Do not advertise NORMALIZATION as calibrated inverse map variance. |

Eight scratch-table probes exercised the **actual TolProj reader**: valid accepted;
missing array, nonfinite amplitude and wrong amplitude units rejected; wrong error
units, false eligibility, fractional array IDs and zero error accepted. Acceptance
of zero error is a documented current behavior, not permission to fabricate an
unavailable error. No actual project products were modified for these probes.

### Calibration factors and the exact selected APTs

The archived machinery computes each bracket's recovered/reference flux ratio,
takes their unweighted arithmetic mean per array, and uses its reciprocal as the
cohort correction. It is not time-interpolated flux calibration. Current defaults
bound the factor to 0.2–5; 20% bracket disagreement is a warning, not rejection.
These are existing rules, not new acceptance thresholds selected for these maps.

The saved reference-flux report and actual archived PPT rows reproduce all six
historical factors exactly:

| SCIENCE observation | POINT brackets | a1100 | a1400 | a2000 |
|---|---|---:|---:|---:|
| 152390 | 152389, 152391 | 0.7661576363184347 | 1.1210333837360158 | 1.4300676124369782 |
| 152392 | 152391, 152393 | 0.951671966009629 | 1.1681551970633939 | 1.52741595135239 |

Report digest and request factor bit patterns match. For each observation, all
5,518 matched APT rows retain UID/array identity, and every finite child `flxscale`
equals the parent value times the array factor (relative arithmetic tolerance
1e-14, not a scientific acceptance threshold). Counts are 3,173/1,368/977 by
array. APT component transport hashes were verified. This is a bounded arithmetic
and identity check, not a fresh canonical bundle validation or issuance.

The current authored `refactor/NGC4449/72_science_observation.yaml` explicitly
selects the archived reference calibrated manifests for 152390/152392 under
`reference/v2-science/apts/v2/fluxcal/`, and the corresponding three archived
POINT directories. Their manifest and APT bytes match the copied project
calibrated children. These are intentional reference-calibration inputs, not
successor-derived calibration. Historical Unity paths in the report are provenance.

Project `flxscale_done` is false; the automatic setup selector would choose matched
APTs in that state. This differs from the already-authored reference YAML and must
be preserved/selected deliberately in the eventual A/B/C setup. No new SCIENCE
low-level configuration exists in this local trial, so this audit establishes
authored selection and historical arithmetic—not the final merged configuration
or exact APT consumed by a new SCIENCE execution.

Only 152389 and 152418 have these saved successor weighted POINT results.
**152391 and 152393 are still needed** for successor calibration of the selected
SCIENCE pair. 152418 is an independent POINT check, not a substitute bracket for
these SCIENCE observations. Do not silently mix legacy and successor brackets.

## Ordered remaining work

1. Settle the POINT-to-calibration product meaning: nominal-beam amplitude,
   usable error, centroid frame/sign, telescope metadata and explicit eligibility.
   Existing saved results support this bounded interface work; a generic temporal
   change detector is not its prerequisite.
2. Prepare the minimal PPT publication/consumer protections with isolated tests:
   discovery, exact identities, units, unavailable measurements, and both flux
   and pointing-offset consumption. No new path configuration is justified by
   this audit; normal project-local POINT discovery already exists.
3. Resolve the specific a1100 transfer/background question before granting
   photometric eligibility. If a new owner-run comparison becomes necessary,
   match sampling, ranks, coverage and calibration first and vary a named element.
   Do not use uniform or V2 amplitude as truth. Keep M1/M2 state attached to the
   observation rather than interpreting lower peak as detector failure. No such
   run has been prepared or launched by this audit.
4. Once that boundary is qualified, prepare owner-run successor POINT 152391 and
   152393 through the same project interface, then issue a separate successor
   calibration via established machinery. Retain 152418 as the independent check.
5. Preserve A/reference, prepare B/successor SCIENCE with compatible reference
   calibration, and C/successor SCIENCE with successor-derived calibration for
   the same 152390/152392 observations and real coadd. Verify final TolTECA config
   and APT bindings at that stage. M1 changes may limit calibration transfer;
   agreement in a peak-calibration calculation alone cannot establish identical
   beam response for extended SCIENCE emission.

No new harm thresholds, uncertainty coverage claims, acceptance decisions or
detector changes follow from these descriptive measurements. Grant's next
scientific decision is the required amplitude/error and calibration-transfer
standard, after a concrete bounded proposal; this report does not invent it.

## Evidence and verification

Project root:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`.
Saved sources are the configurable 152389 redu00, disturbance 152418 redu01,
weighted 152389/152418 redu00, and archived V2 POINT/calibration products. Exact
paths, run/config/executable hashes, map hashes, telescope values, per-array
apertures and calibrated APT bindings are in [saved-metrics.json](saved-metrics.json)
and [consumer-metrics.json](consumer-metrics.json).

`audit_saved.py` verifies actual executable/config bytes against RUN records,
request identities and published map-to-fit hashes; it computes only saved-image
arithmetic. `audit_consumer.py` exercises pure reader functions, scratch fixtures
and saved calibration arithmetic; it never calls calibrate/setup. Run with the
owner's `/Users/gwilson/tolteca/bin/python -B -W error` and headless
`MPLBACKEND=Agg`, writable task-specific `MPLCONFIGDIR`/`XDG_CACHE_HOME`. Both take
`--output`; the image script also takes `--figure`. Figure was visually inspected.
No build/config/CTest gates apply because application/configuration code is
unchanged. These local audit checks are not Spack/Unity application validation.
Independent exact-SHA review is recorded separately after the evidence commit.
