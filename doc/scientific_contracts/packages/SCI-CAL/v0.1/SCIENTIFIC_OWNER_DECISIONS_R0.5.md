# SCI-CAL Scientific Owner Decisions for r0.5/r0.4

Status: approved scientific-owner authority for Q01--Q09

Decision date: 2026-08-20

Scientific owner: Grant Wilson

This record consolidates the owner's interactive dispositions and the
previously approved, content-bound atmosphere authority recovered from the
SCI-CAL atmosphere-model work. It supersedes the open-state interpretation of
Q01--Q09 in the r0.4/r0.3 pair. A decided question can still carry an explicit
unavailable numerical product or deferred producer mechanism.

## Q01 - ordinary xs

Ordinary `xs` is the dimensionless KID fractional fitted-resonance-frequency
change, `delta_f/f_res`. The fitted resonant frequency is expected to match the
active probe tone. Increasing `xs` means increasing absorbed optical power.
The stream measures changes in total optical loading and has no absolute DC
sensitivity. No additive baseline subtraction or other normalization occurs
inside SCI-CAL. PTC, downstream of CAL, owns atmospheric/common-mode cleaning
and DC removal. The response is treated as linear over expected operational
loadings. A wrong Tune invalidates that premise; displacement beyond about
half a resonance width is a high-noise quality condition, not a silent change
of observable.

## Q02 - pipeline boundary

SCI-CAL applies the multiplicative absolute factor and target-atmosphere
correction to the ordinary `xs` stream before PTC. PTC owns downstream DC
removal and correlated-atmosphere cleaning. SCI-CAL performs no additive
baseline operation.

## Q03 - source flxscale

The frozen SCI-BEAM contract is the normative producer authority. Beammap fits
the standardized detector response in `delta_f/f`, uses its declared
finite-source-convolved effective core model, and supplies a top-of-atmosphere
nominal-beam source flux. The source APT carries

`flxscale = F_TOA,nom / A_TOA-equivalent`

in `mJy beam_nom^-1 / (delta_f/f)`, together with calibrator/epoch, source
model, nominal beam, passband/spectrum, source-atmosphere treatment, fit
support, acceptance, and available uncertainty/covariance. SCI-CAL consumes
this value; it does not rederive it.

## Q04 - transfer and optional TolProj rescaling

The default is the temporally closest scientifically accepted source APT;
Tune, loading, focus, and detector-state equality are not additional matching
requirements, and no universal maximum age is imposed. The reducing scientist
may direct TolProj to create a child APT with one per-array pointing-source
photometric rescale when a sufficiently recent independent ALMA, SMA, planet,
or well-calibrated asteroid reference is judged suitable. The same array
factor applies to every admitted detector in that array. TolProj records the
source, authorities and epochs, inferred and reference fluxes, spectral
convention, factor, and uncertainty when available. Primary-mirror efficiency
variation with aberration is a leading hypothesis, not an established
mechanism.

## Q05 - photometric and spectral convention

Per-array reference frequencies are those used by `toltec_beammap`: a1100
272 GHz, a1400 214 GHz, and a2000 150 GHz. Calibration-source spectra are
source dependent and are commonly represented by `S_nu proportional to
nu^alpha`. BEAM/toltec_beammap supplies one per-array `mJy/beam` value to
`flxscale`. SCI-CAL applies that factor without a second source color
correction. Target-source color correction is a downstream scientist action.
The uncorrected product retains its declared reference-spectrum convention.
The atmosphere authority uses one modeled array-average passband per array;
detector/network variation is unavailable.

## Q06 - WVR and atmosphere operator

The LMT water-vapor radiometer is the opacity source. Raw `tau225` readings are
written into the observation's `tel*.nc` files at approximately five-minute
cadence and are interpolated in time onto the detector sample grid. The
content-bound operator is
`am12_fixed_djf25_piecewise_linear_los_tau_v1`, with fixed
`LMT_DJF_25.amc` reference profile, support `0 <= tau225 <= 0.25` and
`25 <= elevation_deg <= 80`, analytic unity at zero, full sample airmass,
`X_ref=0`, shape-preserving PCHIP in elevation at each opacity anchor, and
piecewise-linear interpolation in line-of-sight optical depth across opacity
anchors. It has no switch at `tau225=0.15` and fails closed outside support.

The frozen TolTECA-v1 passband set is integrated with the declared reference
spectrum:

`T_eff = integral R(nu) S_alpha(nu) T_AM(nu) dnu /
         integral R(nu) S_alpha(nu) dnu`.

SCI-CAL applies `C=1/T_eff` once. Precomputed reference-spectrum surfaces
exist for `alpha={-1,0,2,4}`, defaulting to zero; interpolation or extrapolation
in alpha is not authorized. The machine contract is commit
`7156881bd1a47e8cece97b8c541a013c93ac03e1`, contract digest
`7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a`,
and node-table digest
`fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f`.

## Q07 - observation-level opacity policy

The `0.15` and `0.25` limits are flexible experience-based operational
guidance, not physical discontinuities or demonstrated error thresholds. CAL
assigns one class to the whole observation and never splits it into science
and engineering pieces. CAL owns the deterministic classification and allows
a `0.025` tolerance so momentary excursions do not invalidate the whole
observation. This tolerance does not authorize extrapolation of the numerical
operator: individual samples outside operator support are unavailable while
brief unsupported samples do not by themselves invalidate supported samples.
Changing the policy requires scientific-owner approval.

## Q08 - uncertainty products

Measurement-noise estimation is downstream; CAL does not create a statistical
variance or weight. CAL should ultimately report an array-dependent systematic
calibration uncertainty for scientist use. BEAM should ultimately attach
`flxscale` uncertainty to source APTs; none is presently supplied. CAL must
estimate WVR/atmosphere uncertainty because the WVR reports no measurement
uncertainty. A TolProj pointing-source rescale must ultimately carry its
uncertainty in the child APT; that mechanism is not yet defined. Systematic
scale terms are common across detectors within an array. The WVR driver is
observation-common across arrays, with array-dependent response; numerical
cross-array covariance is unavailable. Until the producer mechanisms and
covariance model exist, total uncertainty and total significance are
unavailable, not zero.

## Q09 - realizable closure and transfer validation

Validation follows the actual calibration workflow:

1. reduce calibrated Beammaps and produce source APTs containing Beammap
   `flxscale` values;
2. use `toltec_beammap` to validate those APTs as scientifically acceptable
   calibration products;
3. enter accepted APTs into `apt_library`;
4. use TolProj's normal machinery to associate library APTs with observations;
5. reduce the matched observation through the ordinary science path and
   measure recovered source flux.

Matching a source APT back to its generating Beammap and reducing that Beammap
as science is an end-to-end numerical closure test, not independent absolute
calibration evidence. Applying the Beammap-derived APT to its associated
pointing observation is a calibration-transfer test when the pointing source
has an adequate independent ALMA, SMA, planet, or well-calibrated asteroid
flux. Available tests are repeated and reported separately for every array.
The record identifies observations, source and child APTs, TolProj
associations, references, arrays, atmosphere, recovered/input flux ratios or
residuals, repeatability, absolute recovery, and observed condition
dependence.

The one-percent representation, five-percent repeatability, and five-to-ten-
percent absolute-accuracy figures are reporting benchmarks, not automatic
contractual pass/fail ceilings. The job is to report achieved performance
honestly and as accurately as possible. No combinatorial validation matrix or
arbitrary minimum sample size is required. Final scientific acceptance is an
explicit owner decision based on the resulting evidence.
