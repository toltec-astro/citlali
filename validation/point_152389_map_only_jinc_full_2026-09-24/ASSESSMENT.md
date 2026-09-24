# POINT 152389: assessment of the selected map-only JINC realization

This JINC realization produces smoother maps but does not establish a better
measurement representation for this POINT result. It preserves the shoulder
response deficit, increases peak phase dependence, and does not consistently
improve response-referenced additive precision. Keep this a completed diagnostic
comparison; no production or feedback change follows from it.

## Same inputs and numerical fidelity

All 15 original adaptive C+D states used the same authenticated processed
samples, original means/bases, applied m9 operands, uniform gamma 1 coefficients
and native continuous AST coordinates. Ordinary signal/numerator/hits replay
and all 15 original measurement vectors are exact. No REFSEQ state entered.
The executed call paths only export saved-input AST and accumulate maps; they
contain no observation Learn/ALS, new candidate or feedback application.

All 75 actual JINC accumulations passed the componentwise signal, denominator
and Q/T checks. The largest bound/tolerance ratio was 0.03521, below 1; all
contributing pixels passed. No-member pixels remain unavailable. Negative C
was retained when certified. Three M0 exact identities and 12 T10 rejoin checks
passed, with maximum discrepancy 2.774e-11 mJy/nominal beam.

The first FITS serialization shortened CDELT and changed the existing stamp
selector. A separately recorded header-only repair restored the exact original
WCS in 231 files while preserving all 1,692 image-plane byte sequences and 1,125
native output files. No fitter or accumulated value changed. Failed preparation
attempts and the initial render are preserved.

## Source response, shape and phase

T10 fitted peaks change from 937.42 to 896.05(a1100),664.73 to 638.93(a1400),
and 813.62 to 814.00(a2000), in mJy/nominal beam. The nominal calibration beams
remain 5/7.5/10 arcsec. Major/minor FWHMs change from 6.353/5.513 to 6.511/5.707,
7.388/6.916 to 7.651/7.271, and 9.679/9.372 to 9.763/9.487 arcsec. Centroid changes
are at most 0.034 arcsec in magnitude. These are descriptive gridding changes,
not recalibration or evidence of a different physical optical beam.

On the original full supported benchmark domains:

| Array | Shoulder g, MAP | Shoulder g, JINC | eta, MAP | eta, JINC |
|---|---:|---:|---:|---:|
| a1100 |0.97437|0.97363|0.07571|0.06896|
| a1400 |0.91107|0.90920|0.14559|0.14235|
| a2000 |0.84988|0.84933|0.22117|0.21240|

Compact positive/negative g remains near unity relative to each operator's own
injection reference. Smaller residual norms mainly show smoothing; they do not
restore information removed by the saved adaptive histories. The a1100 ordinary
shoulder ideal-addition fit is constrained and retained. No signed-difference
map was fitted as a positive source.

Across the nine continuous-profile offsets, peak range/mean increases from
0.239% to 0.913%,1.540% to 3.309%, and 0.489% to 2.724%. Maximum centroid errors
are 0.0166/0.0161 arcsec(MAP/JINC) for a1100,0.0561/0.0734 for a1400, and
0.0300/0.1169 for a2000. All 54 fits succeed without constraints. JINC introduces
small signed fixture lobes; phase shape and original-aperture values are saved.
This outcome concerns the single selected kernel and 16-bin midpoint scheme.
It is not evidence about all possible JINC configurations.

JINC's arithmetic support expands to 30,879/30,828/36,466 pixels versus ordinary
science support 9,829/9,780/10,601. All original comparison pixels remain
supported. The added edge region includes 2,609/3,040/4,441 negative-C pixels at
REF T10 and can have very large normalized excursions. On JINC's entire native
support, shoulder eta is 0.254/15.779/4.206. These are explicit domain-dependent
edge effects, not arithmetic certificate failures. Native and common-domain
results must not be conflated or hidden by a new retrospective cut.

## Does lower pixel scatter improve measurement precision?

Common-background T10 centered scatter decreases from 16.05 to 10.82,12.85 to 7.54,
and 21.73 to 7.83 mJy/nominal beam. The covariance becomes more correlated; its
propagation largely removes the apparent measurement advantage.

| Array | Peak input-referenced scatter, MAP | Peak input-referenced scatter, JINC |
|---|---:|---:|
| a1100 |8.022|8.980|
| a1400 |6.432|6.021|
| a2000 |7.015|7.166|

These units are mJy/nominal beam of the tested pre-PTC compact input. Template
and aperture comparisons are mixed as well. All 1,536 prescribed trials
completed with zero failures;20 finite constrained fits remain in the results.
Per-scatter Monte Carlo relative standard error is approximately 4.43%; an
independent scatter ratio has about 6.3% Monte Carlo precision. The small
between-method differences do not establish physical JINC-minus-MAP significance.

All six ensembles are qualified conditional model calculations. The source fit
contains 1,257 pixels; depth lies outside the learned background range for
2/8/67 ordinary pixels and 204/783/1,257 JINC pixels. The a2000 JINC calculation
therefore extrapolates its amplitude model over the entire source fit footprint.
Do not present it as an empirically validated source-location uncertainty.
The retained a2000 regional-model qualification and a1100 nw 3/scan 5 condition
remain. Finite-domain/negative-correlation/PSD handling is unchanged; a2000
JINC's negative spectral mass before projection is 3.79%. Both base and larger
mask analyses are available. Direct covariance, generation, background-coupling
and exact trial-regeneration checks pass. A Gaussian processed-source baseline
omits the observed non-Gaussian structure and does not validate PTC recovery.

## Cost and bounded completion

Preparation took 60.87 s in the successful attempt, reusing independently verified
native AST/cache work. The three cache enclosure walls were 6.64/9.83/17.76 s,
run concurrently. The grid campaign took 617.28 s wall,4,891.98 s summed native CPU,
and 15.02 GiB sampled aggregate RSS with at most 12 independent jobs and one
accumulation/library worker each. Preparation process high-water RSS was 4.64 GiB;
it is a different statistic. Native accumulation plus bound evaluation accounts
for 4,734.38 summed process-wall seconds; cache startup accounts for 2.76 s and
loading 177.27 s. The measured bottleneck is contribution accumulation and
certification, not the finite-cache reference. Source/response/phase diagnostics
took 2.80 s; covariance and six ensembles 16.55 s. Detailed receipts separate
observed timings; no unmeasured noise-stage RSS is inferred from the grid peak.

All 30 primary maps,18 logical references and 54 phase maps are completed; the
six 256-trial ensembles are completed with qualification. All 12 background cells
and larger-mask analytic calculations are available with recorded transfer
limits. No required original source domain is unavailable. Detailed matrix
status, raw/constant-subtracted control tables and trial-level flags accompany
this assessment.

## One recommended next action; do not execute

Recommend a **correlation-aware admission experiment on the established ordinary
MAP representation**. The current result motivates using measurement-scale
correlation information for admission; it does not motivate making this JINC
realization the feedback representation. Define that next experiment separately
without changing the present threshold, source model or weights here.

For any later JINC-feedback investigation, first define the candidate quantity,
signed/missing-support semantics and its sample-space projection, then test the
full projection/regridding round trip. A JINC map is a signed normalized local
estimate of processed brightness, not unconvolved sky truth. Regridding a
projected JINC raster can apply its response again; that must be accounted for
explicitly. No inverse, deconvolution, new projector or candidate policy is
implemented by this experiment.
