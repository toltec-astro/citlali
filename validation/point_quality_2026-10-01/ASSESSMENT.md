# POINT quality checkpoint — 2026-10-01

**Improvement over earlier processing is not established.** The executed successor
recovers more peak brightness through feedback, with very small within-trajectory
centroid changes, but the available R/S comparison changes population, calibration
application, projection, weighting and cleaning. It cannot establish better flux,
pointing accuracy, morphology fidelity or sensitivity. This is a completed saved-product
checkpoint, with three narrowly justified current-S152420 roles prepared for owner
launch. Final current-S152420 results remain pending. No new observation has run.

## What calibration resolution changed

The factor difference is reconstructed rather than unexplained pipeline loss.
Historical redu09 used beammap152307 with3896.367905/amp(Gaussian); successor
matchedAPT uses explicitly selected148670 with3127.216073/cal_amp(empirical).
Source-flux ratio0.802598 and detector response differences jointly produce
nw1–4 matched median ratios0.4205,0.4464,0.4471,0.4004. All1342commonvalid rows
reconstruct within6.66e-16relative. Successor canonical lineage admission executes;
no duplicated/missing downstream correction was found. Template/ownGaussian
ratios near0.96–0.99 do not explain the factor of two.

[Completed lineage assessment](../point152420_amplitude_investigation_2026-09-30/APT_LINEAGE_ASSESSMENT.md)
remains authoritative. Why148670 is scientifically preferable and the original
March producer/matcher execution/selection rationale remain unproven. The original
sourceAPTs are locally present. No scalar fitted to POINT is applied to maps.
R-V2 and S share148670matchedAPT, but R disables extinction while S applies target
atmosphere (~1.0333 for152420), and detector/sample populations differ. EqualAPT
therefore does not prove equal effective calibration; shape metrics are not declared
calibration-independent. Three old unstarted M0calibration controls were retired,
with freeze/deferral/retirement records preserved and zero starts.

## Exact evidence and scorecard

Machine-readable [saved-scorecard.json](results/saved-scorecard.json) binds seven
original maps bySHA256, frame/units, free fits, apertures, radial residuals,
background tails/scales/correlations/artifacts, source asymmetry and original support. Figures use
identical display scales, fixed commanded-origin40arcsec fit and60–100arcsec
background regions, native/6/12arcsec scale summaries. No image registration,
independent recentering, fitted scaling or cross-frame difference was performed.

| Observation/product | Free peak above constant (mJy/nominal beam) | Major×minor FWHM (arcsec) | Gaussian angular integral (mJy arcsec²/beam) |
|---|---:|---:|---:|
|152420 R-V2|251.702|8.156×5.936|13807|
|152420 executed B M0|223.278|10.814×4.842|13246|
|152420 executed B T5|291.810|10.980×5.193|18853|
|152430 R-V2|298.489|6.434×5.004|10889|
|152430 executed B M0|359.798|5.796×4.115|9724|
|152430 executed B T5 remeasured|431.200|6.213×4.625|14042|
|152420 historical redu09|574.588|12.567×7.312|59821|

These integrals are descriptive brightness/angular measures, not flux density.
FreeGaussian+constant is a measurement, not a source-truth assertion. Radial
residuals and complete10/20/40arcsec apertures retain non-Gaussian contamination.
Native-frame centroids are inJSON: R is AltAz; S is FK5tangentoffset. Their numerical
components cannot be subtracted as pointing error. Within S, M0→T5 displacement is
0.0553arcsec(152420) and0.0757arcsec(152430); this is conditional feedback stability,
not independent pointing accuracy or a significance test.

152420 maps executed source518e05891cf8eb2e84621cdcfcea83525f3d1c49.
The repaired periodic-angle fit changed some applied candidates by up to0.02358
mJy/beam; no material membership change, but old products are not silently renamed
current-S. New REF/compact± roles resolve that exact version gap.
152430 B source9d28b2c6e134375b6f377ae59acc4e9f967ac3d4 reproduces originalB.
[Independent saved audit](results/reviewer-152430-feedback-audit.json) checks M0–T5
with the repaired measurement operator: applied membership unchanged, candidate
roundoff≤1.083e-7mJy/beam. T5peak429.451→431.200 changes only unappliedm5.
[Rebound endpoint](results/152430-rebound-measurement.json) recomputes local Jacobian
and covariance: peakσ15.369mJy/beam, nativecx/cyσ0.1148/0.06935arcsec.
These are conditional background-covariance propagation, with high-coverage
continuation, not independently calibrated uncertainties, model-learning error,
absolute calibration error or paired-R/S significance.

| Measurement |152420 assessment|152430 assessment|
|---|---|---|
|Pointing accuracy|Not established: frame registration and independent subsets absent|Not established: same; conditional endpoint errors available|
|Photometry/source fidelity|Not established versusR; Bfeedback raises peak30.69%, angular integral42.33%|Not established versusR; Bfeedback raises peak19.84%, angular integral44.41%|
|Residual noise/sensitivity|Not established: unequal supported domains and no R preRTC response|Not established: same; background scatter larger in savedS, not a matched sensitivity comparison|
|Supported area|Operationally degraded:37404vs97278arcsec²|Operationally degraded:33380vs98094arcsec²|
|Practical cost|174.51s oldB raw-to-final,90.11initial+82.54feedback|165.44s B raw-to-final, sampled process-treeRSS2.782GB|

Area ratios are descriptive domain loss, **not registered common-pixel loss**;
Rincludesnw0 and uses different coverage/weighting rules. S full source fit disk is
supported in both anchors; M0/T5 science support is identical, no lost/gained pixels.
The JSON common-mask comparison includes same-domain M0/T5 backgrounds and paired
changes without assuming independence. Runtimes are actual earlier local executions,
not matched R/S speedups. New evaluation wall time is separately recorded.

Native background standard deviation R/S-T5:28.83/29.46mJy/beam(152420),
11.62/24.50(152430); robustMAD22.55/24.55 and10.06/22.53. Cross-frame/grid/domain
and coefficient differences prevent sensitivity claims. At6/12arcsec smoothing,
complete kernel support shrinks152420 S background to1402/321pixels,152430 to
1671/580pixels. Only~8Gaussian12arcsec beam areas remain for152420; comparative
large-scale noise inference is inadequate. Reported tails/extrema are descriptive,
not calibrated false-source probabilities. Uninjected maps retain conspicuous
outer structure; injected-minus-ownREF can cancel it and is insufficient alone.

Existing fully learned B compact± spatial gains are0.99090/0.98951(152430), with
shape errors0.08299/0.02149; preserved shoulder gain0.84860. Old152420 gains
1.0281/1.0232 and shoulder0.8608 remain earlier-version evidence. These pixel-space
gains are **not** the fitted amplitude gaing_A. Do not divide the fitted amplitude
sigma by them. Compatible current152420 estimator/response extraction is included
in the three-role batch; comparable R response remains unavailable. No new shoulder
reconstruction or covariance ensemble is planned.

## Consequential gaps, split and work accounting

R is actual V2redu00: per-networkPCA5,1arcsecJINCAltAz,validatedweights,noFRUIT,
noextinction. a1100includesnw0 plus1–4; scoped R/S is unavailable. The native
retained `_purge_retained.nc` files explicitly omit samples. Legacy `chunk_select`
selects products, not a independently relearned population; current supported
successor dispatch does not expose a validated R subset/preRTCroute. The legacy
injection hook followsRTC/CAL. A second legacy pipeline was not constructed.

One split fixed before new outcomes:152420 segments0/2/4 vs1/3/5.
[Coverage audit](results/subset-geometry.json):4,101,358vs4,062,357retained
contributions,573,561vs667,577source-disk contributions, both1257/1257diskpixels.
Physical intervals/gaps are retained in the audit. This supports one plausible
balanced split, not independent reductions. Matched R/S subset stability is
**requested but not established through the available interfaces**, not geometrically
impossible. Conditional submaps would not close that gap and were not substituted.

[Matrix](MATRIX.json): only three new S152420 roles, onefull+twocompact,15feedback
applications and3endpoints; noQ/fixed11/noOOF. New starts0, failedstarts0,
replacements0. The overall14start ceiling is not stacked with oldcampaigns;
unused slots remain parked. Source/inputs/review/binary/resource checks precede
owner launch. Exact counts and native completed/unknown receipt facts will be in
campaignACCOUNTING.json, including failures. No automatic retry.

Owned completedQ binary scratch7,816,846,444bytes was removed after preserving
maps, inputs, receipts, feedback/noise state and compact response/consequence reports;
[hashed removal manifest](results/retired-scratch.json.gz). It is not a resumable
checkpoint. Original raw/tunes/APTs/historical maps were untouched.

Largest scientific limitation is lack of a calibration/population/frame-matched
R response and independent stability control. This limits a universal improvement
claim, not the ability to move development toOOF. No further POINT tuning starts
automatically. See [OOF handoff](../../handoff/POINT_TO_OOF_2026-10-01.md).

Exactacquisition/tune/detector/config/cost bindings are in
[point-inventory.json](results/point-inventory.json): bothPOINTacquisitions are
obs/0/2 and each uses its ownobs/0/1processedTune. Networks1–4 are explicitly
enumerated with canonicalAPTmanifest and immutabledependencyreceipt bindings.
The new preparation rehashes152420originaldependencies before science; the saved
152430 audit preserves executed input identities without pretending to rerunraw.

Prelaunch review caught and repaired the existing physical-raster layout adapter:
current shapes usePRIMARY whereas older response files useINJECTION. The shared
response reader now accepts both explicit layouts, tested against an actual bound
current shape. No reduction was spent on this reporting defect. The asymmetry
descriptor measures180degree odd structure within paired40arcsec support, reflecting
about the freefit center solely for shape diagnosis. It includes noise/interpolation
and is not a registered pointingcomparison or truth-based fidelityscore.
