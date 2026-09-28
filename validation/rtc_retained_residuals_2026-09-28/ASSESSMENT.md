# Retained contamination assessment — partial evidence after failed replay

**B remains the working baseline; fixed11 remains parked. No quarantine was
executed.** The authorized B replay completed RTC/CAL/PTC M0, then failed the
native scope check at T1. The owner explicitly allows no replacement. Full
M0–T5 equivalence, full retained T5 population assessment, and the candidate
comparison are therefore **blocked**, not passed or shown unnecessary. No
exposure recovery, unmasking, containment narrowing, or production change occurred.

## Answers to the owner’s five questions

1. **What survives?** Across all2,069 detector/physical runs,1,386 have accepted
   M0 samples (9,805,999 actual MAP contributions);683 are unavailable. The
   complete CAL/M0 summaries use one rule on all four networks. They expose
   off-source retained M0 extrema up to+14,081mJy/beam and sustained structure.
   These are measured residuals, not a validated contamination classifier.
   Full-population T5 survival is unavailable. The prior32 targeted T5 excerpts
   and final B map remain available and are explicitly distinguished from this
   new partial replay.
2. **What is handled, consequential, or unresolved?** Prior retained transitions
   and the n1_w5 CAL trough are strongly suppressed by PTC. Its surviving
   +743.36mJy/beam residual at native row6965 is150.11arcsec from the fitted
   source. All367 excerpt samples enter the native map, but zero enter final
   science-supported pixels or the40arcsec POINT fit disk. Its maximum selected
   contribution is18.54mJy/beam in the mapped, non-science-supported domain.
   This is a concrete support limit for that excerpt, not a universal contamination
   bound. Other retained-transition excerpts contribute up to4.82 and3.08mJy/beam
   within the fit disk, using the actual T5 operator and full normalization.
   These sums include residual noise; they do not isolate an event bias or predict
   a refitted veto effect. See [generated accounting](generated/DIAGNOSTICS.md).
3. **Did stricter rejection improve the result?** Untested. The baseline replay
   failed beforeT1; no selection manifest or veto implementation was created,
   and none of the four candidate trajectories was launched. This is not a
   finding that no defensible contaminant class exists.
4. **What happened to source response, measurement, noise, support and decisions?**
   New upstream/M0 science and discrete decisions match B exactly. No newT5 or
   response result exists. Saved B T5 peak is429.45mJy/beam with conditional
   sigma15.33; centroid(4.148,7.254)arcsec; fitted sigma axes(2.634,1.981)arcsec,
   or FWHM(6.203,4.664)arcsec. The fit is accepted with angle at its lower bound.
   Template conditional sigma is17.51mJy/beam; aperture2 sigma is2,132.06
   mJy/beam arcsec². The saved source-disk fit residual has19.98mJy/beam scatter
   versus median local noise20.92mJy/beam; normalized scatter0.947. The saved
   background scatter is24.50mJy/beam across7,923 pixels. Spatially correlated
   patches remain visible; this analysis establishes no attribution or universal
   cleanliness bound. Earlier compact gains0.9895–0.9909 and broad gain0.8486
   remain historical B controls, not new comparison results. Positive-injection
   support changes and the inferred normalization mechanism remain as recorded
   in the previous assessment; no threshold or normalization change was made.
5. **Baseline disposition?** Retain B because no valid treatment comparison was
   completed. This result supports neither production promotion nor exposure
   recovery. Continuing science requires a separately authorized work order;
   the unused candidate allowance is not permission to replace the failed B run.

## Evidence and interpretation

[Population and twelve new diagnostic windows](generated/DIAGNOSTICS.md) cover
all networks. Target rules choose each network’s largest absolute off-source
M0 sample and largest fully supported0.5second mean departure on another
 detector; the third is a median-MAD detector at mid-run. These controls are
**typical-scale**, not certified quiet; n1_new2 includes a source crossing.
The prior quiet controls remain available in the previous32-window record.
Selection is outcome-conditioned and provides no catalogue precision/recall.
Adjacent-difference MAD is descriptive, never uncertainty on a step contrast.
Physical gaps are respected; numerical processing-scan boundaries are not new
physical runs. CAL uses its eligibility; M0 uses native cause-zero retained
outputs. No sky-restoration operand is added to the residual diagnostics.

**Observed:** n1_new0/D451 has a localized original-x excursion with opposite
original-r structure and a surviving+5,126mJy/beam M0 peak; nearby detector-index
traces do not show the same waveform. n2_new1/D88 and n3_new1/D64 also show
structured x/r episodes and retained M0 structure. These support a plausible
non-astronomical disturbance interpretation, but T5 persistence is unmeasured.
N2D79’s largest sample occurs in a noisy stretch; isolated noise is not proven
contamination. N3D104 and n4D262 peaks likewise do not establish a distinct class.
N4D179’s shared oscillation and n1D346’s mixed source/common/event context do not
justify assigning all residual structure to a detector defect.

Eleven new excerpts have neither a raw Learn candidate nor event-trial overlap.
N1D346 overlaps event251 with one raw candidate, classified
`isolated-admission-unavailable`; retained parts outside its exclusion remain.
Thus absence of an event-group seed is never equated with absence of raw Learn
candidates. The older n3_w4/n4_w4 selection-label correction is preserved.
No native Learn/Consider/Apply authority was changed.

**Inferred:** some detector-local disturbances plausibly survive into M0;
retained residual morphology alone does not distinguish all noise, sky,
atmosphere, common-mode or fitted-component effects. **Untested:** their complete
T5 persistence, counterfactual map/POINT bias, response/noise change under a
whole-run quarantine, and out-of-sample selector validity.

The oldT5 operator-pixel vector and every segment’s cause mask were authenticated
against the immutable deletion hashes and found identical to the available M0
operator/support. This permits exact occurrence tracing of the saved32 T5
excerpts without regenerating a fullT5 stream. T5 normalization equals native
unit-coefficient HITS. The independently summed M0 residual projection reproduces
its science map within7.96e-13mJy/beam. These are contribution diagnostics, not
independent astronomical maps; zeros outside a selected excerpt mean no selected
contribution, never zero missing sky signal.

## Execution, validity and limits

Science source c517444861022f70c95dde95ed13e6a6a259249b; tree
6e43ba687ace1aae9ec57e0e24a56fb9c3ae1544. Original predecessor cleanup head
17f0b9b9bcfdc605161b2c67dba7ccdf52ffdf72 and previous science839f31b28 remain
unchanged. One charged parent/trajectory;4 initial networkApply starts and4
completions;1 attempted feedback, **zero completed feedback**;0 endpoints,
0 RTC-only captures,0 Q trajectories,0 replacements. Internal refit/remeasure/
reassessment coordinate requests are counted separately in compact execution
accounting; they are not extra trajectories. Wrapper `completed_cycle:1` records
the attempted loop index and must not be read as completedT1 science.

The partial invocation took70.49seconds,82.86 user and7.62 system CPU-seconds,
with sampled aggregate peak RSS2.644GB;1 workload/2 networkworkers/1 numeric
thread/1 feedbackworker. Population extraction took about11seconds separately;
read-only comparison/excerpt/packing costs are diagnostic overhead, not speedup
measurements. The6.919GB single-run reservation remained below the30GB ceiling
with2GB free reserve. Exact root and compact footprints are in the cleanup manifest.

All25,142 completed upstream/M0 binary science hashes match the prior sealed
control. Event/support records and initial measurement/admission agree exactly;
allM0 map/candidate planes and frozen noise arrays are exact. **This is partial
 equivalence only.** T1 raised one critical `FRUIT foreign array/observation`
exception, propagated to the wrapper. The new Python campaign scope was absent
from the native FRUIT scope recognition; upstream request equivalence did not
exercise that route. Both author and preflight review missed it. The defect is
preserved, not repaired through another run; see [route failure](ROUTING_FAILURE.md).

Local gates before launch:207 POINT/budget tests,207 baseline tests,1,314 CTest
passes with one pre-existing disabled test, full config preflight. These passing
gates did not cover the downstream scope mismatch. Local AppleClang/Homebrew
results are supplemental; no Unity/Spack run, push, merge or default change.

Compact diagnostics, maps, configs, native receipts, failure logs and exact
original/source bindings are retained. [Reproduction](REPRODUCTION.md) renders
saved figures without raw inputs, native scratch, fitting or a noise ensemble.
Scratch deletion is a separate, exact-manifest proposal awaiting new owner
approval; previous campaign approvals do not apply.
