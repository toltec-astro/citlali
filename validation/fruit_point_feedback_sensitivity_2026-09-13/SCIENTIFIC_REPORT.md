# POINT paired feedback sensitivity

**The image-peak statistic overstates the effect on the next pointing/peak
measurement, but the reconstruction differences are not harmless.** They
survive the actual projector, change learned PTC subspaces, and sometimes
change source morphology or leave substantial off-source structure in the next
total map. This supports proposing one targeted constraint on unsupported
fine-scale feedback, rather than clearing the existing candidate or choosing
a remedy from its image-peak percentage alone.

The [owner-authorized protocol](PROTOCOL.md) used the saved nominal/tighter
solutions without changing or rerunning their optimizer. The evaluator repair
and failed stopping qualification are preserved. This is a six-branch diagnostic,
not a new candidate, a full FRUIT trajectory or production qualification.

## The bounded comparison

All **141 available pairs** were examined: 57 nonempty and 84 exact-zero pairs.
The three missing tighter solutions remain unavailable. No failed case was
silently substituted. All projections use the existing map-to-timestream path,
actual valid samples and inherited calibration conventions, with no smoothing.

The three replay states were selected before any new cleaning:

1. H, seed 20260911, after saved pass 0: ordinary compact source.
2. T, seed 20260911, after saved pass 1: the largest projected difference inside
   the original data-fitted source core among eligible compact/mild states.
3. Real123424, after saved pass 0: fixed real-data state.

T applies the existing 20% a1400 response/loading proxy, including nuisance;
the source remains compact. Its maximum array core-difference energy was
34.90 million in squared inherited sample units, versus 11.46 million for the
next-ranked state. Selection used neither injection truth nor replay outcomes.
The exact ranking and selected hashes are in `SELECTED_STATES.json`, copied into
[DECISION_EVIDENCE.json](DECISION_EVIDENCE.json).

Each state received exactly two one-step branches from its same hash-verified
immutable parent and preceding FRUIT state. Both branches relearned rank-5 PTC
per existing network/chunk and used the normal model-restoration path. No basis
was frozen and no next feedback model was inferred.

## Where the image differences live

The locations differ materially across cases; this is not universally an edge
or low-coverage problem. Only **5/27** compact/mild saved pairs have more than
half their squared difference in the fixed 52–60 arcsec rim. All pairs' maxima,
coverage, footprints, core/wings, pixel concentration and scale diagnostics are
retained in the evidence.

| Example | Nominal maximum → tighter maximum, arcsec | What changes |
| --- | --- | --- |
| H/a1100 | (20,−8) → (20,−8) | All difference energy lies inside the data-fitted source core; the finest wavelet band dominates. |
| H/a2000 | (18,−10) → (60,0) | Peak 201.0 → 614.3; 94.65% of difference energy lies in the rim. The new maximum has 199 valid contributions, about the 77th coverage percentile, and complete wavelet footprints. |
| T/a1400 | (16,−12) → (28,12) | Peak 95.8 → 252.4; 17.25% of difference energy is in the source core and 19.85% in the rim. The new maximum is at ordinary coverage, around the 59th percentile. |
| Real/a1400 | (16,−6) → (−46,−38) | Peak 2298.8 → 8337.0; 86.56% of difference energy is in the rim. Here the new maximum has low coverage (64 contributions, about the 2.4th percentile) and lacks complete footprints for bands 3–5. |

The finest analysis band carries the largest difference norm in each of these
examples. In all six synthetic array problems selected for replay, **no finest-band
coefficients were selected by the objective**. There are also changes in broader
bands, particularly a1400. The bands are not orthogonal, so their norms are not
claimed as additive fractions of image energy. No diagnostic filtering was
inserted into the feedback projector.

## The constrained evidence changes much less than the image

For the nine selected array pairs, the uncertainty-scaled selected-coefficient
change is **0.026–0.247% of the selected target norm**. This supports weakly
constrained image directions, not proof of an exact null space. The objective
does improve; both relative normalizations matter:

| Example | Nominal objective → tighter objective | Remaining objective removed | Reduction / zero-model objective |
| --- | ---: | ---: | ---: |
| H/a1100 | 353.6756 → 353.6460 | 0.0084% | 0.000156% |
| H/a1400 | 0.025671 → 0.000213 | 99.17% | 0.000603% |
| T/a1400 | 0.116417 → 0.038876 | 66.61% | 0.001162% |
| Real/a1400 | 337774.79 → 337742.82 | 0.00947% | 0.001488% |

Thus the a1400 synthetic solves still make substantial progress relative to
their small remaining residual. Calling that progress zero would be wrong.
But sizeable image changes accompany very small changes relative to the
constrained data. These results do not isolate a unique conditioning defect or
establish that a tighter minimizer is a better astronomical model.

## The differences survive projection and change learning

The actual projector does not eliminate them. Among the selected synthetic
array pairs, projected difference norm divided by nominal projected-model norm
ranges from **9.84% to 175.00%**. This is a model-relative comparison, not a
sample-noise test. Every available pair has per-group source-crossing and
shared-waveform diagnostics: **6,816 group records** with contributor counts.
The energy in the simple shared detector mean is often small, yet the actual
learned subspace still changes; a whole-observation RMS or mean-coherence number
would be an inadequate clearance rule.

Maximum principal subspace angles for the selected synthetic arrays range from
**0.58° to 12.61°**. Real/a1400 reaches **88.02°** in one group, with normalized
projector distance 0.853; its median maximum angle across 36 groups is 0.603°.
These compare rank-5 subspaces, not eigenvector signs or ordering. The
reconstructed-total difference norm for real/a1400 is about **75%** of the direct
restored-model difference norm on the original map domain. The off-source
perturbation is therefore not merely an invisible optimizer detail.

See [feedback and next-map differences](FEEDBACK_AND_NEXT_MAP.png). The largest
changes can lie away from the fitted source even while they substantially affect
PTC and the returned map.

## Next-map source measurements are mostly steadier

Changes below are tighter minus nominal, with centroid movement unsigned.
The **0.5% and 0.1-arcsec numerical allocations** remain reference scales;
the **5% and 1-arcsec operational budgets** are also shown in the
[measurement figure](NEXT_MAP_SENSITIVITY.png). Neither replaces the old gate.

| State | Array | Fitted peak change | Centroid movement | Major/minor FWHM change |
| --- | --- | ---: | ---: | ---: |
| H, after pass 0 | a1100 | +0.101% | 0.00567″ | −0.050% / +0.027% |
| H, after pass 0 | a1400 | −0.270% | 0.01073″ | −0.151% / +0.237% |
| H, after pass 0 | a2000 | +0.051% | 0.02873″ | +0.065% / −0.736% |
| T, after pass 1 | a1100 | −0.067% | 0.00937″ | −0.308% / +0.391% |
| T, after pass 1 | a1400 | **−1.195%** | **0.04351″** | **+13.720% / −7.981%** |
| T, after pass 1 | a2000 | −0.196% | 0.03340″ | −0.064% / −0.654% |
| Real, after pass 0 | a1100 | +0.006% | 0.00007″ | −0.00007% / −0.006% |
| Real, after pass 0 | a1400 | +0.204% | 0.02119″ | +0.068% / +0.068% |
| Real, after pass 0 | a2000 | +0.027% | 0.00253″ | −0.030% / −0.004% |

All nine raw centroid differences are below 0.1 arcsec; eight raw peak
differences are below 0.5%. All raw differences fit inside the full operational
scales. Availability remains a separate judgment: **seven pairs have jointly
usable peaks and eight have jointly usable centroids**.

For T/a1400, the nominal peak is withheld because its fixed-domain sensitivity
is 1.034%, just above the retained 1% limit; the tighter peak is available.
Both centroids are available and both shape warnings remain. The raw 1.195%
peak change is not presented as a qualified comparison of two usable peaks.
Its substantial fitted-width change nevertheless identifies a source-morphology
consequence worth retaining, not relabeling away. Real/a2000 remains shape/support
limited in both branches; its small raw differences do not make it trustworthy.

Truth scores do not consistently favor the tighter solution. For example,
H/a1100 peak bias changes from +2.77% to +2.87%; H/a1400 from +9.04% to +8.74%;
T/a1400 from +14.78% to +13.40%. All 12 synthetic centroid errors are below
1 arcsec (maximum 0.593″), but neither solution supplies universal 5% absolute
peak recovery here. Full width errors, residual norms and aperture brightness
are retained for both. H and T have different preceding pass indices, so their
outputs are not used to manufacture a matched operational H/T gain ratio.

## Disposition and next owner decision

This experiment supports **consequential freedom in the reconstruction**, with
an explicit limit on that conclusion: much of its effect is outside the central
peak/centroid readout, and only one selected source-shape case shows a large
fitted morphology change. It does not establish a mathematical null space,
prove a specific regularizer is necessary, or exclude incomplete numerical
progress as part of the explanation.

I recommend proposing **one explicit image-selection constraint targeting
unsupported fine-scale structure**, keeping source amplitude, position and
shape free. That recommendation now rests on projected and returned-map effects,
not the 67% pixel-peak statistic alone. Its exact form and any execution would
be a separate scientific owner decision. A future stability criterion should
judge source response and returned residual structure together, while preserving
the original failed gate as history. It should neither demand that every pixel
be identical nor ignore off-source structure that changes learning.

No constraint, regularizer, solver change or revised gate was implemented here.
No full trajectory, independent pointing or method-policy qualification follows
automatically. The current candidate remains unaccepted and still must show a
useful scientific benefit at acceptable cost.

## Execution and verification

The run took **41.21 seconds**, with **2.59 GiB** peak RSS and **six** cleaning
calls. Branch wall times were 0.97–1.16 seconds, including map measurement and
output; these timings exclude the original saved-model optimization cost and
do not predict full FRUIT latency. There were **zero new feedback optimizations**.
All 130 external payloads (40,325,819 bytes) are bound by
[RUN_PRODUCT_MANIFEST.json](RUN_PRODUCT_MANIFEST.json).

[Verification](VERIFICATION.json) passes: six exact saved-model applications,
three same-parent pairs, 864 newly learned mean/covariance/eigen identities,
432 independent subspace comparisons, all waveform checks and synthetic truth
scores. The maximum normalized eigen residual is 2.64e-15. Verification made
no extra cleaning calls or feedback solves. Three focused unit tests pass;
both figures were inspected. All 877 prior external payloads, 170 prior packet
payloads, 57 frozen ordinary-MAP payloads and both protected archive states are
preserved. [Workspace and freeze provenance](PROVENANCE_NOTE.md) records the
restoration of missing temporary-worktree files and the hash-freeze ordering.
