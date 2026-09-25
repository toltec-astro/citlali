# POINT 152389: a2000 saved-product diagnosis

The saved histories identify a selection-dependent feedback path consistent with CORR's remaining seed dependence. They do not establish a unique cause or an asymptotic fixed point. The seed's effect on the fitted central peak is small, but it leaves a larger localized map residual. The compact-positive aperture effect is larger than either peak effect. The shoulder-recovery benefit remains real within this experiment.

Only the 15 archived a2000 histories were analyzed. All comparisons below use the same policy's REF at the same stage. No fits, candidate evaluations, covariance learning, scientific processing or feedback were run. T10 was produced with m9; m10 and all nine same-map cross-candidates remain unapplied.

## 1. Where does seed dependence enter?

The imposed m0 seed is identical across policies. Their T1 seed-minus-REF maps are also identical: seed coefficient 0.4225434882, maximum difference 16.5638745 mJy/nominal beam. Thus policy-dependent map divergence does **not** begin in T1.

The first unforced difference is selection from that T1 map. All policies have the same three baseline membership differences from REF (pixel IDs 710, 801, 834), and shared selected brightness differs by up to 14.7753723 mJy/beam. BASE-POINT publishes 62 pixels. CORR adds 29 supplemental pixels, DIAG 26; all these proposals survive connectivity at m1. The three CORR-versus-DIAG selection differences are 790, 882 and 1003. Their CORR margins are +0.01892, +0.01713 and +0.05576; their DIAG margins on the identical state are −0.08314, −0.25900 and −0.03537. The saved numerator is shared at this fixed state, while variance expressions and calibrated thresholds differ. These m1 candidates first affect T2: seed coefficients become 0.08779246, 0.14516933 and 0.14141246 for BASE-POINT, CORR and DIAG.

Pixel 882 subsequently moves from supplemental into baseline membership at CORR m2; it is not a persistent supplemental pixel. Pixel 790 drops at m2. Pixel 1003 remains selected in the CORR seed branch through m10, while its matched REF never selects it. At m10 its brightness is 12.85535 versus 10.95426 mJy/beam, normalized neighborhood evidence 12.00497 versus 11.09080, with identical saved CORR variance 10.25595 and effective weight count 41.05298. CORR margins are +0.05865 and −0.22680. This late crossing accompanies different realized brightness/evidence, not a changed saved variance.

CORR's final nine seed-only members are **916, 959, 962, 1000, 1001, 1002, 1003, 1042, 1043**, at east 2–12 and north 4–10 arcsec relative to the AST origin, north of the source. All are supplemental, below the pointwise threshold, and source connected. Their matched REF has the same baseline mask. Eight members form the late differing set by m5; 916 crosses again at m6 (margin +0.00039) and the nine-pixel difference stays unchanged through m10. Earlier switches matter: REF temporarily admits several of these pixels, and REF pixel 1043 is proposed but rejected by connectivity at m4 despite its positive score margin. Equal pixel counts alone would conceal this history.

The complete ledger retains all 100 implicated pixels across all 165 states, including signed evidence, availability, pointwise and neighborhood margins, fitted constants, effective weight counts, exclusions and connectivity. The short ledger includes the union of the nine final differences and all three initial CORR-versus-DIAG differences. Individual neighbor weights/contributions and separate full-footprint failure flags were not saved. They were not reconstructed. Consequently early changes in brightness versus changed exclusion-dependent weights/normalization cannot be completely separated. Negative saved evidence is preserved; no negative-neighbor clipping is inferred or introduced.

The nine archived same-map cross-candidates independently show that both the selection rule and the prior trajectory matter. On BASE-POINT's REF endpoint, CORR/DIAG add 19/11 pixels; on CORR's endpoint they add 13/7; on DIAG's endpoint they add 6/1. These were never applied. Comparing raw threshold values would not explain these differences.

## 2. Is the seed difference still changing?

BASE-POINT and DIAG continue toward very small differences through T10 (coefficients 6.37485e−6 and 6.17661e−6). Their final masks match their own REF from m4 and m5, respectively, while brightness differences continue to decay.

CORR decreases monotonically from T5 coefficient 0.01875843 to T8 0.01037514, T9 0.01026497 and T10 0.01023426. It is nearly stationary over the recorded late interval, but still decreases; there is no late sign switch. The candidate coefficient changes from 0.02033191 at m9 to 0.02030964 at m10. These observations do not justify extrapolation, a fitted contraction rate or a claim about distinct limiting solutions.

T9→T10 changes in CORR's seed-minus-REF difference have full-map norm 0.0100142 and maximum 0.00190378 mJy/beam; candidate norm 0.00825784 and maximum 0.00182640. The coefficient changes by −3.07026e−5. Fitted-peak difference changes by −0.00084143 mJy/beam, fitted constant by −0.00007738, and the large-aperture difference by −0.374208 brightness-area units. These are changes in the paired difference, not each branch's full drift.

CORR REF and seed maps still change separately: their T9→T10 norms are 0.0206236 and 0.0108077, maxima 0.00370861 and 0.00180484. Their candidate norms are 0.0166404 and 0.00871386. Their fitted peaks change by +0.00196754 and +0.00112611; their large apertures by +0.791827 and +0.417618. The measurement tables include every branch's own terminal change and machine-readable own changes at every stage.

At T10 the exact CORR candidate partition is:

| Term | Pixels | Seed coefficient | Norm | Large-aperture difference | Fitted-constant part of aperture |
|---|---:|---:|---:|---:|---:|
| Shared selected brightness | 83 | 0.00345721 | 3.11784 | +51.80162 | −23.70420 |
| Seed-only members | 9 | 0.01685243 | 61.11026 | +564.41076 | −59.78535 |
| REF-only members | 0 | 0 | 0 | 0 | 0 |

The signed terms close exactly. Their fitted-constant coefficient contributions are −0.00174671 and −0.00179646; the higher seed fitted constant reduces published candidate brightness. This is arithmetic bookkeeping of the realized model, not a physical causal separation. Candidate aperture differences are not total-map aperture differences.

Saved cleaning/restoration provides the next part of the chain. At T10, restored applied-m9 seed coefficient +0.02033191 and cleaned residual coefficient −0.01009765 sum to +0.01023426. Their full-map maxima are 33.6541 and 30.0934 mJy/beam; substantial cancellation leaves a 4.70609 maximum total difference. Saved detector means differ by up to 2.26553 in the inherited native brightness units and are not restored by the parent convention. All compared a2000 segment records converge at rank 10 with no application failures, eligibility changes or output-cause changes. Accounting/remap discrepancies remain below 1.57e−11 versus the inherited 1e−8 tolerance. Receipt basis distances refer to each branch versus the original parent, **not** a computed seed-versus-REF basis distance. The products establish this accounting chain, not which cleaning or admission change independently caused persistence.

## 3. Does this explain compact asymmetry?

There is overlapping behavior, but no established common cause. Compact-plus/minus maps and responses remain identical across policies through T2; m2 selections differ, followed by policy-dependent T3 maps. That is one stage later than the seed-policy map divergence. CORR's positive distortion falls to 0.0343888 at T4, then grows to 0.0911916 at T10; its even-response RMS rises from 0.368160 to 0.936557 mJy/beam over that interval. BASE-POINT's final positive distortion is 0.0467917 and even RMS 0.429297. DIAG ends at 0.0547467 and 0.823125.

All nine late CORR seed-membership differences overlap the applied m9 compact-plus differences (25 pixels). Eight overlap compact-minus differences. These are consistent with a related sensitivity at the northern admission boundary. However compact differences extend to other pixels and keep changing after the seed mask stabilizes. DIAG has strong compact asymmetry while its seed mask difference vanishes. The same seed persistence is therefore not necessary for compact asymmetry, and spatial overlap does not prove causation.

The saved compact references are exact signed opposites on the inherited domain; ideal even contribution is zero at every stage. The measured even residual is therefore not an unnoticed nonzero ideal reference. Minus is a decrement to a bright source. All 99 signed-response records preserve the original 10,601-pixel benchmark without lost/gained support or changed normalization. Even RMS uses the 1,681-pixel displayed source stamp; seed-shape norms use 1,257 finite seed pixels. These domains must not be interchanged.

Cached T10 ideal-addition fits distinguish response changes from the descriptive fitter. CORR compact-plus fitted peak increment is +80.50248 versus ideal +81.35425 (discrepancy −0.85177 mJy/beam, −1.047% of ideal increment). The corresponding fixed-template discrepancy is +1.35059 (+1.575%), but large-aperture discrepancy is **+776.46457 (+9.284% of its ideal compact increment)**. Minus gives peak discrepancy −0.14827 and aperture discrepancy +206.43959 (−2.468% of its signed negative ideal aperture). These differences are not new uncertainty estimates.

For CORR plus, actual-minus-ideal east/north shifts are −0.003799/+0.003461 arcsec, major/minor FWHM changes +0.021698/+0.026461 arcsec, and constant change +0.253390 mJy/beam. All policies, both signs, physical ellipse components, cached guard flags and raw/background-separated apertures remain available. No early-stage ideal fits were made.

## 4. Practical size and retained shoulder benefit

| T10 seed minus own REF | BASE-POINT | CORR | DIAG |
|---|---:|---:|---:|
| Maximum absolute full-map difference, mJy/beam | 0.000435 | 4.706088 | 0.000426 |
| Fitted-peak difference, mJy/beam | +0.000280 | −0.103601 | +0.000277 |
| Peak difference / own REF peak, % | +0.0000344 | −0.0127448 | +0.0000341 |
| Large-aperture source-relative difference | +0.083063 | +173.555237 | +0.080749 |
| Aperture difference / own REF aperture, % | +0.0000980 | +0.203416 | +0.0000951 |

For CORR the actual sampled seed-parallel peak is **0.412740 mJy/beam**, 0.050729% of the original REF peak. This is distinct from the fitted-peak change and from the map maximum. The seed-domain total norm is 9.48148, consisting of parallel norm 2.98448 and orthogonal norm 8.99952 in quadrature; the orthogonal component dominates. Full supported-map norm is 9.91466 over 10,601 pixels. The coefficient 0.010234 is a fraction of the artificial shoulder, never a 1.02% whole-source error or an upper bound on spatial error.

CORR seed centroid changes are +0.000478 east and +0.002633 north arcsec. Major/minor FWHM changes are +0.004067/+0.003633 arcsec (0.041884%/0.038698% of their own REF widths); fixed-template amplitude changes by +0.223595 mJy/beam. The fitted constant rises +0.071398. In the 18.868344-arcsec aperture, raw difference +252.95001 minus constant contribution +79.39477 gives +173.55524. These aperture values have units mJy/nominal beam arcsec², not calibrated total source flux. Smaller fixed apertures and their support remain in the tables.

The data-injected shoulder benefit is preserved: T10 gain/distortion is 0.849881/0.221173 for BASE-POINT, 0.992104/0.113269 for CORR and 1.004496/0.131261 for DIAG. Large-aperture actual-minus-ideal discrepancies improve from −1951.09981 (−14.452% of ideal shoulder aperture) to +100.88993 (+0.747%) and +100.97111 (+0.748%). This does not imply every measurement improves: CORR/DIAG shoulder fitted-peak discrepancies are −1.47392/−1.59576 mJy/beam, versus BASE-POINT +0.38108. The report retains the different measurement responses without combining them into a score.

The approximately 5% calibration/PSF goals are not pass criteria for these post-RTC/CAL perturbations. a2000 source-location noise transfer remains unvalidated; frozen-C0 guard errors are conditional reference scales. No paired-difference significance, random error bar from initialization sensitivity, new weight or FRUIT threshold follows.

## Deliverables, limits and next action

NUMERICAL_REPORT.md gives checkpoints/partitions and the five inspected figures. MEASUREMENT_TABLES.md and PIXEL_LEDGER.md provide practical quantities and a short complete-by-rule pixel subset; RESULTS.json, PIXEL_EVIDENCE.json, ARRAYS.npz and INTERPRETIVE_TABLES.json retain all stages and overlap identities. REPRODUCE.md records exact revisions, inputs, commands, runtime and availability. FINAL_REVIEW.md records independent arithmetic, interpretation and figure review; the derivative seal is referenced by repository COMPLETION.json.

Supported explanation: early supplemental selection changes the subsequent restored/cleaned path, and CORR retains a stable late set of nine additional members with different brightness and partially cancelling cleaning/restoration. The saved evidence cannot isolate the causal influence of early mask choice, adaptive cleaning and later selection.

**One next bounded action, recommended but not executed:** a preregistered a2000 CORR seed-branch counterfactual that removes exactly the three CORR-only m1 pixels (790, 882, 1003), preserves every retained m1 brightness value, and continues the unchanged recipe only through T10. Compare against the archived CORR seed and its own REF. This would test the downstream influence of the earliest policy-specific membership decision; it would not validate noise significance or prove a general mechanism. No new branch, processing or feedback for that test was started here.
