# POINT peak failures: noise dependence and readout sensitivity

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the bounded [saved-product scope](SCOPE.md). Starlet remains **parked for
POINT**. All registered outcomes, thresholds and the earlier failed numerical
image-stability qualification are preserved.

**The strongest supported effect is cancellation of shared noise-dependent
peak errors in matched ratios.** Starlet's favorable matched H/D result does
not carry over to crossed-noise pairs. Availability is a separate limitation,
with identifiable score and fit-domain failures. The saved products also prove
a contribution from the common source readout, but do not determine how much
of the larger between-noise variation originates in the total maps versus the
free Gaussian-plus-plane measurement.

## Accuracy and availability are different failures

The primary H/D/T population has 18 terminal array measurements per arm. All
have source evidence, usable centroids and completed feedback decisions.

| Terminal primary peaks | Pixelwise P | Starlet C |
| --- | ---: | ---: |
| Usable | 10/18 | 11/18 |
| Withheld solely because peak / outer MAD <=5 | 6 | 4 |
| Withheld solely because inner-domain peak changes by >1% | 2 | 3 |
| Withheld by a shape warning alone | 0 | 0 |
| Raw absolute peak within 5% of truth | 9/18 | 7/18 |
| Usable **and** raw absolute peak within 5% | 3/18 | 6/18 |

The score is empirical map contrast, not calibrated uncertainty. A usable
measurement is not a certificate of 5% accuracy: seven usable P peaks and five
usable C peaks miss that absolute budget. Conversely, six withheld P peaks and
one withheld C peak meet it. These facts do not authorize changing either gate.

C's first-seed D rejections are specifically: a1100 domain change **2.552%**,
a1400 score **4.750**, and a2000 domain change **2.101%**. Second-seed D a2000
has domain change **1.192%**. C's other three primary low-score cases are
second-seed H/D/T a1400, scores **4.908 / 4.426 / 4.953**. P's two domain
rejections are second-seed a1100 H and T, both **3.240%**. Every primary rejection
has exactly one failed predicate. The expanded compact population adds
shifted-H cases; its exact combinations and values are in the
[withholding table](WITHHOLDING_AND_PASS_TRACES.md). Null, coma, boundary and
real-data limitations remain in all 714 saved array-pass records.

## Matched noise hides much of the between-observation variation

Matched pairs share the same nuisance realization; crossed pairs combine seed
1 in one state with seed 2 in the other. These are dependent recombinations of
two existing realizations, not new observations or six independent trials.

| H/D terminal ratios | P matched | P crossed | C matched | C crossed |
| --- | ---: | ---: | ---: | ---: |
| Raw ratio within 5% | 5/6 | 3/6 | 6/6 | 2/6 |
| Both peaks usable | 2/6 | 2/6 | 1/6 | 1/6 |
| Usable and within 5% | 2/6 | 1/6 | 1/6 | 1/6 |
| Largest absolute raw error | 5.962% | 16.183% | 0.800% | 7.905% |

For C a1400, changing seed lowers the H peak by **5.536%** and D by **4.850%**;
the matched gain changes only **0.721%**. For C a2000 the corresponding peak
changes are **-6.584%** and **-7.821%**, with matched-gain change **+1.343%**.
P a2000 behaves similarly: H **-11.665%**, D **-10.945%**, matched gain
**-0.809%**. Shared-seed variation cancels algebraically in the ratio; it remains
in an unchanged-source or crossed-seed comparison. This narrows the claim of
improved ratio fidelity without changing the original matched-test result.

The response-loss T/H diagnostic agrees with this interpretation. Raw
within-5% counts fall from 6/6 to 2/6 for P and from 6/6 to 4/6 for C when
noise is crossed. Exact per-array, per-direction ratios and all retained-pass
values are in [the ratio record](MATCHED_AND_CROSSED_RATIOS.md).

## What the pass traces establish

The [saved trajectories](SAVED_PEAK_TRACES.png) and numerical trace table retain
peak, both widths, central and outer backgrounds, score, domain sensitivity,
warnings and elapsed time. No favorable iteration was selected.

Across the two H seeds, a2000 fitted width area increases **5.563% for P** and
**3.464% for C** while peaks decrease. Their Gaussian integrals fall less than
their peaks: **6.751% for P**, **3.348% for C**. Peak, width and fitted background
therefore vary together. This is not evidence that integrated brightness should
replace the owner's peak metric, or that one background term alone causes the
error. For C a2000, fitted background at the source center shifts by -0.559 map
units while the outer-plane value there shifts by +0.058; the two background
estimates are different quantities and are not interchangeable.

C's first-seed a1400 H fit makes a gross excursion at pass 4, with the saved
support/domain checks withholding it; later passes recover. The corresponding
width/background changes remain visible. Successful feedback solves and a
usable final centroid do not establish a uniformly stable readout sequence.

A more specific readout effect is identifiable in P. The affected-array T maps
are the expected 0.8 H maps to approximately 1e-13 relative agreement in the
central domain throughout all seven passes. Yet the saved Gaussian-plus-plane
fits need not scale identically. At first-seed pass 6:

- fitted peak differs from exact 0.8 scaling by **+0.8168%**;
- major/minor widths change **-11.04% / +7.24%**, despite the maps differing
  only by the known scalar to numerical precision;
- T's saved fit SSE is **0.32485% higher** than the feasible scaled H fit's SSE;
- empirical score moves from **4.9838** to **5.0245**, making T usable while H
  is withheld.

Thus the saved fitting procedure selects different feasible solutions on an
effectively identical normalized map, and that changes availability. This is a
common evaluator effect, not a starlet-feedback failure or a missing PTC replay.
The saved three-start fit reports alone cannot establish its numerical cause.
It also does not explain all 6–16% crossed-noise errors. C's T/H total maps
actually differ from simple scaling after feedback, so its analogous fit
changes cannot all be assigned to the readout.

## Exactly one proposed next experiment

**Run one controlled readout comparison on the same saved terminal H/D maps.**
Compare the retained free Gaussian-plus-plane peaks with a linear amplitude
measurement using the exact injected source shape as a fixed unit-peak template,
plus a freely fitted plane. Use both existing domains (60 and 52 arcsec), all
24 H/D terminal array maps across both arms and seeds: **48 small linear fits**.
Amplitude remains unconstrained and is compared with truth only afterward.

This is an explicitly truth-assisted diagnostic ruler, not a proposed POINT
estimator, feedback model, selector or availability rule. Do not feed its result
back, promote its apparent stability or accuracy to deployability, or change
existing labels. It needs no PTC or feedback refit. Detailed bindings and the
one scientific question are in [the proposed experiment](NEXT_EXPERIMENT.md).

If the crossed-noise error remains in the fixed-template amplitude, a substantial
component is already present in the source-aligned content of the total maps.
If it falls substantially while the retained free-fit peak varies, the free
shape/background readout materially amplifies the noise dependence. Either
result would narrow the outstanding map-versus-readout distinction; neither
alone would qualify a production measurement. **This proposed experiment has
not been run.** No OOF campaign or other next experiment is proposed.

## Verification

All 714 saved peak-availability labels reproduce from their recorded predicates.
The diagnosis computes 504 matched/crossed ratios, checks 84 exact ratio
identities, and compares 84 array-pass scaling cases using 56 saved H/T map
files. All 237 prior packet payloads, 1804 prior external products, 57 frozen
method-authority payloads and opaque review-archive statuses are preserved.
[Verification](VERIFICATION.json) records zero pre-PTC reads, cleaning calls,
refits, optimization calls, threshold changes or reserved-observation use.
The [input bindings](INPUT_BINDINGS.json) descend from clean commit
`4742e0ad561f7e64b25b9225179818caf5ec01b9`.
