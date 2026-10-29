# Decision separation: synthetic outcome

**Disposition: useful separation, but revise the temporal veto before activation.**
The [candidate rule](DECISION_RULE.md) preserves 0.10 and separates full-observation
response from subset stability. The standalone prototype has no application
imports, operational entry point, configuration activation or reduction access.
Neither existing runtime nor observational detector membership changed.

## What was tested

47 focused tests pass. **Two additional scientific acceptance assertions fail**
for known stationary-response counterexamples and are retained as strict XFAILs;
this is not a scientific acceptance pass. The existing native optical learner's
10 synthetic tests also pass, as unchanged-owner regression evidence only.

The independent experiment generates six blocks of known optical signal, two
peer reference panels and additive noise; full/early/late fits condition on
frozen robust weights. It enumerates exact multinomial resampling probabilities:
462 count combinations for six blocks, ten for three. It preserves constant
blocks and makes singular resampling explicitly unavailable. It does not execute
Citlali or reproduce its held-out-reference construction, source masking, binning,
or finite 2,000-draw RNG. No native-conformance or uncertainty-coverage claim.

Twelve declared physical cases were tested at seed 42 and across seeds 0–15,
giving 204 evaluations. No seed, gain, noise level, response cut, confidence
level or expected decision was changed after seeing the failures. Related
scenarios reuse the same underlying noise realization; these are not independent
replications of a population false-rejection rate.

| Known physical case | Candidate eligible / 16 | Old six-interval rule eligible / 16 | Interpretation |
| --- | ---: | ---: | --- |
| Healthy normal response | 14 | 16 | Two spurious temporal-review vetoes |
| Quiet healthy response | 14 | 16 | Quietness does not imply a dark detector; same veto issue |
| Noisy healthy response | 14 | 16 | Normal response generally retained; weighting remains separate |
| Limited excitation, true gain one | 14 | 0 | Separation rescues the intended subset-uncertainty case, with the same two veto failures |
| Quiet weak response, gain 0.03 | 0 | 0 | Rejected despite low noise |
| Reversed response, gain −1 | 0 | 0 | Rejected |
| Genuine positive gain change 1 → 1.8 | 0 | 16 | Separate temporal analysis finds a change that a minimum-response cut alone misses |
| Late weak response | 0 | 0 | Withheld |
| Late reversed response | 0 | 0 | Withheld |
| No reference excitation | 0 | 0 | Unavailable, not confidently weak |
| One constant-reference block | 0 | 0 | Required singular resample is unavailable; block is not deleted |
| Response at the cut with broad uncertainty | 0 | 0 | Unresolved |

## Counterexample and consequence

The healthy detector has physical gain exactly one. At seed 7, full-panel
intervals are approximately [0.999, 1.003]; early intervals [1.0014, 1.0062]
and late intervals [0.9899, 1.0008] do not overlap. At seed 15 the same mechanism
occurs. A non-overlap veto mistakes finite-noise fluctuation for a reason to
withhold a genuinely stationary detector. Both panels share the target noise,
so requiring both panels to flag it would not repair these counterexamples.
Noise amplitude changes preserve this failure pattern in the related fixtures.

The provisional rule correctly distinguishes imprecision from a confidently
weak subset, and catches the injected large gain change. But separate percentile
interval non-overlap is not a validated change test. Its scientific gate remains
open. Do not claim that 2/16 is a calibrated real-world false-rejection rate, or
use these fixtures to choose a new confidence level or tolerated-change number.

## Recommended next decision

Keep full-observation response as the optical eligibility question and retain
the 0.10 physical cut. Keep uncertain short subsets explicit, not automatically
equivalent to dark detectors. Preserve non-overlap as a review diagnostic only
until a temporal-change assessment has an adequately tested null behavior and a
scientific definition of harmful change. It must respect shared-reference and
target-noise dependence and limited block excitation. A direct change estimate
with block-level support is a candidate for that next scientific design; it is
not implemented or selected as authority here.

For now, do not wire this candidate into the pipeline or restore the 179 saved
a2000 candidates. Their inclusion might help, but this synthetic study supplies
no map-level evidence. Continue to preserve the frozen trial and existing
rejection decisions. No arbitrary duration, correlation, noise cap, confidence
retuning or material-change threshold is introduced.

## Reproduction and environment

From the source checkout, using the required local Python environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/gwilson/tolteca/bin/python -B -W error -m pytest -q -rx -p no:cacheprovider validation/optical_decision_separation_2026-10-29/test_decision_rule.py
/Users/gwilson/tolteca/bin/python -B -W error validation/optical_decision_separation_2026-10-29/synthetic_study.py --output /tmp/optical-separation-synthetic.json
/Users/gwilson/tolteca/bin/python -B validation/optical_decision_separation_2026-10-29/summarize_study.py /tmp/optical-separation-synthetic.json /tmp/optical-separation-results.json
```

[RESULTS.json](RESULTS.json) records compact cases, counterexamples, seed-sweep
counts, full result digest and actual Python/NumPy/host versions. This is local
supplemental synthetic evidence, not Unity/Spack or observed-map validation.
The initial small matrix multiply raised host numerical warnings despite finite
results; explicit bounded summation replaced it, with non-finite checks and no
warning suppression. Final study/tests run with warnings as errors. Unrelated
auto-loaded pytest/Jupyter plugins raised a startup deprecation warning; plugin
autoload is disabled for this standalone test module, which needs no plugins.

Full CTest/config/baseline gates were not triggered because application, native
owner, build and YAML code are unchanged. Required validation here is the focused
study, syntax/lint, diff checks and independent exact-source review. No new
reduction, Unity access, frozen-runtime replacement, integration or push occurred.

## Conformance disposition

- Scientific/behavioral: scoped tests executed; temporal automatic-veto candidate
  **fails scientific readiness**, explicitly preserved with its counterexamples.
- Architecture: isolated pure prototype and independent synthetic experiment;
  no Engine/state growth, pipeline dependency or activation.
- Repository/evidence: own bounded validation directory plus current-status
  entry; historical trials, data, application code and other thread preserved.
- Independent exact-SHA review: recorded separately after candidate commit.
