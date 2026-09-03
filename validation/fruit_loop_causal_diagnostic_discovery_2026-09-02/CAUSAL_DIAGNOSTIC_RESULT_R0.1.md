# SCI-FRUIT causal-diagnostic discovery result r0.1

Result: **one specific warning mechanism was found, but no reliable general
stopping signal was established**

This is an exploratory read-only analysis of the completed EL-F1 and EL-F2
development runs. It performed no new reduction and changed no FRUIT
algorithm. Injected-minus-control measurements label whether an update helped
or harmed the known injected source. Every possible warning signal was then
calculated from one trajectory at a time, using only information an ordinary
run could possess. The analyzer did not use truth to choose a threshold.

The 51 array/iteration rows and their exact inputs are bound by
`CAUSAL_DIAGNOSTIC_MANIFEST_R0.1.yaml` and recorded in
`CAUSAL_DIAGNOSTIC_METRICS_R0.1.csv`. The tested map-update region is the
40--120 arcsec annulus already used to expose the compact-source residuals.
These two already exposed pointing observations are development cases, not a
population from which a stopping policy can be qualified.

## What the failed EL-F2 update told us

The large a1400 failure from iteration 4 to 5 did leave a warning that existed
before iteration 5 ran. At the end of iteration 4, the injected trajectory
added this penalty to its carried checkpoint state:

| Field | Value |
| --- | --- |
| Producer | `mapdiag:raw_obs` |
| Reason | `map_pixel_outlier_detector_dominance` |
| Array | a1400 |
| Scan | 5 |
| Detector UID | 4460 |
| Score | 4 pixels |
| Penalty factor | 0: complete exclusion |

The paired control did not add this penalty. Its iteration-4 learning record
found UID 4460 contributing to three targeted pixels at approximately
107--109 arcsec from the source. The injected record found the same three and
one additional pixel at row 144, column 281, sample 264. The added record had
value 111.971 and leave-one-out z-score 1.78356. The configuration requires
four contributing pixels and then assigns factor zero. Thus the controlled
injection changed this decision from three pixels, below threshold, to four
pixels, at threshold.

The implementation evidence agrees with that chronology.
`include/citlali/core/pipeline/mapdiag_workspace_learning_emit.h` calls
`record_detector_penalty(..., true)`, explicitly allowing this record during
the apply phase. `include/citlali/core/engine/learning.h` immediately merges
an accepted record into the effective detector-penalty state. That state is
in the iteration-4 checkpoint and is therefore seen by iteration 5. This
source inspection explains the observed state transition; it is
implementation evidence, not scientific authority.

The next injected a1400 update is the conspicuous one:

| Quantity, iteration 4 to 5 | Injected | Control |
| --- | ---: | ---: |
| Feedback-map annular update growth | 1.7553 | 0.7933 |
| Output-map annular update growth | 1.6587 | 0.8504 |
| New a1400 penalties applied | 1 | 0 |

Over the same transition, known-source central recovery fell from 0.890451 to
0.822828 and annular residual increased by a factor of 3.8408. This is a
strongly localized association between a discrete learned-state change and
the failure. It is **not yet proof that the exclusion caused the failure**:
the required counterfactual, in which the same state is advanced without
applying that newly learned penalty, has not been run.

Ordinary summary products alone make the failure much less obvious. Injected
a1400 pointing-fit signal-to-noise changed from 103.199 to 102.742, only
-0.44 percent, while map median RMS rose 4.76 percent and fitted amplitude
rose 4.15 percent. A bright pointing source can therefore conceal a large
loss in the much smaller injected response.

## Why this is not a stopping rule

None of the simple signals tested separates helpful and harmful updates across
both development cases:

- Update growth catches the EL-F2 a1400 event, but misses both late EL-F1
  a1100 failures. At EL-F1 alpha 1.25 iteration 6, annular truth residual grew
  by 1.2609 while feedback and output updates shrank to 0.4509 and 0.4933 of
  their preceding size. At alpha 1.50, truth residual grew by 1.4394 while the
  two updates shrank to 0.4591 and 0.4381.
- Growth also has direct counterexamples. EL-F1 alpha 1.25 a1100 iteration 4
  had feedback growth 2.2800 while its annular truth residual improved by
  7.58 percent. Its a2000 iteration-6 feedback grew by 1.1595 while that array
  passed the inherited scientific screen.
- A newly applied hard penalty accompanies EL-F2 a1400 and EL-F1 alpha 1.25
  a1100 failures, but EL-F1 alpha 1.50 a1100 fails without one. EL-F1 a2000
  also receives a new penalty while remaining acceptable. The event is
  therefore a mechanism candidate, not a universal warning.
- Update direction is not consistent either. The failed EL-F2 feedback update
  has positive successive-update cosine 0.184, while the failed EL-F1 alpha
  1.50 a1100 update has cosine -0.833.

There is also no successful EL-F2 early-stop result hidden at iteration 4.
Against the frozen alpha-one iteration-6 reference, every all-array EL-F2
candidate state from iteration 1 through 5 fails at least one scientific
protection. Avoiding the iteration-5 collapse might improve a1400, but would
not make fixed alpha 1.25 an acceptable result on this case.

The bounded conclusion is therefore:

> No tested simple, single-run diagnostic is a reliable general FRUIT stop
> signal across EL-F1 and EL-F2. A late hard detector exclusion is a plausible
> mechanism for the specific EL-F2 a1400 collapse and merits a causal test.

This does not say that no usable causal signal exists. It says that selecting
a rule now would overstate two exposed development cases and would ignore
known counterexamples.

## Recommended next experiment

The smallest useful next step is a counterfactual of the EL-F2 event, not a
new tuning sweep:

1. Start both alpha-1.25 EL-F2 trajectories from their exact iteration-3
   checkpoints, before the UID 4460 penalty exists.
2. Use a development-only variant that still records new map-diagnostic
   penalties discovered during the apply phase but holds them out of the
   effective state. Preserve all penalties already present in the checkpoint.
3. Require iteration-4 maps to reproduce the original maps. The only intended
   difference at that boundary is whether the new penalty enters future
   state.
4. Advance once to iteration 5 and compare the injected response. The control,
   which did not acquire the penalty, is an internal unchanged-result check.

If a1400 recovers without the exclusion, the experiment supports a causal
penalty-policy problem. If it still collapses, the penalty is only a marker of
another instability. Either result is informative. Because EL-F1 already
shows a failure without such a penalty, even a positive causal result would
not establish a complete stopping rule or generally safe recurrence.

That counterfactual requires separate owner authorization before any prototype
or reduction is made. No method promotion, qualification, production change,
or Stage B authoring follows from this diagnostic result.

## Evidence identity

| Artifact | SHA-256 |
| --- | --- |
| Analyzer | `accc3db78ef12fc163ec26157e771800e8abc8e699b76965119d31553e7221d0` |
| Analyzer tests | `de5ed81a1d5f98baf0c80b171595b4c763daad57cfccff699797eb65497ca4a4` |
| Diagnostic manifest | `1cad3cf7e0511a9272aa599399ab3bd2a95a088d9c40594ce1cf8c923c129594` |
| Diagnostic metrics | `0ae36f14b00bc11e00595e355ffe7497901294402dff98dc5f9dd297e19da778` |
| EL-F1 truth metrics | `90aeadeed0535668105517b963287c780e87421128591e02269495ccb6b1a1f9` |
| EL-F2 truth metrics | `3728e5de118c0799ce797a6074b376b22d76a45faf8a9f20b7bbff4cd213ca58` |
| EL-F2 control iteration-4 learning record | `7f25a0a3228167d51a11a7b05fc68f1e292600697b8e73e0a72f5a2645f69fa9` |
| EL-F2 injected iteration-4 learning record | `0336bb9adf62da4f6c0c2a5ccd607129017862aaca2ec34b75f0af4caecaf14e` |
| EL-F2 control iteration-4 checkpoint | `0eb7a0e9d8b35a4168f542c07142f34dff048244a92dc6fa718cd8812e2cd351` |
| EL-F2 injected iteration-4 checkpoint | `c9eee5fada65fe7d9172d39ba84fb275b4124eea635933c95c29b101e6c2192f` |
| EL-F2 injected merged configuration | `05ab51cc0976ea316a30335e3c6ec8f2a4a57479fd85475eaf91ad0c36387db2` |

The large development products remain outside the repository at the exact
roots named in the diagnostic manifest. The original data and all reduction
products were left unchanged.

## Repository verification

All 198 baseline and FRUIT-loop Python tests pass, including four focused
tests for the new analyzer. Ruff and Python byte-compilation pass. A fresh
51-row analyzer output is byte-for-byte equal to the recorded CSV, both YAML
records parse successfully, and `git diff --check` passes. Compiled CTests and
configuration preflight were not rerun because this result changes no C++ or
configuration behavior.
