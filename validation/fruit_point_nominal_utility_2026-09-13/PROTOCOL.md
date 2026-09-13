# Fixed nominal-estimator POINT utility comparison

2026-09-13 · r0.1 · Prospective, before execution.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md) and
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
Adopt the frozen original-parent, residual-relearning, reconstructed-total
replacement recurrence and PTC/MAP boundaries. This is isolated development
outside scientific authorship, implementation conformity and production.
Cite the original [operational trial](../fruit_point_operational_gate_trial_2026-09-12/PROTOCOL.md),
[failed saved-problem qualification](../fruit_point_bounded_repair_2026-09-12/SCIENTIFIC_REPORT.md)
and [six-branch diagnostic](../fruit_point_feedback_sensitivity_2026-09-13/SCIENTIFIC_REPORT.md)
as preserved evidence. The new owner direction prospectively supersedes the
recommendation to constrain fine scales; none of those old results is relabeled.

## Owner decision and question

`SCI-FRUIT-POINT-NOMINAL-UTILITY-2026-09-13` records the supplied
[owner directive](OWNER_DIRECTION.txt). It authorizes this one bounded full
comparison with the already tested nominal estimator. Does C provide useful
POINT peak-response fidelity or availability, preserve pointing, and finish
at acceptable cost? The previous tighter-solution qualification remains failed.
We test a fixed bounded approximate estimator, without asserting image stability,
uniqueness, numerical uncertainty, generalization or production qualification.

## Exact estimator, recurrence and comparison

C uses the existing positive central starlet objective, coefficient eligibility,
five empirical-scale selection, band/coverage calibration, source domain,
outer-plane background treatment, and normalization. Import those definitions
unchanged from the operational candidate. Import the tested nominal path from
`../fruit_point_bounded_repair_2026-09-12/stopping.py`: zero initialization,
L-BFGS-B, ftol=gtol=0, maxcor=10, maxls=20, and the first accepted iterate with
finite values, nonnegative variables and relative projected gradient <=1e-4.
The denominator is the fixed zero-start gradient infinity norm, floored at
1e-12. Work caps remain 3000 iterations and 30000 function evaluations;
retain the path's existing 180-second callback check. No tight solve is run.
No optimizer success flag substitutes for this explicit completion condition.
Empty coefficient support supplies a valid zero replacement; other unavailable
solves retain their trial and terminate that trajectory, never applying it.
Check finite, nonnegative, exact-domain output after restoring physical units.
Background is recorded separately and is never astronomical feedback.

P is the same positive total-map pixel selection above three current outer MAD
scales on original D. C uses original D intersect radius <=60 arcsec. This
named policy comparison includes that existing domain difference. Both use the
same immutable pre-PTC input, flags, uniform weights, 2-arcsec containing-pixel
maps, inherited network/chunk groups, and rank 5. On every pass subtract the
current model from the immutable parent, relearn PTC from that residual,
reconstruct the total, and infer full replacement feedback from that total.
No expected source amplitude, position or width enters feedback or usability.
There is no fine-scale constraint, taper, regularizer, new optimizer or sweep.

## Fixed population and execution limits

Retain the original [case definitions](CASES.json): N, B, H, H-shift, D, C, T,
E for seeds 20260911 and 20260912, plus real123424. Keep all three arrays.
These are the existing compact, displacement, peak-loss/broadening, response-loss,
null/plane, gross-coma and boundary cases. No new truth, input or OOF campaign.
The calibration seed is reused, so this is development, not held-out validation.

Rerun both arms for credible same-harness timing: 34 trajectories, seven passes
each including bootstrap, at most 238 new cleaning calls. Alternate synthetic
arm order exactly as before. Compare all repeated P arrays/maps/models/learned
states and all parent hashes against saved controls where present. Reuse saved
geometry, calibration, detector scales and truth definitions; hash-bind inputs
and code before execution. Use fresh output paths and preserve all old products.

Pass 6 is the sole primary terminal, with all seven feedback decisions required
available (including the final inferred model). No truth-selected earlier map,
early stopping, silent zero continuation or successful-only denominator.
Retain all passes and failures. One hour overall, 600 seconds per trajectory
checked between passes, 8 GiB peak RSS, 4 GiB output, four numerical threads.
No automatic cap extension or replacement campaign follows an adverse result.

## Common evaluator and predeclared POINT decision

Apply the existing repaired data-only evaluator unchanged to both arms. It
separates source evidence, centroid usability, peak usability, shape warnings
and support limitations, including source association and fixed-inner-domain
sensitivity. Truth is passed only to the external scorer. Widths remain free;
shape warnings are not automatically centroid vetoes. Retain the evaluator's
existing compact-domain and support limitations for precision measurements.

Retain original U1--U5 and G1--G3 finite-case questions with that repaired
availability interpretation:

- H, H-shift and D: every required centroid usable and <=1 arcsec external
  error. Report displacement error and two-seed mean vectors separately.
- H/D: joint usable peak ratio within 5% of 1/0.9. D/H: within 5% of 0.9 and
  below 0.95. Unchanged H(seed2)/H(seed1): jointly usable and in [0.95,1.05].
- T/H: jointly usable ratios within 5% of [1,0.8,1], with affected a1400 below
  0.9. Withheld affected peaks remain warnings, not accurate recovery or a
  demonstrated response-loss detection when H was already unavailable.
- Gross C: source evidence and a shape/support warning, without a precision
  coma or focus claim. E: support warning and no reliable correction.
- N/B: zero candidate feedback admissions and zero source reports at every
  retained pass; count baseline false feedback separately. Do not excuse C
  by comparison with an inferior baseline. Total support/contributors remain
  unchanged; C feedback outside its declared domain is exactly zero.

These 1-arcsec and 5% budgets are experimental. Absolute fitted peaks and widths
are also reported against truth; an additional absolute-recovery claim must
meet 5%, rather than confusing a larger peak with improved recovery. Account
for every required array/case in availability, including trajectory failures.
Report unavailable readouts and their raw errors separately; never present raw
accuracy as operational yield.

For this new question, the earlier standalone exterior-RMS <=1.10 gate and
feedback-image/tighter-solution agreement are diagnostics, not prerequisites.
This prospective applicability change is authorized by the owner's POINT
hierarchy; it does not change old gate results. Residual structure, model changes,
widths and learned state remain retained. They block advancement when they
cause a required POINT output, honest warning/null behavior or reliable completion
to fail. Do not start another explanatory morphology/PCA study during this run.

## Timing, disposition and stopping

Charge all cleaning, mapping, inference and admission work. Record evaluation,
output, full cumulative wall time to every map and completed trajectory, and
algorithm-component sums separately. Show errors and availability versus
cumulative wall time without selecting a favorable terminal using truth.
The identified OG control remains contextual: seven passes in 205.38 seconds
with different upstream/JINC/weighting/one-thread boundaries. It supplies no
matched speed ratio; no historical environment reconstruction is required.

Advance only for a demonstrated operational subset whose required gates pass,
with preserved pointing and honest availability, and a substantive benefit:
a required usable 1-arcsec/5%-ratio case that P fails, or equivalent useful
accuracy at lower measured cumulative wall time. Report any accuracy gain that
misses cost. More than 2x matched full-trajectory wall time blocks advancement
for that case; availability changes alone cannot hide failed required pairs.
A richer/different image or mere solver completion is insufficient.
Otherwise park C for POINT. No automatic repair, parameter, second revision or
new experiment follows. Preserve evidence for possible later OOF study without
claiming OOF suitability. Independent-pointing replication needs separate
approval before policy recommendation; reserved 129081 remains unopened.

Routine implementation defects may be repaired and documented within the same
method, population, inputs, gates, resource and cleaning-call bounds. Stop for
a scientific change. No Unity, production, weighting exploration or UID 4460
study is included.
