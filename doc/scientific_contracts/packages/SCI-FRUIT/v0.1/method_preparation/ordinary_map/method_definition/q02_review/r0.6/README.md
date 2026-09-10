# Compare maps before choosing a weighting policy

Q02 review r0.6, 2026-09-10. **Prepared for owner review; implementation and
execution are pending.**

## Program adherence and prior-work recovery

Continue under the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [frozen prior-work recovery](../../r0.4/PRIOR_WORK.md). Adopt the unchanged
Q01 definition, Q02-A/B substance and provisional 20% precision objective.
Cite the completed [precision diagnostic](../../../../../../../../../../validation/fruit_q02_weight_precision_2026-09-09/SCIENTIFIC_REPORT.md)
as empirical evidence. Its result supports testing a four-chunk candidate; it
does not select that candidate or certify its errors.

The owner's “I agree with the proposed next step” authorizes this concrete
mapping-test preparation. The new comparator, fallback, input use and numerical
screen below remain proposed. The disposition table in the
[protocol](MAPPING_SCREEN_PROTOCOL.md) identifies every changed earlier choice.
Preserve all earlier packets and products. This is manager experiment
preparation, outside the independent scientific-author channel; no new author
reference or dispatch is admitted.

## The proposed test

Answer a practical question: **Does uniform weighting leave enough map quality
on the table to justify pursuing a more complicated rule?**

Compare only two policies on identical admitted occurrences:

- **U:** every occurrence has relative weight one.
- **N4U:** inverse centered training scatter, estimated separately for each
  detector over each four-chunk window. A group with no usable training keeps
  uniform relative weight one. Other groups are normalized together so the
  full array's occurrence-weighted mean remains one.

That fallback is a new scientific choice for review. It retains both missing
152389 groups and every other admitted detector. Their total and central-map
influence will be visible. It does not treat missing training as good noise
precision. The 20% objective is not a veto, clipping rule or fallback trigger.
Four chunks pool weight training only; PCA chunk lengths stay unchanged.

| Part | Exact scope | What the result can establish |
| --- | --- | --- |
| T1: synthetic tests | Nine existing noise/quality cases, 1,024 paired trials each, plus fixed arithmetic/fallback checks | Bounded losses under the declared synthetic laws, including changing noise, bursts, calibration error and training contamination |
| T2: discovery maps | 123424 and 152389, all three arrays, fixed geometry and source guards; full and three window maps per policy | Observed changes in background structure, source appearance, support and cost; exact fixed-state mapping response checks |

T1 retains the proposed 5% RMS loss, 2% amplitude/width and 0.02-width centroid
limits and a 242-bound extension of its Monte Carlo prescription, with the
same per-tail error allocation. T2 uses
explicit observed-change alerts, **not** confidence intervals made from pixels.
A quieter T2 map cannot certify noise improvement, true source recovery, or
uniform acceptability. PTC transfer, calibration uncertainty and real-data
noise covariance remain unmeasured by this screen.

The [protocol](MAPPING_SCREEN_PROTOCOL.md) binds the fallback algebra, exact
inputs and adapter, 2-arcsec containing pixels, support threshold, map counts,
measurements, alerts, verification, resources and failure behavior. The
[manifest](REVIEW_MANIFEST.json) locks that proposal and its source versions.
No further method choices are delegated to the implementer.

## One decision and a finite stopping point

Proposed owner decision: **SCI-FRUIT-Q02-MAPPING-SCREEN-R0.6**, currently **open**.
Recommend approval of this exact T1 plus exploratory T2 screen, including its
limited empirical input permission, isolated local implementation, prescribed
checks and one campaign. The combined bounds are two hours, 4 GiB peak aggregate
memory and 2 GiB new output, one worker and one numerical-library thread.
Approval would also cover routine implementation repairs within these choices,
with every attempt preserved; it would not require another implementation vote.

A scientific failure in one synthetic stress case is retained and does not
cancel the other declared cases. Arithmetic or provenance failure blocks
interpretation. After the complete screen, make one owner disposition: the
case for continuing is concrete enough to prepare reserved-pointing replication,
or close this weighting exploration and return to the simpler reference and
contract. An unresolved result stays unresolved; it does not trigger an
unbounded search over weights, windows or observations. Any new estimator or
uncertainty experiment would require a distinct proposal and decision.

129081 remains unopened and reserved for new weighting outcomes. Replication
and adequate uncertainty evidence precede any pointing-policy recommendation.
T3 feedback, historical-control replay, production changes and qualification
are separate decisions. Historical Citlali remains the mandatory FRUIT control;
JINC is untouched. Q02-C and its bundled MAP handoff remain open. A/B substance
needs no repeat vote; this empirical screen does not adopt their successor
sources. `FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.

This revision contains documents and identity checks only. No new scientific
array was opened, coefficient computed, map made, numerical harness implemented,
reduction changed, Unity accessed or GitHub push performed. Frozen authority,
the diagnostic attempt and both opaque review archives are preserved.
