# POINT bounded repair and retest

**Retain the evaluator repair; reject the completion-only repair at the declared
tolerance. The numerical qualification fails, so no operational trajectory was
rerun.** This closes the authorized attempt at its saved-problem gate.

The [protocol](PROTOCOL.md) was frozen at `377164001` before execution. It
implements the owner's one-repair direction and preserves the
[previous registered trial](../fruit_point_operational_gate_trial_2026-09-12/SCIENTIFIC_REPORT.md).
This is experimental method evidence, outside the frozen scientific core and
production. It does not qualify a starlet policy or reject wavelets generally.

## The evaluator recovers useful pointing information

All 167 saved total maps were reassessed identically for reference P and
candidate C. The original 501 array measurements and truth scores remain
unchanged. The new rule uses central source evidence, association with the
fitted source, support, and sensitivity of a free Gaussian-plus-plane fit to
two fixed domains. No injected position, amplitude, width, case name or arm
identity enters that rule. Truth evaluates the resulting labels afterward.

| Saved-map finding | Result |
| --- | --- |
| P compact/mild terminal centroids usable | **18/18**, previously 10/18 |
| External centroid error range for those 18 | **0.0336–0.5888 arcsec** |
| Shape warnings on those maps | All **7** retained |
| P compact/mild terminal peaks usable | **11/18** |
| Null/background source reports or usable measurements | **0/168** array-pass records, both arms combined |
| Near-boundary records | All **48** retain support warnings; no usable corrections |
| Gross-coma P terminal maps | Source evidence and shape warning in **6/6** |

This supports separating centroid usability from photometry and shape warnings.
It does not make the photometry generally adequate. P still supplies only **2/6**
available H/D gain ratios meeting the 5% target, and **2/6** available D/H
degradation ratios meeting their target. Ungated H/D fit ratios meet 5% in five
of six cases; those raw values are retained, not silently promoted. Fitted width
changes and all original residual/support diagnostics remain in the records.

No unchanged-H pair both has usable peaks and passes the 5% repeatability test:
a1100 and a1400 are withheld, while usable a2000 shows a **11.67% apparent peak
loss** between nuisance seeds. The affected-array T/H raw ratios are 0.80653 and
0.80000, but neither pair has jointly usable peaks. A warning already present
on H is not evidence that the imposed T degradation was detected. Three of four
unaffected-array comparisons are available and equal unity; the fourth is
withheld. The revised evaluator is more informative, not uniformly more lenient.

Two gross-coma reference maps receive provisional centroid labels together with
their shape warnings. This experiment has no validated coma-centroid ruler;
those labels do not establish 1-arcsec accuracy for grossly distorted sources.
Their peaks remain withheld. Near-boundary limitations remain explicit.

On real123424, the saved P a1100/a1400 maps receive provisional centroid and
peak labels with shape warnings; P a2000 remains support/shape limited. Saved C
a1400 receives those labels; C a1100 fails fit-domain stability (2.18 arcsec
centroid and 50.1% peak changes), and C a2000 remains support/shape limited.
There is no real-data truth or calibrated uncertainty here. C still has **no
completed synthetic-source trajectory**: revised labels cannot create missing
terminal maps. The full [evaluator evidence](EVALUATOR_EVIDENCE.json) preserves
availability, raw measurements, warnings and scores separately.

## The gradient threshold does not adequately stabilize the feedback image

All **144** retained candidate array problems were used: 60 nonempty and 84
empty-support cases. Selected coefficients, objective, background, source
domain and normalization were checked against the old records. Empty support
remains exact zero; repeated null maps are not independent trials.

For each nonempty problem, the same L-BFGS-B path started at zero. The first
finite, nonnegative iterate meeting relative projected gradient <=1e-4 was
captured; that path continued to <=1e-6 without restarting or changing solver
families. The operational cap remained 3000 iterations. The diagnostic tighter
comparator alone could use 12000.

Every nonempty problem reached the declared threshold in **21–611 iterations**.
Thus a completion rule can resolve the optimizer-success mismatch within the
existing cap. But numerical accuracy, rather than the success flag, is decisive:

| Saved problem group | Problems | Tighter comparator available | All applicable stability checks pass |
| --- | ---: | ---: | ---: |
| Real123424, retained passes | 21 | 21 | 9 |
| Compact, mildly broadened, displaced or response-loss cases | 27 | 27 | **0** |
| Gross coma | 6 | 5 | 0 |
| Near boundary | 6 | 4 | 1 |
| **Total nonempty** | **60** | **57** | **10** |

The registered numerical allocations were **0.5% peak and 0.1 arcsec centroid**,
one tenth of the operational budgets. Two readouts answer different questions:

- **Actual feedback image:** across the 27 compact/mild problems, sampled-peak
  differences are **0.94–67.27%** and positive-brightness centroid movements are
  **0.0022–16.73 arcsec**. No compact problem passes the sampled-peak allocation.
- **Gaussian source readout of that image:** differences are at most **2.49% in
  fitted peak** and **0.189 arcsec in fitted centroid**. Fifteen of 27 pass both
  smaller numerical allocations; all 27 stay within the full 5% and 1-arcsec
  operational budgets.

The second finding prevents an overstatement: this is **not a demonstrated
failure of compact-source operational accuracy**. It is a failure to establish
that the declared tolerance consumes only a small part of those budgets and
leaves the actual feedback image sufficiently stable. FRUIT would subtract the
whole feedback image before relearning PTC; a stable Gaussian summary alone
cannot establish that subsequent iterations are unaffected.

For example, on H/first-seed/a2000, the sampled model peak changes **67.27%**
and its brightness centroid moves **3.48 arcsec**, while its Gaussian peak
changes **2.49%** and fitted centroid only **0.040 arcsec**. The
[saved model comparison](FEEDBACK_STOPPING_MAPS.png) shows the additional
structure. Lower objective and tighter gradient are not evidence of better
astronomical recovery. These results are consistent with weakly constrained
image directions; they do not prove nonuniqueness or identify its full cause.

Three problems lack a 1e-6 comparator at the diagnostic cap: first-seed coma
a2000, and a1400 in both boundary cases. Eight other comparisons have a Gaussian
readout limitation. The extreme boundary changes remain in the evidence and
[stability figure](STOPPING_STABILITY.png), but are not used to claim typical
compact-source performance. Even excluding them descriptively, the 27 compact
problems fail the registered full-image stability condition. No cases were
removed from the gate or replaced by a convenient earlier iterate.

## Consequence, timing and preservation

The evaluator saved-stage gate passes. The stopping qualification fails, so
the conditional 34-trajectory comparison is **not admitted**. There were **zero
new pre-PTC input reads, zero cleaning calls and zero operational reruns**.
No peak-recovery improvement, preservation of degradation through a new
recurrence, or speed advantage is established. The previous measured C/P real
trajectory times (43.51/8.07 seconds) remain historical results, not timing of
this proposed completion rule. The identified OG benchmark and its broader
205.38-second timing scope are preserved without a direct speed-ratio claim.

Saved-map work took **272.11 seconds**, including **230.34 seconds** of numerical
solves, with **146.28 MiB** peak RSS. It retained 293 external payloads totaling
4,903,651 bytes, including all stopping-point models, traces and reassessments.
Full source/input and product bindings are in [FREEZE.json](FREEZE.json) and
[RUN_PRODUCT_MANIFEST.json](RUN_PRODUCT_MANIFEST.json).

[Verification](VERIFICATION.json) checks 144 fixed problems, 60 original
objective/gradient records, 177 new checkpoint records, 345 finite nonnegative
supported model arrays, three directional derivatives and 17 matched P/C
evaluator bootstraps. The three focused unit tests pass. A verification-only
BLAS warning was [repaired and recorded](ROUTINE_REPAIR.md); both attempts are
retained, with no optimization or cleaning rerun. Both figures were inspected.
All 584 old reduction payloads, 139 prior review payloads and 57 frozen
ordinary-MAP payloads are unchanged. Protected worktree states and the two
opaque review archives are preserved; the archives were not opened or hashed.

**Recommendation:** keep the separate evaluator judgments as development
evidence and close this completion-only candidate repair. Another tolerance,
objective, source-support rule or image-selection procedure would be a new
scientific method decision, not routine maintenance under this authorization.
The existing POINT reference and frozen contract remain available foundations.
This result does not authorize another sweep, independent-pointing evaluation,
numerical policy, Unity run or production change.
