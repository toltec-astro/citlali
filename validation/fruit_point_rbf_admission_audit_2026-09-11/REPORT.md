# RBF reassessment: admission, application and processed detectability

## Program adherence and prior-work recovery

This [owner-requested reassessment](OWNER_REASSESSMENT.md) follows the
[program charter](../../doc/scientific_contracts/README.md) and adopts the frozen
[RBF experiment](../fruit_point_rbf_feedback_2026-09-11/SCIENTIFIC_REPORT.md).
Both estimator–admission combinations remain rejected. This audit adds a
narrow interpretation correction and reads saved products only. It performs
**zero new cleaning passes** and does not open 129081 or change an estimator.
The [diagnostic plan](AUDIT_PLAN.md) and code were recorded before its scores
were inspected. A [single starlet preliminary screen](STARLET_SCREEN_PROPOSAL.md)
is prepared for the next owner decision; it has not been implemented or run.

**The criticism is justified.** The earlier sentence attributing low agreement
to fitted noise was too causal. The measured fact is that the candidate fields
failed the declared whole-model agreement rule. That alone does not separate
noise following, estimator variance, coverage effects or insufficient signal.
The underlying basis passed representation tests; the stronger penalty itself
introduced large noiseless model error. Neither fewer nor broader basis functions
is established as a remedy. Those distinctions supersede the causal wording,
without changing the frozen numerical results or rejection decisions.

**The cosine condition blocked recognizable compact sources.** All compact
failures include the cosine rule. In many cases both held-out scores already
exceed five. For example, R1 seed 20260911 a1100 has scores 8.75/8.53 but cosine
0.644; R2 raises the scores to 10.03/10.20 and cosine to 0.779, still below 0.8.
This supports testing detection separately from the ability to reproduce the
entire field. It does not authorize lowering 0.8 after seeing those numbers.
Pixel-parity subsets share upstream processing and are not independent samples.

| Injection | Array | R1 failed gates, maps 1–7 | R2 failed gates, maps 1–7 |
|---|---|---|---|
| 20260911 | a1100 | cosine | cosine |
| 20260911 | a1400 | score_1, cosine | score_1, cosine |
| 20260911 | a2000 | cosine | cosine |
| 20260912 | a1100 | cosine | none; admitted after every map |
| 20260912 | a1400 | score_0, cosine | cosine |
| 20260912 | a2000 | cosine | none; admitted after every map |

Score labels 0/1 identify the two held-out directions, each with threshold five.
Cosine threshold is 0.8. Both coma levels fail all three gates in every array
and every map for both candidates. The mismatch fails cosine alone in a1100 and
a2000, and both scores plus cosine in a1400. Complete case/array/pass accounting
is in [GATES_BY_PASS.csv](GATES_BY_PASS.csv), including null, background and real
maps. The failing-gate classes do not change over the seven maps in these
synthetic trajectories.

**Late admission does not explain the two applied compact failures.** R2 admits
seed 20260912 in a1100 and a2000 immediately after map 1. The nonzero model is
first applied to map 2 and is applied for all six remaining maps. The final
amplitude errors of −13.45% and −11.77% therefore follow six actual feedback
applications, not an unused model first discovered after map 7. The next model
always equals the following map's applied model exactly. See the
[application timelines](ADMISSION_TIMELINES.csv).

For synthetic trajectories with rejection throughout, all seven total maps
are bitwise identical, including supported pixels and unavailable locations.
The zero-model loop repeats the same deterministic operation on the same parent;
it does not accumulate fresh evidence. This is verified behavior, not a reason
to admit unsupported content. The 378 historical passes are two configurations
× nine cases × three arms × seven successive passes, **not 378 independent
scientific tests**. There are only two nuisance realizations.

**The processed-template diagnostic supports a mixed coma interpretation.**
The template is the actual first source-minus-paired-null output, after the
learning/cleaning response it experienced. Three signed template projections
cover the predeclared core, shoulder and tail. Their covariance is estimated
from the other null realization at 41–62 coverage-compatible positions, with
10% diagonal shrinkage. Coverage normalization and the covariance between
projections replace the earlier unprocessed-truth-norm/pixel-scatter proxy.
No source map trains that covariance. Every first source map is identical
across P/G/R and both RBF campaigns, so the diagnostic is shared, not six tests.

All numbers below are **empirical amplitude-scale units, not calibrated sigma
significances or detection probabilities**. The source score includes its
realized nuisance; the response-only score uses the paired difference; the
paired-null score shows the nuisance contribution. The latter two sum to the
source score.

| Coma level | Array | Source score | Processed response alone | Paired-null score |
|---|---|---:|---:|---:|
| bright | a1100 | 5.96 | 5.82 | 0.14 |
| bright | a1400 | 2.57 | 4.22 | -1.64 |
| bright | a2000 | 5.27 | 4.96 | 0.31 |
| half | a1100 | 3.05 | 2.91 | 0.14 |
| half | a1400 | 0.48 | 2.11 | -1.63 |
| half | a2000 | 2.79 | 2.48 | 0.31 |

Compact sources score 17.04–32.13; their paired-null scores are −0.64 to +1.14.
Mismatch sources score 16.44–24.46. Thus recognizable compact-source evidence
is available under this favorable oracle diagnostic even when RBF admission
fails. Bright coma is suggestive in a1100/a2000 but weak in a1400; the a1400
nuisance realization subtracts about 1.64 empirical scales. Half-bright coma
remains weak. The reference-null extrema for coma span approximately −2.20 to
+3.00 across arrays, but that finite set is not a false-alarm calibration.

The oracle knows location and the realized processed morphology, including its
dependence on nuisance-driven learning. Its three-projection covariance assumes
approximate stationarity after coverage normalization; the translated placements
overlap. It is favorable and limited, not a deployable selector, an optimal
full-pixel covariance detector, or independent detection validation. Consequently
we still have not established a reliably detected comatic regime across all
arrays. The audit distinguishes that limitation from the much clearer compact
admission failure without overclaiming either.

**Recommended next decision.** Approve only the proposed saved-map starlet
screen, with signed multiscale evidence, nonnegative source reconstruction,
explicit background and coarsest-band treatment, and no whole-field cosine.
The proposal includes one fixed 4× brighter-coma bootstrap with PTC learning
rerun, to test the intended regime rather than silently relabel the old cases.
It retains the old faint cases. A failed preliminary candidate stops before full
trajectories; a pass leads to a separate owner review of the matched comparison.
A new estimator and a new injection require the scientific-method/input owner
decision under the existing scope boundary. No full campaign is preauthorized.

Two focused covariance/normalization tests pass. All 275 retained files used
in this audit and all 49 payloads of the frozen RBF report packet remain
byte-identical. The [machine results](AUDIT_RESULTS.json) preserve every gate,
timeline, covariance, score and input hash; [DIAGNOSTIC_FEATURES.npz](DIAGNOSTIC_FEATURES.npz)
retains the actual projection filters and reference samples. Both protected
review archives, previous reductions and frozen scientific contracts remain
untouched. The concentration diagnostic's earlier empirical revise disposition
also remains unchanged.
