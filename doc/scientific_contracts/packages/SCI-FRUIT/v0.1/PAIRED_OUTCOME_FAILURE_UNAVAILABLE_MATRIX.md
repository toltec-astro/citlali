# SCI-FRUIT v0.1 — Paired Outcome, Failure, And Unavailable Matrix

Status: **Stage A candidate reporting rule; no outcome has been observed**

The target population is frozen before outcomes are opened. Every admitted
unit remains in accounting under exactly one state:

| Candidate state | Historical-control state | Scientific interpretation | Paired contrast |
| --- | --- | --- | --- |
| succeeds | succeeds | absolute candidate and historical metrics available | available on the declared common comparison support |
| succeeds | fails | candidate-only rescue endpoint; historical failure retained | unavailable, not imputed as improvement |
| fails | succeeds | candidate regression/failure | candidate failure; no favorable imputation |
| fails | fails | joint failure | unavailable |
| scientifically unavailable | any non-inapplicable state | exact unavailable party and cause retained | unavailable |
| any non-inapplicable state | scientifically unavailable | exact unavailable party and cause retained | unavailable |
| prospectively known inapplicable | prospectively known inapplicable | excluded only by the frozen target-population rule | not in target denominator; reported separately |

An analysis must not retain only complete pairs after observing outcomes. It
must not silently merge unavailable with failure, improvement, or exclusion.
On the same declared target population and with exact denominator rules, report

```text
p_improved
p_practically_unchanged
p_degraded
p_failed
p_unavailable
```

plus known-inapplicable accounting. The categories, practical-change bands,
weights, failure semantics, unavailable causes, and catastrophic-regression
limits are frozen prospectively. Candidate-rescue and candidate-regression
rates remain separately visible even when a paired metric is unavailable.
