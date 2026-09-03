# SCI-FRUIT EL-F4 feedback-model-bypass result

Disposition: **mechanism_helpful_but_regressive**

This is development evidence only; it does not qualify a FRUIT method.

## Primary mechanism gate

- UID 4460 penalty absent: `True`
- a1400 iteration-5 recovery: `0.901542068`
- a1400 iteration-5 annular residual: `0.00288776008`
- recovery reversal fraction: `1.164010658`
- annular-residual reversal fraction: `1.170200485`
- full reversal: `True`

## New protected regressions

| Obs | Iter | Array | Failed metric | Complete | Candidate | Registered comparison |
|---:|---:|---|---|---:|---:|---:|
| 123424 | 3 | a2000 | kernel_residual | 0.462081753 | 0.707511056 | 1.531138 (candidate_over_complete_ratio; limit 1.100000) |
| 123424 | 4 | a2000 | kernel_residual | 0.46977599 | 0.580345655 | 1.235367 (candidate_over_complete_ratio; limit 1.100000) |
| 123424 | 5 | a2000 | kernel_residual | 0.48804592 | 0.550567627 | 1.128106 (candidate_over_complete_ratio; limit 1.100000) |
| 152389 | 4 | a1100 | annular_residual | 0.000600952071 | 0.000726372094 | 1.208702 (candidate_over_complete_ratio; limit 1.100000) |

## Terminal candidate versus complete-map response

| Obs | Array | Complete recovery | Candidate recovery | Complete annular | Candidate annular |
|---:|---|---:|---:|---:|---:|
| 123424 | a1100 | 0.869745 | 0.869745 | 0.00030476192 | 0.00030476192 |
| 123424 | a1400 | 0.822828 | 0.901542 | 0.021474106 | 0.0028877601 |
| 123424 | a2000 | 0.734453 | 0.738135 | 0.0041373778 | 0.0043282756 |
| 152389 | a1100 | 0.972687 | 0.972038 | 0.0006638095 | 0.00059268408 |
| 152389 | a1400 | 0.980134 | 0.980134 | 0.00028652712 | 0.00028652712 |
| 152389 | a2000 | 0.966833 | 0.966835 | 0.00058444759 | 0.00058863074 |

## Inherited absolute screen

This screen is reported for context and does not decide whether the bypass caused a new regression.

| Obs | Array | Result | Failed protections |
|---:|---|---|---|
| 123424 | a1100 | fail | major_width, minor_width, annular_residual |
| 123424 | a1400 | fail | major_width, minor_width |
| 123424 | a2000 | fail | major_width, minor_width, centroid, annular_residual, kernel_residual |
| 152389 | a1100 | fail | annular_residual |
| 152389 | a1400 | pass | none |
| 152389 | a2000 | pass | none |

## Execution performance

| Obs | Complete pair mean (s) | Candidate pair mean (s) | Candidate change |
|---:|---:|---:|---:|
| 123424 | 177.740 | 178.570 | +0.467% |
| 152389 | 279.675 | 277.625 | -0.733% |

## Validity and bounds

- Default-off bitwise-compatible image planes: `234`
- Candidate iteration-0 bitwise-equal planes: `18`
- Completed first-attempt trajectories: `8`
- Aggregate wall time: `1827.22 s`
- Maximum resident memory: `0.910 GiB`
- Retained output: `2.983 GiB`
- Error/critical messages: `0`
- Configuration, FITS provenance, checkpoint policy, and candidate evidence traces: verified
- Penalty comparison (candidate versus complete-map evidence): retained `2`, retained_value_changed `0`, timing_changed `0`, removed `15`, added `0`

## Interpretation

The bypass reproduces the earlier causal rescue on observation 123424, but the wholesale policy is too broad. It introduces three preregistered a2000 kernel-residual regressions on observation 123424 and raises the observation-152389 a1100 iteration-4 annular residual by more than the preregistered ten-percent allowance. The registered disposition therefore stops advancement of this exact policy. The result supports a narrower future hypothesis that distinguishes injection-sensitive feedback-driven penalties from repeatable ordinary detector evidence; it does not authorize another run.

The terminal effective map-diagnostic penalty inventory contains `19` records across the eight trajectories; see `PENALTY_INVENTORY_R0.1.csv` for exact identities.
