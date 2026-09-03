# FRUIT EL-F5 off-source injection result

Disposition: **same event replicated off source**

The new disabled-injection control reproduced all `54` registered EL-F4 signal, kernel, and weight planes bitwise before the off-source result was interpreted.

The complete injected-minus-control response was retained in `18` FITS maps. Across all three arrays and injected iterations, the largest fitted transfer-centroid distance from the declared position was `0.229` arcsec and the largest kernel-centroid distance was `0.239` arcsec.

## Iteration-4 to iteration-5 response

| Location | Array | Recovery k=4 | Recovery k=5 | Annular k=4 | Annular k=5 | Registered loss direction |
|---|---|---:|---:|---:|---:|---|
| off source | a1100 | 0.981249 | 0.981291 | 0.00049468388 | 0.00043785103 | no |
| off source | a1400 | 1.043975 | 1.037055 | 0.0032704798 | 0.023134428 | yes |
| off source | a2000 | 0.958046 | 0.967963 | 0.0034311288 | 0.0034853414 | no |
| centered EL-F4 | a1100 | 0.851657 | 0.869745 | 0.0003829023 | 0.00030476192 | no |
| centered EL-F4 | a1400 | 0.890451 | 0.822828 | 0.0055910619 | 0.021474106 | yes |
| centered EL-F4 | a2000 | 0.724708 | 0.734453 | 0.0034204768 | 0.0041373778 | no |

## Penalty association

Target UID 4460 event replicated: `yes`.
Injection-specific iteration-4 hard penalties: `1`.
The target detector has `3` qualifying map pixels in the control and `4` in the injected run; four is the configured hard-penalty threshold.

| Iteration | Scan | UID | Array id | Score | Factor |
|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 4460 | 1 | 4 | 0 |

## What changed in a1400

The off-source Gaussian/kernel-normalized central response falls only from `1.043975` to `1.037055` (an absolute change of `-0.006920`, or `-0.663%`). Because both values exceed unity, that change is slightly *closer* to unit recovery, not an amplitude degradation. The whole-kernel projection changes from `1.056239` to `1.056704`. The actual degradation is response shape/leakage: kernel-residual relative RMS rises from `0.320673` to `0.727804` (`2.270x`) and the registered annular residual rises from `0.00327048` to `0.0231344` (`7.074x`).

## Execution

Both first-attempt trajectories completed in `344.67` aggregate seconds with zero error/critical messages. Maximum resident memory was `0.911` GiB.

## Interpretation

The preregistered classification is **same event replicated off source**. This rules out overlap with Neptune's central source core as a necessary condition for the UID 4460 event in this observation. It strengthens the narrower hypothesis that the penalty responds to injection-altered complete-map state under this detector/scan geometry. It does not yet show that the interaction is generic across observations or prove that the reapplied penalty caused the off-source iteration-5 degradation.
The classification label follows the prospectively registered sign test; it must not be read as evidence that the off-source central amplitude itself became less accurate.

## Interpretation limit

This one-location pointing result does not establish a blank-field or isolated-source response and does not prove that any detector penalty caused a response change. It does not qualify or select a FRUIT recurrence, penalty policy, or production configuration.
