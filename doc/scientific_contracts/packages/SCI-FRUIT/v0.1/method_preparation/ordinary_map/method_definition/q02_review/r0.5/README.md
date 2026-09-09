# Measure weight precision before choosing a training window

Q02 review r0.5, 2026-09-09. **Concrete diagnostic proposal; execution pending.**

## Program adherence and prior-work recovery

Continue under the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [frozen prior-work recovery](../../r0.4/PRIOR_WORK.md). Adopt the frozen
Q01 definition, Q02-A/B decisions and
[approved 20% precision objective](../WEIGHT_PRECISION_DIRECTION_2026-09-09.md).
Retain the [U/N test design](../r0.2/WEIGHTING_TEST_PLAN.md). Cite the
[r0.4 input preflight](../r0.4/INPUT_PREFLIGHT.md) as implementation-informed
evidence. Its proposed standalone T1 and observation-wide pooling are deferred
while this smaller diagnostic is reviewed. Earlier packets remain unchanged.

The owner's “Let's proceed. What's next” directs completion of this preparation.
The new uncertainty prescription below is a proposal, not an already approved
scientific decision. This is manager experiment preparation; no new independent
author, core derivation, upstream family or numerical FRUIT method is admitted.

## What happens next

Use the two existing discovery reductions, 123424 and 152389, to answer:

**Can we estimate useful relative detector weights with roughly 20% precision
without averaging over meaningful changes in the noise?**

| Comparison | What changes | What we learn |
| --- | --- | --- |
| One original chunk, about 5 seconds | Use its reserved training half | Precision and availability at the finest proposed time resolution |
| Two chunks, about 10 seconds | Pool only their training halves | Whether more training improves precision without hiding changing scatter |
| Four chunks, about 20 seconds | Pool only their training halves | The same tradeoff over a longer interval |

These are elapsed window labels. Actual usable training is shorter, irregular,
and detector dependent. PCA chunk lengths stay as reduced. No new reduction or
download is needed for this diagnostic.

The [exact protocol](WEIGHT_PRECISION_PROTOCOL.md) supplies the input hashes,
recovered row intervals, source guards, normalization population, correlation
prescription, reporting rules and execution bounds. It proposes an inexpensive
first-order uncertainty estimate that includes squared-signal correlations and
joint normalization across detectors. Six fixed correlation spans expose how
dependent the estimate is on that choice; the shortest estimate is never
selected just because it falls below 20%.

The result will be one per-array comparison of availability, estimated precision,
and scatter changes with time. Missing weights stay missing. A fraction below
20% is descriptive evidence, not a certified pass or a detector veto. The
finite-data uncertainty of the uncertainty estimate itself may remain unknown;
the report must say so. This screen cannot establish optimal weights or map
benefit from a small error bar.

## One bounded execution decision

Proposed decision identity: `SCI-FRUIT-Q02-WEIGHT-PRECISION-SCREEN-R0.5`.
**Pending owner approval.**

Recommend approving the exact protocol for implementation, deterministic
verification and one local discovery diagnostic, including its limited legacy
input permission. Bound the run to two hours, 4 GiB peak aggregate memory and
2 GiB of new output, one worker and one numerical-library thread. Approval would
cover ordinary implementation choices and routine defect repair within these
bindings, without another implementation vote.

This decision resolves the diagnostic's method, input and execution permission
together. It does not select a weight window or policy. T1, T2 maps, reserved
129081 access and T3 feedback retain their separate decisions. No source
recovery, morphology, leakage, map support, convergence or cost requirement is
removed from the eventual comparison; the non-map quantities cannot yet answer
those questions. Independent-pointing replication still precedes a policy
recommendation. Historical Citlali remains the mandatory FRUIT control.

After the diagnostic, make one practical disposition: prepare the U/N mapping
test around a supportable training design, identify the specific missing evidence,
or stop this weighting branch and return to the simpler reference. Do not turn
an inconclusive result into an open-ended estimator search.

Only documents and preserved packet identities were checked to prepare r0.5.
The [review manifest](REVIEW_MANIFEST.json) binds the two proposal documents and
their repository sources; its adjacent SHA-256 file binds the manifest itself.
No new signal analysis, coefficient calculation, mapping, replay, implementation,
Unity activity or push occurred. Existing reductions, freezes and both opaque
review archives are preserved.
`FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.
