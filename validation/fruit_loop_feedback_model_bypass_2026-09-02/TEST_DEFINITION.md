# FRUIT EL-F4 feedback-model-bypass penalty screen

Status: **registered before implementation or execution**

Test ID: `SCI-FRUIT-EL-F4-FEEDBACK-MODEL-BYPASS-R0.1`

## Question

Can the accepted feedback model be prevented from supplying the evidence for
a future hard map-diagnostic detector exclusion, while retaining the ordinary
complete map products and diagnostic archive and without introducing a new
protected scientific regression in the two exposed pointing cases?

This is a mechanism-specific candidate-policy screen. It is not a stopping-
rule test, a detector-quality judgment, or a qualification experiment.

## Candidate operation

At completed absolute iteration `k`, let:

- `Q_k` be the ordinary complete unfiltered map used by current mapdiag;
- `F_(k-1)` be the exact feedback model loaded, projected, and restored during
  iteration `k`; and
- `W_k` be the ordinary complete-map weight.

The candidate defines the map-domain penalty-evidence signal

`E_k = Q_k - F_(k-1)`.

At iteration 0, where no accepted feedback model exists, `E_0 = Q_0`.
`Q_k` and `F_(k-1)` must have identical map count, grid, and finite support;
an unavailable or incompatible required model fails closed.

The complete `Q_k` map, all mapdiag products and statistics, the complete-map
outlier record archive, and the D19 next-iteration target selection remain
unchanged. Only `map_pixel_outlier_detector_dominance` penalty evidence is
recomputed from `E_k` with the existing `W_k`, core and source-radius masks,
minimum effective samples, robust median/MAD, minimum absolute z, `top_n`,
contribution trace, four-pixel dominance threshold, scan scope, and factor
zero. All other learned masks, penalties, weights, and processing policies are
unchanged.

The option is explicit and default-disabled. Default-disabled execution must
retain numerical identity. The enabled state and evidence-view identity are
written to configuration/provenance and the restart policy snapshot.

`E_k` is not called a literal sample-domain residual. Equivalence to a map of
the cleaned residual is unavailable because the model's projection,
mapmaking response, and weighting have not been proved to compose to identity.

## Fresh-run matrix

All trajectories start from raw inputs with one configured thread,
`grppiex: seq`, the same newly built executable, `alpha=1.25`, 100 mJy/beam
centered injection, and saved diagnostics/checkpoints.

| Order | Observation | Evidence view | Variant | Absolute iterations |
|---:|---:|---|---|---|
| 1 | 123424 | complete map | control | 0--5 |
| 2 | 123424 | feedback excluded | control | 0--5 |
| 3 | 123424 | feedback excluded | injected | 0--5 |
| 4 | 123424 | complete map | injected | 0--5 |
| 5 | 152389 | complete map | control | 0--6 |
| 6 | 152389 | feedback excluded | control | 0--6 |
| 7 | 152389 | feedback excluded | injected | 0--6 |
| 8 | 152389 | complete map | injected | 0--6 |

Orders 5--8 run only if the observation-123424 primary gate below passes.
The complete-map trajectories are same-build compatibility controls, not
historical truth.

## Validity and primary gate

1. The new executable with the option disabled must reproduce every prior
   corresponding EL-F1-R1/EL-F2 signal, kernel, and weight image bitwise at
   every iteration. Failure invalidates the test.
2. Within every method pair, control and injected iteration 0 must be bitwise
   equal before injection begins.
3. Required products must be finite on identical support and carry the
   expected observation, absolute iteration, alpha, injection, and evidence-
   view identities.
4. Logs must end normally with no unexpected error- or critical-level message.

For observation 123424, the candidate must omit the effective UID 4460 a1400
penalty at iteration 4. The iteration-5 recovery and annular-residual reversal
fractions retain the EL-F3 definitions and frozen original values. The primary
gate passes only when both reversal fractions are at least 0.5. Both fractions
at least 1.0 are again reported as full reversal. Failure stops before the
observation-152389 matrix.

## Cross-case regression screen

For every common observation, array, and post-injection iteration, compare the
candidate injected-minus-control response with the same-build complete-map
response. A new protected regression is present if any one of these holds:

- absolute central-recovery error grows by more than 0.01;
- absolute major- or minor-width error grows by more than 0.01;
- centroid error grows by more than 0.1 arcsec;
- annular residual grows above 1.10 times the complete-map value; or
- full-kernel residual grows above 1.10 times the complete-map value.

The inherited absolute EL-F1/EL-F2 screen is also reported but does not define
whether this narrow policy caused a new regression. All effective mapdiag
penalty additions, removals, and timing changes are inventoried. No penalty-
count minimum is imposed after seeing the parent results.

The prospective disposition is:

- `advance_to_broader_policy_testing` if the primary gate passes and there is
  no new protected regression across either completed observation;
- `mechanism_helpful_but_regressive` if the primary gate passes but a new
  protected regression occurs; or
- `do_not_advance` if the primary gate does not pass.

Advancement means only that a new, independent and more varied test may be
proposed. It does not qualify or promote the policy.

## Bounds and stop rules

- output root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f4-feedback-model-bypass-r0.1`;
- at most eight fresh trajectories in the fixed gated order;
- no parameter or threshold tuning and no replacement for an unfavorable
  result;
- at most one environmental/input replacement per trajectory;
- 2 hours and 64 GiB per trajectory, 12 hours and 12 GiB retained overall;
- stop on build/test failure, default-off mismatch, primary-gate failure,
  non-finite required output, unexpected error/critical logging, or resource
  breach; and
- no later iteration, candidate, observation, or fallback follows
  automatically.

All source inputs and prior reduction products remain immutable. Unity access
is neither needed nor authorized.

## Claim limit

Even the advancement disposition would concern one explicit map-domain
penalty-evidence policy on two already exposed bright compact-source pointing
cases. It would not establish sample-domain residual equivalence, extended-
source fidelity, faint-signal recovery, a universal detector policy, a
stopping rule, historical superiority, or production readiness.
