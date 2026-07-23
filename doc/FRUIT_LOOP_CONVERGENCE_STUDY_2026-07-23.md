# Fruit-Loop Convergence Study

Date: 2026-07-23

## Decision

Citlali retains its hard `max_iters` fruit-loop bound without production early
stopping. The offline evidence path is now ready, but the available active
NGC4449 sequence does not establish a scientifically defensible stopping rule.

This is a triggered retained-debt investigation under D15, not a change to the
production algorithm.

## Question

Can Citlali recognize that source-model feedback has stopped changing the
accepted science enough to justify ending before `max_iters`?

A useful answer must distinguish actual convergence from:

- a temporarily small map change before learned state takes effect;
- one array stabilizing while another continues to change;
- changing valid-pixel support or map weights;
- a reset or transition in the learning lifecycle; and
- an invalid no-op feedback request.

## Evidence Inventory

The local NGC4449 archive contains three relevant classes of evidence:

| Sequence | Use | Disposition |
| --- | --- | --- |
| `reduced_full_spatial_learning/redu00` through `redu04` | Five active, saved spatial-feedback iterations for five observations and all three arrays | Primary convergence sequence |
| `reduced_full_spatial_learning_continue_r04/redu00` through `redu02` | Map-path continuation from the final primary map | Analyze separately; do not append to the primary sequence |
| Earlier ten-iteration full reduction | Statically empty fruit-loop request whose late maps repeat to roundoff | Negative contract evidence only; not scientific convergence calibration |

The primary sequence is suitable because every retained iteration:

- was produced by Citlali `e97de3fd`;
- has the same complete low-level configuration;
- uses active `local_snr_floor: 2.5` spatial feedback with `map_center`;
- contains the same raw coadd product for `a1100`, `a1400`, and `a2000`;
- contains signal, weight, and coverage planes with stable identity and units;
  and
- records the final effective learning-state counts and iteration runtime.

The continuation predates the state-complete restart checkpoint. It loads the
final map through `fruit_loops.path`, but its learning phase restarts at
`learn`, and its effective mask and penalty counts differ from the primary
sequence. Combining those directories into one nominal eight-iteration
sequence would confound continued map feedback with reset learning state.

## Offline Analyzer

[`analyze_fruit_loop_convergence.py`](../tools/baseline/analyze_fruit_loop_convergence.py)
consumes a checked manifest and performs no Citlali execution. For every
consecutive iteration and every requested array it verifies product identity
and measures:

- valid support Jaccard similarity;
- map-wide relative L2 change;
- relative L2 change within a map-centered scientific aperture;
- peak fractional change within that aperture;
- map-weight relative L2 change; and
- RMS map change expressed on the formal difference-weight scale.

The analyzer also reads effective sample-mask and detector-penalty counts from
the Citlali log. A learning transition is stable only when both counts are
unchanged and the current phase is `apply`.

Candidate rules are explicit manifest data. They are simulated independently
for every array and are always labeled
`candidate_only_not_production_approved`. The tool reports a stop only after
the configured minimum iteration and consecutive-pass count. Missing maps,
identity changes, shape/unit changes, unexpected config differences, or
unavailable common support fail the study rather than being skipped.

The portable checked manifest uses `NGC4449_ROOT`:

```bash
export NGC4449_ROOT=/path/to/NGC4449
$HOME/tolteca/bin/python \
  tools/baseline/analyze_fruit_loop_convergence.py \
  validation/fruit_loops/ngc4449_full_spatial_learning_study.json \
  --json-out /tmp/ngc4449-convergence.json \
  --report-out /tmp/ngc4449-convergence.md
```

The measured machine-readable result is retained in
[`ngc4449_full_spatial_learning_result.json`](../validation/fruit_loops/ngc4449_full_spatial_learning_result.json).

## Primary Sequence Results

The worst array metric in each transition is:

| Transition | Min support Jaccard | Max map relative L2 | Max aperture relative L2 | Max peak change | Max weight relative L2 | Learning stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 -> 1 | 0.998433 | 0.0481785 | 0.0602557 | 0.148889 | 0.00336499 | No |
| 1 -> 2 | 0.982237 | 0.267260 | 0.270943 | 0.0773757 | 0.0260332 | Yes |
| 2 -> 3 | 0.999496 | 0.0263146 | 0.0330215 | 0.0270956 | 0.00167224 | Yes |
| 3 -> 4 | 1.000000 | 0.0043496 | 0.00649999 | 0.00901853 | 0.0000553 | Yes |

The non-monotonic sequence is important. Iteration 0 to 1 changes moderately
while learning is not stable. Iteration 1 to 2 then changes substantially as
the learned state takes effect. Only the final two transitions show a
consistent approach toward stability.

The exploratory rule requires:

- completion through at least absolute iteration 2;
- two consecutive passes;
- all three arrays to pass independently;
- stable valid-pixel support and weights;
- bounded full-map, aperture, and peak changes; and
- stable effective learning state in the `apply` phase.

It does not stop this five-iteration sequence. Transition 2 to 3 misses the
exploratory full-map bound in `a1400`; transition 3 to 4 is the first complete
pass, and no later saved iteration exists to provide the required confirmation.
The result therefore claims zero saved iterations and zero demonstrated
runtime savings.

The five measured iterations consumed approximately 4.4 hours in total. A
weaker one-pass rule could appear to save roughly the final 54-minute
iteration, but this archive provides no following full-state iteration with
which to test whether that decision preserves the accepted final science.
Adopting that rule would replace evidence with optimism.

## Production Safeguards

Any future production rule should retain all of these properties:

1. `max_iters` remains a hard upper bound.
2. Every active array must pass; arrays are not averaged together.
3. A minimum number of completed feedback iterations is required.
4. More than one consecutive transition must pass.
5. Learning state and valid map support must be stable.
6. A metric that cannot be assessed is a failed convergence check.
7. The decision uses the raw feedback product, not a downstream filtered map.
8. The final provenance records estimator version, thresholds, per-array
   metrics, pass count, absolute stop iteration, and terminal reason.

## Evidence Still Needed

Before D15 can close:

1. Collect state-complete sequences using the current restart checkpoint,
   enough iterations to include at least two post-stability transitions.
2. Include representative compact, extended, bright, and faint science fields
   rather than calibrating on NGC4449 alone.
3. Test learning-disabled and learning-enabled policies separately.
4. Compare every simulated early-stop product with the corresponding full
   sequence using the accepted science comparator and an approved successor
   profile.
5. Select thresholds from those measurements with the scientific owner.
6. Add production convergence state and versioned provenance only after the
   rule and failure policy are approved.

Until then, the analyzer is a collection and review tool. It is not a reason
to broaden the production fruit-loop implementation.
