# SCI-FRUIT v0.1 — Recovered Historical Recurrence Baseline

Status: **Stage A implementation/documentation/test/study evidence; exact at the
bound refs below; non-authoritative for scientific authorship**

Baseline identifier: `SCI-FRUIT-HISTORICAL-RECURRENCE@f70701ad`

This file answers what the recovered Citlali procedure did. It does not say
that the procedure is scientifically correct, that every current product is an
admissible feedback model, or that v0.1 must preserve it. The historical
implementation reference
`f70701ad488444f3e2528c6bbe3e798863c9e301` and the Stage A launch commit
`7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` are byte-identical for the core
recurrence, observation-rerun, map-loading, learning, restart, and focused-test
files cited here. Output orchestration evolved after the historical reference;
both exact snapshots preserve the raw/filtered observation/coadd route choice.

## 1. Owner-Facing Identity

For completed iteration `k`, define:

- `D_o`: the original input for observation `o`; it is reread for every
  iteration. A previous residual timestream is not the next iteration's input.
- `Q_k`: the complete map bundle selected by the configured historical route
  from completed iteration `k`: observation/raw, observation/filtered,
  coadd/raw, or coadd/filtered. The bundle supplies map signal and the
  associated map fields used by feedback loading, including kernel, weight, and
  noise/RMS information where present.
- `S_k(Q_k)`: the selected feedback model derived when the loaded bundle is
  projected. Historical selection may depend on sign mode, S/N-like or flux
  gates, adaptive support, map weight/RMS, detector grouping, and kernel policy.
- `F_k = (Q_k, S_k, support_k, response_k, identity_k)`: an owner-facing name
  for the accepted feedback-model state. Historical files did not expose this
  typed state; they carried `Q_k` and reconstructed the selected projected
  model under the fixed requested policy.

The exact numerical map object carried from `k` to `k+1` was therefore one
selected **complete route map bundle** `Q_k`, not a residual timestream, not a
stored residual map, not a separately stored update increment, and not an
explicit sum of all prior increments. The feedback actually removed was the
selection-supported projection of that bundle, not necessarily every pixel in
the complete map.

For an unseeded first iteration there is no predecessor map. A configured
map-only seed supplies an external `Q_-1`-like initializer for a new sequence.
An exact restart supplies the selected complete product from completed
iteration `k` together with separate causal operational state and resumes at
`k+1`.

## 2. Exact Recovered Transition

For scan `s` of observation `o` in iteration `k+1`, concise pseudocode is:

```text
x = RTC_k+1(original observation D_o; prior applicable masks/penalties)
q_minus = project(select(Q_k); pre-PTC flags, geometry, interpolation)
z = x - q_minus
r = PTC_k+1(z; fixed policy, coefficients relearned on z,
            prior applicable masks/penalties)
w_res = estimate/reset weights from r
noise_products = map residual r with w_res, when requested
q_plus = project(select(Q_k); post-residual flags, same map/policy/geometry)
y = r + q_plus
w_final = w_res, or recompute/reset weights from y, by configured policy
apply post-cleaning and pre-mapmaking exclusions
H_k+1 = make complete observation/coadd map bundle from y and w_final
Q_k+1 = select configured raw or post-map-filtered observation/coadd product
F_k+1 = versioned accepted state whose numerical predecessor is Q_k+1
```

In operator notation, with all iteration- and observation-resolved state shown
as `A_k`, the historical reference is

\[
\begin{aligned}
  X_{k+1} &= \operatorname{RTC}_{k+1}(D;A_k),\\
  q^-_{k+1} &= P^-_{k+1}S_k(Q_k),\\
  r_{k+1} &= C_{k+1}(X_{k+1}-q^-_{k+1};A_k),\\
  q^+_{k+1} &= P^+_{k+1}S_k(Q_k),\\
  H_{k+1} &= M_{k+1}(r_{k+1}+q^+_{k+1};w_{k+1},A_k),\\
  Q_{k+1} &= B_{k+1}(H_{k+1}),\\
  F_{k+1} &= \mathcal U^{\mathrm{hist}}_k(F_k,R_{k+1}).
\end{aligned}
\]

Here `C` is the residual-domain PTC operation, `M` includes the complete
observation/coadd map construction and normalization, and `B` selects the
historical raw or filtered observation/coadd route. `R_{k+1}` denotes the
complete realized iteration record: residual processing, weights/masks,
complete map result, selected route, and state updates. The general transition
notation does not assert additivity.

`P^-` and `P^+` use the same loaded map, interpolation policy, selection gates,
and geometry. They are written separately because residual processing and
weight resetting may flag samples between subtraction and restoration. The
historical study measured a small but nonzero reduction in restored support in
one controlled run. Exact sample-by-sample equality of `q^-` and `q^+` is not a
general recovered invariant.

## 3. Operator Ordering And Bypass Semantics

The historical semantics are more than the literal array calls:

1. RTC starts from the original observation. Any applicable pre-RTC learned
   masks/exclusions act before map subtraction.
2. Applicable pre-PTC masks/exclusions act before map subtraction.
3. The selected model `q^-` is removed before residual-domain atmosphere/line
   validation, PTC cleaning, residual-weight estimation/reset, and residual
   noise-map accumulation. Those operations see the residual, not the accepted
   astronomical model.
4. The accepted model bypasses the PTC cleaner and the residual-only
   weight/noise pass. It is reprojected and restored after those operations.
5. The restored stream is then subject to post-cleaning detector removal,
   either retained residual weights or optional post-add-back weight
   recomputation, learning diagnostics, pre-mapmaking exclusions, complete
   mapmaking/normalization, and any selected post-map filter.
6. The response carried into `Q_{k+1}` is therefore the response of the
   restored projection followed by the actual final weighting, support,
   mapmaking, normalization, and route-selection/filter chain. It is not
   automatically unit response.

The recovered operator domains are:

| Operation | Data/model domain it sees |
| --- | --- |
| Prior learned pre-RTC mask/exclusion application | Fresh original-observation RTC state, before any feedback subtraction |
| RTC processing and RTC learning diagnostics | Fresh original-observation path; the feedback model has not yet been removed |
| Prior learned pre-PTC mask/exclusion application | Fresh RTC-derived PTC state, still before feedback subtraction |
| Model-protected atmosphere/line validation, PTC cleaning, residual weight estimation/reset, and residual noise-map pass | The source-subtracted residual `X-P^-S(Q_k)` or its processed descendant |
| PTC learning collection | Executed after restoration; it also consumes summaries produced by residual cleaning, so it is not a purely residual-only observation |
| Pre-mapmaking exclusions, complete mapmaking/normalization, map diagnostics, and optional filtering | The restored stream/result after the accepted model rejoins |

Thus neither all RTC/PTC work nor all learning is generically “residual-only.”
The contract must name the exact subset above rather than use a blanket label.

A future implementation may fuse these operations or work in another domain
only if it proves the same model removal, residual-only operator exposure,
bypass, rejoin point, support behavior, and next-map response. The scientific
contract need not prescribe the two array mutations, but it cannot classify
their ordering and bypass meaning as an implementation detail.

## 4. Fixed, Reused, Reset, And Relearned State

| Object | Recovered historical behavior between iterations | Carried numerically? |
| --- | --- | --- |
| Original observation | Reopened/reprepared and processed again each iteration | Original input identity/content, not a residual |
| Complete predecessor `Q_k` | Selected by route and loaded for iteration `k+1`; overwritten in-place when all iterations are not saved, or read from the prior reduction directory when they are | Yes; this is the predecessor map bundle |
| Feedback selection/support | Re-evaluated from `Q_k` under requested sign, threshold, weight/RMS, adaptive-support, grouping, and kernel policy | Policy and any derived source/support state; no stored increment history |
| RTC | Rerun on the original observation each iteration under fixed configuration; applicable prior learned state may alter flags/exclusions | RTC output/coefficients are not the carried predecessor |
| PTC cleaning | Rerun for every scan on `X-P^-S(Q_k)`; current cleaning coefficients are learned/recomputed from that residual under fixed PTC policy | Coefficients are not serialized as an inter-iteration map state |
| Learned sample masks | Accumulated during learning phases; only records from earlier iterations are eligible when apply phase is active; not cleared at each iteration | Effective intervals are carried and checkpointed |
| Learned detector penalties/exclusions | Accumulated/deduplicated; earlier-iteration records may apply pre-RTC, pre-PTC, or pre-mapmaking according to policy | Effective penalties are carried and checkpointed |
| Detector/sample flags | Recreated from each observation plus applicable learned state; residual weight/reset and cleaning can add flags before restoration | Not a prior residual; causal learned sources must be carried |
| Detector weights | Estimated/reset on the residual before restoration; either retained or recomputed/reset after restoration | Per-scan weights are regenerated; accumulated validation state is carried |
| Weight-validation state | Begun/finalized per iteration; sums/counts/penalties may accumulate and later finalize | Accumulated/finalized state is checkpointed |
| Map buffers/coadds | Reinitialized for an iteration, accumulated from restored timestreams, and normalized into complete products | Only the selected completed route product becomes `Q_k+1` |
| Post-map filters | Rerun after complete raw map/coadd creation when filtered output is requested | Filter policy/parents must remain compatible; filtered result is carried only for a filtered route |
| Diagnostic counters/histories | Per-iteration feedback counters reset; bounded event histories retained for reporting | Not required for restart only if causally inert |

The phase defaults and configuration names are implementation facts, not
scientific defaults. The scientifically important recovered fact is the phase
structure: learning can accumulate causal state in earlier iterations, while a
later apply phase reuses it. That state is independent of the carried map
bundle.

## 5. What Must Exist For A Historical Exact Restart

The recovered v2 continuation path needs, at minimum:

- the selected complete `Q_k` product at the checkpoint reduction directory;
- completed iteration `k` and next absolute iteration `k+1`;
- the selected route/type and ordered observation identities;
- the exact learning-policy and processed-PTC-policy snapshots checked by the
  loader;
- effective sample-mask intervals and effective detector penalties;
- accumulated/finalized weight-validation sums, counts, penalties, and validity;
- the original observation inputs and all external calibration/configuration,
  geometry, map/filter, and executable dependencies needed to repeat RTC/PTC
  and map construction; and
- the generation/branch and predecessor identity needed to distinguish exact
  continuation from a map-only seed.

The writer stores the compact causal fields listed above in the checkpoint and
relies on the completed reduction directory for `Q_k`. It does not embed every
historical map or diagnostic event. Focused tests show round-trip restoration of
the represented learning/weight-validation state and a synthetic five-plus-two
versus seven-iteration learning continuation. A controlled full reduction study
reported equality after the v2 repair for its tested path. These establish
evidence, not a universal proof that the v2 field list is scientifically
complete. The Stage A causal-completeness test remains controlling: retain or
reconstruct exactly the state whose omission can change a later required
result. This does not require permanent retention of every intermediate map.

## 6. Evidence And Limitations

| Evidence class | Exact recovered source | What it supports | What it does not authorize |
| --- | --- | --- | --- |
| Historical implementation | `f70701ad...` versions of `reduction_iteration_loop.h`, `reduction_observation*.h`, `lali_run_impl.h`, `pointing_run_impl.h`, `lali_fruitloop_impl.h`, `pointing_fruitloop_impl.h`, map-loading/path/output files, learning state, and restart checkpoint | Original-observation rerun; complete-map carry; residual-only PTC; restoration; next complete route product; causal state behavior | Scientific correctness, route admission, response, or a new recurrence |
| Focused tests | `f70701ad...:tests/test_fruit_loop_recurrence.cpp` and `tests/test_learning_and_fruit_contracts.cpp` | Immediate subtract/add round trip; controlled linear convergence seam; effective-state and checkpoint round trips | Production nonlinear equivalence, calibrated increments, or universal restart completeness |
| Architecture documentation | [`doc/adr/0006-fruit-loop-restart-checkpoint.md`](../../../../adr/0006-fruit-loop-restart-checkpoint.md) at `7f9307ff...` | Map seed versus exact restart, absolute iteration, v2 operational history | Scientific recurrence authority |
| Historical studies | [`FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md`](../../../../FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md), [`FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md`](../../../../FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md), and calibration/population follow-ups at `7f9307ff...` | Recovered production-recurrence description, support/weight/learning/projection sensitivities, restart failure/recovery evidence | Universal calibration, stopping, response, or scientific approval |

All implementation, tests, documentation history, and studies in this table
remain excluded from an implementation-blind author packet. Stage A uses them
to identify the v0.1 compatibility baseline and the questions the owner must
answer.
