# SCI-NOI-001 coordinator decision brief — 2026-08-06

Status: coordinator review accepted the bounded Phase 2 documentation audit at
`bf6de403c2c50d55f54b8486424aea5543cdd346`. The package remains `amend`:
contract `proposed`, implementation `nonconformant`, validation `incomplete`,
and production `existing_use_only`. No repair, recipient dispatch, evidence
request, reduction, Unity action, or production change is authorized.

## Coordinator-accepted framework and provenance requirements

1. R3 remains the sole independent equation authority. The final audit is a
   read-only exact-`d5015fe716971bf8ea617e8a187311bf5af05185` source trace;
   all eight findings remain open at P1.
2. Keep NOI-001 separate from NOI-002. NOI-002 exclusively owns finite-N
   normalization, empirical variance/weight calibration and formula, S/N,
   significance, threshold, feedback, aperture estimator, and production
   count/default policy.
3. Require compact reproducibility provenance when a successor is considered:
   versioned key/namespace, coherence partition, stable ordering, mode,
   completion identity, and digest joins. Do not require dense per-sample IDs,
   sign vectors, or N-by-N sign-correlation matrices.
4. Route the NOI-002 and FLT handoffs as post-core material. The FRUIT source
   fact is also post-core evidence, not pre-core authority; it cannot be used
   to seed a FRUIT independent derivation.
5. Retain all evidence designs as unrequested. Any costly work remains held
   for FRAMEWORK-NUM-001. Count 64 is optional validation capacity only.

## Owner decisions

1. **Realization identity policy (F001/F002/F008) — approved 2026-08-06,
   pending repair.** Every realization must use a compact, versioned,
   deterministic key containing observation identity, ensemble mode,
   conditioning iteration/pass, realization number, and coherence-unit
   identity. A detector/channel identity is the stable observation-scoped
   realized channel identity in the reduction/operator chain, not a
   design-detector match. Distinct observations must receive independent sign
   assignments; repeated cross-observation prefixes are not desired behavior.
   Beammap signs are generated once per named mapmaking pass/iteration and
   reused across active map slots, independent of active-map ordering/history.
   Assignments must repeat across sequential/OpenMP scheduling. Provenance must
   record policy/version, stable ordering/partition, completed realization IDs,
   ensemble mode, and digest joins. Dense sign vectors, per-sample IDs, and
   N-by-N Q matrices are not required. This selects no RNG implementation or
   seed, repair SHA, production count, estimator, product validity, evidence
   run, or production authorization.
2. **Zero-realization admission (F005) — approved 2026-08-06, pending
   repair.** When noise realizations are enabled, requested/effective/realized
   count must be at least one; enabled with zero is invalid. Disabled is the
   sole supported no-ensemble state: effective and realized counts are zero,
   no realization-derived products, weights, or diagnostics are promised, and
   no realization-generation or downstream noise-product work occurs.
   Real-time pointing and OOF quicklook reductions may use this disabled lane
   to perform the minimum computation needed for quicklook because neither
   depends on noise maps. This selects no default, scientifically adequate, or
   production count; no estimator, product validity, repair SHA, or evidence
   plan. NOI-002 retains count/default and estimator ownership.
3. **Ensemble-mode identity (F003) — approved 2026-08-06, pending repair or
   dependent contract.** Retain the ordinary ensemble as
   `source_imprinted_current`: randomization of cleaned `x=s+n`. Individual
   realizations and their moments may retain deterministic astronomical signal,
   including literal negative-source realizations; that is expected under this
   identity and is not a source-free physical-noise ensemble. Existing
   realizations remain restricted diagnostics only; NOI-002 retains every
   variance, precision, empirical-weight, S/N/significance, threshold,
   feedback, aperture-uncertainty, and production decision.

   Reserve distinct `final_pre_readdition_residual` identity for a future
   realization mode after subtraction of an SCI-FRUIT-001-approved source
   model. Do not infer residual unbiasedness; retain source-model error and
   physical-noise overfit/removal risks. SCI-FRUIT-001 must establish the
   residual state, source-model identity, and readdition contract before such
   a mode is implemented or authorized. This approval does not implement a
   mode, authorize the legacy fruit path as that product, select repair SHA,
   count/default, estimator, evidence plan, or change production status.
4. **Evidence admission — owner decision still required.** Decide whether any exact-d501 evidence case merits
   a separate FRAMEWORK-NUM-001 cost/readiness review after a successor policy
   is chosen. Recommendation: hold all cases now; authorize none until the
   requested scientific question, comparison set, cost, and acceptance policy
   are separately approved.

The FLT edge-preprocessing fact is routed only for a future conditioned
interface disposition; it does not reopen filter mathematics. JINC remains
conditioned on SCI-MAP-002, and residual interface facts remain conditioned on
SCI-FRUIT-001.
