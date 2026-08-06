# SCI-NOI-001 current and future-residual evidence design — 2026-08-06

Status: **proposal only; not requested, not launched, and not authorized for
execution**.

This document preserves two separate designs. The first is bounded to exact
application SHA `d5015fe716971bf8ea617e8a187311bf5af05185` and its implemented
current ensemble. The second is a future paired residual A/B design for which
no approved implementation SHA exists. Nothing here defines a scientific
estimator, numerical tolerance, production realization count, Unity action, or
permission to modify code.

## Governing equations and ownership

The sole equation authority is
`doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex`, SHA-256
`27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da`.
The relevant numbered equations are:

- current and residual ensembles: R3 Eqs. (15)–(16);
- finite mean, second moment, and covariance: R3 Eqs. (17)–(21), (40)–(43);
- deterministic-signal imprint: R3 Eqs. (26)–(27);
- residual bias trade and source readdition: R3 Eqs. (28)–(32);
- coadd and cross-observation terms: R3 Eqs. (33)–(37);
- filtering: R3 Eqs. (38)–(39);
- assignment cardinality, effective uniqueness, RNG identity, and adequacy:
  R3 Eqs. (44)–(54).

`SCI-NOI-002` exclusively owns finite-stack normalization, empirical
variance/weight calibration, S/N, significance, thresholds, feedback,
aperture uncertainty, and production count/default policy. `SCI-FRUIT-001`
owns approval of a residual state/model contract. Costly numerical work is
held for `FRAMEWORK-NUM-001`.

## Design A — exact-d501 current ensemble

### Identity gate

Any coordinator-approved request derived from this design must first require:

- clean exact application SHA
  `d5015fe716971bf8ea617e8a187311bf5af05185`;
- checksum-pinned inputs and authored plus fully resolved configuration;
- reduction type, observation ordering/identity, scan partition/order,
  observation-scoped detector/channel ordering, array/network membership,
  map grouping/method, fruit mode, coadd mode, filter mode, and
  `randomize_dets`;
- executable/build/compiler/dependency identity, including Boost and Eigen
  versions, host identity, execution policy, and thread count;
- complete logs, standard requested/effective/realized provenance, product
  index, FITS/NetCDF paths and digests, HDU inventory, and failure/omission
  report;
- explicit confirmation that no application patch, helper, schema, wrapper,
  verifier, download, or substituted SHA was used.

If exact sign reconstruction cannot be performed from those facts, the return
must classify that provenance gap. A digest alone verifies bytes but is not a
lossless sign representation. Dense per-sample identities, persisted sign
vectors, and dense sign-correlation matrices are not required.

### Bounded cases

The cases below are questions, not an acceptance campaign. Counts are chosen
only as needed to expose a named analytic limit. No count is a production
recommendation.

| ID | Exact-d501 case | Equation-linked question | Required returned facts |
| --- | --- | --- | --- |
| C01 | One ordinary naive observation, `randomize_dets=false`, sequential policy | Do signs remain constant within each realized scan unit, and what are the on-demand empirical sign mean, selected entries of `Q_epsilon`, raw unique count, and complement-unique count? R3 Eqs. (5), (40)–(48). | Resolved scan partition, realization count/order, reconstruction method or explicit gap, compact sign-summary values, standard products and digests. |
| C02 | Exact C01 input/config twin with only `randomize_dets=true` | Does the coherence unit change only to scan × observation-scoped channel, with no hidden array/network namespace? R3 Eqs. (5), (9), and (49)–(50). | Stable channel-order identity/digest, array/network membership, same summaries as C01. |
| C03 | Exact C01/C02 sequential–OpenMP twins and an exact repeat | Is RNG assignment identical across policy/thread changes and repeats, distinct from allowed map-summation differences? R3 Eqs. (49)–(50). | Build/host/thread identity, product and provenance digests, sign reconstruction or gap, map comparison under a separately preapproved MAP acceptance policy. |
| C04 | At least two observations with controlled equal and unequal scan/channel shapes, coadd enabled | Which pseudorandom prefixes repeat, and which cross-observation terms survive in the realized coadd? R3 Eqs. (33)–(37). | Ordered observation IDs, per-observation shapes/order, coadd admissions/weights, reconstruction or gap, observation and coadd product digests. |
| C05 | Existing checksum-pinned source-bearing and source-free/blank-support inputs, ordinary current ensemble | Is the literal negative source confined to individual realization HDUs as predicted by R3 Eq. (21), and how do source and blank regions differ in finite mean/second-moment diagnostics under R3 Eqs. (26)–(27), (40)–(43)? | Named existing inputs, source/blank region authority, raw realization HDUs if configured, on-demand summaries, no estimator interpretation. |
| C06 | Exact filter twins with edge guard disabled, enabled with zero background, and enabled with nonzero realized background where an existing input supplies those states | Does the common filter core satisfy R3 Eqs. (38)–(39), and is the affine background asymmetry exactly localized to preprocessing? | Realized filter/template/denominator/edge state, raw-parent joins, science/realization products, no re-audit of filter mathematics. |
| C07 | Existing Beammap configuration with noise explicitly admitted by the coordinator, exact repeat, and controlled active-map/pass histories | Which sign assignment survives the per-active-map overwrite and is it reproducible from compact pass history? R3 Eqs. (44), (49)–(50). | Active-map sequence, pass/iteration history, PTC chunk/channel order, reconstruction or gap; JINC conclusions remain conditioned on `SCI-MAP-002`. |

### Acceptance boundary

The coordinator must approve the valid comparison set and numeric acceptance
policy before any run. This design supplies no tolerance. A complete return
can establish only the named ensemble and operator facts at exact d501. It
cannot validate a variance, weight, S/N, significance, threshold, feedback,
aperture estimator, production count, JINC contract, filter contract, or fruit
model.

### Cost status

Static identity review and inspection of already-existing returned artifacts
would be low cost. Any new astronomical reduction, multi-observation paired
run, high-count stack, or Beammap/fruit study is **held, not admitted**, until
`FRAMEWORK-NUM-001` supplies a cost decision and the coordinator issues a
separate request. A 64-realization case is optional capacity only after that
decision; it is not required and does not define acceptance.

## Design B — future current-versus-residual A/B

Status: **design-only and blocked**. There is no approved residual-mode
implementation SHA. The narrow d501 fruit path demonstrates a pre-readdition
state interface but is not substituted for a successor with explicit ensemble
identity and provenance.

### Entry prerequisites

Before a residual A/B can be requested, all of the following are required:

1. `SCI-FRUIT-001` approves the final source-subtracted state, model identity,
   readdition boundary, and conditioning facts without claiming unbiasedness.
2. A successor application SHA explicitly selects and records `current` versus
   `final_pre_readdition_residual` ensemble mode.
3. Compact RNG key/coherence/assignment policy and completed realization IDs
   are bound by digest to both A and B products.
4. The same checksum-pinned input, observation membership/order, eligible
   sample state, realization keys, map/coadd/filter operators, and admitted
   product identities are pairable. Any unavoidable data-derived weight
   difference is recorded as a difference, not silently normalized away.
5. `SCI-NOI-002` and `FRAMEWORK-NUM-001` approve respectively the consumer
   question and the cost/acceptance plan.

### Paired questions

- Compare current `A D_epsilon x` with residual
  `A D_epsilon (x - shat)` using the same admissible realization assignments
  (R3 Eqs. (15)–(16)).
- At named source locations, compare empirical mean and second-moment imprint,
  including literal anti-source realizations, without interpreting either
  stack as a variance estimator (R3 Eqs. (21), (26)–(29), (40)–(43)).
- In source-free/blank regions, test whether the residual model suppresses
  genuine noise, explicitly retaining the perfect-removal and complete-overfit
  limits (R3 Eqs. (30)–(31)).
- Verify that fixed source readdition changes the science product but not the
  residual realization stack (R3 Eq. (32)).
- Repeat the comparison after fixed coadd and fixed filter only when their
  realized state and identity are equal and admitted (R3 Eqs. (33)–(39)).
- Report use-specific distribution, covariance, and scale adequacy separately
  (R3 Eqs. (51)–(53)); do not collapse them into one pass/fail label.

### Prohibitions and stop

Do not invent tolerances, repair code, treat the legacy d501 fruit path as the
missing approved SHA, request Unity, or launch a reduction from this design.
Do not infer that reduced source imprint proves correct physical-noise
covariance or an unbiased residual. Stop at the returned evidence bundle and
route interpretation through coordinator review and `SCI-NOI-002`.
