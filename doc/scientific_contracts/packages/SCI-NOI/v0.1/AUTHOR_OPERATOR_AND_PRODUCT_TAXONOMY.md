# SCI-NOI v0.1 — Collision-Free Operator And Product Taxonomy

Status: proposed sanitized Stage B author input; ODQ-101, ODQ-102A/B/C,
ODQ-103/104/105A/105B/106/107/108/109/110A/110B incorporated; exact bytes await owner approval

The semantic prefixes are:

- `NOI-GEN`: realization-ensemble generation;
- `NOI-UNC`: empirical uncertainty inference; and
- `NOI-STD`: derived signal standardization.

Bare `G`, `U`, and `Z` are prohibited as NOI family names. They collide with
MAP projection notation, the PTC removed component, and the PTC transformed
sample. `Z_i^PTC` remains reserved exclusively for the transformed PTC sample.

## Conditioning Class Is Not Complete Method Identity

ODQ-101 approves fixed-state conditional-sign GEN as the ordinary v0.1
conditioning class. ODQ-102A selects the PTC-to-frozen-MAP method as the
ordinary route. Relearned GEN is a separate class and is unavailable until its
scientifically consequential rerun/relearn graph and changed resulting state
are fully specified. This is scientific method identity, not exhaustive
implementation provenance. Neither conditioning class spans
different parents or insertion points.

The route-specific method candidates remain distinct:

| Candidate identity | Scientific role | Present state |
| --- | --- | --- |
| `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` | One observation-scoped detector assignment applies to all its admitted samples at the PTC-to-MAP boundary; detector coefficient mass is balanced separately within each readout network before one frozen MAP operator; output is an NOI realization map, not ordinary MAP science | **Selected ordinary route/coherence/balance family; numerically unavailable** pending exact Stage B mechanics delegated under ODQ-102D and later accepted, plus MAP numerical gates |
| `NOI-GEN/PTC-TO-FROZEN-JINC-CONDITIONAL-SIGN@1` | Randomize exact PTC parent quantities before one frozen JINC operator | Unselected and unavailable under JINC numerical gates |
| `NOI-GEN/REALIZED-MAP-CONDITIONAL-SIGN@1` | Randomize an exact already realized MAP product under a declared map-product coherence law | Unavailable pending exact method and parent authority |
| `NOI-GEN/REALIZED-JINC-CONDITIONAL-SIGN@1` | Randomize an exact complete realized JINC product under a declared role/coherence law | Unavailable pending exact method and numerical JINC parent |
| `NOI-GEN/OWNER-TRANSFORMED-CONDITIONAL-SIGN@1` | Apply exactly the deterministic transformation defined by the appropriate upstream/downstream scientific process to every admitted compatible randomization; NOI owns binding/application, not transformation choice or definition | ODQ-110A ownership/parity rule selected; unavailable pending an exact owner-supplied transformation authority and parity interface |
| `NOI-GEN/RELEARNED@1` | Replay an exact authorized learn/resolve graph that identifies all scientifically consequential rerun/relearned stages and resulting changed state per member | Unavailable; no consequential-state graph approved |

The exact DAG and required route-specific identity are in
[`NOI_GEN_PARENT_OPERATOR_GRAPH.md`](NOI_GEN_PARENT_OPERATOR_GRAPH.md). Parent
numerical coincidence or a shared sign law never merges method identities.

ODQ-110B creates no NOI-owned Wiener algorithm identity. An exact Wiener
transformation frozen by its scientific owner follows the owner-transformed
candidate above. Owner learning/selection/update from a prior NOI product
creates a separately versioned successor transformation, science-product,
GEN, and UNC generation. Per-member Wiener learning is a distinct
`NOI-GEN/RELEARNED@1` method. Every numerical Wiener route remains unavailable
pending its exact owner authority and route-specific boundary.

Inline consumption by MAP is a permitted representation, not transfer of
scientific ownership. NOI owns assignment/design and realization identity;
MAP owns only conforming application within frozen ordinary accumulation.

## UNC Operator Families

| Candidate family | Logical role | Explicit limit |
| --- | --- | --- |
| `NOI-UNC/CONDITIONAL-RANDOMIZATION-SECOND-MOMENT` | Owner-approved ordinary primary pointwise second-moment representation on the common all-member domain | Conditional assignment-law quantity in squared signal units; pointwise/diagonal-like shape does not make it covariance, physical-noise variance, precision, or significance; numerical realization remains gated |
| `NOI-UNC/EMPIRICAL-SCALE` | One declared scalar or projected scale and uncertainty-of-scale state | Not covariance or significance by existence |
| `NOI-UNC/DIAGONAL-VARIANCE` | Separately authorized marginal variance diagonal on an exact domain | Not supplied by the initial second moment by shape; no off-diagonal or precision claim |
| `NOI-UNC/STRUCTURED-COVARIANCE` | Separately authorized stationary/kernel, block, spectral, low-rank, sparse, or another exact structure | Representation, member population, rank/null space, omissions, and regularization are method identity; missing entries are not zero |
| `NOI-UNC/PROJECTED-UNCERTAINTY` | Uncertainty for one exact fixed statistic/operator | Not portable to another statistic |
| `NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE` | Owner-approved `W_hat_cond=1/V_hat_cond` on the finite strictly positive parent domain | Inverse squared signal units; not inverse variance, precision, validity, support, exposure, or a PTC/MAP coefficient; unavailable outside domain; regularization is separate |
| `NOI-UNC/MARGINAL-INVERSE-VARIANCE` | Reciprocal of a separately authorized finite positive marginal variance | Initial conditional second moment is not an eligible variance parent by shape |
| `NOI-UNC/PRECISION` | Exact inverse/generalized inverse of an authorized covariance on a declared domain/subspace | Reciprocal covariance diagonal is not precision by default; rank/null/conditioning/regularization are identity |
| `NOI-UNC/CONSUMER-EFFECTIVE-WEIGHT` | Weight for one exact named estimator/projection/response/domain | Not portable, not a PTC/MAP coefficient, and not an instruction to mutate a parent |

Exact target, estimator, design, domain, covariance, rank, null-space, and
effective-information fields are in
[`FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md`](FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md).

## STD Operator Family

The selected initial method
`NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1` combines the exact immutable
normalized real-observation MAP signal with canonical `sqrt(V_hat_cond)` on
their exact compatible finite-positive valid-domain intersection. Its output
is `empirical_scale_standardized_signal` with unit `1`. It means only “MAP
signal standardized by the stated conditional randomization second-moment
scale.” The algebraic inverse-scale route is not a second implicit method, and
JINC standardization remains separately identified and unavailable.

Exact numerator, transformation, compatibility, dependence, local behavior,
unit, and prohibited claim fields are in
[`STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md`](STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md).

## Exact Supporting Scientific Objects

- [`ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md`](ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md)
  controls assignment design, stable ordering, canonical key identity,
  equivalence, duplicate detection, design rank, completion, and source imprint.
- [`PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md`](PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md)
  controls atomic GEN, UNC, and STD roles, producer/policy ownership, and the
  plan-selected persisted, compact-regeneration, or streaming-sufficient-
  statistic lifecycle without silent fallback.
- [`SCI-NOI_VAL_PROFILE_DRAFTS.md`](SCI-NOI_VAL_PROFILE_DRAFTS.md) contains the
  NOI-owned use-specific profile candidates.

No product automatically realizes the next operator role. Every requested
product publishes its complete exact identity or a typed unavailable/failed
state. Later products never mutate an immutable GEN member or MAP/JINC/PTC
parent.
