# SCI-NOI v0.1 — Collision-Free Operator And Product Taxonomy

Status: proposed sanitized Stage B author input; ODQ-101 incorporated; exact
bytes await owner approval

The semantic prefixes are:

- `NOI-GEN`: realization-ensemble generation;
- `NOI-UNC`: empirical uncertainty inference; and
- `NOI-STD`: derived signal standardization.

Bare `G`, `U`, and `Z` are prohibited as NOI family names. They collide with
MAP projection notation, the PTC removed component, and the PTC transformed
sample. `Z_i^PTC` remains reserved exclusively for the transformed PTC sample.

## Conditioning Class Is Not Complete Method Identity

ODQ-101 approves fixed-state conditional-sign GEN as the ordinary v0.1
conditioning class. Relearned GEN is a separate class and is unavailable until
its complete rerun graph is owner-approved. Neither class is one complete
method across different parents or insertion points.

The route-specific method candidates remain distinct:

| Candidate identity | Scientific role | Present state |
| --- | --- | --- |
| `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` | Randomize exact PTC parent quantities before one frozen MAP operator | Unavailable pending route choice and MAP numerical gates |
| `NOI-GEN/PTC-TO-FROZEN-JINC-CONDITIONAL-SIGN@1` | Randomize exact PTC parent quantities before one frozen JINC operator | Unavailable pending route choice and JINC numerical gates |
| `NOI-GEN/REALIZED-MAP-CONDITIONAL-SIGN@1` | Randomize an exact already realized MAP product under a declared map-product coherence law | Unavailable pending exact method and parent authority |
| `NOI-GEN/REALIZED-JINC-CONDITIONAL-SIGN@1` | Randomize an exact complete realized JINC product under a declared role/coherence law | Unavailable pending exact method and numerical JINC parent |
| `NOI-GEN/FIXED-FLT-CONDITIONAL-SIGN@1` | Use an exact fixed deterministic filter route | Unavailable pending a content-bound FLT boundary |
| `NOI-GEN/RELEARNED@1` | Replay a complete exact owner-approved learn/resolve graph per member | Unavailable; no complete graph approved |

The exact DAG and required route-specific identity are in
[`NOI_GEN_PARENT_OPERATOR_GRAPH.md`](NOI_GEN_PARENT_OPERATOR_GRAPH.md). Parent
numerical coincidence or a shared sign law never merges method identities.

## UNC Operator Families

| Candidate family | Logical role | Explicit limit |
| --- | --- | --- |
| `NOI-UNC/EMPIRICAL-SCALE` | One declared scalar or projected scale and uncertainty-of-scale state | Not covariance or significance by existence |
| `NOI-UNC/DIAGONAL-VARIANCE` | Marginal variance diagonal on an exact domain | No off-diagonal or precision claim |
| `NOI-UNC/STRUCTURED-COVARIANCE` | Stationary/kernel, block, spectral, low-rank, sparse, or another exact structure | Representation and regularization are method identity |
| `NOI-UNC/PROJECTED-UNCERTAINTY` | Uncertainty for one exact fixed statistic/operator | Not portable to another statistic |
| `NOI-UNC/EMPIRICAL-INVERSE-OR-WEIGHT` | Marginal inverse variance, precision, or consumer-effective weight after an exact authorized transform | Every meaning is separate; none is a PTC/MAP coefficient |

Exact target, estimator, design, domain, covariance, rank, null-space, and
effective-information fields are in
[`FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md`](FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md).

## STD Operator Family

`NOI-STD/EMPIRICAL-SCALE-STANDARDIZED-SIGNAL` combines one exact immutable
signal numerator with one authorized finite positive scale in the numerator's
signal unit. Its output is `empirical_scale_standardized_signal` with unit `1`.
It means only “standardized by the stated empirical scale.”

Exact numerator, transformation, compatibility, dependence, local behavior,
unit, and prohibited claim fields are in
[`STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md`](STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md).

## Exact Supporting Scientific Objects

- [`ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md`](ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md)
  controls assignment design, stable ordering, canonical key identity,
  equivalence, duplicate detection, design rank, completion, and source imprint.
- [`PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md`](PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md)
  controls atomic GEN, UNC, and STD roles and producer/policy ownership.
- [`SCI-NOI_VAL_PROFILE_DRAFTS.md`](SCI-NOI_VAL_PROFILE_DRAFTS.md) contains the
  NOI-owned use-specific profile candidates.

No product automatically realizes the next operator role. Every requested
product publishes its complete exact identity or a typed unavailable/failed
state. Later products never mutate an immutable GEN member or MAP/JINC/PTC
parent.
