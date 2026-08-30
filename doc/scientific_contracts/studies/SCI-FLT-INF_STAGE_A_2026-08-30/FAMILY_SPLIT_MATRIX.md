# SCI-FLT-INF candidate family split matrix

Matrix identity: `SCI-FLT-INF-FAMILY-SPLIT v0.1/r0.1`

Status: Stage A recommendation for owner review; names are provisional and no
row is an approved package or method

## Split rule

An operator implementation is not the unit of scientific authority. A
separate package or explicitly versioned method is required when estimand,
prior, learned state, response, uncertainty/covariance, support, failure, or
NOI lifecycle changes. Lifecycle overlays may share a base operator only when
the operator and estimand remain exact and the dependency graph is explicit.

## Candidate rows

| ID | Provisional family | Estimand/claim | State and dependence | Recommended disposition | Readiness |
| --- | --- | --- | --- | --- | --- |
| `INF-A` | normalized template-amplitude field | local amplitude of one declared template under one declared inverse-noise/weight model | template, parent coefficient field, spectral model, approximation and support; may be learned on the real parent then frozen | candidate first package if ODQ-001 selects this estimand; provisional name `SCI-FLT-TAMP` or owner-selected source/estimator name | not Stage B ready |
| `INF-B` | Wiener/posterior sky reconstruction | posterior mean or other explicitly named reconstructed sky field | exact signal prior, noise likelihood/covariance, hyperparameters, boundary, regularization and posterior state | separate package from `INF-A`; retain only if owner confirms this is a desired scientific product | no recovered complete method; not ready |
| `INF-C` | matched/GLS scalar or catalog amplitude | scalar/source-local amplitude and covariance for a declared template/location or fit domain | fixed template/covariance or separately learned source state; catalog/location selection may be external | likely SRC/estimator-owned successor rather than generic map filter; may share sanitized GLS math with `INF-A` but not product identity | partial reusable math; boundary not ready |
| `INF-D1` | externally declared fixed state | exact `INF-A`, `INF-B`, or `INF-C` operator with state fixed before parent use | state from immutable external authority, not learned on target parent | lifecycle variant of selected base method; bind exact state source | operator-dependent |
| `INF-D2` | parent-learned then frozen state | conditional product under state learned from the real observation/coadd parent | immutable learning generation followed by one frozen application generation | separate versioned method/generation graph; not a free mode toggle | not ready |
| `INF-D3` | NOI-informed successor state | product after owner learning/selection/update consumes prior NOI output | prior UNC, owner-learning generation, new state, new science product, new GEN and successor UNC are immutable distinct generations | mandatory frozen-NOI successor graph; never mutate prior products | not ready |
| `INF-D4` | per-member-relearned state | ensemble distribution after each admitted randomization reruns the exact declared learning graph | member-specific state and possibly support/response | separate NOI-GEN method under ODQ-104; cannot mix with fixed-state members | not ready |
| `INF-E` | data-thresholded spectral mode selection | map after input-dependent selection/removal/retention of Fourier modes | selected modes depend on the parent and threshold law | separate nonlinear/adaptive filtering package if activated; provisional `SCI-FLT-MODESEL` | inactive and scientifically immature |
| `INF-F` | automatic selector/fallback | output of a declared method-selection policy, including failed-primary handling | request, candidate methods, selection facts and realized method | separate orchestration/policy identity; every realized output retains the selected underlying method identity | current silent substitution is not admissible; not ready |
| `INF-G` | adaptive edge/background conditioning | parent conditioned by learned support/window/background, or a selected estimator applied after that conditioning | masks from weight/coverage; background from signal; optional taper | separate preprocessing method or explicit component of a selected estimator contract; provisional `SCI-FLT-EDGE-ADAPT` | historical precedent exists; current authority absent |
| `INF-H` | NOI-based coefficient calibration | empirically scaled normalization/coefficient field and exact dependent standardized products | depends on a frozen NOI ensemble and admitted region; may mutate current runtime weight field | route to NOI/consumer-derived-product contract with an FLT boundary; not part of the base map estimator | frozen NOI prevents automatic promotion; not ready |
| `INF-I` | source-learned transformation | map or estimator after template/state learned from a source model or fitted source population | SRC-owned model/fit generation, source-selection effects, calibration and covariance | separate SRC-to-FLT successor; do not infer from configured analytic templates | no active route recovered; defer |
| `INF-J` | ordered FIXED/INF composition | output of `T_FIXED o T_INF` or `T_INF o T_FIXED` | binds both exact operators, state generations and order | distinct composition identity; never imply commutation | no active route recovered; defer |

## Recommended package structure

The study recommends against approving `SCI-FLT-INF` as one combined package.
After owner decisions:

1. create one base package for the selected primary estimand (`INF-A` or
   `INF-B`, or both as separate packages);
2. place source-local/catalog amplitude estimation (`INF-C`) at an explicit
   FLT/SRC ownership boundary;
3. encode fixed, parent-learned, NOI-informed, and per-member-relearned cases
   as exact lifecycle/method variants rather than a generic `learned` flag;
4. leave `INF-E`, `INF-F`, and `INF-G` as separate packages or explicit
   preprocessing/selection authorities; and
5. route `INF-H` through a frozen-NOI-compatible derived-product contract.

## Why the split matters

Calling every row a filter would erase distinctions that change the claimed
quantity. A template-amplitude field may have a unity response to one
template but is not a reconstructed sky map. A posterior mean can be biased by
its prior and requires posterior uncertainty. A learned-once result is
conditional on learned state; a per-member-relearned ensemble targets a
different population. A selector's fallback output is not evidence that the
requested method succeeded. An adaptively filled/tapered edge has a different
support and response from a full-footprint fixed operator. A NOI-calibrated
coefficient is not precision by construction.

## Cross-row non-equivalences

- `INF-A != INF-B` unless a future derivation proves an exact equivalence for
  one explicit model and preserves both product claims.
- `INF-A != INF-C` when one is a whole-map field and the other is a selected
  source/catalog scalar, even if their local algebra matches.
- `INF-D2 != INF-D4`; learning once on the real parent and relearning per
  member target different conditional populations.
- `INF-E != fixed high-pass`; input-dependent mode selection is not a fixed
  Fourier transfer.
- `INF-F != successful primary method`; a selector owns the selection record
  and the output retains the realized underlying method.
- `INF-G != numerical-only padding`; learned background and support affect
  admitted scientific response.
- `INF-H != covariance or precision`; that promotion requires separate
  authority.
