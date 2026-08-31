# SCI-FLT-INF candidate family split matrix

Matrix identity: `SCI-FLT-INF-FAMILY-SPLIT v0.1/r0.7`

Status: Stage A recommendation for owner review; names are provisional and no
row is an approved package or complete method

## Split rule

An operator implementation is not the unit of scientific authority. A
separate package or explicitly versioned method is required when estimand,
prior, learned state, response, uncertainty/covariance, support, failure, or
NOI lifecycle changes. Lifecycle overlays may share a base operator only when
the operator and estimand remain exact and the dependency graph is explicit.

## Candidate rows

| ID | Provisional family | Estimand/claim | State and dependence | Recommended disposition | Readiness |
| --- | --- | --- | --- | --- | --- |
| `INF-A` | **owner-selected matched-template map filter** | filtered version of an exact admitted ordinary-MAP observation or coadd product; local samples have the ODQ-001 optimal matched-template amplitude-estimator identity and matching-amplitude unbiasedness under declared assumptions; published role remains a filtered map | ODQ-003 admits observation-local and coadd-local identities separately; ODQ-004 delegates exact noise/covariance, spectral-weighting, and coefficient-role options to both contract views; historical radial-average map PSD is candidate only; ODQ-005 selects one fixed immutable template-response product per application, sourced from the exact parent-bound point-source response or another explicitly supplied scientific template; ODQ-006 selects `A_hat=<t,Qm>/<t,Qt>` as the exact reference operator, permits approximation only inside a later owner-selected quantitative conformance envelope, and makes unresolved/nonpositive normalization null or unavailable; support remains open | selected map-domain filtering package under ODQ-002; final name remains unapproved; no observation/coadd equivalence; exact reference evaluation is conformant and outside-envelope changes are separate methods | not Stage B ready; noise model, quantitative conformance envelope, and support not selected |
| `INF-B` | genuine Wiener/posterior sky reconstruction | posterior mean or other explicitly named reconstructed sky field | exact signal prior, noise likelihood/covariance, hyperparameters, boundary, regularization and posterior state | expressly not the historical path; separate future package only if later desired | unselected; no recovered complete method; not ready |
| `INF-C` | detected-source, selected-candidate, peak, fitted-source, or catalog inference | source-local or catalog quantity and covariance for a declared selection or fit domain | would require its own selection/model state, calibration, covariance, support, and validity | excluded from the selected matched-filter package with no present ownership assignment; a future independent contract may consume `INF-A` maps | no active Citlali tranche; deferred |
| `INF-D1` | declared fixed state | exact selected `INF-A` operator with state fixed before method application | state from immutable external or parent-owned authority, not learned from the target by this method | lifecycle variant of selected base method; bind exact state source | operator-dependent |
| `INF-D2` | parent-learned then frozen state | conditional product under state learned from the real observation/coadd parent | immutable learning generation followed by one frozen application generation | separate versioned method/generation graph; not a free mode toggle | not ready |
| `INF-D3` | NOI-informed successor state | product after owner learning/selection/update consumes prior NOI output | prior UNC, owner-learning generation, new state, new science product, new GEN and successor UNC are immutable distinct generations | mandatory frozen-NOI successor graph; never mutate prior products | not ready |
| `INF-D4` | per-member-relearned state | ensemble distribution after each admitted randomization reruns the exact declared learning graph | member-specific state and possibly support/response | separate NOI-GEN method under ODQ-104; cannot mix with fixed-state members | not ready |
| `INF-E` | data-thresholded spectral mode selection | map after input-dependent selection/removal/retention of Fourier modes | selected modes depend on the parent and threshold law | separate nonlinear/adaptive filtering package if activated; provisional `SCI-FLT-MODESEL` | inactive and scientifically immature |
| `INF-F` | automatic selector/fallback | output of a declared method-selection policy, including failed-primary handling | request, candidate methods, selection facts and realized method | separate orchestration/policy identity; every realized output retains the selected underlying method identity | current silent substitution is not admissible; not ready |
| `INF-G` | adaptive edge/background conditioning | parent conditioned by learned support/window/background, or a selected estimator applied after that conditioning | masks from weight/coverage; background from signal; optional taper | separate preprocessing method or explicit component of a selected estimator contract; provisional `SCI-FLT-EDGE-ADAPT` | historical precedent exists; current authority absent |
| `INF-H` | NOI-based coefficient calibration | empirically scaled normalization/coefficient field and exact dependent standardized products | depends on a frozen NOI ensemble and admitted region; may mutate current runtime weight field | route to NOI/consumer-derived-product contract with an FLT boundary; not part of the base map estimator | frozen NOI prevents automatic promotion; not ready |
| `INF-I` | source-learned transformation | map or estimator after template/state learned from a fitted source model or selected population | would require an independently governed fit/model generation, selection effects, calibration, and covariance | excluded from the selected package with no present ownership assignment; do not infer from configured analytic templates | no active route recovered; defer |
| `INF-J` | ordered FIXED/INF composition | output of `T_FIXED o T_INF` or `T_INF o T_FIXED` | binds both exact operators, state generations and order | distinct composition identity; never imply commutation | no active route recovered; defer |

## Recommended package structure

The study recommends against approving `SCI-FLT-INF` as one combined package.
Under ODQ-001 through ODQ-003, the ODQ-004 author delegation, and ODQ-005
through ODQ-006:

1. create one narrow map-domain filtering package for owner-selected `INF-A`;
   its published signal role is a matched-filtered map and its final name is
   not yet approved;
2. admit both ordinary-MAP observation and coadd parents as distinct
   observation-local and coadd-local identities, with no equivalence,
   commutation, or filter-owned cross-observation combination;
3. require the future author to develop shared-identity noise/covariance,
   spectral-weighting, and parent-coefficient options in both contract views;
   admit the historical radially symmetrized average map noise PSD only as a
   candidate for analysis;
4. require one immutable template-response product per application with exact
   amplitude convention, units, parent compatibility, grid/WCS, phase,
   support/tails, array/beam/calibration identity, validity, and provenance;
   admit parent-bound point-source and explicitly supplied scientific-template
   sources, treat Gaussian/Airy only as complete-product construction, and
   defer learned templates plus high-pass/delta;
5. make `A_hat=<t,Qm>/<t,Qt>` the exact reference operator, conditional on
   the ODQ-004 `Q` and ODQ-007 support; permit approximations only inside an
   owner-selected quantitative conformance envelope binding normalization,
   template response, support/null behavior, and uncertainty, and make
   unresolved or nonpositive normalization null/unavailable rather than zero;
6. exclude source detection, candidate selection, peak interpretation,
   deblending, fitting, and catalog construction without introducing a current
   source-estimation or SRC ownership boundary; a future independent contract
   may consume an exact matched-filtered map;
7. encode fixed, parent-learned, NOI-informed, and per-member-relearned cases
   as exact lifecycle/method variants rather than a generic `learned` flag;
8. leave `INF-E`, `INF-F`, and `INF-G` as separate packages or explicit
   preprocessing/selection authorities;
9. route `INF-H` through a frozen-NOI-compatible derived-product contract; and
10. leave `INF-B` outside the historical path, requiring a wholly separate
   future recovery/contract if ever requested.

## Why the split matters

Calling every row a filter would erase distinctions that change the claimed
quantity. The selected filtered map may have a unity response to one template
under its exact normalization but is not a reconstructed posterior sky map or
a source catalog. A posterior mean can be biased by its prior and requires
posterior uncertainty. A learned-once result is
conditional on learned state; a per-member-relearned ensemble targets a
different population. A selector's fallback output is not evidence that the
requested method succeeded. An adaptively filled/tapered edge has a different
support and response from a full-footprint fixed operator. A NOI-calibrated
coefficient is not precision by construction.

## Cross-row non-equivalences

- `INF-A != INF-B`; ODQ-001 expressly selects matched-template amplitude and
  excludes a posterior-sky estimand for the historical path.
- `INF-A != INF-C`: the selected product is a whole matched-filtered map and
  performs no detection, selection, fitting, peak interpretation, or catalog
  inference.
- observation-local `INF-A(P_MAP_OBS)` is not equivalent to coadd-local
  `INF-A(P_MAP_COADD)`; no filtering/coaddition commutation is presumed.
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
