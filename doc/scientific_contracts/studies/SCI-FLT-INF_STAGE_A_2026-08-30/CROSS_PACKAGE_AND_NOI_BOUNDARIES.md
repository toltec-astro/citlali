# SCI-FLT-INF cross-package and NOI boundaries

Boundary identity: `SCI-FLT-INF-BOUNDARIES v0.1/r0.5`

Status: Stage A owner-review candidate; it changes no frozen authority

## MAP to inference-bearing method

SCI-MAP owns the exact immutable normalized map bundle, its parent estimator,
grouping, WCS/grid, support/validity facts, response/covariance declarations,
and observation/coadd identity. The inference-bearing method owns its own
admission policy, state, estimand, operator, response, support/null space,
uncertainty/covariance, products, and failure.

The MAP coefficient field is not precision unless exact MAP authority says so.
No INF method may infer `C_PARENT^{-1}` from a field name, positivity, or
historical use. If the method requires a covariance/inverse-noise operator that
MAP does not supply, the method is unavailable until an exact scientific
source is admitted.

The ordinary numerical MAP route remains unavailable at its frozen PTC
coefficient and numerical `coverage_cut` gates. This study does not bypass
those gates.

ODQ-001 identifies the historical full path as an optimal matched-template
amplitude estimator. Its map-domain output is the amplitude of the exact
supplied template as a function of position; a point-source-response kernel
specializes this to a matched point-source amplitude field. The normalization
must be unbiased for a matching amplitude under the exact declared model and
support assumptions. This identity supplies neither the required parent
covariance nor a numerical route.

ODQ-003 admits both exact immutable normalized ordinary-MAP observation bundles
and exact immutable normalized ordinary-MAP coadd bundles. ODQ-002 fixes their
published product role as a matched-filtered map preserving applicable parent
map-domain structure and semantics. Observation filtering is observation-local;
coadd filtering is coadd-local and binds the exact contributing-observation set
and coadd generation. The filter performs no independent coaddition, and no
equivalence or commutation between the two graphs is presumed. Exact
inheritance, units, response, uncertainty, validity, and bundle details remain
later decisions.

ODQ-004 delegates the exact noise/covariance, spectral-weighting, and parent-
coefficient choice to bounded two-view author development. The historical
radially symmetrized average map noise PSD is a candidate only. It supplies no
current MAP covariance, stationarity/isotropy proof, default, or precision
meaning. Observation and coadd candidates remain separately accountable, and
the MAP coefficient field remains nonprecision/unavailable for estimator
weighting unless a later owner-selected option supplies exact authority.

## JINC to inference-bearing method

SCI-JINC owns a separate signed-coefficient observation estimator and complete
per-array bundle. It is not an alternate serialization of ordinary MAP.
Any INF method over JINC would require a JINC-specific parent, response,
support, unit/beam, covariance, and product boundary. A method defined for MAP
is not automatically applicable to JINC. ODQ-003 does not admit JINC in v0.1.

JINC's numerical route remains unavailable under its frozen gates. This study
does not create a JINC numerical route or cross-observation JINC coadd.

## SCI-FLT-FIXED boundary

SCI-FLT-FIXED is the neighboring strict-linear same-grid transformation
candidate at the base commit. Its exact 17 Stage A author objects and manifest
are protected. This study neither amends nor interprets any active Stage B
work.

ODQ-003 does not admit SCI-FLT-FIXED derivatives in v0.1. If a future INF
method consumes a fixed-filter product, the exact fixed
operator, parent, response, covariance, support, order, and product generation
must be bound. If a future fixed operator consumes an INF product, that is a
different ordered chain. No commutation or response reuse is presumed.

Ordinary source-shaped convolution remains SCI-FLT-FIXED-like deterministic
transformation science when it otherwise satisfies that package's scope. It
is not the owner-selected matched estimator merely because the same kernel is
used: the matched estimator additionally requires its exact noise weighting,
normalization, estimand, and validity contract.

## Frozen SCI-NOI boundary

The following rules are imported from exact frozen SCI-NOI authority at
`f28d7a2617160febca85c1c40e6f7ba7494e266e` through the object bindings in
[`FROZEN_AUTHORITY_AND_SOURCE_BINDING.md`](FROZEN_AUTHORITY_AND_SOURCE_BINDING.md):

1. the transformation owner, not NOI, owns purpose, algorithm, operator,
   parameters/learned state, order, support/edge, units, response, validity,
   lifecycle, and failure;
2. NOI applies the exact owner-authorized transformation to every admitted
   compatible member when estimating uncertainty for the exact transformed
   product;
3. a method learned once and frozen is a fixed-state NOI route even when the
   state is data-derived;
4. if NOI informs learning, selection, or update, the prior UNC input,
   owner-learning generation, resulting state/transformation, transformed
   science product, transformed GEN, and successor UNC are separate immutable
   generations;
5. the prior UNC is dependent input, not independent validation, and cannot be
   mutated;
6. member-specific relearning is a distinct NOI-GEN method under the complete
   consequential-state graph;
7. fixed-state and relearned members cannot be mixed absent a separately
   authorized mixture estimand; and
8. every transformed numerical route remains unavailable until exact owner
   authority and parity are content-bound.

### Required parity cases

| Case | Science state | Member state | UNC meaning | Identity |
| --- | --- | --- | --- | --- |
| declared fixed | exact external state fixed before parent/member use | identical state for all members | conditional on state | base method + `DECLARED_FIXED` |
| parent-learned frozen | state learned once from real parent | identical frozen state for all members | conditional on learned real-parent state | base method + learning generation |
| NOI-informed successor | new state learned using prior UNC | identical successor state for successor members unless separately relearned | conditional successor UNC with explicit dependence | new state/science/GEN/UNC generation chain |
| per-member relearned | real product uses its declared learning graph | each member reruns the exact consequential graph | full-procedure member population, subject to its declared conditioning | separate NOI-GEN method |
| selector | realized science method chosen by exact policy | selector and/or underlying method applied under an exact declared member graph | depends on whether selection is fixed or rerun per member | separate selector parity identity |

Adaptive edge/background learning, PSD shaping, template learning, method
selection, convergence selection, and coefficient calibration must each be
classified in the consequential-state graph; none may hide behind a generic
`fixed filter` label.

## NOI-derived coefficient calibration

Frozen NOI authority distinguishes conditional second-moment products, their
finite-positive reciprocal scale, covariance, precision, and consumer-
effective weight. A future FLT/NOI boundary must therefore state whether an
empirical scalar changes only a diagnostic coefficient, defines a new
versioned estimator normalization, or produces a consumer-specific weight.
It may not mutate an immutable parent or promote a reciprocal to precision by
analogy.

Any standardized product must bind the exact immutable numerator, exact NOI
scale, compatible response/unit/WCS/support/domain, and dependence. It cannot
claim significance, detection probability, completeness, purity, or catalog
authority. Those behaviors are outside this package and require a future
independent scientific contract if ever pursued.

## Deferred source-analysis exclusion

The selected package has no source-estimation or SRC ownership boundary. It
owns a map-domain matched-filter operation and publishes matched-filtered maps.
It does not detect sources, select candidates, construct catalogs, interpret
peaks, deblend, fit sources, or assign significance, completeness, purity, or
morphology.

A future independently governed source-analysis method may consume an exact
matched-filtered map if later authorized. That possibility creates no current
dependency, ownership assignment, product role, handoff, or validation
profile. Source-learned filter/template state is likewise outside the selected
package.

## VAL boundary

Each future INF producer/consumer owns policy for its named use. SCI-VAL may
register and evaluate exact immutable profiles but may not invent INF parent
admission, method support, publication, covariance use, response use, or
fallback policy. At least separate future profiles are likely for:

- INF parent admission;
- learned-state admission;
- public signal/estimand publication;
- uncertainty/covariance use;
- standardized-product use.

No profile identity or rule is approved by this study.

## CAL boundary

CAL owns calibrated signal transfer, unit/beam basis, atmosphere/passband/
color corrections, and calibration covariance. INF may not recover missing
CAL authority from a template, response kernel, or signal-unit label.
Any future cross-band amplitude inference remains outside this package and
conditional on exact CAL authority and covariance.

## RTC and PTC boundary

RTC temporal filtering/destriping and PTC correlated-mode cleaning occur
before MAP/JINC and remain owned there. A map-domain spectral selector cannot
silently reinterpret or compensate for an upstream temporal transfer. The
full ordered response must retain the upstream authority and the selected
map-domain method.

## FRUIT boundary

FRUIT owns source modeling, subtraction/add-back, recurrence, learning,
stopping/restart/selection, response, support, validity, lifecycle, failure,
and interpretation. An exact frozen FRUIT terminal/residual product could be a
future parent under its own boundary. NOI-informed FRUIT continuation and
per-member replay follow the frozen SCI-NOI generation rules. This study does
not define FRUIT science.
