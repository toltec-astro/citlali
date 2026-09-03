# SCI-POINT Proposed Operator And Product Taxonomy

Status: Stage A candidate; names and roles require owner approval

## Operator Roles

### `POINT-FIT`

Fits the declared known-source model on one exact observation-local array-map
parent. A complete method identity includes:

```text
(parent family/product/version/generation,
 observation, array, frame/WCS/grid,
 terminal FRUIT method/iteration/generation lineage when applicable,
 source identity and expected-center relation,
 model, seed/search rule, fit domain,
 support/validity rule, weight/covariance use,
 parameter constraints, response/calibration state,
 estimator version)
```

Changing any scientifically consequential element creates a different method
or an explicitly compatible successor. “Gaussian fit” alone is not a complete
method identity.

Owner-approved ODQ-004 selects the base-v0.1 estimator identity
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`: the established
zero-background elliptical Gaussian with amplitude, two-coordinate centroid,
two widths, and orientation angle. Stage B must state that working method
completely without silently changing it. Owner-approved ODQ-005 preserves its
configurable expected-center/central-search, weighted-peak initialization,
global fallback, bounded fit domain, and parameter constraints, with distinct
requested, effective, and realized state.

### Cross-array aggregation — external

Owner-approved ODQ-001 keeps this role outside SCI-POINT v0.1. A named
pointing-support producer may combine a declared set of per-array measurements
into one observation-level displacement measurement. It must name
participating arrays, weights, dependence/covariance treatment, exclusion
policy, estimand, failure behavior, method/version, and exact POINT ancestry.

### External correction construction

Transforms one POINT measurement or external aggregate into a pointing-
correction record, including sign, telescope user/paddle offsets, selection,
and native support. Owner-approved ODQ-002 assigns this operation to the named
pointing-support producer, not POINT. SCI-AST retains application authority.

## Product Roles

| Proposed product | Cardinality | Meaning | Not equivalent to |
| --- | --- | --- | --- |
| `POINT-FIT-RESULT` | observation × array × exact parent/method | full fitted parameter result with status and available marginal formal errors; joint covariance may be unavailable | applied correction; intrinsic beam; source catalog |
| `POINT-DISPLACEMENT-MEASUREMENT` | observation × array × exact parent/method | fitted tangent-plane source displacement | absolute celestial position; correction vector |
| `POINT-EFFECTIVE-SHAPE-DIAGNOSTIC` | observation × array × exact parent/method | required fitted effective source widths/orientation and telescope/observing-condition QC metrics for this parent response | SCI-BEAM intrinsic/effective detector PSF authority; unique causal diagnosis |
| `POINT-AMPLITUDE-DIAGNOSTIC` | observation × array × exact parent/method | required fitted amplitude and QC metric with exact unit/calibration/response limitations; eligible for separately authorized CAL/TolProj use | universal source flux; standalone absolute calibration |
| `POINT-FIT-QUALITY-CONTROL-METRICS` | observation × array × exact parent/method | centroid, amplitude, effective shape, and honest fit state for telescope/observing-condition QC | automatic acceptance decision; unique physical-cause attribution |
| `POINT-LEGACY-DYNAMIC-RANGE` | observation × array × exact parent/method | amplitude/full-map RMS | statistical significance |
| `POINT-FORMAL-STANDARDIZATION` | observation × array × exact parent/method | amplitude/formal amplitude error | empirical S/N, detection probability |
| external aggregated displacement | observation × declared array set | pointing-support-producer product under ODQ-001; not a POINT product | per-array POINT result; correction record |
| unavailable result | each requested role | typed reason and affected claim | numerical zero or successful fit |

Under ODQ-007, marginal formal parameter errors and full joint covariance are
distinct representations. Absence of joint covariance is not zero, diagonal,
or independence. Later uncertainty representations attach as separately
versioned companions to the immutable fit result.

## Atomicity

Under owner-approved ODQ-006, each requested array receives one independently
atomic complete, diagnostic-only, or unavailable `POINT-FIT-RESULT` state.
Every limited or unavailable state identifies the affected use or claim and
reason. POINT does not synthesize a missing array result. A table or file may
carry multiple arrays, but file co-location does not merge their scientific
atomicity.

An external aggregate has a separately atomic lifecycle. A failed downstream
aggregate does not erase usable per-array POINT results, and a successful
aggregate must not conceal excluded or unavailable array members.

## Named-Use Evaluation Roles

Owner-approved ODQ-009 requires separate policies for POINT fit completeness,
pointing-support displacement use, telescope/observing QC use, and CAL/TolProj
amplitude use. Their respective named-use owners author them. A product may
have different outcomes under different policies; diagnostic-only is an
explicit use-specific outcome. VAL registers/evaluates the policies and does
not author or compose them. Exact collision-free identifiers and mechanics are
assigned to the Stage B author for later owner approval. No POINT aggregate
profile exists in base v0.1.

## Compatibility Baseline

The proposed base method name is
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`. It captures the established
six-parameter model while allowing Stage B to state the estimator cleanly.
ODQ-004 and ODQ-005 preserve the estimator and its scientifically consequential
configurable search/support/constraint state without freezing incidental
solver implementation.
