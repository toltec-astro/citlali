# SCI-FLT-INF-ODQ-002 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-002`

Date: `2026-08-30`

Scientific owner: Grant Wilson

Status: approved; closes ODQ-002 only

## Approved ownership and product role

The owner-selected optimal matched-template method belongs to a narrow map-
domain filtering package. The method owns the filtering operation and its
scientific products. Its published signal product is a **matched-filtered
map**: a filtered version of the exact admitted input map product or products,
preserving the applicable map-domain structure and semantics of its parent.

The filter remains scientifically defined by ODQ-001 as an optimal matched-
template amplitude estimator. Its local values use the exact supplied
template, declared noise weighting, and normalization required to return an
unbiased estimate of a matching template amplitude under the stated
assumptions. That estimator identity does not change the published product
role from a filtered map into a detected-source, selected-candidate, fitted-
source, peak, or catalog product.

The package must concentrate on the filtering operation itself, including:

- exact admissible input map products and grouping;
- template or kernel identity and discretization;
- noise model, covariance authority, and weighting;
- normalization, units, beam, and response;
- uncertainty propagation or explicit unavailability;
- support, edge, missing/nonfinite, invalid, and null behavior;
- output validity and atomic product lifecycle; and
- fixed-state, successor-generation, or per-member NOI realization handling.

The exact choices within those roles remain open under ODQ-003 onward.

## Explicit source-analysis exclusion

This package does not own or require source detection, candidate selection,
catalog construction, peak interpretation, deblending, source fitting, or any
other source-analysis behavior. No source-estimation package or SRC ownership
boundary is introduced by this decision. Citlali is not implementing source
detection in this tranche.

A future independently governed source-analysis method may consume an exact
matched-filtered map if later scientific authority admits that parent. Such a
future possibility creates no current source contract, dependency, product
role, validation profile, or ownership assignment.

## Posterior-reconstruction exclusion

A genuine prior-bearing Wiener/posterior sky reconstruction remains a
distinct deferred method. It must not be folded into the matched-filter
package and would require its own recovery, Stage A decisions, scientific
contract, response, covariance, products, and lifecycle.

## Consequences

- `SCI-FLT-INF-ODQ-002` is closed with map-domain filtering ownership.
- The required published signal role is a matched-filtered map, not a source
  estimate or catalog-facing product.
- Candidate source-local, catalog, detection, fitted-source, and source-
  learned families remain outside the selected package without any present
  ownership assignment.
- A posterior/Wiener reconstruction remains separate, deferred, and
  unavailable.
- ODQ-003 is the next owner gate: exact admitted input map parent or parents
  and observation/coadd grouping.

## Nonclaims

This decision does not approve a final package name, author packet, Stage B
launch, input parent, grouping, numerical operator, noise/covariance object,
template instance, discretization, optimality proof, approximation,
regularization, edge/support method, units, response, uncertainty product,
NOI lifecycle, output bundle details, implementation conformity, validation,
performance, readiness, production, freeze, or Unity action. It changes no
SCI-FLT-FIXED or frozen SCI-NOI byte.
