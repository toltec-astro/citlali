# SCI-FLT v0.1 Ownership And Boundary Classification

Date: `2026-08-30`

Status: Stage A proposal for scientific-owner review

## Program Adherence And Prior-Work Recovery

This classification follows package-specific recovery and does not reopen or
modify any frozen or approved adjacent authority. “Owns” below means the
scientific package that must define the fact; it does not assert that a usable
numerical route currently exists.

## Ownership Table

| Fact or decision | Owner | SCI-FLT relationship |
| --- | --- | --- |
| Parent map estimand, normalization, grid/frame, map units, response identity, support/validity, and covariance availability | SCI-MAP or SCI-JINC, as named by the parent | FLT consumes an immutable, content-bound parent and cannot strengthen its claims. |
| Timestream temporal filtering, notch/high-pass/FIR state, and temporal-filter flags | RTC | Outside map-domain FLT. |
| Transformed timestream, contribution coefficients, and pre-map facts | PTC | Upstream input to MAP/JINC; not redefined by FLT. |
| Absolute calibration, passband/color correction, and calibration covariance | CAL | FLT carries/binds available authority; it cannot infer missing calibration from units or a kernel. |
| Exact map-domain transformation or estimator purpose and method identity | Proposed SCI-FLT method package | Core FLT ownership. |
| Operator parameters, template/kernel/prior identity, fixed or learned state, order, domain, support, edge/padding/missing policy, normalization, units, response, lifecycle, and failure policy | Proposed SCI-FLT method package | Core FLT ownership, content-bound for every applied product. |
| Transformed signal/product identity and immutable parent lineage | Proposed SCI-FLT method package | Core FLT product ownership. |
| Transformed response/kernel/transfer identity | Proposed SCI-FLT method package, with input facts from MAP/BEAM/CAL as applicable | FLT defines how the selected method transforms the admitted response object. |
| Filter-specific support and scientific validity policy | Proposed SCI-FLT method package | Parent validity remains distinct; VAL may later register/evaluate exact FLT-owned policy. |
| Deterministic propagation of a declared parent covariance through a fixed operator | Proposed SCI-FLT deterministic package | May define the mathematical propagation and honest output representation; cannot invent unavailable parent covariance. |
| Randomization design, empirical conditional second moment/covariance, inverse conditional scale, and standardized signal | SCI-NOI | FLT supplies the exact frozen transformation and compatible target product; NOI owns inference and attachment. |
| Selection or learning of a Wiener/noise-dependent operator | Proposed SCI-FLT inference-bearing package at an explicit NOI/noise-model boundary | NOI information may be an input, but NOI does not choose or define the filter. |
| Source model, source location, fitted amplitude, morphology, fit background, and source-use policy | SCI-BEAM for frozen Beammap scope; future SCI-SRC/mode package otherwise | FLT must not turn a transformed amplitude into a fitted-source claim. |
| Beammap effective PSF and Beammap calibration/sensitivity products | SCI-BEAM | May be an explicitly versioned template/response input or downstream consumer. |
| Pointing and OOF scientific interpretation | Future mode-specific authority | They may consume a named FLT product but own their fitted corrections/interpretation. |
| FRUIT source model, subtract/add cycle, learning, recurrence, stopping, restart, response, validity, and failure | SCI-FRUIT | FLT can supply a named transformed product only; it does not authorize iterative use. |
| Named-use policy evaluation and Registry binding | SCI-VAL evaluator plus the fact/policy owner | VAL does not invent FLT policy. |

## Proposed FLT Internal Ownership

The `SCI-FLT` tranche should be split before Stage B:

### `SCI-FLT-DET`

Own fixed deterministic map-domain transformations for which the complete
operator state is fixed before application to the admitted parent random
field. Initial candidate methods are fixed convolution and a genuinely fixed
low-pass transform. Each method still has its own purpose, transfer/response,
normalization, support, and product identity.

This package does not automatically own local normalization, parent-derived
templates, or adaptive masks. If those make the applied coefficients depend on
the input random field, the total method must be explicitly classified rather
than called fixed by reference to its final numerical convolution step.

### `SCI-FLT-INF`

Serve initially as a tranche, not necessarily one final contract, for methods
whose estimand or operator depends on a prior, covariance/noise model, learned
state, source model, or target-data statistic. Candidate method identities are:

- Wiener map transformation;
- matched or generalized least-squares template-amplitude estimator;
- source-learned or source-sensitive filtering; and
- input-spectrum-thresholded map-domain destriping.

These should split further before or during their own Stage A work unless the
owner establishes that they share the same estimand, prior/state semantics,
response, uncertainty, and lifecycle. A shared implementation class or FFT
mechanism is not sufficient.

## Boundary Rules

1. Every FLT output names one immutable parent MAP or JINC product and one
   exact applied transformation generation.
2. Parent facts are carried as parent facts; transformed support, validity,
   response, and uncertainty require their own identities.
3. A fixed operator may be reused across parent science and admitted NOI
   members only when exact compatibility and parity are bound.
4. Learning, updating, or selecting an operator creates a new immutable state
   generation. Applying a frozen learned operator and learning independently
   for each NOI member are distinct methods.
5. Filtering observations and filtering a coadd are distinct product methods
   unless an explicit scientific contract proves the applicable equivalence.
6. A source-shaped kernel does not by itself create a matched estimator or a
   flux product.
7. Unknown or unavailable covariance, response, validity, or calibration is
   recorded as such and never represented by numerical zero or inferred from
   a label.
8. Downstream Pointing, OOF, Beammap, source-fitting, NOI, and FRUIT uses
   require an exact consumer-owned admission of the FLT product.

## Open Ownership Questions

The package split, deterministic-family membership, treatment of adaptive
support/normalization, Wiener and matched-estimator separation, legacy
empirical-scale placement, and downstream admission are owner decisions in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
