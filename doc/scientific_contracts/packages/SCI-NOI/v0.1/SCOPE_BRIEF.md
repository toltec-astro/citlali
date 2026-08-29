# SCI-NOI — Noise Realizations And Empirical Uncertainty Scope Brief

Status: Stage A owner-review candidate; not owner-approved

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-29`

Starting source identifier:
`codex/scientific-contract-library@5f206cf46bb2868aadb00f37dbbbc3944ac4ec8c`

Approved source identifier: unavailable until owner approval

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It starts after, and does not alter, frozen SCI-MAP v0.1/r0.7.1 or frozen
SCI-JINC v0.1/r0.3.

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager on `2026-08-29`; scientific-owner review
  pending
- Existing science cited: `SCI-NOI-001` R3 conditional-sign ensemble core and
  `SCI-NOI-002` finite-ensemble/covariance core
- Existing science adopted provisionally: general coherence-unit/sign-law
  formalism; randomization-versus-physical-noise distinction; source-imprint
  caveat; fixed observation/coadd/filter propagation; finite-design,
  centering, rank, covariance-domain, projected-uncertainty, and use-specific
  adequacy reasoning
- Existing owner decisions abstracted: compact deterministic assignment
  identity; no dense sign-stream requirement when exact reconstruction exists;
  enabled-positive versus disabled-zero cardinality; truthful current
  source-imprinted conditional-stack labeling; distinct S/N-like identities;
  and package-level provenance joins
- Existing material superseded for this package: the old NOI-001/NOI-002
  package split; fixed-state-only scope; any implication that generation
  establishes uncertainty; any treatment of `sig2noise` as uncertainty; and
  any promotion of empirical NOI weights into PTC/MAP coefficients
- Existing material deferred or excluded: implementation, audits, findings,
  repairs, re-audits, tests, accepted runs, Unity, achieved performance,
  integration, readiness, production status, defaults, and unapproved
  estimator recommendations
- Genuinely new Stage B work: reconcile the recovered mathematics with the
  frozen MAP/JINC parent contract; define one package with three separately
  typed operations; add relearned-method identity without silently selecting
  it; and resolve only the owner questions listed below
- Proposed author inputs: this Scope Brief, both exact recovered cores under
  [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md),
  [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md),
  and
  [`AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md)
- Author exclusions: recovery and dossier records, raw owner/audit/repair/
  validation material, implementation, adjacent full packages, internal draft
  recommendations, and every unlisted source

Confirm that this opening and exact packet were reviewed before launching
scientific authorship: **no**. Stage B is blocked on explicit owner approval.

## 1. Scientific Purpose

SCI-NOI defines how declared pseudo-realizations are generated, how a named
empirical uncertainty quantity may be inferred from their completed ensemble,
and how an immutable signal parent may be standardized by an authorized
uncertainty product.

It exists because these operations answer different questions:

- a realization generator defines a conditional random or designed ensemble;
- an uncertainty estimator defines a target law, finite-ensemble estimator,
  domain, and limitations; and
- a standardizer combines an already existing signal estimand with an already
  authorized scale product.

No arrow between those roles is automatic. The contract must state exactly
which generated ensemble is admitted by which inference method and which
uncertainty product is admitted by which standardization method.

## 2. Primary Families And Hard Boundary

### Family G — realization-ensemble generation

A Family G method starts from immutable input parents and a requested method
plan. It produces a declared finite ensemble or an explicit unavailable/failed
state. Its authority includes method identity, conditioning state, randomized
unit, joint assignment law, grouping, balance/pairing, cardinality, seed/key
policy, operator route, support/validity preservation, completed membership,
QC, persistence/reconstruction state, and lineage.

At least two conditioning families must remain distinguishable:

- **fixed-state generation**, which holds the realized reduction and learned
  operator state fixed and estimates variation conditional on that state; and
- **relearned generation**, which repeats named learning/resolution steps and
  attempts to include their variation.

They are different methods, answer different questions, and cannot share one
ensemble identity. Stage A does not select either as the ordinary baseline.

### Family U — empirical uncertainty inference

A Family U method consumes one exact declared realization ensemble. It owns
the estimand/target law, centering, finite-design normalization or correction,
missingness, dependence/effective-information treatment, covariance domain,
pooling or regularization, calibration, output units, support, validity, and
limitations.

Generation alone supplies none of those claims. A completed ensemble may be
useful only as a diagnostic; may support a conditional randomization variance
but not physical-noise variance; or may support one projected scale while
failing a covariance, precision, or tail-law use.

### Family Z — derived standardized signal

A Family Z method combines one immutable MAP or JINC signal parent with one
authorized Family U uncertainty/scale product for the exact same estimator,
response, domain, support, and identity. It has a separate method and product
identity from both parents.

The default claim class is **standardized by the stated empirical scale**.
Gaussian significance, detection probability, false-alarm rate, completeness,
purity, or a universal “N-sigma” interpretation requires separate null,
selection, search, multiplicity, and validation authority.

## 3. Typed Inputs And Prohibited Conflations

The contract must keep distinct:

1. producer-owned sample validity and causes;
2. PTC/MAP analysis or gridding coefficients;
3. realization signs, partitions, or randomized assignments;
4. realization availability, support, and QC;
5. empirical variance or covariance;
6. empirical marginal inverse variance, precision, or consumer-effective
   weight; and
7. standardized signal.

Numerical resemblance, inverse-square units, filenames, or historical field
names cannot establish an identity join. In particular, an empirical NOI
weight is not a MAP-facing PTC coefficient. Any future use of an NOI product
to alter mapmaking or coaddition is a separately authorized feedback or
successor method, not ordinary Family U inference.

## 4. Parent And Operator Boundary

One NOI product binds exact immutable parent generation(s), including:

- MAP or JINC method and product identity;
- observation or coadd membership;
- RTC, CAL, PTC, AST, and VAL parent/source identities carried by that map;
- coefficient, coordinate, WCS/frame, grouping, normalization, support,
  response, covariance-status, validity, and lifecycle identity;
- deterministic filter identity when the ensemble is evaluated after a
  filter; and
- every learned or selected state held fixed or explicitly rerun.

NOI does not rewrite the claims or validity of the parent. A later empirical
variance, covariance, response-qualified uncertainty, or standardized signal
is a new versioned companion. Absence of complete covariance does not
invalidate the parent or prohibit later scientific analysis.

Frozen SCI-MAP permits the same fixed map/coadd operator for an admitted
realization only when membership, projection, grouping, coefficients,
normalization, support-row selection, boundary, and atomic coadd admission are
fixed or identically prescribed. Re-estimation defines a different named NOI
method. Frozen SCI-JINC currently authorizes no ordinary numerical route and
defers response/covariance companions; a future JINC-attached NOI product must
not fill those unavailable states by analogy.

## 5. Realization-Generation Questions The Contract Must Cover

For every method, define or explicitly type unavailable:

- randomized object: detector/channel, sample, scan, subscan, observation,
  coadd member, residual block, or another exact coherence unit;
- insertion point: before MAP/JINC, after observation maps but before coadd,
  after coadd, after deterministic filtering, or another exact route;
- fixed/relearned state across RTC, PTC, AST, MAP, JINC, filtering, source
  model, and consumer selection;
- requested/effective/observation-resolved/realized activation and count;
- assignment-space cardinality, balance, pairing, complement, duplicate,
  replacement, and cross-observation rules;
- seed/key derivation, algorithm/version, stable identities, and
  scheduling-independent regeneration;
- source-cancellation target and known leakage, including deterministic sky,
  scan-synchronous residuals, filter effects, and source-model error;
- common or variable support, preservation of producer validity/causes, and
  realization-local availability/QC; and
- persistence plan, sufficient statistics, reconstruction capability, and
  audit limitation.

No per-sample provenance stream or stored sign cube is required when exact
regeneration from immutable parents, a bound algorithm/version, configuration,
and seed/key policy is possible.

## 6. Empirical-Uncertainty Questions The Contract Must Cover

For every method, define or explicitly type unavailable:

- the exact target: conditional randomization scatter/covariance, repeated
  physical-noise uncertainty, calibrated empirical null, consumer-projected
  uncertainty, or another named law;
- empirical or known centering, second moment versus covariance, finite
  normalization/correction, and completed rather than requested membership;
- effects of correlated pseudo-realizations, balance, complements, duplicates,
  incomplete support, and use-specific effective information;
- output domain: per-pixel diagonal, stationary/kernel summary, structured or
  low-rank covariance, full covariance, retained empirical ensemble, fixed
  consumer projection, or unavailable;
- calibration data, overlap/independence, response, pooling, regularization,
  inverse-bias treatment, and uncertainty of the uncertainty estimate;
- units, shape, frame, indexing, support, validity, parent, method, and
  limitations; and
- whether an output is variance, covariance, marginal inverse variance,
  precision, consumer-effective weight, descriptive scale, or diagnostic.

No full pixel covariance matrix is universally required. The representation
must be sufficient only for its stated claim and honest about omitted terms.

## 7. Standardized-Signal Questions The Contract Must Cover

The Family Z operation must bind:

- immutable signal-parent identity and signal estimand;
- exact authorized uncertainty/scale product and its target/domain;
- numerator/denominator compatibility in method, response, units, WCS,
  support, validity, and lifecycle;
- zero, negative, nonfinite, unavailable, and incompatible-scale behavior;
- the resulting dimensionless product identity; and
- a truthful claim class.

`sig2noise` or any compatibility alias cannot serve as the scientific
identity. A historical name may remain only with exact method metadata and an
unmistakable distinction between descriptive standardized signal and
calibrated statistical significance.

## 8. Dependencies And Consumers

- **MAP** owns the ordinary map estimator, base/coadd identity, nonprecision
  coefficients, support, response/covariance disclosure, and immutable parent
  claim. NOI owns later empirical companions.
- **JINC** owns its signed estimator and current limited product authority.
  NOI cannot infer missing JINC coefficient, response, or covariance authority.
- **Deterministic filtering/FLT** owns its transfer, edge/support, response,
  and covariance transformation. A fixed filter must be identical for signal
  and compatible realizations; a realization-dependent filter defines another
  method.
- **Wiener filtering** is inference-bearing when its operator depends on an
  estimated noise model. NOI must distinguish a held-fixed Wiener operator
  from a re-estimated operator and cannot validate the filter from its own
  input ensemble.
- **Beammap, Pointing, and OOF** own their fitted/source/mode interpretations.
  Their historical `sig2noise` or fitted-amplitude ratios are not silently NOI
  uncertainty or significance products.
- **FRUIT** owns source subtraction/add-back, recurrence, learning, stopping,
  and adaptive selection. A FRUIT residual ensemble and a complete relearned
  FRUIT procedure are different NOI methods and require the FRUIT boundary.

## 9. Required Product And Lifecycle Results

For every requested method, the package must publish atomically or type
unavailable/failed:

1. exact method family and version;
2. exact parent bundle(s) and immutable joins;
3. requested, effective, observation-resolved, and realized state;
4. exact realized ensemble membership/design identity for Family G;
5. exact target/estimator/domain identity for Family U;
6. exact numerator/scale/claim identity for Family Z;
7. output inventory, unit, shape, support, validity, QC, and causes;
8. persistence/reconstruction state and limitations;
9. omitted covariance, systematic, response, selection, or learning terms;
   and
10. failure state that prevents a required output from being reported as
    realized success.

## 10. Genuine Owner Questions Before Stage B

The bounded open questions are maintained in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
The first and most consequential is whether SCI-NOI v0.1 authorizes only a
fixed-state ordinary realization method, authorizes fixed-state and relearned
methods as peers, or adopts another explicit baseline/availability rule.
The launch direction identifies fixed-state generation as the likely ordinary
baseline, but deliberately leaves that choice to the owner.

Later questions cover permitted randomization/coherence families; target and
source-suppression claims; exact fixed/relearned state; finite-ensemble
estimators; initial covariance representations; empirical weight roles;
standardized-signal identities; persistence; and mode/filter/FRUIT boundaries.
The Stage B author may analyze a bounded question only when the owner explicitly
leaves it open for analysis; it may not silently choose an implementation
default.

## 11. Non-Goals

SCI-NOI Stage A does not:

- draft or freeze the scientific rationale, normative core, engineering
  conformance specification, requirements, predictions, or PDFs;
- implement, audit, repair, optimize, or refactor Citlali;
- alter frozen MAP, JINC, PTC, AST, RTC, CAL, or VAL authority;
- select fixed-state as the default, choose a relearning procedure, or mix
  their realizations;
- select balance, seed, count, centering, divisor, covariance model,
  regularization, threshold, default, or production policy;
- require dense covariance, dense assignment provenance, or persistence of
  every realization;
- authorize an empirical NOI weight as a mapmaking/coadd coefficient;
- define deterministic or Wiener filtering, Beammap, Pointing, OOF, source
  fitting, or FRUIT science;
- infer Gaussian significance, detection probability, catalog authority, or
  feedback validity from a standardized signal; or
- make implementation-conformity, validation, achieved-performance,
  readiness, or production claims.

## 12. Proposed Allowed References

The future implementation-blind author may open only the exact logical items
content-bound in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md):

1. the owner-approved bytes of this Scope Brief;
2. `SCI-NOI-001_INDEPENDENT_CORE_R3.tex` and
   `SCI-NOI-002_INDEPENDENT_CORE.tex`, both paired with the one binding
   supersession cover;
3. the sanitized conventions and ownership extract; and
4. the proposed operator and product taxonomy.

No raw owner decision, audit, repair, validation, implementation, internal
draft, web source, or external paper is proposed.

## 13. Independence Statement

This brief defines scientific questions and ownership boundaries without
prescribing current Citlali source, schemas, filenames, defaults, or observed
products as the answer. The proposed author packet contains two recovered
implementation-independent mathematical references only under an explicit
cover, plus owner-reviewed sanitized boundary material.

If that packet is insufficient, a future author must return one precise
scientific question rather than inspect the repository, import implementation
evidence, or fill a gap from model memory.
