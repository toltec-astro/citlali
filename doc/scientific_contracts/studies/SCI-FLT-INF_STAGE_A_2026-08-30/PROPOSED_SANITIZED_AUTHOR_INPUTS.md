# SCI-FLT-INF proposed sanitized author inputs

Record identity: `SCI-FLT-INF-SANITIZATION-CANDIDATE v0.1/r0.5`

Status: proposed material only; not approved, not exhaustive, not SHA-bound as
an author packet, and not permission to launch Stage B

## Firewall

Any future author packet must be rebuilt package-locally after the owner
resolves the remaining blocking decisions for the selected matched-filter map
operation. The future author must not receive this study's implementation
dossier, code/config/schema,
history, audit/repair/evidence, manager inferences, validation, defaults,
production behavior, or active neighboring work.

The items below describe what may be sanitized. They are not an exclusive
author-input list.

## Candidate shared conventions extract

- requested, effective, observation-resolved, realized, and published state
  are distinct;
- each product binds an immutable exact parent and one authoritative method;
- observation and coadd parents are distinct;
- units, beam, response, covariance, support, validity, missingness, and
  failure are explicit and never inferred from names;
- numerical computability does not imply scientific validity;
- unknown covariance is not zero, diagonal-like is not independence, and a
  reciprocal is not precision by construction;
- learned-once, NOI-informed successor, and per-member-relearned lifecycles are
  distinct; and
- required bundle failure propagates; unavailable does not become zero or a
  silent alternative.

## Owner-approved estimator identity for future sanitization

ODQ-001 authorizes the scientific identity **optimal matched-template
amplitude estimator** for the historical full path. For a future selected
package whose estimand is the local amplitude `A` of an exact supplied
template `t_x` in a parent `m` with exact admitted covariance `C`, an author
may start from

```text
N(x) = t_x^T C^{-1} m
D(x) = t_x^T C^{-1} t_x
A_hat(x) = N(x) / D(x)
```

subject to exact declarations of domain, support, location/indexing, template
normalization, parent beam/response, covariance authority, null space, and
regularization. Under the exact authorized zero-mean model
`m = A t_x + n`, its normalization must give `E[A_hat(x)] = A` for a matching
signal, subject to all declared noise, support, edge, missing/nonfinite,
validity, and response assumptions. Under a complete authorized linear model
with fixed `t_x` and `C`, `Var(A_hat | t_x,C)=D(x)^{-1}` may be derived.
Without those assumptions, `D` is only a normalization coefficient. The
author must state the precise optimality criterion and its conditions.

A point-source-response kernel yields a matched point-source amplitude field;
another scientifically defined kernel yields the amplitude field of that
specified template or shape. This estimator is not ordinary convolution with
the same kernel because convolution lacks the exact noise-weighted,
amplitude-unbiased matched-estimator normalization.

Under ODQ-002, the published signal product is a matched-filtered version of
the exact admitted parent map, preserving its applicable map-domain structure
and semantics. Its local estimator identity does not create detected sources,
selected candidates, interpreted peaks, deblended or fitted objects, or
catalog rows. Those behaviors are excluded without introducing a current SRC
ownership boundary.

## Owner-approved parent and grouping extract

ODQ-003 admits two exact ordinary-MAP parent roles:

- one immutable normalized observation bundle with observation-local learning
  and application; and
- one immutable normalized coadd bundle with coadd-local learning and
  application, binding its exact contributing-observation set and coadd
  generation.

They are distinct realized method/product identities. The author must not
presume that filtering a coadd equals filtering and combining its observations,
must not invent a filtered-map coadd, and must not transfer response,
normalization, support, covariance, uncertainty, state, or validity between
the ordered graphs. JINC, SCI-FLT-FIXED derivatives, and all other derived-map
parents are excluded from v0.1.

## Owner-delegated noise/covariance option assignment

The author packet must include the exact ODQ-004 delegation. Without receiving
implementation or historical mechanics, the author must produce a bounded,
shared-identity option set in both the Scientific Rationale and Contract and
the Engineering Conformance Specification. The options must cover the exact
noise/covariance or weaker spectral-weighting object, parent-coefficient role,
assumptions, approximation, normalization, optimality/unbiasedness
consequences, support/validity, response, uncertainty, NOI handling, required
products, and typed unavailable states for each admitted parent role.

The only historical candidate admitted to that assignment is the owner-
provided fact that Citlali has used a radially symmetrized average map noise
PSD. The author must define and examine that candidate scientifically. No
historical averaging population, radialization order, Fourier convention,
normalization, window/edge rule, coefficient role, threshold, or numerical
mechanic may be supplied or inferred. The candidate is not a default,
covariance authority, or proof of stationarity, isotropy, or optimality.

## Deferred posterior-family exclusion

No posterior-reconstruction material belongs in the selected matched-filter
author packet. If an owner later commissions a separate posterior sky-field
method, that future package and author would require the exact signal prior,
measurement/response operator, noise likelihood/covariance, hyperparameters,
boundary/support, regularization, and posterior quantity. It must derive the
posterior response and covariance from those exact inputs and must not inherit
the template-amplitude equation or an implementation label as a Wiener
specification.

No exact TolTEC posterior method is currently proposed for author admission.
ODQ-001 excludes posterior/Wiener reconstruction as an interpretation of the
historical full path; a future posterior method requires a separate recovery
and scientific contract.

## Candidate learning-generation extract

For any selected base method:

```text
state_g = Learn(parent_g, external_inputs_g)
product_g = Apply(parent_g; state_g)
```

must bind `g`, the learning population, inputs, output state, dependence,
failure, and fixed-state/full-procedure response/covariance. If prior NOI
informs `Learn`, use a new immutable successor chain. If `Learn` runs per NOI
member, define a separate NOI-GEN method and do not mix its members with a
fixed-state ensemble.

## Candidate method-selection extract

A selector is an explicit method with:

- requested primary method;
- eligible alternatives;
- exact observable selection facts;
- deterministic or probabilistic selection law;
- failure and unavailable states;
- realized selected method and exact state; and
- product identity that retains the selected underlying method.

Missing or invalid state must not silently change the method while preserving
the primary label.

## Candidate adaptive-edge extract

An adaptive edge method must specify the exact parent facts used to learn
support/window/background, the learned state, conditional apply operator,
full-procedure response, fill/taper influence region, admitted scientific
region, missing/nonfinite behavior, covariance/parity, and failure. The
historical owner preference for fill as a numerical device with an eroded
scientific region may be offered only under an exact current owner cover; it is
not automatically admitted.

## Candidate parent boundaries

Future sanitized boundary objects may abstract, without copying manager or
implementation facts:

- the frozen MAP observation/coadd parent identities and their explicit
  nonprecision/covariance limitations;
- the frozen JINC identity only to state its v0.1 exclusion;
- the protected SCI-FLT-FIXED identity only to state that derived parents are
  excluded;
- the frozen NOI transformation-owner and fixed/relearned generation rules;
- CAL ownership of signal transfer/calibration/covariance;
- VAL's evaluator/registry-only role; and
- FRUIT/RTC/PTC exclusions.

Each boundary must bind exact source authority and must not make an unavailable
numerical parent available.

## Candidate product/lifecycle extract

The author should receive an owner-approved table containing, for every
required/conditional/optional role:

- exact product identity and units;
- parent and method/state generation;
- response/covariance/support/validity meaning;
- required completion and atomicity;
- disabled, unavailable, failed, and superseded representation;
- retained diagnostics versus public science roles; and
- permitted named consumers.

No current detailed product table is proposed because ODQ-005 and later units,
response, uncertainty, validity, and lifecycle decisions remain open. The
top-level signal role and two distinct ordinary-MAP parent/grouping roles are
already fixed; the ODQ-004 option set is an explicit future-author deliverable.

## Material that must remain excluded

- numerical class/file/function/config/schema/key names;
- observed fallback or sentinel behavior;
- current denominator/PSD/edge algorithms and tolerances;
- historical map-noise PSD mechanics beyond the exact owner-provided candidate
  statement;
- current output names and historical science labels;
- historical validation, adverse results, re-audit gates, and production use;
- historical owner decisions not re-admitted under a current exact cover;
- implementation-derived claims about why the active path is amplitude-like;
- source detection, candidate selection, peak interpretation, deblending,
  source fitting, catalog construction, or an SRC ownership boundary;
- active SCI-FLT-FIXED Stage B material; and
- any unapproved default, template, prior, covariance, threshold, selection
  law, edge rule, response, or uncertainty interpretation.

## Author-packet construction gate

After ODQ-005 onward supplies the required package-local owner decisions,
create a new package directory and a package-specific
`PRIOR_WORK.md`, sanitized `SCOPE_BRIEF.md`, exact boundary objects, operator/
product/lifecycle tables, owner decision record, and exclusive SHA-bound
author manifest. If the selected package still needs implementation evidence
to define its science, stop and return one precise owner question.
