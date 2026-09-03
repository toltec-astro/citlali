# SCI-NOI v0.1 Stage B Engineering Conformance Specification

Document identity: `SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.1`

Status: implementation-blind draft specification. This document defines what
an implementation-facing conformance review would have to demonstrate; it does
not report that any implementation conforms.

## 1. Conformance object

A conformance candidate is one exact tuple:

```text
(implementation identity,
 SCI-NOI contract version,
 route method and generation,
 immutable parent identities,
 requested/effective/resolved/applied/realized plans,
 product inventory,
 evidence generation).
```

Changing any tuple member creates another candidate. A finite numerical payload
does not establish conformance when identity, lifecycle, or required states are
missing. Review of one method, product, persistence mode, observation, array,
or execution lane does not imply another.

## 2. Required plan record

The GEN plan record shall expose, without relying on a hidden default:

- route method/version and fixed versus relearned class;
- exact earliest parent and PTC-to-MAP insertion boundary;
- exact parent coefficient, projection, support, response, WCS, validity, and
  lifecycle identities;
- canonical observation, detector, array, group, and readout-network IDs;
- exact rational `tau_h` for every observation/network stratum;
- exact random-bit generator content identity and algorithm version;
- opaque seed/key bytes, key namespace, canonical serialization version, and
  counter-domain rules;
- `B_requested` and positive retry cap;
- persistence mode plus requested/effective/applied/realized states;
- terminal failure policy; and
- exact profile/source versions for every named admission decision.

Missing or conflicting fields make the candidate unavailable before numerical
execution. This is the engineering projection of `NOI-REQ-002` through
`NOI-REQ-011`, `NOI-REQ-017`, and `NOI-REQ-029` through `NOI-REQ-034`.

## 3. Deterministic design construction

For each canonical observation/network stratum, a conforming design builder
shall:

1. construct the stable detector domain from scientific identities, never
   encounter order;
2. derive every positive `B_d` from exactly the frozen admitted parent
   contributions;
3. represent each persisted contribution and `tau_h` as an exact rational and
   evaluate the integer/rational cross-multiplied predicate without an
   order-dependent reduction;
4. obtain independent uniform candidate bits from disjoint canonical
   member/stratum/candidate-counter key domains;
5. map bit `0` to `-1` and bit `1` to `+1`;
6. accept the first candidate satisfying the stratum predicate;
7. fail the complete design if a required stratum exhausts its positive retry
   cap; and
8. compose accepted stratum vectors in canonical observation/stratum/unit
   order.

Candidate rejection shall not create a member identity. No builder may relax a
tolerance, balance across strata, alter a parent population, force a complement,
drop a duplicate, or silently change RNG, cap, count, or persistence mode.

The target-law statement is scientific, while any empirical demonstration of
generator behavior belongs to a later evidence plan. This draft makes no such
demonstration.

## 4. Design publication and exact reconstruction

A successful design publication shall include:

- canonical ordered coherence domain and digest;
- exact mass inputs or their content-bound reconstructing parent identity;
- every tolerance and exact comparison representation;
- generator/version, seed/key namespace, and key/counter derivation;
- assignment-design generation and each member identity;
- exact assignment bytes or the complete deterministic reconstruction record;
- assignment digest followed by byte-equality duplicate confirmation;
- complement-orbit relation;
- all distinct member counts;
- the uncentered member-by-coherence sign matrix rank and null-space record;
- equal weight `1/B_resolved` for every successfully resolved member; and
- requested/effective/resolved/applied/realized lifecycle states.

Reconstruction shall be invariant to worker count, process scheduling,
traversal order, container layout, and writer arrival. A digest verifies
identity but does not reconstruct omitted values. Compact regeneration is
conforming only when the complete record reconstructs the declared product
under its stated byte-identical or numerical reproducibility class.

## 5. Member production and atomic failure

For every admitted member, the producer shall bind the exact assignment to the
exact admitted PTC occurrences and apply it before identical frozen MAP
arithmetic. Inline and materialized representations are equivalent only under
that equality. The producer shall publish exact parent/operator, unit/beam,
WCS, support, response, source-imprint, QC, persistence, lifecycle, terminal
state, cause, and provenance facts.

An admitted-member failure fails the whole ensemble for UNC. A conforming
implementation shall not publish completed survivors or a partial streaming
accumulator as an UNC-authorized ensemble. Diagnostic retention shall remain
bound to the failed ensemble. Retry or replacement requires another exact
generation.

## 6. UNC construction

Before calculation, the exact member policy shall evaluate every member and
the exact ensemble policy shall evaluate the complete design. Only the named
positive conjunction authorizes the named action; producer completion facts
remain producer-owned.

For the initial estimator, a conforming calculation shall:

- construct `D_common` by exact identity over every admitted member;
- reject survivor, pairwise, zero-fill, interpolation, and domain-extension
  substitutes;
- accumulate `sum_b (1/B_resolved) M_b(p)^2` without empirical recentering or
  a `B-1` correction;
- preserve squared signal units and the exact source-bearing conditional
  randomization label;
- publish dependence, counts, rank, effective-information, and estimator-
  uncertainty state; and
- publish an atomic product or a typed unavailable/failed state.

For the reciprocal, the implementation shall construct only the finite
strictly positive subdomain and preserve unavailability elsewhere. No zero,
floor, cap, clipping, epsilon, shrinkage, inverse-variance, precision, or MAP
coefficient interpretation is conforming under this method identity.

## 7. STD construction

The standardization admission decision shall bind the immutable normalized
real-observation MAP numerator to the exact compatible `V_hat_cond` generation
from the same frozen operator state. A conforming calculation shall use the
canonical square root, exact compatible finite-positive intersection, unit
`1`, and the exact claim in `NOI-REQ-028`. It shall publish unavailable rather
than zero or infinity on invalid divisions. JINC, an algebraic inverse-scale
route, interpolation, substituted response, or another generation requires a
separate method.

## 8. External transformations, Wiener, and FRUIT

Conformance for a transformed route requires exact owner authority and parity
for purpose, algorithm, state, parameters, operation order, domain, support,
edge and missing-data behavior, normalization, unit, response, validity,
lifecycle, and failure semantics. Applying a transform with a similar name or
numerical output is insufficient.

If the transform is learned or updated from a prior NOI product, the candidate
shall publish distinct immutable prior-input, owner-learning, transformation,
science-product, GEN, and successor-UNC generations plus their dependence. A
prior NOI product shall not be presented as independent validation. Per-member
learning or FRUIT replay is a different relearned method and cannot share a
fixed ensemble.

## 9. Static conformance checks required before numerical evidence

A later conformance review shall first establish all of the following without
executing a scientific reduction:

- every identifier and method version is exact and collision-free;
- the ordinary route is the selected PTC-to-frozen-MAP method;
- all unavailable parent gates remain explicit;
- no default is supplied for tolerance, RNG, seed/key, retry cap, count, or
  persistence mode;
- canonical keys contain every required field in the required order;
- all counts, duplicate/complement states, rank, and weights are separately
  represented;
- no partial-success state can project to an UNC action;
- UNC and STD formulae, domains, units, and unavailable behavior are exact;
- every profile projects only its named action;
- parents and prior generations are immutable; and
- forbidden claim labels are absent from realized-success semantics.

Passing static checks would establish only algebraic/structural consistency of
the candidate representation. It would not establish implementation
conformity, numerical fidelity, calibration, physical-noise meaning,
covariance completeness, significance, performance, readiness, or production
authority.

## 10. Requirement and prediction traceability

The normative source of requirements is `NORMATIVE_CORE.md`. The analytic
prediction source is `SCIENTIFIC_RATIONALE.md`. Exact row-level mapping is in
`REQUIREMENT_PREDICTION_TRACEABILITY.csv`. The package-local verifier requires
every `NOI-REQ-001` through `NOI-REQ-037` and every `NOI-PRED-001` through
`NOI-PRED-015` to appear in that crosswalk, rejects duplicate IDs, verifies
that every cited authority code is one of the 17 manifest-admitted objects,
and verifies the deterministic source/PDF build manifest.

The crosswalk is a documentation trace, not evidence that a prediction has
been tested or a requirement implemented.

## 11. Claim ceiling

This specification defines a future review target only. It makes no
implementation-conformity, validation, calibration, physical-noise,
covariance-completeness, Gaussian-significance, achieved-performance,
readiness, freeze, production-suitability, or production-authorization claim.
