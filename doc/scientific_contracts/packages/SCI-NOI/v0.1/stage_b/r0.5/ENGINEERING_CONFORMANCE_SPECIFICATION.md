# SCI-NOI v0.1 r0.5 Engineering Conformance Specification

Document identity: `SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.5`

Scientific owner: Grant Wilson

Date: 2026-08-30

Status: implementation-blind proposed scientific-owner freeze candidate;
scientifically freezeable conditional; not frozen.

Normative authority: the six ordered modules in
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.5`, binding-file SHA-256
`aa59ecaaaa149e2990d07623563d90af76c7b3084ee37c497a06e17ebf0fe213`.
This ECS supplies prospective evidence procedures only and authors no science.

# 1. Evidence identity and result vocabulary

Every result shall identify exact product, method, parent, observation, TolTEC
array, plan, generation, lifecycle, and named use. Review compares serialized
scientific identities and typed states, never filenames, shapes, generic flags,
approximate WCS, display decimals, or undocumented defaults.

Method identity, parent/boundary source closure, changed use-profile authority,
plan resolution, numerical realization, response-bearing realization, random-
generator validation, empirical calibration, implementation conformity,
validation, and production authorization shall be reported separately as pass,
fail, unavailable, or not applicable with exact cause.

# 2. Request lifecycle and cardinality evidence

A `not_requested` fixture shall have request axis `not_requested`, no GEN method
or eligibility proposition, no request/effective-plan identity, no resolved
assignment-design identity, `N_requested=0`, and no downstream object.

An `explicitly_disabled` fixture shall retain exact request and effective-plan
identities, effective state `disabled`, and disabling owner, policy, and cause,
while having no resolved assignment-design identity, `N_requested=0`, member,
UNC, or STD. Evidence shall reject aliasing the two zero-cardinality states.

An enabled-success fixture shall have positive integer request and
`N_resolved=N_requested>0`. If any requested assignment exhausts its cap, the
complete design shall fail and publish no smaller successful ensemble.
Rejected candidates shall appear only as construction events.

# 3. Frozen-parent and lossless-rational evidence

Evidence shall expose retained PTC occurrence, detector/network, coefficient
family/value/QC, exact frozen MAP product/generation, `G_pi`, `gamma_i`
family/generation/payload, exact parent contribution, lossless canonical
rational, representation source/digest, projection, unsigned denominator,
numerical `coverage_cut`, WCS/support, response, and application count.

Decoding the rational shall equal the exact parent `a_pi` bit-for-value under
the declared lossless representation. Evidence shall reject display-text
rounding, independent quantization, re-estimation, undocumented decimal
precision, and another coefficient generation. The rational may determine NOI
balance/design identity only; it shall not replace MAP numerical arithmetic.

Mandatory mismatch fixture:

| Fact | Exact value |
| --- | --- |
| Frozen-parent contribution and lossless rational | `10000000000000001/100000000000000000` |
| Rounded display text | `0.1` |
| Total and tolerance | `T_h=1/1`, `tau_h=1/10` |
| Required exact decision | reject, because the lossless value is above the boundary |
| Prohibited display-derived decision | admit by treating rounded `0.1` as equality |

Missing lossless identity makes design resolution unavailable.

# 4. Exact design and arithmetic evidence

Every design record shall include exact per-stratum admitted sets `A_h`,
canonical stratum order and concatenation, `A=product_h A_h`, stable detector
order, alphabet, ideal probabilities, duplicates/complements, all counts,
ranks/null spaces, reconstruction identity, lifecycle, completion/failure, and
preregistration before assignment or output inspection.

Canonical ASCII rationals shall reject nonpositive denominators, unreduced
pairs, noncanonical zero, nonpositive active mass, and thresholds outside
`[0,1)`. Exact accumulation and integer cross multiplication shall reproduce
every mass, numerator, total, and admitted-set decision without floating
epsilon, tolerance relaxation, or best-failed selection. Equality, one exact
rational step above, one step below, and reordered wide-magnitude accumulation
fixtures shall produce their prescribed exact decisions.

# 5. Target probability law and deterministic replay evidence

Evidence shall represent separately:

1. the scientific `Uniform(A)` randomization measure;
2. the declared ideal independent symmetric candidate-bit/key-selection law;
3. the deterministic replay map `F(K,plan)`.

Exact enumeration or analytic proof shall establish first-accepted conditional
uniformity over each `A_h` and the canonical product `A`. The plan shall bind
generator identity/version, opaque key bytes, namespace, serialization,
observation, array, stratum, member, attempt counter, detector order, and domain
separation. Replays varying scheduling, process/worker count, traversal,
container layout, and persistence shall reproduce the same assignment sequence.

Review shall record that replay equality establishes realized identity only. A
single realized sequence and replay success shall not pass iid-independence,
generator-quality, random-generator-validation, or estimator-uncertainty gates.
Those gates remain unavailable pending separately named methods and evidence.

# 6. Design-ignorability and member-admission evidence

Before assignment, evidence shall serialize every binding predicate in `R_b`
or identify its exact invariance proof. For every enumerated `e in A`, a
deterministic predicate shall return the same outcome, or a probabilistic proof
shall establish EQ-017. Output amplitude, sign, source imprint, morphology,
tail value, and assignment-dependent QC shall be advisory and shall not alter
base membership.

Mandatory fixtures shall show:

- a parent/plan-fixed predicate preserving the target law;
- an advisory output diagnostic changing value without changing membership;
- a sign- or amplitude-dependent exclusion violating EQ-017 and making the base
  method unavailable; and
- a failed design not being regenerated under the same method identity until a
  favorable all-pass ensemble appears.

A scientifically required assignment-dependent gate shall fail the base-method
identity check and require a new selection-conditioned target law, estimator,
weights, uncertainty, and claim. No evidence may report the unconditioned
`Uniform(A)` law after output-dependent selection.

# 7. Conditional UNC realization and information evidence

For a realized initial UNC, evidence shall prove `A_UNC=B_resolved` and
`N_admitted=N_completed=N_resolved=N_requested>0`, every positive exact member
decision, EQ-017, and one common domain. Estimator reproduction shall then use
the exact known-zero-center formula with divisor `N_resolved`, no finite-
ensemble centering, and no `B-1`.

For failed or unavailable UNC, evidence shall retain actual counts, member and
ensemble causes, and any inequalities. It shall prove absence of estimator,
survivor subset, renormalization, and changed divisor. A completed-but-policy-
unavailable member and an assignment-dependent failure are mandatory
unavailable-state fixtures.

Counts, duplicate/orbit facts, `r_sign`, `r_map`, estimator uncertainty, and
effective information shall remain separate. An effective-information product
shall state estimator, target, domain, dependence model, and calculation or be
unavailable. Deterministic replay supplies none of these quantities by itself.

# 8. STD, profiles, and freeze evidence

STD evidence shall bind independently realized `m_MAP` and compatible
`Vhat_cond`, reproduce the scale and unit-one product, and emit only the bounded
claim. Fixed-scale and full-response fixtures shall remain separate; without
separately authorized `delta Vhat_cond`, full response is unavailable. No
reciprocal, interpolation, JINC substitution, or significance claim is allowed.

Every immutable r0.18 profile and Registry/source digest shall reproduce. Each
changed use shall bind the exact revised `@2` bytes, owner approval, and new
SCI-VAL source/Registry record before evaluation. An old profile or proposed
name shall not authorize a changed action; missing authority yields unavailable,
not numerical ineligibility or empty success.

`REQUIREMENT_PREDICTION_OWNER_TRACEABILITY.csv` shall cover every stable ID.
The superseding freeze manifest shall bind all owner directives, modules,
profiles, bindings, views, parity reports, PDFs, and freeze record. This draft
may support a conditional scientific-owner freeze disposition, but supplies no
current freeze, implementation conformity, random-generator validation,
calibration, empirical validation, readiness, or production result.
