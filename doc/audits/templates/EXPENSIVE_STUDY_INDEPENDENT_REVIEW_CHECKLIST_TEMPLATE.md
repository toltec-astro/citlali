# Costly Numerical Study Independent Review

Review ID: `TEMPLATE-INDEPENDENT-REVIEW-001`

Study ID: `TEMPLATE-STUDY-001`

Reviewer role: independent audit reviewer

Decision: `pending`

The reviewer must be independent of every person or agent that authored the
runner, condition register, preflight implementation/report, or evaluator.
Record exact paths and SHA-256 digests beside every reviewed artifact. A
checked box without supporting evidence is not approval.

## Frozen bindings

- [ ] Protocol, runner, evaluator, frozen case set, condition register,
  preflight report, salvage policy, and this review have exact SHA-256
  bindings.
- [ ] The register and preflight identify the same study and case-set tuple
  count.
- [ ] Raw model execution, raw admission, downstream evaluation, and the
  scientific decision have separate identities and validity states.

## Guard and tolerance review

- [ ] A condition-ID dispatcher or complete static source inventory accounts
  for every abort, invalidation, and scientific-failure site, plus any warning
  route that affects execution, evidence, or admission and any guard explicitly
  demoted to warning. Ordinary informational log messages are outside this
  register.
- [ ] The source inventory contains no unregistered abort-capable guard.
- [ ] Each source site's actual route and implemented action agree with both
  the frozen register and the preflight-discovered action; no `require`,
  assertion, or exception has been relabeled as a warning on paper.
- [ ] A new or revised harness uses the condition-ID dispatcher. Any legacy
  static-inventory exception has a specific manager approval record.
- [ ] Every Class B abort or invalidation has an operation-specific analytic,
  conditioning, interval, ULP, or propagated-impact derivation; no arbitrary
  epsilon controls execution.
- [ ] Every Class C condition traces to an approved scientific contract and
  produces a scientific result rather than corrupting otherwise valid raw
  evidence.
- [ ] Every condition that remains Class D is warning-only. A proved identity,
  derived-numerical, or scientific-acceptance consequence is reclassified as
  A, B, or C and satisfies that class; impact never gives D veto authority.
- [ ] The registered actions precisely state their failure and salvage scope.
- [ ] Every numeric comparison preserves its canonical literal,
  representation, units, and comparison fingerprint; the preflight observed
  the exact same values.
- [ ] Every quantified effect enumerates all affected final metric IDs and a
  canonical bound for each; exact-integrity and diagnostic cases do not claim
  invented numeric bounds.

## Model-free preflight review

- [ ] The preflight made zero scientific-model calls.
- [ ] Every frozen tuple was enumerated and exercised for each deterministic
  guard.
- [ ] Boundary neighbors, constants, Decimal/binary conversion paths,
  coordinate paths, candidate dispatch, output formats, and guard branches
  are complete.
- [ ] Discovered, registered, and abort-capable condition-ID sets agree, with
  empty unregistered and unknown sets.
- [ ] Every remaining data-dependent condition and the cost exposed if it
  fires are named in the readiness certificate.
- [ ] Every data-dependent condition has a reviewed input-availability basis
  and a passing model-free synthetic or fault-injection test bound by digest.
- [ ] The parser/admission validity state is separate from raw execution,
  evaluator validity, and the final scientific decision.

## Salvage and authorization review

- [ ] The pre-execution salvage plan names a durable, read-only reuse
  authority, binds provenance, distinguishes reused from new computation, and
  requires one frozen successor evaluator for the union. If the source cache
  is temporary or writable, a verified preserved copy must exist before reuse.
- [ ] No decisive partial scientific result was inspected before freezing any
  successor evaluator, or the resulting loss of confirmatory independence is
  explicitly dispositioned.
- [ ] The certificate establishes readiness only for the named frozen study;
  it is not itself launch authorization and does not authorize application
  repair, production disposition, Unity access, or re-audit.

## Findings and decision

Unregistered hard-guard count: `TEMPLATE`

Unsupported-threshold count: `TEMPLATE`

Unresolved-finding count: `TEMPLATE`

Evidence notes:

`TEMPLATE`

Final decision (`approved` or `rejected`): `TEMPLATE`

Reviewer identity/role and UTC date: `TEMPLATE`
