# Costly Numerical Study Preregistration Checklist

Study ID: `TO_SET`

Package ID: `TO_SET`

Manager: `TO_SET`

Status: `draft`

This checklist is documentary support for `FRAMEWORK-NUM-001`. The schemas and
mechanical launch gate remain authoritative.

## Scientific freeze

- [ ] The exact protocol, candidates, case/tuple manifest, support/domain,
      scientific metrics, pass/fail gates, and approved authorities are bound
      by SHA-256.
- [ ] No candidate result or partial decisive metric was inspected before this
      freeze.
- [ ] Each acceptance tolerance traces to a scientific decision, analytic
      precision, finite-sample model, repeatability evidence, or other named
      authority.
- [ ] The study states what a valid scientific failure means and confirms that
      it will not be mislabeled as corrupt evidence.

## Condition register

- [ ] A source-level census covers every assertion, exception, abort, early
      return, invalidation route, and scientific-failure route in the runner,
      parser, and evaluator.
- [ ] Every route with aborting or invalidating authority has one stable
      registered condition ID.
- [ ] Each source route records its actually implemented action, and that
      action matches the registered action; a source abort is not merely
      relabeled as a warning.
- [ ] Every Class B abort has either a derived analytic, conditioning,
      interval, or ULP bound for the actual arithmetic path, or a quantified
      propagation into every affected final metric.
- [ ] Every Class C condition traces to the frozen scientific contract.
- [ ] Every condition that remains Class D warns only; a proved identity,
      derived-numerical, or scientific consequence is explicitly reclassified
      as Class A, B, or C and satisfies that class before gaining authority.
- [ ] Exact equality is limited to byte/discrete identity and is not assumed
      between merely equivalent floating-point constructions.
- [ ] Every numerical abort has either a derived bound or a quantified maximum
      propagated effect on each affected final metric.

## Model-free preflight

- [ ] The dry-run performs zero scientific-model calculations.
- [ ] It enumerates every frozen tuple and exercises all deterministic guards.
- [ ] It covers constants, Decimal/binary conversions, coordinate transforms,
      boundary neighbors, candidate dispatch, branches, and output formatting.
- [ ] It reports the exact discovered condition IDs and fails on any
      unregistered abort-capable guard.
- [ ] Every declared data-dependent guard states why its input is unavailable
      before execution, has independent-review acknowledgement, and passes a
      synthetic or fault-injection test.
- [ ] Condition, tuple, boundary, branch, conversion, dispatch, and format
      coverage are complete.

## Cost and salvage readiness

- [ ] Estimated wall time, compute, memory, storage, external scheduling, and
      human cost are recorded with the basis available before execution.
- [ ] Raw-model generation and downstream evaluation have separate versioned
      identities and validity records.
- [ ] The salvage plan defines exact integrity, warning-admission, parsing,
      completeness, provenance, and independence gates for reuse.
- [ ] New output cannot mutate the frozen raw cache; missing computation goes
      to a distinct delta location.
- [ ] The protocol forbids inspection of partial decisive metrics before a
      successor evaluator is frozen.

## Authorization

- [ ] An independent reviewer who did not author the runner/register has
      approved the source census, classification, threshold derivations,
      preflight coverage, and salvage boundary.
- [ ] The readiness certificate binds the exact protocol, runner, evaluator,
      case set, register, preflight, review, and salvage plan by SHA-256.
- [ ] The mechanical `--launch-gate` passes against those exact artifacts.
- [ ] Any separate project-owner, coordinator, external-infrastructure, or
      resource authorization has been recorded.

Until every applicable item is complete, certificate status remains `draft`
or `denied` and no costly model execution is authorized.
