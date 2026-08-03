# SCI-CAL-001 TAU025 runner-preparation authorization — 2026-08-03

Status: bounded execution-preparation clarification under the owner-approved
`CAL-ATM-D007` direct-AM study. This is not a new scientific decision.

Decision ID: `CAL-ATM-D007-RUNNER-001`

Authority: coordinator, applying the project owner's approved exact execution
request.

## Finding

The approved request makes a deterministic TAU025 runner a readiness-gate
requirement, but the CAL task correctly found that no such runner or equivalent
preflight implementation exists. The prior approval authorized the defined
study; it did not expressly state that the small task-local implementation
needed to enact its frozen request could be created. This is a workflow
omission, not a new scientific, numerical, or data-quality finding.

## Bounded authorization

CAL may add only the task-local runner, preflight support, and focused tests
needed to enact the already approved execution request beneath
`validation/sci_cal_001_atmosphere_operator_2026-08-01/`. It must implement,
without changing the approved study:

- the exact 1,275-run inventory and the 225 scale-trace inventory;
- immutable input/profile/passband binding and exact requested-literal checks,
  including the approved `1e-12` derived-provenance annotation bound;
- the registered fresh-cache and lock checks, with cache creation deferred until
  all readiness checks succeed;
- the prescribed AM invocation, raw-output/sidecar digest pairing, and
  `WARN-001` disposition; and
- fail-closed preservation of evidence on any run or policy failure.

The runner's focused local tests and dry-run/preflight validation must not
create the selected cache root or invoke AM. Once those tests pass, CAL may
rerun the exact seven-gate readiness preflight. If every gate passes, the
existing owner approval directly authorizes it to create the selected fresh
cache and execute the frozen study; it need not stop again merely to request
permission for that already approved action.

## Boundaries

No Citlali or TolTECA application change, candidate/operator fitting or
selection, numerical-result interpretation beyond the preregistered checks,
Unity activity, repair, re-audit, adoption, production-status change, or
output-format expansion is authorized. Any failure outside the runner's
bounded implementation or the registered gates remains a stop-and-report
condition.
