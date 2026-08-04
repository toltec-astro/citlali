# SCI-CAL-001 TAU025 fresh-retry-root authorization — 2026-08-03

Status: bounded retry authorization under the owner-approved `CAL-ATM-D007`
study. No AM result has yet been accepted.

Decision ID: `CAL-ATM-D007-RETRY-ROOT-001`

Authority: coordinator, applying the already owner-approved direct-AM study and
its fresh-root rule.

## Preserved first attempt

The original selected root

```text
/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_001_root
```

contains only failure-context provenance from the pre-AM runner error. It is
forensic evidence, must not be reused, modified, deleted, or treated as a
partial result. It contains no raw AM outputs, sidecars, failed AM attempts, or
scientific result.

The runner-only correction at task commit
`1e4f96f321d230b8578e2042b5687c7147d75d24` repaired that serialized-inventory
error and passed the required actual-serialization, no-AM command-construction,
inventory, policy, and non-creation regression checks.

## Selected retry root

The selected retry root is:

```text
/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_002_root
```

At authorization recording, its parent exists, the selected root is absent,
and the parent filesystem has `323930752` KiB available. CAL must independently
recheck immediately before creation that this exact root remains absent,
writable, unlocked, and provisioned with at least the registered 12-GiB minimum.
Any failure remains fail-closed; no substitution, reuse, deletion, or inferred
path is permitted.

## Continuation

After the runner correction's committed-source preflight and all seven
registered readiness gates pass for this exact selected root, the existing
owner approval authorizes creation of that root and execution of the unchanged
1,275-grid direct-AM study. The frozen tuple inventory, literals,
derived-provenance rule, `WARN-001` policy, profile/passband bindings, result
schema, and all prohibitions remain unchanged.
