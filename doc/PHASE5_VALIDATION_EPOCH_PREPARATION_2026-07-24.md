# Phase 5 Validation Epoch Preparation - 2026-07-24

## Decision

The historical Phase 4 validation epoch remains active and immutable. A
successor epoch, `phase5-v2.1-candidate-2026-07-24`, is now registered as
**preparing** for the self-contained TolPROJ V2.1 suite. It is not an accepted
baseline, does not supersede the Phase 4 records, and cannot be reported as
accepted by the validation command.

This split lets validation evolve with the new portable project layout without
rewriting history. It also prevents old project paths from becoming part of
the scientific contract.

## Config Comparison Boundary

The merged low-level config still has to match the selected baseline exactly,
except for a small versioned set of environment bindings:

- input data and calibration paths must retain the same filename;
- the KIDs fit-report directory must retain the same final directory name;
- the Beammap prior must retain the same filename; and
- the output root may move but must remain a non-empty string.

All scientific choices, output switches, algorithm parameters, observation
selection, and filenames remain exact. The policy is
`tolteca-native-project-bindings-v1` in
[`config_binding_policies.json`](../validation/config_binding_policies.json).
There are no general ignored paths.

## Prepared Profiles

| Mode | Preparing profile | Product comparison |
| --- | --- | --- |
| Point | `phase5-point-152389-v2` | Exact, including timestream products |
| OOF | `phase5-oof-152385-152387-v2` | Exact, including timestream products |
| Science | `phase5-science-152390-152392-v2` | Versioned scientific equivalence |
| Beammap | `phase5-beammap-148670-v2` | Versioned scientific equivalence |

Preparing profiles deliberately have no accepted ledger baseline. Until
promotion, `validate_reduction.py` requires an explicit `--baseline` and labels
a passing result `prepared gates pass (not accepted)`.

## Provisional Fixture Audit

The available self-contained suite runs were evaluated as fixture smoke
evidence. These checks prove that the prepared contracts and comparison tools
can consume the real products; self-comparison is not numerical promotion
evidence.

| Mode | Fixture commit | Audit | Config | Contract | Products |
| --- | --- | --- | --- | --- | --- |
| Point | `cfae989c` | Blocked: runtime provenance V1 | Pass | Pass | Pass |
| OOF | `e97de3fd` | Blocked: runtime provenance V1 | Pass | Pass | Pass |
| Science | `7ca0be50` | Blocked: runtime V1 and no pointing provenance | Pass | Pass | Pass |
| Beammap | `cfae989c` | Blocked: runtime provenance V1 | Pass | Pass | Pass |

The OOF fixture exposed a historical contract mismatch: current
PSF-preserving diagnostic maps correctly use the seven-HDU
`formal_standardized_signal` schema. Successor contract
`phase5-oof-products-v2` records that behavior without changing the active
historical contract.

The machine-readable record is
[`phase5_validation_readiness.json`](../validation/phase5_validation_readiness.json).
Render and validate it with:

```bash
$HOME/tolteca/bin/python tools/baseline/phase5_readiness.py
```

Use `--require-ready` only for the final promotion gate; it intentionally
returns nonzero while blockers remain.

Rerun all four fixture checks and verify that their actual pass/fail outcomes
match this record with one command:

```bash
$HOME/tolteca/bin/python tools/baseline/phase5_readiness.py \
  --verify-fixtures \
  --fixture-output-dir /tmp/citlali-phase5-fixtures
```

## Promotion Gate

Promotion requires all of the following:

1. Freeze one full Citlali commit after the deferred build-integration review.
2. Build that commit once with the accepted dependency environment.
3. Run point, OOF, science, and Beammap from the self-contained suite without
   changing the executable, dependencies, or numbered configs.
4. Require runtime provenance V2 and every mode-specific provenance record.
5. Pass config, product-contract, and numerical-product gates with no skips or
   unexpected error-level messages.
6. Add reviewed immutable ledger records and baseline paths for all four runs.
7. Change the successor epoch and its profiles from `preparing` to `active`;
   only then mark the historical epoch superseded.

A repaired failure creates a new candidate SHA and restarts the four-mode
matrix. Existing accepted profiles are never loosened to admit a candidate.
