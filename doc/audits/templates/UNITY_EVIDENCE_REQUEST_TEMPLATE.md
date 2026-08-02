# Unity external-evidence request

This document is completed by the auditor before candidate results are
inspected. It tells Grant exactly what to execute and what evidence to return.
Codex does not connect to Unity and an unreturned request is not evidence.

## Request identity

| Field | Required value |
| --- | --- |
| Request ID | `TO_SET_PACKAGE_ID-UNITY-NNN` |
| Package ID and audit commit | `TO_SET` |
| Governing/candidate source SHA | full 40-character SHA; no moving branch name |
| Permitted dirty state | `clean` only; reject a dirty run as same-SHA evidence |
| Required Citlali version output | exact `citlali --version` text |
| Compiler/build type/options | `TO_SET` |
| Direct dependency/lock identity | `TO_SET` |
| Runtime resources | node/partition, CPUs, affinity, OpenMP policy, memory policy |
| Evidence owner | Grant or named human operator |

If implementation or contract remediation changes the requested SHA, commit
it and issue a new request. A dirty checkout, even with hashed changes, is not
same-SHA evidence. Do not combine products from different SHAs into one
same-SHA claim.

## Exact configurations

List every numbered authoring file and the generated low-level Citlali input.
Return collision-safe copies and SHA-256 digests.

| Mode/case ID | Ordered configuration paths | File digests | Canonical merged digest | Purpose |
| --- | --- | --- | --- | --- |
| `TO_SET` | `TO_SET` | `TO_SET` | `TO_SET` | `TO_SET` |

State requested, effective, observation-resolved, and realized values relevant
to the audited estimator. No undocumented overlay or interactive edit is
permitted after the digests are recorded.

## Observations, inputs, and arrays

| Case ID | Observation(s) and scan selection | Input/APT/calibration identity | Arrays/networks | Coverage/edge condition |
| --- | --- | --- | --- | --- |
| `TO_SET` | `TO_SET` | `TO_SET` | a1100, a1400, a2000 or justified subset | `TO_SET` |

For every omitted array, network, observation, or required input, give a
scientific rationale. A missing required dataset is a gap, not a passing gate.

## Noise realizations and seeds

| Case ID | Realization count | Generator/seed policy | Exact seeds or derivation | Fixed versus realization-dependent masks/weights |
| --- | ---: | --- | --- | --- |
| `TO_SET` | `TO_SET` | `TO_SET` | `TO_SET` | `TO_SET` |

State what randomness the realization represents, what it conditions upon,
whether realizations are independent/exchangeable, and which coherent or
systematic effects they cannot measure.

## Injections and blank controls

Pre-register every control before results are viewed.

| Case ID | Blank or injected template | Input location/amplitude/units | Repetitions | Quantities recovered |
| --- | --- | --- | ---: | --- |
| `TO_SET` | compact, beam-shaped, resolved, extended, edge, or blank | `TO_SET` | `TO_SET` | amplitude, response, uncertainty coverage, S/N, support, morphology |

Record template normalization, beam/epoch identity, insertion stage, source
positions including edge distance, and whether the injection changes
selection, weights, or calibration. Blank controls should measure the
empirical distribution and spatial/temporal covariance, not only map RMS.

## Required output products and provenance

| Product/provenance artifact | Required identity, shape, units, and metadata | Completeness/cardinality rule | Returned digest/path |
| --- | --- | --- | --- |
| `TO_SET` | `TO_SET` | `TO_SET` | operator completes after execution |

Require, as applicable:

- raw and transformed signals, formal variance/inverse variance, empirical
  variance, covariance summaries, response, support, validity, hits, coverage,
  confidence, and fit/feedback products;
- exact configuration-source, requested/effective/realized, runtime,
  mapmaking, noise, post-processing, and mode provenance;
- complete required FITS HDUs, NetCDF variables/attributes, tables, product
  indices, logs, and retained diagnostics; and
- zero unexpected error-, critical-, or fatal-level records and zero silent
  required-data skips.

## Comparisons and pre-registered tolerances

| Comparison ID | Candidate quantity | Reference or analytic prediction | Metric | Acceptance bound and justification |
| --- | --- | --- | --- | --- |
| `TO_SET` | `TO_SET` | `TO_SET` | exact, absolute, relative, coverage, false rate, covariance, recovery | `TO_SET` |

Specify comparator version, exact options, stable product matching, volatile
allowlist, and all tolerances before viewing the candidate. Bounds must follow
analytic precision, finite-sample distributions, repeatability, controls, or a
recorded scientific decision. Do not loosen an accepted profile to admit a
candidate.

Include applicable checks for:

- exact configuration and product inventory;
- sequential/OpenMP or other parallel equivalence;
- formal-versus-empirical residuals and uncertainty coverage;
- response and amplitude recovery by source/template class and edge distance;
- false S/N/support and connected-region behavior in blanks;
- output covariance/correlation or effective independent-mode count;
- unaffected-product regression against the accepted predecessor; and
- performance/RSS only when the package or a defined trigger requires it.

## Cost and execution-readiness controls

The audit manager classifies this request before dispatch:

| Field | Required value |
| --- | --- |
| Study ID | `TO_SET` |
| Costly study | `true` or `false` |
| Cost basis and estimate | `TO_SET` |
| Tolerance-and-Stop-Condition Register path/SHA-256 | `TO_SET_OR_NOT_APPLICABLE_WITH_BASIS` |
| Model-free preflight report path/SHA-256 | `TO_SET_OR_NOT_APPLICABLE_WITH_BASIS` |
| Readiness certificate path/SHA-256 | `TO_SET_OR_NOT_APPLICABLE_WITH_BASIS` |
| Independent review path/SHA-256 | `TO_SET_OR_NOT_APPLICABLE_WITH_BASIS` |
| Evidence-salvage plan path/SHA-256 | `TO_SET_OR_NOT_APPLICABLE_WITH_BASIS` |

If `costly study` is true, this request is not executable until the exact
register, preflight report, review, and certificate pass
`tools/audits/validate_expensive_study_controls.py --launch-gate` and the
human-mediated Unity dispatch is separately authorized. The preflight must
invoke no Citlali reduction or other scientific model. It must enumerate all
frozen cases and exercise every deterministic guard, configuration conversion,
boundary, dispatch branch, and output-format path. An unregistered aborting
condition is a launch stop.

Raw reduction products, parser/admission records, comparison/evaluator
products, and the final audit decision must be written to distinct,
digest-bound locations with separate validity states. The request must state
what can be retained if a later parsing, comparison, or packaging step fails.

## Commands Grant should execute

Give copy/paste-ready commands using the project-required SSH host alias
`unity_toltec` only for commands Grant runs from another host. Codex does not
execute them. Include:

```text
TO_SET_BUILD_AND_VERSION_COMMANDS
TO_SET_REDUCTION_COMMANDS
TO_SET_AUDIT_AND_COMPARISON_COMMANDS
TO_SET_DIGEST_AND_BUNDLE_COMMANDS
```

Commands must save the governing SHA, version, environment/dependency
identity, complete log, exact config inputs, output inventory, audit reports,
and exit statuses.

## Evidence Grant must return

Return one immutable bundle or manifest containing:

1. request ID, full source SHA, `git status --short`, binary version, build and
   dependency identity;
2. exact commands and exit statuses;
3. ordered configuration bytes/digests and canonical merged input;
4. observation, array/network, APT, calibration, seed, blank, and injection
   identities;
5. complete logs and product/provenance inventory with sizes and SHA-256
   digests;
6. machine-readable audit/comparison outputs and human summaries;
7. every changed, missing, extra, skipped, invalid, or unreadable record;
8. all measured metrics with pre-registered bounds and pass/fail results;
9. durable locations for large products that are not returned directly; and
10. the frozen condition register, preflight report, readiness certificate,
    independent review, raw/parser/evaluator validity records, and every fired
    condition ID when the request is costly; and
11. operator name, date/time, and any deviation from this request.

The auditor records supplied artifacts under `supplied_external_evidence` only
after checking identity and completeness. Until then the ledger retains an
outstanding request and no Unity claim.
