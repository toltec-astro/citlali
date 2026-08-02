# FRAMEWORK-NUM-001 independent framework review — 2026-08-02

Record ID: `FRAMEWORK-NUM-001-INDEPENDENT-REVIEW-001`

Decision: `approved`

Scope: framework policy, salvage policy, manager instructions, package and
external-evidence templates, three machine schemas, semantic launch gate,
positive fixture, adversarial tests, CAL incident disposition, canonical
ledger integration, and living status

## Role separation

The policy/schema adversarial reviewer and the document/ledger coherence
reviewer were read-only tasks. They did not author the runner, validator,
schemas, register/preflight/readiness templates, incident records, or canonical
ledger edits. Machine-control implementation and mutation-test authorship were
also separated. The coordinator resolved findings and requested a final fresh
review after all live edits stopped.

## Material findings resolved before approval

The review rejected successive drafts until all of the following were fixed:

- an unsatisfiable `source_artifact` schema composition;
- a source hard stop that could be relabeled as a registered warning;
- `data_dependent: true` acting as a guard-coverage escape;
- lossy JSON-number thresholds and unbound comparison fingerprints;
- self-declared branch coverage and unverified source-site digests;
- missing parser/admission validity and failure/salvage scope;
- a readiness certificate whose names or fixture text implied launch authority;
- a Class B gate that incorrectly required both a derived bound and a final-
  metric mapping instead of the approved either/or rule;
- Class D diagnostics retaining a path to veto instead of requiring explicit
  reclassification as A, B, or C;
- free-form approval roles instead of class-specific authority;
- ordinary informational warnings being pulled into the register;
- stale CAL execution authority and stale machine-record digests in the
  canonical ledger; and
- the current CAL cache being described as retained/read-only although its
  only live copy is writable and volatile under `/private/tmp`.

## Final verification

The final read-only review verified:

- `42/42` adversarial validator tests pass;
- the positive synthetic costly-study bundle passes the semantic launch gate;
- the draft templates pass schema validation but fail the launch gate;
- all three Draft 2020-12 schemas pass meta-schema validation;
- all seven relevant YAML records/templates parse with duplicate-key
  rejection;
- Ruff and `git diff --check` pass;
- the repository config preflight passes all 123 unit tests and every required
  compatibility/authority/boundary check;
- no stale authorization/attestation field names or generated `__pycache__`
  directories remain;
- the CAL disposition digests in the ledger exactly match the final Markdown
  and YAML records; and
- changes are confined to audit governance documentation, templates, schemas,
  validator, and tests. No Citlali application or TolTECA code changed.

## Verdict and boundary

`approved`: no P0/P1 framework or machine-gate gap remains. The package is
ready for one coherent framework commit.

This approval does not establish execution readiness for a future SCI-CAL
successor. It authorizes no cache copy, AM call, candidate evaluation,
application repair, Unity access, production change, or re-audit. The next
open authorization remains `FRAMEWORK-NUM-D001`, bounded to durable evidence
preservation and model-free successor preparation.
