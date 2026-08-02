# FRAMEWORK-NUM-001 Corrective Action

Date: 2026-08-02

Status: framework controls integrated; replacement SCI-CAL execution held

Authority: project-owner direction to treat the SCI-CAL EL25 stop as a
framework-level corrective action

## Scope and immutable boundaries

This action governs the design, authorization, execution, and salvage of
costly numerical audit studies. It does not modify Citlali application code,
TolTECA, frozen candidates, the selected passband authority, the scientific
domain, or the one-percent representation-fidelity gate. It does not approve
the composition-closure decisions `FRAMEWORK-COMP-D005` or
`FRAMEWORK-COMP-D006`.

The reviewed SCI-CAL records are:

- preregistration commit
  `fe3b3a1f7885334c50337382d97a84121dbe57c0`;
- stopped-execution/failure commit
  `5d1597ca2d18f5e35519f6e62b5a014aea736fad`; and
- evidence branch `codex/sci-cal-001-atmosphere-operator`.

No replacement AM calculation, partial candidate evaluation, CAL repair,
Unity request, or re-audit was performed by this corrective action.

## Incident and systemic root cause

The EL25 runner completed 12 of 16 cases and 672 of 896 full AM grids. It had
already established exact equality between the parsed atmospheric
transmission and the frozen target. A later, redundant reconstruction compared
two equivalent binary64 construction paths:

- runtime value `1.34988558021834626e-01`;
- frozen Decimal-cast value `1.34988558021834681e-01`;
- two-ULP difference `5.55111512312578270e-17`; and
- unregistered hard bound `5.0e-17`.

The source-only check protected no named scientific or integrity requirement,
had no conditioning or ULP derivation, had no propagation to the one-percent
calibration metric, and could have been exercised over all frozen cases before
the first AM call. It nevertheless used the same abort mechanism as genuine
integrity gates and thereby invalidated the whole study.

The immediate defect was not simply a too-small epsilon. The governance system
lacked five separations:

1. registered scientific/integrity conditions versus source diagnostics;
2. exact identity versus numerical equivalence;
3. deterministic preflight versus expensive runtime reachability;
4. raw-model validity versus evaluator/decision validity; and
5. safe execution stop versus retroactive evidence invalidation.

The audit inventory found the same costly-repeat pattern elsewhere in CAL:
documentary-key mismatch after 1,025 requests, incomplete execution binding
after numerically exact native regeneration, and harness/cache/memory defects
after substantial H2O-scale computation. It also found narrower risks in MAP
regression bounds and exact baseline comparisons. ALIGN, AST, the CAL
q-boundary preflight, and convolve contain useful examples of proportional
preflight, quarantine, or correctly scoped exact identity.

## Corrective controls

The canonical policy is
`doc/audits/NUMERICAL_PROPORTIONALITY_AND_COST_CONTROL_POLICY.md`. It adds:

- a four-class condition taxonomy separating exact integrity, derived
  numerical correctness, scientific acceptance, and engineering diagnostics;
- a machine-readable register for every aborting, invalidating, or failing
  condition and every diagnostic whose implemented route or admission role can
  affect evidence;
- a mandatory scientific-model-free preflight over every frozen tuple and
  deterministic guard path;
- an audit-manager readiness certificate with exact artifact digests;
- independent review of source guard coverage and numerical proportionality;
- separate raw, evaluator, and scientific-decision validity states; and
- missing-only evidence salvage when provenance and independence survive.

The schemas, templates, and validator under `doc/audits/schemas/`,
`doc/audits/templates/`, and `tools/audits/` make these controls mechanically
checkable before launch. They are deliberately narrow: they do not automate or
replace the general hand-maintained scientific-audit ledger.

## SCI-CAL disposition

The stopped confirmation remains invalid and supplies no operator adoption,
domain approval, candidate ranking, or scientific pass/fail result. The 672
full grids remain salvageable raw, warning-bearing evidence because their
inputs, AM executable/profile identities, raw payloads, warning admission,
sidecars, and provenance bindings were verified independently of the failing
diagnostic. No partial passband integration, candidate error, maximum error,
or ranking was inspected.

The only live cache found during this review is a writable 6.4-GiB tree under
`/private/tmp`. Its recorded digests support salvage, but it is not yet a
durable immutable authority. A successor-preparation task must first make a
byte-preserving copy to a named durable location, reverify the complete
manifest and aggregate digest, and protect that copy before admission.

A successor may preserve all scientific tuples, candidates, passbands,
domain, and the one-percent gate; change scientific/numerical semantics only
for the redundant consistency guard by making it a registered diagnostic; add
the non-scientific condition-ID dispatcher, preflight, and layered-validity
plumbing required by the framework; preflight all 16 cases without invoking
AM; bind and independently re-admit the durably preserved cache; and generate
only the missing evidence. Subject to successful salvage verification, that
means three unstarted scale searches and 224 missing full grids. The final
successor evaluator must recompute all scientific metrics over the complete
896-grid union. The exact disposition is recorded separately and is not a
launch authorization.

## Verification and next authorization

The final
[independent framework review](FRAMEWORK-NUM-001_INDEPENDENT_REVIEW_2026-08-02.md)
records `approved` with no remaining P0/P1 gap. All three JSON schemas
meta-validate; the draft templates pass schema validation and fail execution
readiness; the fully bound positive fixture passes the launch gate; all 42
adversarial tests pass; relevant YAML parses with duplicate-key rejection;
Ruff, `git diff --check`, and the full required config preflight pass. The
ledger's CAL disposition digests match the final artifacts exactly.

The next permissible authorization is owner approval to preserve the current
cache and begin a model-free successor-preparation task. The coordinator may
then prepare—not execute—a successor SCI-CAL preregistration, condition
register, model-free preflight, independent review, salvage manifest, and
readiness certificate.
Only after those exact artifacts pass the launch gate may the owner separately
authorize missing-only AM execution. No new scientific decision is currently
required.
