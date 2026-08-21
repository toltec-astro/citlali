# SCI-VAL v0.1 — Surgical r0.3 Manager Review

Status: owner-approved surgical revision implemented; scientific authority
not frozen

Date: `2026-08-21`

## Authority And Scope

The r0.3 revision is governed by
[`REVISION_DIRECTIVE_R0.3.md`](REVISION_DIRECTIVE_R0.3.md), which is bound to
the owner feedback SHA-256
`9e04c73f8cad5731536720e741d78c53541fe8a378490e16d9aefda9a9c56635`.
The original four-file packet and r0.2 directive remain immutable authority in
the verifier chain. No implementation, audit, test, validation, Unity, or
production-status source was admitted as scientific authority.

## Surgical Content Review

- The active canonical key is `SCI-VAL:independent_exposure@1`. The former
  draft key is unregistered and has no inferred compatibility alias. The
  namespace identifies the registering package, not the policy owner.
- Narrative prose refers to the exact owner in the immutable registry binding;
  the registry itself retains the exact scientific owner and source.
- Noncanonical profiles in worked cases are explicitly hypothetical, and
  their results are conditional on exact registration, applicability, and
  satisfaction.
- Every aggregate proposition has a distinct registered profile and exact
  compatible atomic source-profile binding. Base atomic inputs remain
  homogeneous; no heterogeneous transformation is supplied.
- Structural conflicts make applicability unknown and the decision
  unavailable. In an established domain, decisive exclusion dominates an
  unrelated non-gating unknown/conflict; otherwise a required conflict is
  unavailable. Exception conflict cannot neutralize the restriction.
- Response and uncertainty facts use the exact owner-supplied roles
  `structural_gate`, `required_permission`, `decisive_exclusion`, and
  `advisory`.
- Both views preserve the package title and add the subtitle clarifying that
  VAL governs producer facts and use-specific eligibility, not final map
  validity. Adjacent source tables are r0.3 snapshots of the continuing
  `SOURCE_BINDING_REGISTER.md` authority.

The engineering companion was mechanically reviewed for the complete list in
`VAL-R03-D010`: Core/Registry authority, nonexceptionable origin, profile
binding and invalid-profile handling, four axes, conflict precedence,
response/uncertainty roles, aggregate identity, homogeneous aggregation,
generation boundaries, package-qualified names, source replay, and review and
exception provenance.

## Normative Stability And Layered Disposition

R0.3 preserves all `SCI-VAL-REQ-001--049` and
`SCI-VAL-PRED-001--024` identities and appends none. Totals remain `49`
requirements, `24` predictions, and `73` exact crosswalk rows.

No general SCI-VAL scientific question remains open from
`SCI-VAL-OWNER-QB001--QB006`. QB001 and QB003 retain exact serialization work
as engineering-deferred. QB006 retains sufficient-summary, associative
combine, and exact partition-equivalence declarations as profile-local work.
No package-local profile or representation choice is falsely resolved.

## Final Artifact QA

- Scientist-facing rationale: `8` letter pages; SHA-256
  `6cc42e5802fa0bab938613ace377be5ce87f37a96df921996e4c9914600f9bfd`
- Engineering conformance specification: `20` letter pages; SHA-256
  `07c35ea33b11a8375b28428b3e09973d5402df2c2c9943c2f88d34af0bf141c0`
- Durable verifier: original packet hashes, r0.2/r0.3 directive hashes,
  canonical/no-alias rule, aggregate schema, four roles, continuing source
  bindings, ledger dispositions, sequential IDs, exact crosswalk, dual-view
  split, PDF text, and page expectations pass
- Poppler render and inspection: all `28` pages inspected after the final
  build; no clipping, overlap, broken table, bad glyph, split normative ID, or
  unreadable content found
- Whitespace/conflict-marker check: the temporary library no longer contains
  Git worktree metadata, so ordinary `git diff --check` is unavailable.
  Scoped `git diff --no-index --check /dev/null <file>` checks produced no
  findings for every authorized changed text file. No content was relocated.

This review makes no claim of implementation conformity, representation
fidelity, observational validation, scientific freeze, production readiness,
or adjacent-package readiness.
