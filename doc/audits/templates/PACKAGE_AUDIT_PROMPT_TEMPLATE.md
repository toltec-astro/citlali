# Citlali package scientific-contract audit prompt

Replace every `TO_SET` field before dispatch. This prompt starts one bounded
audit; it does not authorize implementation repair, integration, a push, or a
Unity connection.

## Assignment

Audit package:

- package ID: `TO_SET_PACKAGE_ID`
- package name: `TO_SET_PACKAGE_NAME`
- tier: `TO_SET_A_OR_B_OR_C`
- canonical repository: `/Users/gwilson/GitHub/citlali-refactor`
- authority branch: `codex/refactor-mainline`
- governing source SHA to assess: `TO_SET_FULL_SHA`
- audit branch: `codex/audit-TO_SET_PACKAGE_SLUG`
- suggested isolated worktree: `/private/tmp/citlali-audit-TO_SET_PACKAGE_SLUG`
- coordinator ledger snapshot/commit: `TO_SET_FULL_SHA`

The audit package is one scientific transformation, not a source directory.
Its included scope is:

`TO_SET_INCLUDED_SCOPE`

Explicit exclusions and adjacent package IDs are:

`TO_SET_EXCLUSIONS`

Upstream dependencies, the exact facts required from them, and the consequence
while open are:

`TO_SET_DEPENDENCIES`

Known implementation paths are listed below only so they can be quarantined
until the independent core is frozen:

`TO_SET_IMPLEMENTATION_PATHS`

## Hard boundaries

1. Read and follow the repository `AGENTS.md`, the TolTEC context skill, and
   the project-level authorities it routes you to.
2. Before creating anything, verify and report every relevant worktree path,
   branch, full HEAD, parent/upstream relationship, and dirty state. Report any
   material mismatch with this prompt and do not alter another checkout.
3. Create a fresh isolated audit worktree from the exact governing SHA. Do not
   move an existing branch or reuse an implementation worktree.
4. Do not inspect the contents or diffs of the quarantined package
   implementation paths until the independent-core freeze described below.
   Repository-level architecture, scientific conventions, product intentions,
   and upstream contract documents may be read first. Record any unavoidable
   prior implementation exposure.
5. Do not modify Citlali application code, tests, build files, production
   documentation, candidate branches, other audit branches, or dirty
   worktrees. Audit documents and audit-specific evidence manifests only.
6. Do not push, merge, rebase, cherry-pick a candidate, install/download
   software, or use the network unless Grant separately authorizes it.
7. Unity is human-mediated external infrastructure. Prepare an exact evidence
   request for Grant; do not connect to Unity or claim to have inspected its
   state.
8. Preserve mature RTC/PTC/JINC/Wiener internals for Tier B work unless the
   interface/response evidence supplies a specific reopen trigger.
9. Historical output, a regression pass, or a plausible map is evidence, not
   the mathematical authority.

## Phase 1: independent core before source inspection

First define the claimed estimator from the product intention and approved
upstream abstractions. Use one or more of:

- affine: `y = A x + c`;
- nonlinear/data-dependent: `y = F(x, theta)`; or
- iterative/stateful: `s_(n+1) = G(s_n, x)`, `y = H(s_n, x)`.

Create an exact companion file such as
`doc/audits/packages/TO_SET_PACKAGE_ID_INDEPENDENT_CORE.tex`. Before opening any
quarantined implementation source, it must contain:

- input identities, shapes, ordering, units, frames, indexing, validity, and
  upstream assumptions;
- classification of consequential variables as random, fixed, fitted,
  selected, data-derived, or conditioned upon;
- estimator, expectation, formal variance, full covariance, empirical and
  calibration/systematic uncertainty, and response uncertainty;
- applicable spatial, temporal, template/beam, extended-mode, photometric, and
  iteration response;
- strict distinctions among variance, inverse variance, support, validity,
  hits, coverage, confidence, and response;
- requested, effective, observation-resolved, and realized state;
- analytic limiting cases and pre-registered falsification tests; and
- provisional consumer allowlist/restrictions based only on the claimed
  contract.

For every standard validation method that is omitted, include a specific
`not_applicable` rationale. Tier alone is not a rationale.

Freeze the exact bytes before implementation inspection:

```bash
shasum -a 256 doc/audits/packages/TO_SET_PACKAGE_ID_INDEPENDENT_CORE.tex
git add doc/audits/packages/TO_SET_PACKAGE_ID_INDEPENDENT_CORE.tex
git commit -m "docs: freeze TO_SET_PACKAGE_ID independent core"
```

Record the digest, freeze commit, timestamp, and exact first source-inspection
event in the final audit. If a commit is not possible, stop and obtain an
equally immutable, reproducible freeze method from the coordinator. Do not
hash an evolving whole document. After the freeze, corrections to the
independent core are successor revisions: preserve the frozen bytes and record
what prompted each change.

## Phase 2: implementation and evidence audit

Only now inspect source at the exact governing SHA. Trace the complete package
through:

- science signal and every alternate/parallel operator;
- formal variance/covariance and inverse-variance paths;
- empirical/noise realizations and simulations;
- flags, masks, selection, fill, support, and non-finite behavior;
- sequential and OpenMP/other parallel paths;
- configuration request, effective resolution, observation state, and
  realized lifecycle;
- output writers, product identity, units, metadata, and provenance;
- downstream consumers, aliases, feedback paths, and fail-closed checks; and
- existing analytic, deterministic, injection, blank, regression, same-SHA,
  and astronomical evidence.

Compare each source operation to numbered independent equations. Distinguish
the full mathematical object from any Citlali approximation. A conditional
conformity conclusion must name a stable dependency ID, exact assumption, and
falsifiable test. A known mismatch is `nonconformant`, not conditional.

Classify findings independently as `implementation_defect`, `contract_gap`,
`scientific_policy_decision`, `evidence_gap`, or `dependency_gap`. Use P0-P3,
`observed`/`derived`/`suspected`/`owner_decision`, and confidence separately.

## Required audit artifact

Use `doc/audits/templates/SCIENTIFIC_CONTRACT_AUDIT_TEMPLATE.tex` as a guide,
but keep the document proportional to the package. Create:

`doc/audits/packages/TO_SET_PACKAGE_ID_SCIENTIFIC_CONTRACT_AUDIT.tex`

Compile it standalone with already available offline tools and inspect the
rendered PDF. Do not add a tool or dependency merely to compile the document.
Bulky external scientific data stay outside Git.

The audit must include:

- exact SHA/worktree/branch and independence record;
- scope, estimator, variable classification, response, uncertainty, support,
  units/frames/indexing, and state;
- source/equation conformity trace;
- product and consumer matrix in the form
  `input -> operator -> estimator -> units -> metadata -> consumer`;
- analytic limits and all applicable validation methods or explicit N/A
  rationales;
- open findings, scientific decisions, stable dependencies, owners, and
  closure gates;
- exact Unity/external evidence request with pre-registered comparisons and
  tolerances;
- downstream allowlist, restrictions, and fail-closed uses;
- the four independent status axes and one verdict; and
- a final machine-readable YAML proposal matching the package record in
  `doc/audits/audit-ledger.yaml`.

The YAML is a proposal for coordinator review. Do not edit the canonical
ledger from a parallel package-audit branch.

## Status and verdict vocabulary

Use only the controlled values in `doc/audits/README.md`. In particular:

- validation `complete` does not imply production `approved`;
- `existing_use_only` does not authorize a new interpretation or consumer;
- `amend` can conclude a complete report but does not mean the package is
  done; and
- blockers are findings/dependencies, not an omnibus status.

## Final report and stop condition

Commit audit documents only after the independent freeze and proportional
offline checks. Report:

1. exact audit branch, commit(s), parent SHA, worktree, and clean state;
2. independent-core path, SHA-256, freeze commit, and first inspection event;
3. final LaTeX/PDF paths and compile/render commands;
4. compact claimed estimator and exact implementation operator;
5. every finding, unresolved decision, and dependency;
6. contract, implementation, validation, production statuses and verdict;
7. local evidence and the exact unsupplied external evidence;
8. allowed, restricted, and fail-closed consumers; and
9. confirmation that no application code, frozen worktree, external lane,
   mainline, or canonical ledger was modified or integrated.

Stop after the report. Do not repair the implementation, launch another
package audit, push, integrate, or claim production authorization.
