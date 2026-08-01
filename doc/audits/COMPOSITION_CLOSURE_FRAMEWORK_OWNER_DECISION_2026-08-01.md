# Scientific-audit composition-closure framework owner decision — 2026-08-01

Status: `FRAMEWORK-COMP-D001`--`FRAMEWORK-COMP-D004` approved;
`FRAMEWORK-COMP-D005`--`FRAMEWORK-COMP-D006` held for later approval

Scope: scientific-audit framework and package-boundary policy only

## Decision authority and reviewed state

The project owner approved items 1--4 from the read-only composition-framework
reassessment on 2026-08-01. The reassessment used the clean scientific-audit
coordination line at
`2273d61cb2b2347116d91df73ffb8978fc3cceec` and did not modify or steer any
active task, application source, frozen prompt, manifest, handoff,
independent-core artifact, framework document, or ledger record.

The decision preserves these active-lane boundaries:

- `SCI-AST-001` continues its bounded independent audit against its explicit
  abstract `SCI-ALIGN-001` interface. Its governing application SHA remains
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`; its independent core was frozen
  at `17d683ada3856ecb5f0a5c42eed744cb219a3586` before quarantine was lifted.
- `SCI-ALIGN-001` continues mandatory phase zero on
  `codex/repair-sci-align-001` from exact application SHA
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`. Application edits remain barred
  until the required owner review.
- No composition review, repair, re-audit, Unity evidence request, or
  production authorization is created by this decision.

## FRAMEWORK-COMP-D001 — Terminology and governing principle

Decision: approved.

The framework will retain **bounded-operator scientific-contract audits** and
add explicit **composition obligations** plus separate, claim-specific
**composition-closure reviews**. The project does not rename every package a
"compositional audit" and does not expand a package until it owns a scientific
pipeline.

The governing principle is:

> Each package owns one bounded operator and its typed domain and codomain.
> Local conformity does not establish compatibility with adjacent operators
> or closure of a composed estimator. Material cross-operator claims require a
> named closure gate bound to exact contracts, implementation SHAs, evidence,
> and tests.

## FRAMEWORK-COMP-D002 — Active-task protection and sequencing

Decision: approved.

- Do not interrupt, broaden, amend, restart, or otherwise steer the active
  `SCI-AST-001` audit or `SCI-ALIGN-001` phase-zero task for this framework
  improvement.
- The current AST prompt and ALIGN repair authority are complementary; no
  genuine contradiction presently requires an immediate steer.
- Consider the documentation-only framework amendment only after AST
  completes and ALIGN returns phase zero for owner review.
- A contract-interface comparison may then precede repair, but exact
  composed-estimator closure must bind the relevant repaired and re-audited
  implementation identities.
- The framework amendment must be in place before downstream RTC dispatch and
  before any production or conformity claim that a repaired ALIGN/AST pair
  closes detector-sample-to-sky astrometry.

## FRAMEWORK-COMP-D003 — Pilot composition closure

Decision: approved.

Use `SCI-COMP-ALIGN-AST-001` as the first composition-closure pilot, with the
named conditional operator

\[
A_{\mathrm{detector\rightarrow sky}}
  = A_{\mathrm{AST}} \circ A_{\mathrm{ALIGN}}.
\]

The pilot owns exact producer-codomain/consumer-domain compatibility and joint
falsification only. It owns neither ALIGN nor AST implementation and does not
introduce a third estimator.

Treat AST-to-MAP as a separate seam and later closure identity rather than
expanding `SCI-COMP-ALIGN-AST-001` into a three-package audit. The pilot may
name MAP-facing validity and registration consequences, but it cannot close
the AST-to-MAP contract.

Use two stages within the eventual closure record:

1. contract-interface compatibility after the package audits and owner
   dispositions; and
2. exact-implementation evidence closure after applicable repairs and fresh
   re-audits.

## FRAMEWORK-COMP-D004 — Status and ledger model

Decision: approved.

Do not add a fifth package-wide status axis. Interface compatibility and
composed-estimator closure are relationship- and claim-specific.

The future documentation-only amendment may use:

- existing implementation status for local exact-SHA conformity;
- dependency/interface gates for producer-consumer compatibility;
- named composition-closure records for end-to-end estimator claims; and
- existing production status for operational authorization.

A minimal optional `composition_closures` registry or dependency-level closure
references may reuse `open`, `conditioned`, `satisfied`, and `superseded`.
No ledger schema change is authorized by this decision alone.

## Held decisions

### FRAMEWORK-COMP-D005 — Scientific definitions and tolerances

Status: held for project-owner approval when the applicable AST and ALIGN
evidence is available.

The later decision must address only unresolved scientific facts and
preregistered tolerances, including as applicable:

- the exact physical event and time scale denoted by `TelUtc`;
- integration-start versus integration-midpoint semantics;
- latency sign, magnitude or bound, reference, and application stage;
- allowed predicted sky displacement and two-observation registration error;
- validity/support propagation through AST and into MAP-facing products; and
- source-crossing, centroid, and PSF compatibility tolerances derived from
  existing empirical repeatability rather than selected after candidate
  results are viewed.

### FRAMEWORK-COMP-D006 — Documentation patch and closure launch

Status: held for later explicit project-owner authorization.

Until approved, do not:

- amend the framework README, templates, handoff rules, or canonical ledger;
- create the composition-closure template or canonical closure registry;
- alter any frozen package artifact;
- launch `SCI-COMP-ALIGN-AST-001`; or
- use this policy decision as repair, re-audit, Unity, integration, conformity,
  or production authority.

## Return trigger

The coordinator should return to the project owner when both of these are
available:

1. the completed, committed `SCI-AST-001` audit and its owner-decision list;
2. the completed `SCI-ALIGN-001` phase-zero evidence package and owner-review
   brief.

At that point, present only the evidence-backed `FRAMEWORK-COMP-D005`
scientific choices and the exact documentation-only
`FRAMEWORK-COMP-D006` patch/dispatch scope for approval. Preserve all active
branch, SHA, digest, quarantine, and Unity-access boundaries in the meantime.
