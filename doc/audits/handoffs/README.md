# Cross-audit handoff registry

This directory is the canonical, coordinator-maintained exchange for Citlali
scientific-contract audit packages. Each package has a logical inbox at
`doc/audits/handoffs/<TARGET-PACKAGE-ID>/`; the directory is created when its
first handoff is integrated.

## Record ownership

- A source auditor proposes a handoff on the source audit branch.
- The audit coordinator verifies and integrates the record here.
- A recipient auditor proposes a disposition on the recipient audit or
  re-audit branch.
- The coordinator records the accepted disposition in the canonical record.
- No audit edits another audit's report, branch, worktree, or canonical inbox.

The reusable record is
`doc/audits/templates/CROSS_AUDIT_HANDOFF_TEMPLATE.yaml`. Use one source and
one target per file. The stable record ID and path are:

```text
<TARGET-PACKAGE-ID>-XAUD-NNN
doc/audits/handoffs/<TARGET-PACKAGE-ID>/<TARGET-PACKAGE-ID>-XAUD-NNN.yaml
```

Do not duplicate the record into a physical outbox. The source package lists
the emitted ID in its audit and ledger proposal; the target directory is the
durable inbox.

Freeze each dispatch set with
`doc/audits/templates/CROSS_AUDIT_INBOX_MANIFEST_TEMPLATE.yaml`. The manifest
records the canonical registry commit, every record path and digest, and the
pre-core/post-core partition.

## Required semantics

Every submission states:

- exact source and target package IDs;
- originating audit, candidate, evidence, and finding identities as
  applicable;
- a bounded claim, what the evidence supports, and what it does not establish;
- requested recipient actions;
- exact artifact paths and SHA-256 digests, or an explicit hash-only/missing
  availability classification;
- priority, evidence basis, and review phase; and
- supersession relationships.

`pre_core_authority` is allowed only for an approved contract, owner decision,
or canonical dependency fact identified by exact commit. All observations,
downstream manifestations, suspected or derived defects, questions, and
unapproved proposals are `post_core_evidence` and stay closed until the
independent core is frozen.

A recipient disposition does not rewrite the immutable submission. It records
the target audit commit, rationale, affected findings/dependencies, and one or
more controlled actions. A resolved handoff cannot by itself approve a
contract, close a finding, or authorize production.

## Dispatch and late-arrival rule

Before an audit is dispatched, the coordinator freezes an inbox manifest with
the canonical commit, every handoff ID and file digest, and its review phase.
The auditor records which pre-core records were opened, freezes the independent
core, then records the first opening of post-core evidence.

A later handoff is not silently added to the active evidence set. The
coordinator chooses and records one of three routes:

1. hold it for a fresh re-audit;
2. authorize a dated amendment that preserves the frozen core and records the
   new exposure; or
3. impose an urgent operational restriction outside the audit while retaining
   the handoff for later scientific disposition.

## First integrated record

`SCI-MAP-001/SCI-MAP-001-XAUD-001.yaml` records the bounded convolve-to-MAP
raw-validity exchange. It is deliberately post-core evidence: it establishes a
downstream consequence of an unresolved raw-map handoff while explicitly not
establishing a raw numerical regression, a valid filtering estimator, or
production authorization.

## SCI-MAP-001 final recipient dispositions

The completed final re-audit at
`8fc716557ca78b0d220200a92be46fa3545797e9` resolves the late CAL
`SCI-MAP-001-XAUD-002` and AST `SCI-MAP-001-XAUD-003` recipient dispositions.
Both records are incorporated as dependency, interface-test, and consumer-
restriction inputs to SCI-MAP-001-F013. Their source submissions remain
immutable. The resolutions do not close SCI-CAL-001 or SCI-AST-001, establish
upstream production eligibility, or expand MAP production use; F013 remains
open and conditioned, and production remains `existing_use_only`.
