# Scientific-Contract Program Manager Handoff

Date: `2026-09-04`

Status: **manager-transition record; scientific authority and implementation
state are unchanged**

This record transfers management of the Citlali scientific-contract program
to a fresh Codex task without relying on the preceding long conversation.  It
is a routing and continuity document.  It is not a scientific contract,
scientific-owner decision, implementation-conformity finding, validation
result, readiness claim, or production authorization.

## Start Here

The incoming manager must read, in this order:

1. repository-local `AGENTS.md`;
2. this handoff and
   `HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM_MANIFEST.json`;
3. `doc/scientific_contracts/README.md`;
4. `doc/scientific_contracts/INDEX.md`;
5. `doc/scientific_contracts/PRIOR_WORK_REGISTRY.md`;
6. `doc/scientific_contracts/PILOT_PROCESS_REVIEW_2026-08-16.md`;
7. `doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md`;
8. `doc/SCIENTIFIC_CONVENTIONS.md`;
9. `doc/REFACTOR_STATUS.md`; and
10. the exact candidate sources named under **Immediate Review Candidate**.

Run `handoff/verify_scientific_contract_program_handoff_2026_09_04.py`
before acting.  Compare all moving refs and worktree states with the dated
snapshot below and report drift.  Commit identities and content digests are
the stable evidence; branch names alone are not.

## Governing Approach That Must Not Drift

The program charter controls the complete method.  The compact manager view
is:

1. Begin every package by citing the program charter, applicable roadmap,
   predecessor authorities, and package-specific prior-work recovery.
2. Search existing contracts, decisions, derivations, handoffs, and evidence
   before doing new work.  Classify each recovered item and explicitly adopt,
   cite, abstract, supersede, defer, or exclude it.  Rewording established
   reasoning is not progress.
3. Keep implementation-informed Stage A discovery separate from the
   implementation-blind Stage B scientific author channel.
4. Obtain owner approval for the sanitized scope and allowed author sources
   before Stage B.
5. Maintain one scientific authority through a scientist-facing rationale,
   an engineering-facing conformance view, one shared formal core, and an
   exact crosswalk/owner-decision ledger.
6. Do not let implementation, tests, validation results, convenient field
   names, or historical behavior author scientific meaning.
7. Preserve producer, transformer, and consumer ownership.  VAL evaluates and
   registers owner-authored policy; VAL does not author that policy.
8. Treat absence and unavailability honestly.  Missing response, covariance,
   uncertainty, or policy is not zero, unity, independence, or permission.
9. Freeze scientific authority separately from implementation conformity,
   numerical validation, achieved performance, readiness, and production.
10. Never silently edit frozen science.  A genuine scientific correction
    requires an explicitly authorized versioned successor or erratum.

The current charter still specifies a fresh GPT-5.6 Ultra task for a Stage B
scientific author.  This manager transition to GPT-6 Astra does not amend that
standing author-channel rule.  Any change to the Stage B model policy requires
separate owner authorization and a charter update.

## Accepted Program State At This Snapshot

Canonical accepted application/integration ancestry is
`codex/refactor-mainline` at commit
`9f42d348298d76c5d5145aaf0c3eace1f3e154c1`, tree
`e51f22760c64454ce7233c45dd740aa710777bae`.  The owner reported that this
state was pushed.  The local remote-tracking ref is stale and no network or
remote verification was performed for this handoff.

The exact package-level status, open questions, and next package-local action
are recorded in `doc/scientific_contracts/INDEX.md`.  At this snapshot:

- SCI-CAL, SCI-MAP, SCI-JINC, SCI-BEAM, SCI-ALIGN, SCI-AST, SCI-RTC,
  SCI-PTC, SCI-VAL, SCI-NOI, SCI-FLT-FIXED, SCI-FLT-MATCHED, and SCI-POINT
  have the frozen or conditionally frozen scientific-authority states stated
  by their package-local records and the index;
- frozen authority does not establish implementation conformity, validation,
  achieved response/covariance/performance, readiness, or production use;
- blank-field source detection and faint distributed-source fitting remain
  deferred as a future SRC package;
- SCI-OOF remains eligible for a future recovery-first Stage A, but work was
  deliberately paused until the map-space audit knowledge is operationalized;
  and
- SCI-FRUIT remains an independent active Stage A/empirical-lane development
  line and is not represented by canonical mainline alone.

## Repository And Worktree Snapshot

| Role | Branch / worktree | Exact snapshot | Disposition |
| --- | --- | --- | --- |
| Canonical accepted line before this handoff commit | `codex/refactor-mainline`; `/private/tmp/citlali-refactor-mainline-after-map-space-repair-2026-09-04` | commit `9f42d348298d76c5d5145aaf0c3eace1f3e154c1`; tree `e51f22760c64454ce7233c45dd740aa710777bae`; clean before handoff preparation | Manager authority and integration base |
| MAP-space contract-to-implementation candidate | `codex/map-space-contract-to-implementation-conformance-001`; `/Users/gwilson/.codex/worktrees/8bf8/citlali-refactor` | commit `93c2b4591bb5d0cf8efe4491975c31e5f8fb5903`; tree `e0b51383cdeb4ad318d3548b05ad803dd9ef1cf4`; direct clean child of canonical snapshot | Review candidate only; not accepted or integrated |
| Active SCI-FRUIT line | `codex/sci-fruit-v0.1-empirical-lane`; `/Users/gwilson/.codex/worktrees/4c31/citlali-refactor` | commit `d39d4685baa0aeda036305031491121ce97008ec`; tree `4034b9b3037ec577dfaea7078c93561953edd364`; two untracked owner-review archives | Independent and moving; inspect separately; do not merge or clean by inference |
| Historical ALIGN checkout | `codex/sci-align-001-lissajous-timestream-fit`; `/Users/gwilson/GitHub/citlali-refactor` | commit `353b11887ff04dfd7bca12915917495f81a587fa`; tree `0212e37395c86cdd34b4d824ba6ac91bb5a5220e`; tracked and extensive untracked material | Historical evidence; protect; never use as the current project checkout |

At the snapshot, FRUIT contained 93 commits not reachable from canonical
mainline, while canonical contained 188 commits not reachable from FRUIT.
These counts establish divergence, not scientific status.  FRUIT's two
untracked review archives were:

- `SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz`; and
- `SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz`.

Do not delete, move, unpack into tracked paths, or infer authority from either
archive without a separate request.

## Immediate Review Candidate

`MAP-SPACE-CONTRACT-TO-IMPLEMENTATION-CONFORMANCE-001` is a bounded,
documentation-only source study at commit `93c2b459...`.  It inspected the
exact canonical snapshot above and changed only its 12 packet files.  Its
durable verifier passes its present structural checks.  The study's central
conclusion is credible but has not received owner acceptance:

> the existing source contains substantial predecessor behavior, while no
> complete end-to-end route yet implements the exact frozen map-space product
> graph.

This means legacy implementation identities must not be advertised as the
new frozen products.  It does not mean the current reduction software is
scientifically useless or operationally broken.

The manager review found four bounded documentation defects that must be
repaired before owner disposition:

1. The route table contains all 32 edges, but `MSP-E027` was omitted from the
   summary count.  The correct edge-state count is 2 source-level conformant,
   4 legacy, 5 unavailable by design, 10 missing authority, 1 missing
   implementation, 9 contradictory, and 1 not applicable.  Update both
   summaries and extend the verifier to derive and validate the totals.
2. `FRUIT_ATTACHMENT_ENVELOPE.md` incorrectly makes MSP-P009/MATCHED the only
   possible FRUIT input.  The active FRUIT authority treats ordinary MAP,
   JINC, FLT-FIXED, and FLT-MATCHED as separately reviewable candidate parent
   families.  Because FRUIT was excluded from the source study, the packet may
   preserve MSP-E030 as one non-exhaustive deferred envelope but may not author
   an exclusive FRUIT input policy.
3. `CTI-OD-007` repeats inherited program sequencing as an open decision.
   Preserve FRUIT independently and OOF as envelope-only for this study;
   record that disposition as inherited/closed rather than reopening it.
4. The recommended repair sequence names a frozen MAP observation-bundle
   unit, but the backlog table does not define that unit.  Add an explicit
   bounded unit, or explicitly and completely bind its requirements to an
   existing unit.  Do not hide MAP bundle identity, admission,
   response/covariance state, coefficient binding, or unique-original
   exposure inside an incomplete exposure-only item.

The exact candidate artifact digests are recorded in the companion manifest.
The primary review files are:

- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/FINAL_REPORT.md`;
- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/OWNER_DECISION_LEDGER.md`;
- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/PRIORITIZED_REPAIR_BACKLOG.md`;
- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/ROUTE_AVAILABILITY_CLASSIFICATION.md`;
- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/FRUIT_ATTACHMENT_ENVELOPE.md`; and
- `doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/verify_packet.py`.

## Incoming Manager Assignment

The fresh manager task must first:

1. verify this handoff and independently refresh the four branch/worktree
   states;
2. report discrepancies before interpreting them;
3. prepare a bounded successor repair containing only the four documentation
   corrections above;
4. preserve all product, edge, trace, finding, and owner-decision identifiers;
5. change no frozen package, scientific convention, application source,
   configuration, Registry, validation record, FRUIT artifact, or ALIGN
   artifact;
6. rerun the corrected packet verifier and repository diff checks; and
7. stop for owner review without merge, push, canonical movement, branch
   deletion, or cleanup.

The original candidate branch and commit must remain recoverable.  The
incoming task may create a dedicated successor branch or isolated temporary
worktree.  It must not treat this handoff as authority to rewrite history or
alter another active task's worktree.

After the corrected candidate is owner-accepted, the next manager actions are:

1. integrate the exact accepted documentation candidate through an explicitly
   reviewed route;
2. update manager indexes/status only after acceptance;
3. let the owner push canonical mainline;
4. walk through the genuinely open `CTI-OD-001` through `CTI-OD-006`
   decisions; and
5. authorize at most one bounded implementation-mapping unit at a time.

SCI-OOF Stage A remains deferred until this operationalization step has an
owner disposition.  SCI-FRUIT continues independently.

## Pickup Acceptance Gate

The incoming task must return a pickup report that states:

- its exact branch, HEAD, parent/base, and tree;
- the verified handoff base and handoff commit;
- the refreshed canonical, candidate, FRUIT, and ALIGN states;
- the authority order it will follow;
- the four bounded candidate repairs in its own words;
- the no-mutation boundaries; and
- the exact first action it proposes or completed.

Archive the preceding manager conversation only after that report agrees with
this handoff or explicitly resolves every discrepancy.  Retain the archived
conversation as historical evidence, not as current scientific authority.
