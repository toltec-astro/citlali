# Citlali Review And Conformance

Status: candidate; not effective until owner-accepted and incorporated into
canonical integration authority

Owner: Citlali project owner

Scope: preflight, reassessment, review, closure, and repository reconciliation
for governed Citlali work

## Review principles

Review MUST produce three separate dispositions:

1. **Scientific and behavioral conformance** — scientific identity, semantics,
   numerical behavior, validity, failure, and evidence.
2. **Architectural conformity and ownership** — dependency direction,
   lifecycle, stage ownership, typed interfaces, and `Engine` boundary.
3. **Repository, branch, and evidence hygiene** — ancestry, scope, dirty state,
   exact identities, reachability, and preservation.

A pass in one category does not imply a pass in another. Tests are evidence,
not architectural authority.

## Risk tiers

### Tier 1 — mandatory core

Use for ordinary local changes that do not alter scientific meaning,
architecture, cross-stage communication, persistent state/schema, hot paths,
or multiple packages. The task MUST report:

- applicable governance read;
- exact base, branch/worktree, and initial dirt;
- bounded purpose and exclusions;
- focused tests and actual environment;
- final changed paths, exact commit, and clean/dirty state; and
- the three review-category dispositions.

### Tier 2 — full conformance

Required for scientific, architectural, cross-stage, persistent-state, schema,
performance-sensitive, multi-package, build-environment, integration, or
evidence-cleanup work. Tier 2 includes the mandatory core plus:

- complete subject-specific authority map;
- explicit owners, producer/consumer interfaces, and lifecycle analysis;
- risk and review-trigger register;
- focused and broader tests reported separately;
- affected-mode or observational-validation disposition;
- time/memory/determinism evidence where relevant;
- exact artifact, source, environment, and evidence identities;
- independent fresh-context exact-SHA review; and
- owner decisions required before integration, push, activation, or cleanup.

## Preflight template

```text
WORK ORDER / PREFLIGHT
Purpose:
Owner:
Risk tier:
Applicable governance read:
Current sequencing authority:
Scientific authority:
Architectural authority:
Exact canonical base:
Branch and worktree:
Initial staged / unstaged / untracked state:
WIP slot:
Owned product/interface and lifecycle:
Included scope:
Excluded scope:
Expected changed paths or ownership areas:
Focused gates:
Broader gates:
Affected-mode / representative-environment gate:
Review triggers:
Stop conditions:
Integration / push / activation / cleanup authority:
```

## In-progress reassessment

A trigger requires conscious reassessment and a recorded conclusion. It does
not automatically require owner intervention when the work can remain
conformant and bounded.

Mandatory triggers include:

- a missing or contradictory scientific contract;
- a newly discovered cross-stage dependency or owner;
- proposed growth of `Engine` or orchestration science;
- a prerequisite defect outside the stated ownership area;
- a new persistent field, schema, identity, cause, or lifecycle;
- unexpected numerical, failure, logging, performance, or memory behavior;
- need for a sibling-repository change;
- a dirty or moving base, branch, or evidence source;
- a second branch or WIP-budget conflict;
- representative-build or observational evidence that differs from local
  results; or
- a cleanup action whose target is not proven reachable and preserved.

The reassessment MUST record: trigger, observed evidence, affected authority,
whether scope remains bounded, required gate changes, and one disposition:
continue, repair within scope, split to a new work order, preserve and stop, or
request an owner exception.

## Completion and conformance template

```text
COMPLETION / CONFORMANCE
Disposition: candidate | evidence after failed gate | accepted integration record
Exact base / parent / candidate / tree SHAs:
Changed paths and content digests:
Scientific authority implemented or preserved:
Architectural owner and boundary:
Focused results and environment:
Broader results and environment:
Representative Spack / Unity result: performed | not performed | not required
Affected-mode result or reason not triggered:
Scientific and behavioral conformance:
Architectural conformity and ownership:
Repository, branch, and evidence hygiene:
Intentional scientific changes:
Scope exclusions and retained limitations:
Unexpected error-level output:
Independent review SHA and disposition:
Integration / push / activation / production status:
Final worktree state:
Next owner decision:
```

## Independent review template

The reviewer MUST begin from fresh context, operate read-only, and review one
exact full candidate SHA. The writer MUST NOT approve their own change.

```text
INDEPENDENT EXACT-SHA REVIEW
Candidate SHA and tree:
Parent/base and ancestry:
Work-order scope:
Authority sources read:
Changed-path inventory:
Scientific/behavioral findings: blocker | major | minor | none
Architecture/ownership findings: blocker | major | minor | none
Repository/evidence findings: blocker | major | minor | none
Gate reproduction and environment:
Scope-exclusion verification:
Verdict: pass | pass with recorded limitations | repair required | reject
```

A review finding MUST identify evidence, affected authority, consequence, and
smallest bounded repair. Style preference alone MUST NOT broaden the candidate.
After a repair, review MUST bind the new exact SHA; a previous verdict does not
transfer automatically.

## Governance exception or amendment template

```text
GOVERNANCE EXCEPTION / AMENDMENT
Owner directive:
Affected rule and document:
Exact scope:
Reason and evidence:
Temporary or permanent:
Duration / exit condition:
Risks and retained obligations:
Independent review:
Canonical incorporation and ledger record:
```

## Branch, worktree, and evidence cleanup template

No cleanup operation may be executed from this template without separate
owner authorization.

```text
EVIDENCE-PRESERVING CLEANUP
Item and exact ref/SHA/path:
Worktree state and fully enumerated untracked paths:
Unique commits and reachability proof:
Stashes / bundles / tags / snapshot or turn-diff refs:
Authority or evidence role:
Preservation target and digest:
Local/remote status:
Recommendation:
Exact proposed operation:
Expected post-operation state:
Owner approval:
Independent verification:
```

Recommended dispositions are: retain canonical, retain active, integrate,
rename, tag/archive/bundle, preserve as evidence, close worktree but retain
ref, candidate for later local deletion, candidate for later remote deletion,
or unresolved. Remote-deletion proposals MUST be isolated from ordinary push
operations.

## Mechanical checks

Repositories SHOULD add small deterministic checks only after the human policy
is stable. Suitable checks include governance-link discovery, required
preflight/completion fields, exact-SHA form, document digest recording,
single-candidate review identity, clean-tree evidence, and branch/WIP inventory.
Mechanical checks MUST NOT decide scientific authority or infer branch safety
from names alone.

```text
MECHANICAL CHECK PLAN / RESULT
Check identity and owner:
Exact inputs / base / candidate:
Deterministic command or rule:
Expected result:
Actual result and artifact identity:
Known limitations:
Scientific-authority decision: NOT PERFORMED
Branch-safety or deletion decision: NOT PERFORMED
```

## Adversarial examples

### Scientific glue proposed in `Engine`

Reject the location even if focused tests pass. Name the scientific owner and
pass a typed product through orchestration. Growth of `Engine` is an
architecture finding separate from numerical correctness.

### RTC reaches into AST state

Stop the reach-through. AST emits a typed, occurrence-bound sky-motion product;
RTC consumes it without selecting or redefining pointing authority.

### A prerequisite defect appears mid-increment

Diagnose and record it. Continue only if repair is owned, bounded, and leaves
the approved outcome unchanged; otherwise preserve the candidate and split a
new work order.

### The convenient checkout is dirty

Do not clean or reuse it. Inventory its evidence and create the authorized
clean worktree from the exact base.

### A sibling prerequisite commit is needed

Do not edit the sibling implicitly. Record the exact prerequisite and obtain a
separate owner-authorized work order or use an already accepted immutable
dependency.

### A second module branch is requested

Reassess the WIP budget. Complete, park, or close the current probe, or obtain a
bounded exception; do not hide the second probe in the spine branch.

### A reusable abstraction has only hypothetical consumers

Do not add it. Implement the concrete owned interface and revisit abstraction
only after real consumers and duplication provide evidence.

### The scientific contract is ambiguous

Codex may enumerate consequences and evidence but must stop before selecting
or modifying scientific authority.

### Branch-budget pressure comes from evidence refs

Classify correctly. Read-only audits, retained evidence, canonical integration,
build adaptation, and pre-implementation contracts do not consume the
application budget, though their dirty state and preservation still require
governance.

### Tests pass under the wrong owner

Report behavioral pass and architectural failure separately. Passing output
does not authorize cross-stage ownership or `Engine` growth.

### Cleanup would remove the only copy of evidence

Stop. Verify unique commits, dirty and untracked content, stashes, bundles,
special refs, tags, and remote reachability; create an accepted preservation
target before proposing deletion.
