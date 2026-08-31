# Citlali Engineering Governance

Status: candidate; not effective until owner-accepted and incorporated into
canonical integration authority

Owner: Citlali project owner

Scope: repository-wide engineering, implementation, review, integration, and
evidence-preservation work

## Normative language

**MUST** and **MUST NOT** state requirements. **SHOULD** states the default
that may be departed from only with a recorded reason and proportionate
review. **MAY** states an allowed choice, not an obligation.

## Discovery and effectiveness

1. Every governed task MUST read and report this document during preflight.
   Timestream Successor work MUST also read and report
   [`TIMESTREAM_SUCCESSOR_GOVERNANCE.md`](TIMESTREAM_SUCCESSOR_GOVERNANCE.md).
   Gate selection and closure MUST use
   [`REVIEW_AND_CONFORMANCE.md`](REVIEW_AND_CONFORMANCE.md).
2. A governance candidate is not effective merely because it exists on a
   branch. It becomes effective only after owner acceptance and incorporation
   into the canonical application integration authority.
3. The incorporation record in `doc/INTEGRATION_LEDGER.md` MUST name the
   accepted full commit SHA and SHA-256 digest of every normative governance
   document. This avoids a circular "effective from the commit containing this
   text" declaration.
4. A durable owner decision MUST be incorporated into the appropriate
   repository authority. Conversation history alone MUST NOT remain its only
   permanent source.
5. Governance changes MUST occur under a dedicated owner-authorized governance
   work order. Ordinary feature work MUST NOT amend governance opportunistically.

## Subject-specific authority

Authority is resolved by subject; there is no single scalar hierarchy that
turns one kind of acceptance into another.

| Subject | Authority | Does not establish |
| --- | --- | --- |
| Current sequencing and WIP | `doc/REFACTOR_STATUS.md`, then `doc/INTEGRATION_LEDGER.md` | Scientific correctness or production use |
| Architecture and ownership | `doc/ARCHITECTURE.md` and accepted records in `doc/adr/` | A new scientific method |
| Scientific meaning | Accepted scientific contracts, owner dispositions, `doc/SCIENTIFIC_CONVENTIONS.md`, and executable product/config contracts | Implementation conformance |
| Implementation evidence | Exact source/test commits and reproducible focused results | Scientific or architectural acceptance |
| Executable acceptance evidence | `validation/validation_profiles.json`, `validation/accepted_runs.json`, `validation/intended_science_changes.json`, and exact named evidence manifests | Integration or production authorization |
| Bounded work-order scope | The current owner-approved work order and its exact recorded base | Permission outside that scope |
| Integration acceptance | Owner disposition recorded on canonical application ancestry | Production authorization |
| Production authorization | Explicit owner operational disposition | Broader science, release, or support claims |

When authorities conflict, the task MUST report the conflict by subject and
stop if conformity cannot be maintained without a consequential scientific,
architectural, ancestry, scope, safety, or exception decision. Existing code,
tests, branch names, commit recency, or passing results MUST NOT silently
override accepted authority.

Codex MAY evaluate implementation against accepted scientific authority. It
MUST NOT create, alter, waive, or select among unresolved scientific authority.

## Owner directives, exceptions, and amendments

An owner directive MAY temporarily waive a governance rule only when it
identifies:

- the affected rule;
- the exact scope;
- the reason;
- the duration or observable exit condition; and
- any evidence or cleanup obligations retained by the waiver.

Silence about a rule is not a waiver. A temporary waiver MUST be recorded in
the task completion report and, if consequential beyond that task, in the
appropriate durable repository authority. Permanent policy changes require a
governance amendment, independent review, owner acceptance, and canonical
incorporation.

## Architectural ownership

### Orchestration and `Engine`

Session and pipeline orchestration MUST own sequencing, lifecycle control,
failure propagation, and invocation of typed stage boundaries.

`Engine` remains an active transitional compatibility aggregate frozen for
growth. New work MUST NOT add to `Engine`:

- scientific calculations or algorithms;
- thresholds, admission decisions, or scientific policy;
- cross-stage ownership or stage-specific learned state;
- process-lifetime run or observation state; or
- new public cross-cutting mutable state.

Extraction from `Engine` MUST have a named lifecycle or scientific owner, a
bounded typed contract or interface, one-way compatibility where necessary,
and validation proportionate to the affected behavior. Moving code without a
clear owner is not an architectural improvement.

### Stage and cross-stage communication

Each scientific behavior MUST have one named scientific owner. Cross-stage
facts MUST travel through concrete typed products or interfaces that state
identity, units, frame, shape, indexing, validity, causes, provenance, and
lifetime where applicable.

A stage component MUST be testable at its typed boundary without constructing
the full `Engine`. Focused tests SHOULD exercise the owner, contract, and
failure behavior directly; a whole-application test alone is not boundary
evidence.

Stages MUST NOT reach through `Engine` or another stage's internals to obtain
scientific facts. Cross-stage policy glue MUST NOT be placed in orchestration.
One-way adapters MAY preserve a validated compatibility boundary; they MUST NOT
create bidirectional authority or synchronize realized state back into a
request.

A product or interface SHOULD be domain-specific and have an identified
producer and consumer. A speculative registry, universal processor interface,
generic data/context/node framework, or abstraction serving only hypothetical
consumers MUST NOT be introduced without a separately approved architectural
case.

## Bounded work and WIP

Every implementation change MUST have one bounded work order stating:

- owner, purpose, and scope;
- exact canonical base and relevant authority identities;
- owned branch and worktree;
- included and excluded behavior;
- expected changed paths or ownership areas;
- risk tier, review triggers, gates, and stop conditions; and
- integration, push, activation, and cleanup authority.

An approved work order authorizes creation and ordinary use of its one bounded
branch and clean worktree without repeated permission. It does not authorize a
second branch, scope expansion, canonical integration, push, production
activation, or cleanup unless those operations are stated explicitly.

**Active** means implementation or repair may still change application source
under the work order. A branch is not active merely because it exists. A
read-only audit, owner-retained evidence ref, separately governed build lane,
canonical integration operation, or pre-implementation scientific-contract
authoring task does not consume an application-implementation slot.

The Timestream Successor budget is defined in its workstream governance. Other
programs SHOULD use the smallest number of simultaneous implementation branches
that preserves safe independent progress. Branch-budget pressure is a
reassessment trigger, not permission to hide work in an existing branch.

## Git, worktrees, and repository authority

1. The canonical application integration branch is the only application
   integration authority. Topic, audit, repair, coordination, evidence, and
   build branches MUST NOT be described as parallel application mainlines.
2. One branch and worktree MUST have one declared owner and bounded purpose.
   Shared dirty checkouts MUST NOT be repurposed for unrelated work.
3. Feature implementation MUST use a clean worktree based on the exact
   approved commit. Unrelated owner changes MUST remain untouched.
4. Read-only audits SHOULD create no branch. A report that must be committed
   MAY use one bounded evidence or governance branch when authorized.
5. Commits MUST be coherent and tied to the work-order boundary. Commit count
   is not a cadence requirement.
6. Merge, rebase, cherry-pick, ref movement, branch rename, worktree removal,
   push, force-push, remote deletion, clean, prune, stash drop, and destructive
   reset require the authority stated by the work order or a later exact
   operation plan.
7. Branch and path names are locators, not scientific, application, or artifact
   identities. Exact full SHAs and content digests MUST identify durable
   evidence.

## Evidence-preserving cleanup

Rejection, supersession, or inactivity is not evidence that a ref may be
deleted. Before cleanup, the responsible review MUST inventory:

- exact branch/ref tips and unique commits;
- reachability from accepted refs and tags;
- all worktrees and detached HEADs;
- staged, unstaged, and fully enumerated untracked paths;
- stashes, bundles, snapshot refs, turn-diff/evidence refs, and meaningful
  tags;
- local versus cached-remote presence; and
- owner-retained operational or scientific evidence.

Every item MUST receive an explicit disposition: integrate, retain active,
preserve as evidence, tag/archive/bundle, close worktree while retaining the
ref, candidate for later local deletion, candidate for later remote deletion,
or unresolved. Deletion is allowed only after preservation and reachability are
proved and the owner approves the exact operation. Remote changes remain
owner-controlled.

## Build and validation environments

Build-environment identity is a distinct evidence axis. The accepted
representative environment for the Native Integration Baseline V2 campaign is
the exact Spack realization bound by ADR 0014 and
`validation/citlali_v2_spack_validation_authority.json`.

A Homebrew, Conan, cached-dependency, syntax-only, or ad hoc local build MAY
provide focused compilation, compatibility, portability, or regression
evidence. It MUST be labeled with its actual environment and MUST NOT be called
a reproduction of the accepted Spack-backed campaign without an authoritative
exact binding. Build-adaptation work has separate scope and WIP, but neither it
nor application work may silently change the other's scientific or persistent
contracts.

## Minimum task discipline

Every governed change MUST complete the preflight, reassessment, review, and
completion rules in `REVIEW_AND_CONFORMANCE.md`. An ordinary low-risk change
uses the mandatory core. Scientific, architectural, cross-stage,
persistent-state, schema, performance-sensitive, or multi-package work uses the
full report and independent exact-SHA review.
