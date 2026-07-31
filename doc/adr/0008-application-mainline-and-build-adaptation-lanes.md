# ADR 0008: Application Mainline And Build Adaptation Lanes

- **Status:** Accepted
- **Recorded:** 2026-07-31
- **Decision owners:** Citlali project owner and engineering

## Context

Citlali application development must continue while the successor Conan 2
build is completed across Tula CMake, Tula, Kidscpp, and Citlali. The active
application history had accumulated on a topic branch whose name no longer
described its contents. Merging the external Citlali milestone wholesale
would combine build migration, compatibility work, and validated application
changes in one difficult review.

The accepted Adapt decision already requires the full refactored application
to be carried into the new build model without replacing it with the smaller
upstream milestone. This ADR defines the Git workflow that enforces that
boundary while allowing both workstreams to progress.

## Decision

`codex/refactor-mainline` is the canonical application integration branch. It
contains the current refactored application, scientific behavior, operational
contracts, tests, configuration, and accepted validation history.

`codex/conan2-adaptation` is an isolated build-integration branch and worktree
created from the exact application mainline. It may change build definitions,
dependency ownership, package recipes, compiler profiles, generated build
metadata, and narrowly required compatibility boundaries. It must not contain
unrelated numerical or scientific algorithm changes.

Application development continues on the mainline or on short-lived topic
branches based on it. The Conan 2 lane regularly incorporates the application
mainline so that it adapts the application being developed rather than a
frozen historical snapshot. The completed build lane returns to mainline only
after the gates in the build integration review pass.

Do not merge `citlali/v4.x_conan2` wholesale. Import its build architecture
deliberately, consume Tula and Kidscpp as versioned dependencies, and review
upstream Citlali source fixes individually. Preserve the existing build as a
fallback until the successor has passed its local, package-consumer, Unity,
provenance, and same-SHA validation gates.

Historical topic branches remain immutable forensic pointers. They are not
parallel application authorities.

## Integration Rules

1. Every workstream records its branch, head, purpose, owner, and gate state in
   `doc/INTEGRATION_LEDGER.md`.
2. A scientific change and a build-system migration are separate commits and
   separate reviews even when one exposes the need for the other.
3. Upstream first-party revisions are pinned by exact identity during review;
   branch names alone are not sufficient evidence.
4. Updating the Conan 2 lane from mainline must not weaken tests, provenance,
   output contracts, or configuration gates to resolve an integration issue.
5. The existing build is removed only in a later cleanup after the successor
   has operational evidence and rollback remains straightforward.

## Consequences

- Application work need not stop while build infrastructure evolves.
- The adaptation branch may need routine conflict resolution when mainline
  adds sources, tests, or generated inputs; this is expected maintenance, not
  a reason to freeze application development.
- The final build integration is a bounded architectural change whose
  numerical behavior can be compared against the same application source.
- Branch names and the ledger, rather than thread memory, identify the current
  authority for each workstream.
- Existing committed study evidence remains in place. A separate housekeeping
  decision may define future evidence storage without rewriting validated
  history as part of build integration.

## Rejected Alternatives

- **Freeze Citlali until the build is ready:** blocks required application
  development for an external schedule without reducing final integration
  risk.
- **Merge the upstream Citlali branch wholesale:** risks replacing validated
  application behavior with a smaller milestone and obscures source changes
  among build changes.
- **Develop both streams in one checkout:** allows branch switches and partial
  build edits to interfere with application work.
- **Keep using topic branches as rolling mainlines:** makes authority and
  validation status depend on historical context rather than explicit policy.

## Supersession

Review this decision after the Conan 2 successor is the accepted operational
build and the fallback build has been retired. A future release-branch policy
may replace the two-lane adaptation workflow, but it must preserve explicit
application authority and exact validated source identity.

## Evidence

- [`../INTEGRATION_LEDGER.md`](../INTEGRATION_LEDGER.md)
- [`../TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`](../TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md)
- [`../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md`](../TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md)
- [`../PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md`](../PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md)
